import torch
from torch import nn
import skimage
import imageio.v3 as iio
import numpy as np
import scipy
import matplotlib.pyplot as plt

from . import utils
from .models import UNet, EdgeNet
from .datasets import EdgeDataset

import cv2

import os
import time
import itertools

import argparse


@torch.no_grad()
def infer_image(model: nn.Module, img: np.array, confidence_thr: float = 0.6,
                check_half=False, device: str | torch.device = 'cuda', show_heatmap: bool = False):
    img_0 = img
    img: torch.Tensor = torch.from_numpy(img[None, ...].astype(np.float32)).to(device)

    if check_half:
        half_img0 = skimage.transform.rescale(img_0, 0.5, anti_aliasing=True)
        half_img: torch.Tensor = torch.from_numpy(half_img0[None, ...].astype(np.float32)).to(device)

    out_img = np.zeros_like(img_0)
    out_img_max = np.zeros_like(img_0)
    out_img_patch_cnt = np.zeros_like(img_0)

    # orig size

    def run(input_img: torch.Tensor, input_scale: int):
        # input_scale: -1 for half size, 1 for double size
        stride = 32
        block_size = 64

        patches = input_img.unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)

        out_patches = torch.empty((*patches.shape[:2], 2, *patches.shape[3:]), dtype=patches.dtype, device=patches.device)
        for i in range(patches.shape[0]):
            out_patches[i] = model(patches[i, :, :, :, :])
        out_patches = out_patches.cpu().numpy()

        coords = np.meshgrid(np.arange(0, patches.shape[1])*stride, np.arange(0, patches.shape[0])*stride)
        for i in range(patches.shape[0] - 1):
            for j in range(patches.shape[1] - 1):
                out_patch = out_patches[i, j, 0]

                r_s = i*stride
                r_e = r_s + block_size
                c_s = j*stride
                c_e = c_s + block_size

                if input_scale > 0:
                    r_s, r_e, c_s, c_e = [x >> input_scale for x in [r_s, r_e, c_s, c_e]]
                    out_patch = skimage.transform.rescale(out_patch, 2**-input_scale, anti_aliasing=True)
                elif input_scale < 0:
                    r_s, r_e, c_s, c_e = [x << -input_scale for x in [r_s, r_e, c_s, c_e]]
                    out_patch = skimage.transform.rescale(out_patch, 2**-input_scale, anti_aliasing=True)

                out_patch = out_patch[:out_img[r_s:r_e, c_s:c_e].shape[0], :out_img[r_s:r_e, c_s:c_e].shape[1]]

                out_img_patch_cnt[r_s:r_e, c_s:c_e] += 1
                weight_block = out_img_patch_cnt[r_s:r_e, c_s:c_e]

                weight_block = 1 / weight_block
                # print(out_img[r_s:r_e, c_s:c_e].sum())
                # print(weight_block.sum())
                out_img[r_s:r_e, c_s:c_e] *= (1 - weight_block)
                out_img[r_s:r_e, c_s:c_e] += out_patch * weight_block

                max_block = out_img_max[r_s:r_e, c_s:c_e]
                out_img_max[r_s:r_e, c_s:c_e] = np.maximum(out_patch, max_block)

    run(img, 0)

    if check_half:
        run(half_img, -1)

    max_val = 10
    # print(out_img.shape, out_img_max.shape)
    # print(out_img.max(), out_img_max.max())
    out_img = out_img / max_val
    # out_img[out_img>=confidence_thr] = 1
    # out_img[out_img<confidence_thr] = 0

    out_img_max = out_img_max / max_val
    out_img = out_img_max

    if show_heatmap:
        out_img_rotated = skimage.transform.rotate(out_img, -90)
        out_img_rotated = np.fliplr(out_img_rotated)
        plt.imshow(out_img_rotated)
        plt.show()

    _, binary = cv2.threshold(out_img, confidence_thr, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers, confidence = [], []

    # Loop over the contours
    for contour in contours:
        # Get the coordinates of the center of the contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # Add the coordinates to the list
            markers.append((cX, cY))
            confidence.append(out_img[cY, cX])

    # hit = val[:,:,0] > confidence_thr
    # markers = val[hit,1:]
    # confidence = val[hit,0]

    return np.asarray(markers), np.asarray(confidence), np.asarray(out_img)


@torch.no_grad()
def refine_checkpoints(edge_model: nn.Module, frame: np.array, markers: np.ndarray, marker_confidence: np.ndarray,
                       edge_confidence_thr: float = 0.6, marker_distance_thr: int = 65, k_nearest: int = 8,
                       device: str | torch.device = 'cuda'):

    if len(markers) == 0:
        return markers, []

    marker_distances = scipy.spatial.distance_matrix(markers, markers)
    marker_distances_order = np.argsort(marker_distance_thr, axis=-1)
    marker_distances_k_nearest = marker_distances.copy()
    marker_distances_k_nearest[:, marker_distances_order[k_nearest:]] = 1000000

    close_markers = np.nonzero(marker_distances_k_nearest < marker_distance_thr)
    markers_in_range = [set() for i in range(len(markers))]
    for idx_0, idx_1 in zip(*close_markers):
        if idx_0 == idx_1:
            continue
        markers_in_range[idx_0].add(idx_1)
        markers_in_range[idx_1].add(idx_0)
    markers_in_range = [list(indices) for indices in markers_in_range]

    print(f'> Close Markers: {len(close_markers)}')

    # prepare images for inference
    edge_imgs = []
    edge_pts = []
    for idx_0, pt_0 in enumerate(markers):
        if len(markers_in_range[idx_0]) < 3:
            continue

        for idx_1 in markers_in_range[idx_0]:
            pt_1 = markers[idx_1]
            edge_img = EdgeDataset.extract_block_(image=frame,
                                                  block_image_size=64,
                                                  block_image_margin=10,
                                                  edge_ends=np.stack((pt_0, pt_1), axis=0), do_augment=False)

            edge_pts.append((idx_0, idx_1))
            edge_imgs.append(edge_img.reshape(1, edge_img.shape[0], edge_img.shape[1]).astype(np.float32))

    if len(edge_pts) == 0:
        return np.zeros((0, 2)), []

    print(f'> Edge Pts: {len(edge_pts)}')

    edge_imgs = torch.from_numpy(np.stack(edge_imgs, axis=0)).to(device)
    edge_pred = edge_model(edge_imgs).sigmoid().cpu().numpy().flatten()

    # adjacency matrix
    graph = np.zeros((len(markers), len(markers)), dtype=int)
    graph[[pt[0] for pt in edge_pts], [pt[1] for pt in edge_pts]] = edge_pred > edge_confidence_thr

    # make it symmetric?
    graph = (graph * graph.T)

    block_candidates = []
    checked_candidates = set()
    for idx_0 in range(len(markers)):
        if len(markers_in_range[idx_0]) < 3:
            continue

        comb = np.array(list(itertools.permutations(markers_in_range[idx_0], 3)))
        idx_0_r = np.full(comb.shape[0], idx_0)

        check = np.logical_and.reduce((
            graph[idx_0_r, comb[:, 0]],
            graph[idx_0_r, comb[:, -1]],
            graph[comb[:, 0], comb[:, 1]],
            graph[comb[:, 1], comb[:, 2]]))

        is_block = np.nonzero(check)[0]

        for i in is_block:
            corner_idx = [idx_0] + comb[i].tolist()

            corner_idx_sorted = tuple(sorted(corner_idx))
            if corner_idx_sorted in checked_candidates:
                continue
            checked_candidates.add(corner_idx_sorted)

            hull = cv2.convexHull(markers[corner_idx])
            if len(hull) < 4:  # concave points is not feasible
                continue

            edges = (hull - np.roll(hull, 1, axis=0)).reshape(4, 2)
            edge_length = np.linalg.norm(edges, axis=-1)
            inner_angles = -(edges * np.roll(edges, 1, axis=0)).sum(axis=-1).astype(float)
            inner_angles /= edge_length * np.roll(edge_length, 1, axis=0)
            inner_angles = np.arccos(inner_angles)

            if (inner_angles.max() > np.pi/5*4) or (inner_angles.min() < np.pi / 5):
                continue

            if edge_length.max() / edge_length.min() > 4:
                continue

            diagonal_length = np.linalg.norm(hull[:2] - hull[2:], axis=-1).flatten()
            if diagonal_length.max() / diagonal_length.min() > 2.5:
                continue

            hull_area = cv2.contourArea(hull)

            convex_order = utils.check_convex_4_points(markers[corner_idx, :])
            corner_idx = [corner_idx[i] for i in convex_order]

            block_candidates.append((hull_area, corner_idx))

    print(f'> Block Candidates: {len(block_candidates)}')

    # now consolidate markers
    checked_marker_indices = list(set(sum((idx for _, idx in block_candidates), [])))
    index_mapping = [-1 for i in range(len(markers))]
    for check_idx, idx in enumerate(checked_marker_indices):
        index_mapping[idx] = check_idx

    checked_markers = markers[checked_marker_indices]
    block_candidates = [(area, tuple(index_mapping[i] for i in block_indices)) for area, block_indices in block_candidates if len(block_indices) > 0]

    print(f'> Checked Markers: {len(checked_markers)}')

    print(f'> Done Refining\n')

    return checked_markers, block_candidates


def infer_markers(conv_model: nn.Module, edge_model: nn.Module, input: str, output: str, device: str | torch.device, marker_distance_thr: int = 65, marker_confidence_thr: float = 0.75, edge_confidence_thr: float = 0.75, start_frame: int = 0, end_frame: int = -1):
    in_folder, in_file = os.path.split(input)
    out_folder, out_file = os.path.split(output)

    in_split = in_file.split('.')
    out_split = out_file.split('.')

    if out_split[0] != '':
        output = f'{out_folder}\\{out_split[0]}'
    elif in_split[0] != '':
        output = f'{out_folder}\\{in_split[0]}'

    # Start
    t_start = time.time()

    def process_frame(idx: int, frame: np.ndarray, output: str, all_markers, ffmpeg_process):
        # Start processing...
        t_process_start = time.time()

        frame = skimage.color.rgb2gray(frame)

        markers, confidence, heatmap_img = infer_image(conv_model, frame, confidence_thr=marker_confidence_thr, check_half=True, device=device, show_heatmap=False)

        checked_markers, block_candidates = markers, []
        checked_markers, block_candidates = refine_checkpoints(edge_model, frame, markers, confidence, edge_confidence_thr=edge_confidence_thr, marker_distance_thr=marker_distance_thr, device=device)

        all_markers.append({
            'markers': markers.tolist(),                             # [n, 2]
            'confidence': confidence.tolist(),                       # [n]
            'checked_markers': np.asarray(checked_markers).tolist(),  # [m, 2]
            'blocks': block_candidates                               # [(area, [m])]
        })

        if output != '':
            frame_1 = np.zeros_like(frame)

            frame_marker = frame[:]
            frame1_marker = frame_1[:]
            block_pts = [checked_markers[list(block_indices)].reshape(-1, 1, 2) for area, block_indices in block_candidates]
            cv2.polylines(frame_marker, block_pts, True, 1, thickness=2)
            cv2.polylines(frame1_marker, block_pts, True, 1, thickness=2)

            frame_marker = utils.draw_markers(frame_marker, checked_markers, radius=5, color=0.5+0.5)
            frame_marker = utils.draw_markers(frame_marker, markers, radius=5, color=confidence*0+0.5)

            frame1_marker = utils.draw_markers(frame1_marker, checked_markers, radius=5, color=0.5+0.5)
            frame1_marker = utils.draw_markers(frame1_marker, markers, radius=5, color=confidence*0+0.5)

            frame_marker = np.concatenate((frame_marker, frame1_marker), axis=1)

            frame_marker = skimage.color.gray2rgb(frame_marker)
            out_img = (frame_marker*255).astype(np.uint8)

            if ffmpeg_process is None:
                import ffmpeg
                ffmpeg_process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{frame_marker.shape[1] if frame_marker.shape[1] % 2 == 0 else frame_marker.shape[1] + 1}x{frame_marker.shape[0] if frame_marker.shape[0] % 2 == 0 else frame_marker.shape[0] + 1}').output(
                    output, pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True))
            ffmpeg_process.stdin.write(out_img)
            # cv2.imwrite(os.path.join(out_dir, f'{prefix}{utils.leading_zeros(idx,leading_zeros)}.jpg'), out_img)

        # End processing ...
        t_process_end = time.time()

        print(f'> Processed frame {idx} in {t_process_end - t_process_start:.2f}s - Total processing time: {t_process_end - t_start:.2f}s\t\t')
        print('---')

        return all_markers, ffmpeg_process

    def finish(output, ffmpeg_process, all_markers):
        if ffmpeg_process is not None:
            t_finish_start = time.time()
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            print(f'\n> Video saved to "{output}_markers.mp4" in {(time.time() - t_finish_start):.2f}s')
            print('---')

        if len(all_markers) == 0:
            print(f'\n> No markers to save!\n')
            print('---')
            return
            
        # End
        t_finish_start = time.time()
        import json
        with open(f'{output}_markers.json', 'w') as f:
            json.dump(all_markers, f, indent=4)
            all = 0
            for frame in all_markers:
                all += len(frame['markers']) 
            avg = all / len(all_markers)
            print(f'\n> Saved {all} markers (Average: {avg:.2f}) to "{output}_markers.json" in {(time.time() - t_finish_start):.2f}s\n')
            print('---')

    if os.path.isfile(input) and in_split[1] == 'mp4':
        all_markers = []
        ffmpeg_process = None
        count = 0

        for idx, fn in enumerate(iio.imiter(input, plugin="pyav"), start_frame):
            if end_frame > -1 and idx > end_frame:
                break

            all_markers, ffmpeg_process = process_frame(idx, fn, f'{output}_markers.mp4', all_markers, ffmpeg_process)
            count += 1

        finish(f'{output}_markers.json', ffmpeg_process, all_markers)

    else:
        subfolders = os.listdir(input)
        if len(subfolders) > 0:
            sorted(subfolders, key=lambda fn: fn)
            subfolders = [os.path.join(input, folder) for folder in subfolders]
        else:
            subfolders = in_folder

        base_output = output

        for idx_folder, subfolder in enumerate(subfolders):
            images = [os.path.join(subfolder, frame) for frame in os.listdir(subfolder)]
            if len(images) > 0:
                sorted(images, key=lambda fn: os.path.basename(fn))

            all_markers = []
            ffmpeg_process = None
            count = 0

            output = os.path.join(base_output, os.path.basename(subfolder))

            if not os.path.isdir(output):
                os.makedirs(output, exist_ok=True)

            output += f'\\{os.path.basename(subfolder)}'

            for id_img, fn in enumerate(images, start_frame):
                if end_frame > -1 and id_img > end_frame:
                    break

                if not os.path.isfile(fn):
                    print(f'> Failed to read "{fn}"')
                    break

                frame = skimage.io.imread(fn)

                all_markers, ffmpeg_process = process_frame(id_img, frame, f'{output}_markers.mp4', all_markers, ffmpeg_process)
                count += 1

            finish(output, ffmpeg_process, all_markers)

    print(f'> Processing markers complete - Total time: {(time.time() - t_start):.2f}s')
    print('---')

class CornerLabeler:
    def __init__(self, corner_model_fn: str, edge_model_fn: str, device: str | torch.device = 'cuda', show_heatmap: bool = True) -> None:
        self.device = device
        self.show_heatmap = show_heatmap

        self.corner_model = UNet(output_channel=2).to(device)

        with open(corner_model_fn, 'rb') as f:
            data = torch.load(f, map_location=device, weights_only=True)
            if 'model' in data:
                self.corner_model.load_state_dict(data['model'])
            else:
                self.corner_model.load_state_dict(data)
            self.corner_model.eval()

        self.edge_model = EdgeNet().to(device)
        with open(edge_model_fn, 'rb') as f:
            data = torch.load(f, map_location=device, weights_only=True)
            if 'model' in data:
                self.edge_model.load_state_dict(data['model'])
            else:
                self.edge_model.load_state_dict(data)
            self.edge_model.eval()

    def label(self, image: np.ndarray, marker_confidence_thr: float = 0.75, marker_check_half_res: bool = True, edge_confidence_thr: float = 0.75, marker_distance_thr=65):
        image = np.asarray(image).squeeze()
        if image.ndim > 2:
            image = skimage.color.rgb2gray(image)

        print('processing... ')

        t0 = time.time()
        markers, confidence, heatmap_img = infer_image(self.corner_model, image, confidence_thr=marker_confidence_thr,
                                                       check_half=marker_check_half_res, device=self.device, show_heatmap=self.show_heatmap)
        print(f'    1st pass: found {len(markers)} corners in {time.time()-t0:0.2f} seconds.')

        t0 = time.time()
        checked_markers, block_candidates = refine_checkpoints(self.edge_model, image, markers, confidence, edge_confidence_thr=edge_confidence_thr,
                                                               marker_distance_thr=marker_distance_thr, device=self.device)
        print(f'    2nd pass: found {len(checked_markers)} corners and {len(block_candidates)} blocks in {time.time()-t0:0.2f} seconds.')

        t0 = time.time()

        # binary = cv2.adaptiveThreshold((image*255).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 89, 5)

        checked_markers_subpix = checked_markers

        if len(checked_markers) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
            checked_markers_subpix = cv2.cornerSubPix(image.astype(np.float32), checked_markers[:, None, :].astype(np.float32), winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

            checked_markers_subpix = checked_markers_subpix[:, 0]
            correction = checked_markers_subpix - checked_markers
            print(f'    3rd pass: subpixel correction: max {np.abs(correction).max():0.4f} pixels in {time.time()-t0:0.2f} seconds.')

        return markers, checked_markers_subpix, block_candidates

# Get model from a snapshot or checkpoint


def getModel(fn, device):
    with open(fn, 'rb') as f:
        data = torch.load(f, map_location=device, weights_only=True)

    m = data
    if 'model' in data:
        m = data['model']
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corner-model', default='', type=str, required=True, help="Trained .pt model")
    parser.add_argument('--edge-model', default='', type=str, required=True, help="Trained .pt model")

    parser.add_argument('--input', default='', type=str, required=True, help="Input video / images")
    parser.add_argument('--output', default='', type=str, required=True, help=".json format")

    parser.add_argument('--marker-distance-thr', default=65, type=int, help="Recommended: 150, Default: 65")
    parser.add_argument('--marker-confidence-thr', default=0.7, type=float, help="Recommended: 0.7, Default: 0.75")
    parser.add_argument('--edge-confidence-thr', default=0.65, type=float, help="Recommended: 0.6, Default: 0.75")

    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=-1, help="-1 for all frames in a folder")
    parser.add_argument('--gpu', type=int, default=0, help="Which gpu to use (if there are multiple)")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load corner net model
    conv_model = UNet(output_channel=2).to(device)
    conv_model.load_state_dict(getModel(args.corner_model, device))
    conv_model.eval()

    # Load edge net model
    edge_model = EdgeNet().to(device)
    edge_model.load_state_dict(getModel(args.edge_model, device))
    edge_model.eval()

    if len(args.output) != 0:
        if not os.path.isfile(args.output) and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

    print(f'> Input: {args.input}')
    print(f'> Output: {args.output}')
    print(f'> Start frame: {args.start_frame}')
    print(f'> End frame: {args.end_frame}')
    print(f'---')

    infer_markers(conv_model, edge_model, input=args.input, output=args.output, device=device, marker_distance_thr=args.marker_distance_thr,
                  marker_confidence_thr=args.marker_confidence_thr, edge_confidence_thr=args.edge_confidence_thr, start_frame=args.start_frame, end_frame=args.end_frame)


if __name__ == '__main__':
    main()
