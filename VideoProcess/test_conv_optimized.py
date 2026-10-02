import torch
from torch import nn
import skimage
import numpy as np
from models import UNet, EdgeNet
import time
import cv2
import scipy
from datasets import EdgeDataset
import itertools
import utils
import concurrent.futures
from os.path import join
from os import makedirs, listdir
import gc
from multiprocessing import Process, Pool


@torch.inference_mode()
def process_conv_inference(all_frames: torch.Tensor, half_all_frames: torch.Tensor, all_out_img: np.ndarray, all_out_img_max: np.ndarray, all_out_img_patch_cnt: np.ndarray, conv_model: nn.Module, stride: int, block_size: int, check_half: bool):
    for i in range(all_frames.shape[0]):
        t_start = time.time()

        out_patches = torch.empty((block_size-1, 75, 2, block_size, block_size), device=all_frames.device, dtype=all_frames.dtype)
        patches = all_frames[i].unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
        for j in range(patches.shape[0]):
            out_patches[j] = conv_model(patches[j, :, :, :, :])
        out_patches = out_patches.float().cpu().numpy()
        all_out_img[i] = combine_patches(all_out_img[i], all_out_img_max[i], all_out_img_patch_cnt[i], out_patches, stride, block_size)

        if check_half:
            half_out_patches = torch.empty((block_size-1, 37, 2, block_size, block_size), device=half_all_frames.device, dtype=half_all_frames.dtype)
            half_patches = half_all_frames[i].unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
            for j in range(half_patches.shape[0]):
                half_out_patches[j] = conv_model(half_patches[j, :, :, :, :])
            half_out_patches = half_out_patches.float().cpu().numpy()
            all_out_img[i] = combine_patches(all_out_img[i], all_out_img_max[i], all_out_img_patch_cnt[i], out_patches, stride, block_size)

        scale = 0.1
        all_out_img[i] = all_out_img[i] * scale
        all_out_img_max[i] = all_out_img_max[i] * scale
        all_out_img[i] = all_out_img_max[i]

        print(f'\t> Process conv model took {(time.time() - t_start):.02f}s')


@torch.inference_mode()
def process_edge_model(all_edge_imgs: list, all_edge_preds: list, edge_model: nn.Module, device):
    while len(all_edge_imgs) > 0:
        t_start = time.time()
        edge_imgs = torch.from_numpy(np.stack(all_edge_imgs[-1], axis=0)).to(device)
        pred = edge_model(edge_imgs).sigmoid().cpu().numpy().flatten()
        all_edge_preds.append((len(all_edge_imgs)-1, pred))
        all_edge_imgs.pop()
        print(f'\t> Process edge model took {(time.time() - t_start):.02f}s')


def combine_patches(out_img, out_img_max, out_img_patch_cnt, out_patches, stride, block_size, input_scale=0):
    for i in range(out_patches.shape[0] - 1):
        for j in range(out_patches.shape[1] - 1):
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
            weight_block = 1 / out_img_patch_cnt[r_s:r_e, c_s:c_e]

            out_img[r_s:r_e, c_s:c_e] *= (1 - weight_block)
            out_img[r_s:r_e, c_s:c_e] += out_patch * weight_block

            max_block = out_img_max[r_s:r_e, c_s:c_e]
            out_img_max[r_s:r_e, c_s:c_e] = np.maximum(out_patch, max_block)

    return out_img


def get_contours_center(contours, out_img):
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
    return markers, confidence


def get_edge_imgs(frame, markers, marker_distance_thr=65, k_nearest=8, debug=False):
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

    # if debug:
    # print(f'> Close Markers: {len(close_markers)}')

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

    # if debug:
        # print(f'> Edge Pts: {len(edge_pts)}')

    return edge_imgs, edge_pts, markers_in_range


def process_edge_imgs(x: list):
    t_start = time.time()

    frame_idx, frame, out_img, marker_confidence_thr, marker_distance_thr = x

    _, binary = cv2.threshold(out_img, marker_confidence_thr, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers, confidence = get_contours_center(contours, out_img)

    edge_imgs, edge_pts, markers_in_range = get_edge_imgs(frame, np.asarray(markers), marker_distance_thr)

    print(f'\t> Process edge imgs took {(time.time()-t_start):.02f}s')

    return frame_idx, edge_imgs, edge_pts, markers, confidence, markers_in_range


def process_block_candidates(x: list):
    all_markers, all_edge_pts, all_edge_pred, edge_confidence_thr, all_markers_in_range = x
    while True:
        if len(all_edge_pred) > 0:
            frame_idx, edge_pred = all_edge_pred.pop()

            t_start = time.time()

            markers: np.ndarray = np.asarray(all_markers[frame_idx])

            # adjacency matrix
            graph = np.zeros((len(markers), len(markers)), dtype=int)
            graph[[pt[0] for pt in all_edge_pts[frame_idx]], [pt[1] for pt in all_edge_pts[frame_idx]]] = edge_pred > edge_confidence_thr

            # make it symmetric?
            graph = (graph * graph.T)

            block_candidates = []
            checked_candidates = set()

            for idx_0 in range(len(markers)):
                if len(all_markers_in_range[frame_idx][idx_0]) < 3:
                    continue

                comb = np.array(list(itertools.permutations(all_markers_in_range[frame_idx][idx_0], 3)))
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

            # now consolidate markers
            checked_marker_indices = list(set(sum((idx for _, idx in block_candidates), [])))
            index_mapping = [-1 for i in range(len(markers))]
            for check_idx, idx in enumerate(checked_marker_indices):
                index_mapping[idx] = check_idx

            checked_markers = np.asarray(markers[checked_marker_indices]).tolist()
            block_candidates = [(area, tuple(index_mapping[i] for i in block_indices)) for area, block_indices in block_candidates if len(block_indices) > 0]

            print(f'\t> Process block candidates took {time.time() - t_start:.02f}s')

            return frame_idx, checked_markers, block_candidates
        time.sleep(.1)


@torch.inference_mode()
def main():
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    conv_model = UNet(output_channel=2).to(device)
    conv_model.load_state_dict(torch.load("./ckpts/0002/conv/best_lr_0.001_bs_128_model_25_acc=0.7881.pt", weights_only=True, map_location=device))
    conv_model.to(dtype=torch.bfloat16)

    conv_model = torch.compile(
        conv_model,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead"
    )

    edge_model = EdgeNet().to(device)
    edge_model.load_state_dict(torch.load("./ckpts/0002/edge/best_lr_0.001_bs_1024_model_20_accuracy=0.9836.pt", weights_only=True, map_location=device))
    edge_model.to(dtype=torch.float32)

    # edge_model = torch.compile(
    #     edge_model,
    #     fullgraph=False,
    #     dynamic=True,
    #     mode="max-autotune"
    # )

    start_frame = 0
    end_frame = 9
    workers = 12
    chunk_size = 32

    stride = 32
    block_size = 64

    marker_distance_thr = 65
    marker_confidence_thr = 0.6
    edge_confidence_thr = 0.6
    check_half = True

    t_global = time.time()

    input = "./Blender/renders/cameras/"
    files = []
    for cam in listdir(input):
        _, f = utils.folder_contents(join(input, cam))
        files.extend(f[start_frame:end_frame+1])

    output = "./Out/synth_01/markers/"

    chunks = int(len(files) / chunk_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers):
            executor.submit(time.sleep(.1))
        for c in range(chunks + 1):
            t_chunk = time.time()
            c_first_frame = c*chunk_size
            c_last_frame = min(len(files), (c+1)*chunk_size)-1
            c_files = files[c_first_frame:c_last_frame+1]
            frames = np.ndarray([len(c_files), 2048, 2448], dtype=np.float32)

            for i in range(len(c_files)):
                frame = skimage.io.imread(c_files[i])
                frame = skimage.color.rgb2gray(frame)
                frames[i] = frame

            gpu_half_frames = None
            gpu_frames = torch.from_numpy(frames).to(device).bfloat16().reshape(-1, 1, frames.shape[1], frames.shape[2])

            if check_half:
                half_frames = np.ndarray([len(c_files), 1024, 1224], dtype=np.float32)
                for i in range(len(frames)):
                    half_frame = skimage.transform.rescale(frame, 0.5, anti_aliasing=True)
                    half_frames[i] = half_frame
                gpu_half_frames = torch.from_numpy(half_frames).to(device).bfloat16().reshape(-1, 1, half_frames.shape[1], half_frames.shape[2])

            all_out_img = np.zeros((gpu_frames.shape[0], gpu_frames.shape[2], gpu_frames.shape[3]))
            all_out_img_max = np.zeros((gpu_frames.shape[0], gpu_frames.shape[2], gpu_frames.shape[3]))
            all_out_img_patch_cnt = np.zeros((gpu_frames.shape[0], gpu_frames.shape[2], gpu_frames.shape[3]))

            print('> Start conv inference...')
            t_conv = time.time()

            p = Process(target=process_conv_inference,args=(gpu_frames, gpu_half_frames, all_out_img, all_out_img_max, all_out_img_patch_cnt, conv_model, stride, block_size, check_half))

            p.start()
            process_conv_inference(gpu_frames, gpu_half_frames, all_out_img, all_out_img_max, all_out_img_patch_cnt, conv_model, stride, block_size, check_half)
            # t_conv = time.time() - t_conv
            # print(f'> Total conv inference took {t_conv:.02f}s (Avg {(t_conv / (len(frames))):.02f}s per frame ({len(frames)} frames))')

            del gpu_half_frames
            del gpu_frames
            if check_half:
                del half_frames
            gc.collect()
            torch.cuda.empty_cache()

            chunk_markers = [0 for _ in range(len(frames))]
            chunk_confidence = [0 for _ in range(len(frames))]
            chunk_edge_imgs = [0 for _ in range(len(frames))]
            chunk_edge_pts = [0 for _ in range(len(frames))]
            chunk_markers_in_range = [0 for _ in range(len(frames))]

            print(f'> Start process edge imgs...')
            t_edge_imgs = time.time()
            for frame_idx, edge_imgs, edge_pts, markers, confidence, markers_in_range in executor.map(process_edge_imgs, [[i, frames[i], all_out_img[i], marker_confidence_thr, marker_distance_thr] for i in range(len(frames))]):
                chunk_edge_imgs[frame_idx] = edge_imgs
                chunk_edge_pts[frame_idx] = edge_pts
                chunk_markers[frame_idx] = markers
                chunk_confidence[frame_idx] = confidence
                chunk_markers_in_range[frame_idx] = markers_in_range
            t_edge_imgs = time.time() - t_edge_imgs
            print(f'> Process all edge imgs took {t_edge_imgs:.02f}s (Avg {(t_edge_imgs / len(frames)):.02f}s per frame ({len(frames)} frames))')

            del all_out_img
            gc.collect()

            chunk_edge_preds = []

            print(f'> Start process edge model...')
            # t_edge_model = time.time()
            thread_edge = concurrent.futures.ThreadPoolExecutor(1)
            fut = thread_edge.submit(process_edge_model, chunk_edge_imgs, chunk_edge_preds, edge_model, device)
            # process_edge_model(chunk_edge_imgs, chunk_edge_preds, edge_model, device)
            # torch.cuda.empty_cache()
            # t_edge_model = time.time() - t_edge_model
            # print(f'> Process edge model took {t_edge_model:.02f}s (Avg {(t_edge_model / len(frames)):.02f}s per frame ({len(frames)} frames))')

            # fut.result()

            chunk_checked_markers = [0 for _ in range(len(frames))]
            chunk_blocks = [0 for _ in range(len(frames))]

            print(f'> Start process block candidates...')
            t_block_candidates = time.time()
            for frame_idx, checked_markers, block_candidates in executor.map(process_block_candidates, [[chunk_markers, chunk_edge_pts, chunk_edge_preds, edge_confidence_thr, chunk_markers_in_range] for _ in range(len(frames))]):
                chunk_checked_markers[frame_idx] = checked_markers
                chunk_blocks[frame_idx] = block_candidates
                torch.cuda.empty_cache()
            t_block_candidates = time.time() - t_block_candidates
            print(f'> Process all block candidates took {t_block_candidates:.02f}s (Avg {(t_block_candidates / len(frames)):.02f}s per frame ({len(frames)} frames))')

            # j = {
            #     "chunk_markers": chunk_markers,
            #     "chunk_confidence": chunk_confidence,
            #     "chunk_checked_markers": chunk_checked_markers,
            #     "chunk_blocks": chunk_blocks,
            # }

            # if output:
            #     makedirs(output, exist_ok=True)
            #     import json
            #     fn = join(output, f'markers_chunk_{utils.leading_zeros(c,4)}_frames_[{c_first_frame}-{c_last_frame}].json')
            #     with open(fn, 'w+', 1) as f:
            #         json_string = json.dumps(j, separators=(',', ":"))  # Compact JSON structure
            #         f.write(json_string)

            # del chunk_checked_markers
            # del chunk_markers
            # del chunk_blocks
            # del chunk_confidence
            # del chunk_edge_imgs
            # del chunk_edge_preds
            # del chunk_edge_pts
            # del chunk_markers_in_range

            t_chunk = time.time() - t_chunk
            print(f'> Chunk [{c}] time {t_chunk:.02f}s (Avg {(t_chunk / (len(frames))):.02f}s per frame ({len(frames)} frames))')

    t_global = time.time() - t_global
    print(f'> Total time for {chunks + 1} chunks: {t_global:.02f}s (Avg {(t_global / (len(files))):.02f}s per frame ({len(files)} frames))')

    return


if __name__ == '__main__':
    main()
