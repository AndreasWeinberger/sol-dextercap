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
from os.path import join
from os import makedirs, listdir, rename, remove
import gc
import concurrent.futures
import json


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
                                                  block_image_size=64,  # TODO: parameterize this?
                                                  block_image_margin=10,
                                                  edge_ends=np.stack((pt_0, pt_1), axis=0), do_augment=False)

            edge_pts.append((idx_0, idx_1))
            edge_imgs.append(edge_img.reshape(1, edge_img.shape[0], edge_img.shape[1]).astype(np.float32))

    if len(edge_pts) == 0:
        return np.zeros((0, 2)), []

    # if debug:
        # print(f'> Edge Pts: {len(edge_pts)}')

    return edge_imgs, edge_pts, markers_in_range


@torch.no_grad()
@torch.inference_mode()
def process_conv_inference_2(all_frames: torch.Tensor, half_all_frames: torch.Tensor | None, all_out_img: np.ndarray, all_out_img_max: np.ndarray, all_out_img_patch_cnt: np.ndarray, conv_model: nn.Module, stride: int, block_size: int):
    for i in range(all_frames.shape[0]):
        t_start_full = time.time()

        out_patches = torch.empty((block_size-1, 75, 2, block_size, block_size), device=all_frames.device, dtype=all_frames.dtype)
        patches = all_frames[i].unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
        for j in range(patches.shape[0]):
            out_patches[j] = conv_model(patches[j, :, :, :, :])
        patches = out_patches.float().cpu().numpy()

        del out_patches
        torch.cuda.empty_cache()

        all_out_img[i] = combine_patches(all_out_img[i], all_out_img_max[i], all_out_img_patch_cnt[i], patches, stride, block_size)

        print(f'\t\t> Conv inference & combine (full) [{i}] took {(time.time() - t_start_full):.02f}s')

        if half_all_frames != None:
            t_start_half = time.time()
            half_out_patches = torch.empty((block_size-1, 37, 2, block_size, block_size), device=half_all_frames.device, dtype=half_all_frames.dtype)
            half_patches = half_all_frames[i].unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
            for j in range(half_patches.shape[0]):
                half_out_patches[j] = conv_model(half_patches[j, :, :, :, :])
            half_patches = half_out_patches.float().cpu().numpy()

            del half_out_patches
            torch.cuda.empty_cache()

            all_out_img[i] = combine_patches(all_out_img[i], all_out_img_max[i], all_out_img_patch_cnt[i], half_patches, stride, block_size)

            print(f'\t\t> Conv inference & combine (half) [{i}] took {(time.time() - t_start_half):.02f}s')

        scale = 0.1
        all_out_img[i] = all_out_img[i] * scale
        all_out_img_max[i] = all_out_img_max[i] * scale
        all_out_img[i] = all_out_img_max[i]

        print(f'\t> Conv inference & combine [{i}] took {(time.time() - t_start_full):.02f}s')


@torch.no_grad()
@torch.inference_mode()
def process_conv_inference_3(frame_idx, frame: torch.Tensor, half_frame: torch.Tensor | None, conv_model: nn.Module, stride: int, block_size: int):
    t_start_full = time.time()
    out_img = np.zeros((frame.shape[1], frame.shape[2]))
    out_img_max = np.zeros((frame.shape[1], frame.shape[2]))
    out_img_patch_cnt = np.zeros((frame.shape[1], frame.shape[2]))

    patches = frame.unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
    out_patches = torch.empty((block_size-1, patches.shape[1], 2, block_size, block_size), device=frame.device, dtype=frame.dtype)
    for j in range(patches.shape[0]):
        out_patches[j] = conv_model(patches[j, :, :, :, :])
    patches = out_patches.float().cpu().numpy()

    del out_patches
    torch.cuda.empty_cache()

    out_img = combine_patches(out_img, out_img_max, out_img_patch_cnt, patches, stride, block_size)

    print(f'\t\t> Conv inference & combine (full) [{frame_idx}] took {(time.time() - t_start_full):.02f}s')

    if half_frame != None:
        t_start_half = time.time()
        half_patches = half_frame.unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
        half_out_patches = torch.empty((block_size-1, half_patches.shape[1], 2, block_size, block_size), device=half_frame.device, dtype=half_frame.dtype)
        for j in range(half_patches.shape[0]):
            half_out_patches[j] = conv_model(half_patches[j, :, :, :, :])
        half_patches = half_out_patches.float().cpu().numpy()

        del half_out_patches
        torch.cuda.empty_cache()

        out_img = combine_patches(out_img, out_img_max, out_img_patch_cnt, half_patches, stride, block_size)

        print(f'\t\t> Conv inference & combine (half) [{frame_idx}] took {(time.time() - t_start_half):.02f}s')

    scale = 0.1
    out_img = out_img * scale
    out_img_max = out_img_max * scale
    out_img = out_img_max

    return out_img


@torch.no_grad()
@torch.inference_mode()
def process_conv_inference_1(all_frames: torch.Tensor, half_all_frames: torch.Tensor | None, all_out_img: np.ndarray, all_out_img_max: np.ndarray, all_out_img_patch_cnt: np.ndarray, conv_model: nn.Module, stride: int, block_size: int):
    # FULL
    out_patches = torch.empty((all_frames.shape[0], block_size-1, 75, 2, block_size, block_size), device=all_frames.device, dtype=all_frames.dtype)
    for i in range(all_frames.shape[0]):
        t_start = time.time()
        patches = all_frames[i].unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
        for j in range(patches.shape[0]):
            out_patches[i][j] = conv_model(patches[j, :, :, :, :])
        print(f'\t> Conv inference (full) [{i}] took {(time.time() - t_start):.02f}s')

    patches = out_patches.float().cpu().numpy()
    del out_patches
    torch.cuda.empty_cache()

    for i in range(all_frames.shape[0]):
        t_start = time.time()
        all_out_img[i] = combine_patches(all_out_img[i], all_out_img_max[i], all_out_img_patch_cnt[i], patches[i], stride, block_size)
        print(f'\t> Combine (full) [{i}] took {(time.time() - t_start):.02f}s')

    # HALF
    if half_all_frames != None:
        half_out_patches = torch.empty((half_all_frames.shape[0], block_size-1, 37, 2, block_size, block_size), device=half_all_frames.device, dtype=half_all_frames.dtype)
        for i in range(half_all_frames.shape[0]):
            t_start = time.time()
            patches = half_all_frames[i].unfold(1, block_size, stride).unfold(2, block_size, stride).permute(1, 2, 0, 3, 4)
            for j in range(patches.shape[0]):
                half_out_patches[i][j] = conv_model(patches[j, :, :, :, :])
            print(f'\t> Conv inference (half) [{i}] took {(time.time() - t_start):.02f}s')

        half_patches = half_out_patches.float().cpu().numpy()
        del half_out_patches
        torch.cuda.empty_cache()

        for i in range(half_all_frames.shape[0]):
            t_start = time.time()
            all_out_img[i] = combine_patches(all_out_img[i], all_out_img_max[i], all_out_img_patch_cnt[i], half_patches[i], stride, block_size)
            print(f'\t> Combine (half) [{i}] took {(time.time() - t_start):.02f}s')

    t_start = time.time()
    for i in range(all_frames.shape[0]):
        scale = 0.1
        all_out_img[i] = all_out_img[i] * scale
        all_out_img_max[i] = all_out_img_max[i] * scale
        all_out_img[i] = all_out_img_max[i]
    print(f'\t> Remapping [all] took {(time.time() - t_start):.02f}s')


@torch.no_grad()
@torch.inference_mode()
def process_edge_model_2(edge_imgs, edge_model, device):
    edge_imgs = torch.from_numpy(np.stack(edge_imgs, axis=0)).to(device)
    pred = edge_model(edge_imgs).sigmoid().cpu().numpy().flatten()
    torch.cuda.empty_cache()

    return pred


@torch.inference_mode()
@torch.no_grad()
def process_edge_model_1(all_edge_imgs: list, all_edge_preds: list, edge_model: nn.Module, device, end: int):
    print(f'> Start process edge model...')
    t_edge_model = time.time()
    count = 0

    while count != end:
        if len(all_edge_imgs) > 0:
            t_start = time.time()
            frame_idx, edge_imgs = all_edge_imgs.pop()
            print(frame_idx)

            edge_imgs = torch.from_numpy(np.stack(edge_imgs, axis=0)).to(device)
            pred = edge_model(edge_imgs).sigmoid().cpu().numpy().flatten()
            all_edge_preds.append(pred)
            torch.cuda.empty_cache()

            count += 1

            print(f'\t> Edge inference [{frame_idx}] took {(time.time() - t_start):.02f}s')
        else:
            time.sleep(.1)
            # print('> Sleeping...')

    t_edge_model = time.time() - t_edge_model
    print(f'> Edge inference [all] took {t_edge_model:.02f}s (Avg {(t_edge_model / end):.02f}s per frame ({end} frames))')


def process_edge_imgs(frame_idx, frame, out_img, marker_confidence_thr, marker_distance_thr):

    t_start = time.time()

    _, binary = cv2.threshold(out_img, marker_confidence_thr, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers, confidence = get_contours_center(contours, out_img)

    edge_imgs, edge_pts, markers_in_range = get_edge_imgs(frame, np.asarray(markers), marker_distance_thr)

    # make sure that the ndarrays have a consistent shape --> enables CUDA graph compile for the edge model
    edge_imgs = np.stack(edge_imgs, axis=0)
    edge_imgs = np.resize(edge_imgs, (4000, 1, 64, 64))

    edge_pts = np.array(edge_pts)
    edge_pts = np.resize(edge_pts, (4000, 2))

    print(f'\t> Edge imgs [{frame_idx}] took {(time.time()-t_start):.02f}s')

    return frame_idx, edge_imgs, edge_pts, markers_in_range, markers, confidence


def process_block_candidates(frame_idx, markers, edge_pts, edge_pred, edge_confidence_thr, markers_in_range):
    t_start = time.time()

    markers: np.ndarray = np.asarray(markers)

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

    # now consolidate markers
    checked_marker_indices = list(set(sum((idx for _, idx in block_candidates), [])))
    index_mapping = [-1 for i in range(len(markers))]
    for check_idx, idx in enumerate(checked_marker_indices):
        index_mapping[idx] = check_idx

    checked_markers = np.asarray(markers[checked_marker_indices]).tolist()
    block_candidates = [(area, tuple(index_mapping[i] for i in block_indices)) for area, block_indices in block_candidates if len(block_indices) > 0]

    print(f'\t> Block candidates [{frame_idx}] took {time.time() - t_start:.02f}s')

    return frame_idx, checked_markers, block_candidates


def export(data, output: str, prefix: str, chunk_first_frame: int, chunk_last_frame: int, chunk_overall_first_frame: int, chunk_overall_last_frame: int, update_output: bool):
    fn = join(output, f'{prefix}_frames_[{chunk_first_frame}-{chunk_last_frame}]_markers.json')

    if update_output:
        update_files = [join(output,f) for f in listdir(output) if f.startswith(prefix)]
        if len(update_files) > 0:
            old_fn = update_files[0]
            with open(old_fn, 'r') as f:
                update_data = json.load(f)

                for i in range(chunk_overall_last_frame+1-chunk_overall_first_frame):
                    if i+chunk_overall_first_frame > len(update_data)-1:
                        update_data.append(data[i])
                    else:
                        update_data[i+chunk_overall_first_frame] = data[i]

            with open(old_fn, 'w+') as f:
                json_string = json.dumps(update_data, separators=(',', ":"))  # Compact JSON structure
                f.write(json_string)

            fn = join(output, f'{prefix}_frames_[{chunk_first_frame}-{chunk_first_frame+len(update_data)-1}]_markers.json')
            if old_fn != fn:
                rename(old_fn, fn)

            return
    delete_files = [join(output,f) for f in listdir(output) if f.startswith(prefix)]
    for f in delete_files:
        remove(f)

    with open(fn, 'w+', 1) as f:
        json_string = json.dumps(data, separators=(',', ":"))  # Compact JSON structure
        f.write(json_string)


@torch.no_grad()
def main():
    t_global = time.time()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    conv_model: nn.Module = UNet(output_channel=2).to(device)
    conv_model.load_state_dict(torch.load("./ckpts/0002/conv/best_lr_0.001_bs_128_model_25_acc=0.7881.pt", weights_only=True, map_location=device))
    conv_model.to(dtype=torch.bfloat16)

    conv_model = torch.compile(
        conv_model,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead"
    )
    conv_model.eval()

    edge_model: nn.Module = EdgeNet().to(device)
    edge_model.load_state_dict(torch.load("./ckpts/0002/edge/best_lr_0.001_bs_1024_model_20_accuracy=0.9836.pt", weights_only=True, map_location=device))
    edge_model.to(dtype=torch.float32)
    edge_model.eval()

    edge_model = torch.compile(
        edge_model,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead"
    )

    # Tuple of ids e.g.: (0,1,2,3,4,5,6) or (0,) for only cam_idx 0
    camera_ids = (0,)
    # when chunks_from_camera_ids is True, this counts for each camera
    cam_first_frame = 0
    # when chunks_from_camera_ids is True, this counts for each camera
    cam_last_frame = 19
    # Update output file
    update_output = False
    # -1 to create only one chunk with all frames
    chunk_size = 10
    # for the ProcessPoolExecutor(), betw. 6-12 is recommended
    workers = 8

    # betw. 64-256 --> 128 performs best with 0.60s / frame w/o check_half on conv inference
    block_size = 128
    # half of block_size for overlapping patches in conv inference
    stride = 64

    # process conv model on half resolution images --> slower, but may be more robust
    check_half = False
    marker_distance_thr = 65
    marker_confidence_thr = 0.6
    edge_confidence_thr = 0.6

    input = "./Blender/renders/cameras/"
    output = "./Out/synth_02/markers/"

    if output:
        makedirs(output, exist_ok=True)

    cameras = listdir(input)

    all_files = [[] for _ in range(max(camera_ids)+1)]
    for idx in camera_ids:
        _, folder_files = utils.folder_contents(join(input, cameras[idx]))
        all_files[idx].append(folder_files[cam_first_frame:cam_last_frame+1])

    if chunk_size == -1:
        chunk_cnt = 1
        chunk_size = cam_last_frame+1-cam_first_frame
    else:
        chunk_cnt = int((cam_last_frame-cam_first_frame) / chunk_size)+1

    if chunk_cnt > 1:
        for chunk_idx in range(len(all_files)):
            if len(all_files[chunk_idx]) == 0:
                continue

            files = all_files[chunk_idx][0].copy()
            all_files[chunk_idx] = []
            for j in range(chunk_cnt):
                start = (j)*chunk_size
                end = min(len(files), (j+1)*chunk_size)
                all_files[chunk_idx].append(files[start:end])

    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        print(f'>>> Running ProcessPoolExecutor() with {workers} workers <<<')
        for _ in range(workers):
            executor.submit(time.sleep, .1)
        for cam_idx in camera_ids:

            print(f'>>> Starting camera idx {cam_idx} | Chunks: {len(all_files[cam_idx])} | Range: [{cam_first_frame}-{cam_last_frame}] <<<')
            t_cam = time.time()

            cam_frames_count = sum([len(all_files[cam_idx][chunk_idx]) for chunk_idx in range(len(all_files[cam_idx]))])

            for chunk_idx in range(len(all_files[cam_idx])):

                t_chunk = time.time()

                chunk_first_frame = (chunk_idx)*chunk_size
                chunk_last_frame = min(((chunk_idx+1)*chunk_size)-1,cam_frames_count-1)
                chunk_overall_first_frame = chunk_first_frame + cam_first_frame
                chunk_overall_last_frame = chunk_last_frame + cam_first_frame

                print(f'>>> Starting chunk idx [{chunk_idx}/{chunk_cnt-1}] | Frames: {chunk_overall_last_frame+1-chunk_overall_first_frame} | Range: [{chunk_overall_first_frame}-{chunk_overall_last_frame}] <<<')

                frames = np.ndarray([len(all_files[cam_idx][chunk_idx]), 2048, 2448], dtype=np.float32)

                for frame_idx in range(len(all_files[cam_idx][chunk_idx])):
                    frame = skimage.io.imread(all_files[cam_idx][chunk_idx][frame_idx])
                    frame = skimage.color.rgb2gray(frame)
                    frames[frame_idx] = frame

                gpu_half_frames = None
                gpu_frames = torch.from_numpy(frames).bfloat16().to(device).reshape(-1, 1, frames.shape[1], frames.shape[2])

                if check_half:
                    half_frames = np.ndarray([len(all_files[cam_idx][chunk_idx]), 1024, 1224], dtype=np.float32)
                    for frame_idx in range(len(frames)):
                        half_frame = skimage.transform.rescale(frame, 0.5, anti_aliasing=True)
                        half_frames[frame_idx] = half_frame
                    gpu_half_frames = torch.from_numpy(half_frames).bfloat16().to(device).reshape(-1, 1, half_frames.shape[1], half_frames.shape[2])

                print(f'> Setup took {(time.time() - t_chunk):.02f}s')

                ### ### ### ###
                ### ### ### ###
                ### ### ### ###

                t_step = time.time()
                chunk_markers = [0 for _ in range(len(frames))]
                chunk_confidence = [0 for _ in range(len(frames))]
                chunk_edge_pts = [0 for _ in range(len(frames))]
                chunk_markers_in_range = [0 for _ in range(len(frames))]
                chunk_edge_imgs = [0 for _ in range(len(frames))]

                chunk_checked_markers = [0 for _ in range(len(frames))]
                chunk_blocks = [0 for _ in range(len(frames))]

                futures = []

                def done_edge_imgs(future):
                    frame_idx, edge_imgs, edge_pts, markers_in_range, markers, confidence = future.result()

                    chunk_markers_in_range[frame_idx] = markers_in_range
                    chunk_confidence[frame_idx] = confidence
                    chunk_markers[frame_idx] = markers
                    chunk_edge_pts[frame_idx] = edge_pts
                    chunk_edge_imgs[frame_idx] = edge_imgs

                def done_block_candidates(future):
                    frame_idx, checked_markers, block_candidates = future.result()

                    chunk_checked_markers[frame_idx] = checked_markers
                    chunk_blocks[frame_idx] = block_candidates

                for frame_idx in range(len(frames)):
                    t_conv = time.time()

                    if check_half:
                        h = gpu_half_frames[frame_idx]
                    else:
                        h = None
                    # 0.86s / frame, GPU constant ~100%, lots of VRAM usage
                    # process_conv_inference_1(gpu_frames, gpu_half_frames, all_out_img, all_out_img_max, all_out_img_patch_cnt, conv_model, stride, block_size)
                    # 0.85s / frame, GPU spikes ~70-80%, constantly low VRAM usage
                    out_img = process_conv_inference_3(frame_idx, gpu_frames[frame_idx], h, conv_model, stride, block_size)

                    fut = executor.submit(process_edge_imgs, frame_idx, frames[frame_idx], out_img, marker_confidence_thr, marker_distance_thr)
                    fut.add_done_callback(done_edge_imgs)
                    futures.append(fut)

                    print(f'\t> Conv inference [{frame_idx}] took {(time.time() - t_conv):.02f}s')

                for f in futures:
                    f.result()

                f = []

                del gpu_half_frames
                del gpu_frames
                if check_half:
                    del half_frames
                gc.collect()
                torch.cuda.empty_cache()

                print(f">>> Step 1 complete: Conv & combine patches took {(time.time() - t_step):.02f}s <<<")

                t_step = time.time()

                for frame_idx in range(len(frames)):
                    t_edge = time.time()

                    pred = process_edge_model_2(chunk_edge_imgs[frame_idx], edge_model, device)
                    # edge_imgs = torch.from_numpy(chunk_edge_imgs[frame_idx]).to(device)
                    # pred = edge_model(edge_imgs).sigmoid().cpu().numpy().flatten()
                    # torch.cuda.empty_cache()

                    fut = executor.submit(process_block_candidates, frame_idx, chunk_markers[frame_idx], chunk_edge_pts[frame_idx], pred, edge_confidence_thr, chunk_markers_in_range[frame_idx])
                    fut.add_done_callback(done_block_candidates)
                    futures.append(fut)

                    print(f'\t> Edge inference [{frame_idx}] took {(time.time() - t_edge):.02f}s')

                del chunk_edge_imgs
                gc.collect()
                torch.cuda.empty_cache()

                for f in futures:
                    f.result()

                print(f">>> Step 2 complete: Edge & block candidates took {(time.time() - t_step):.02f}s <<<")

                if output:
                    j = []

                    for frame_idx in range(len(all_files[cam_idx][chunk_idx])):
                        j.append({
                            "markers": chunk_markers[frame_idx],
                            "confidence": chunk_confidence[frame_idx],
                            "checked_markers": chunk_checked_markers[frame_idx],
                            "blocks": chunk_blocks[frame_idx],
                        })

                    prefix = f"camera_{utils.leading_zeros(cam_idx,3)}_chunk_{utils.leading_zeros(chunk_idx,3)}"
                    export(j, output, prefix, chunk_first_frame, chunk_last_frame, chunk_overall_first_frame, chunk_overall_last_frame, update_output)

                t_chunk = time.time() - t_chunk
                print(f'>>> Chunk idx [{chunk_idx}/{chunk_cnt-1}]" took {t_chunk:.02f}s (Avg: {(t_chunk / (chunk_overall_last_frame+1-chunk_overall_first_frame)):.02f}s) | Frames: {chunk_overall_last_frame+1-chunk_overall_first_frame} (Range: [{chunk_overall_first_frame}-{chunk_overall_last_frame}]) <<<\n')

            t_cam = time.time() - t_cam
            print(f'>>> Camera idx {cam_idx} took {t_cam:.02f}s (Avg: {(t_cam / cam_frames_count):.02f}s) | Frames: {cam_frames_count} (Range: [{cam_first_frame}-{cam_last_frame}]) <<<\n')

    if not update_output:
        delete_files = [join(output,f) for file_idx, f in enumerate(listdir(output)) if f.startswith(f"camera_{utils.leading_zeros(cam_idx,3)}") and file_idx > chunk_cnt-1]
        for f in delete_files:
            remove(f)

    all_frames_count = sum([len(all_files[cam_idx][chunk_idx]) for chunk_idx in range(len(all_files[cam_idx])) for cam_idx in range(len(all_files))])
    t_global = time.time() - t_global
    print(
        f'>>> Processed camera idx {camera_ids} ({chunk_cnt} chunks / cam) in {t_global:.02f}s (Avg: {(t_global / all_frames_count):.02f}s) | Cam frame range: [{cam_first_frame}-{cam_last_frame}] | Frames: {all_frames_count} <<<\n')


if __name__ == '__main__':
    main()
