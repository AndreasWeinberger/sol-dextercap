import scipy.spatial
from . import utils

import skimage
import numpy as np
import scipy
import cv2

import os
import time

from typing import Tuple
import concurrent.futures
import json


def subpixel_refine(gray: np.ndarray, markers: np.ndarray, blocks: np.ndarray, *, win_size: Tuple[int, int] = (5, 5), zero_zone: Tuple[int, int] = (-1, -1), refine_iter: int = 2):
    if len(markers) == 0:
        return markers, blocks, [], []

    if np.issubdtype(gray.dtype, np.integer):
        gray = gray / 255.0
    gray = gray.astype(np.float32)

    num_markers = markers.shape[0]
    markers_subpixel = markers.reshape(num_markers, 1, 2).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)

    for iter in range(refine_iter):
        markers_subpixel = cv2.cornerSubPix(gray, markers_subpixel, winSize=win_size, zeroZone=zero_zone, criteria=criteria)

        correction = markers_subpixel.squeeze(axis=1) - markers
        # print(f'    subpixel correction {iter+1}/{refine_iter}: max {np.abs(correction).max():0.4f} pixels.')

    markers_subpixel = markers_subpixel.squeeze(axis=1).astype(float)

    # some markers may move together during this operation, we can use this result to merge mislabeled markers and blocks
    dist = scipy.spatial.distance_matrix(markers_subpixel, markers_subpixel)
    closed_markers = dist < 1.0  # 1 pixel is small enough
    marker_map = np.arange(num_markers)
    marker_to_remove = np.zeros(num_markers, dtype=bool)
    new_num_marker = 0
    for i in range(num_markers):
        if marker_map[i] != i:
            marker_to_remove[i] = True
            continue
        neighbors = np.nonzero(closed_markers[i, i:])[0]
        marker_map[neighbors+i] = new_num_marker

        new_num_marker += 1

    removed_markers = np.nonzero(marker_to_remove)[0]
    removed_blocks = []

    assert num_markers - new_num_marker == marker_to_remove.sum()
    if num_markers == num_markers:
        # no marker need to be removed
        blocks_refined = blocks
    else:
        markers_subpixel = markers_subpixel[np.logical_not(marker_to_remove)]
        blocks_refined = []
        block_set = []
        for blk_idx, blk in enumerate(blocks):
            new_blk = [marker_map[i] for i in blk]
            new_blk_ck = tuple(new_blk.sort())
            if new_blk_ck in block_set:
                removed_blocks.append(blk_idx)
                continue

            block_set.add(new_blk_ck)
            blocks_refined.append(new_blk)

        print(f'    {num_markers - num_markers} markers and {len(blocks) - len(blocks_refined)} blocks are removed.')

    return markers_subpixel, blocks_refined, removed_markers, removed_blocks


def process_frame(frame_idx: int, fn: str, all_markers: list):
    # t_start = time.time()

    frame = skimage.io.imread(fn)
    gray = skimage.color.rgb2gray(frame)

    markers = np.asarray(all_markers['checked_markers'])
    blocks = np.asarray([blk[1] for blk in all_markers['blocks']])
    block_labels = all_markers.get('block_labels', None)
    assert block_labels is None or len(block_labels) == len(blocks)

    markers_subpixel, blocks_refined, removed_markers, removed_blocks = subpixel_refine(gray, markers, blocks)

    block_labels_refined = block_labels
    if block_labels is not None and len(removed_blocks) > 0:
        block_labels_refined = [lbl for i, lbl in enumerate(block_labels) if not i in removed_blocks]

    refined_markers = markers_subpixel.tolist()
    refined_blocks = [(
        cv2.contourArea(markers_subpixel[list(blk)].reshape(-1, 1, 2).astype(np.float32)),
        blk.tolist()
    ) for blk in blocks_refined]

    refined_bock_labels = []

    if block_labels_refined is not None:
        refined_bock_labels = block_labels_refined

    # print(f'\t> Refine markers [{frame_idx}] | Time: {time.time() - t_start:.02f}s')

    return frame_idx, refined_markers, refined_blocks, refined_bock_labels, len(removed_markers)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, required=True, help="Images or videos")
    # parser.add_argument('-o', '--output', type=str, required=True, help="Yields refined markers and video")

    # parser.add_argument('--override', default=False, action='store_true', help='override current marker and block labels')
    # parser.add_argument('--save-video', type=bool, default=True)
    # parser.add_argument('--start-frame', type=int, default=0)
    # parser.add_argument('--end-frame', type=int, default=-1, help="-1 for all frames in a folder")
    # parser.add_argument('--gpu', type=int, default=0, help="Which gpu to use (if there are multiple)")

    # args = parser.parse_args()

    t_global = time.time()

    # Tuple of ids e.g.: (0,1,2,3,4,5,6) or (0,) for only cam_idx 0
    cam_ids = "0,1,2,3,4,5,6,7,8,9,10,11,12"
    cam_first_frame = 0
    cam_last_frame = 199
    workers = 12
    chunk_size = 64
    render_video = False

    chunks_folder = "./Out/synth_06/block_labels/"
    imgs_folder = "./Blender/renders/cameras_D/"
    output = "./Out/synth_06/refined_markers/"

    if output:
        os.makedirs(output, exist_ok=True)

    cam_ids = tuple(int(x) for x in cam_ids.split(","))

    all_files = utils.get_file_names_with_cam_ids(cam_ids, imgs_folder, cam_first_frame, cam_last_frame)

    if chunk_size == -1:
        chunk_cnt = 1
        chunk_size = cam_last_frame+1-cam_first_frame
    else:
        chunk_cnt = int((cam_last_frame-cam_first_frame) / chunk_size)+1

    if chunk_cnt > 1:
        all_files = utils.chunk_cam_file_names(all_files, chunk_cnt, chunk_size)

    print(f'>>> Running ProcessPoolExecutor() with [{workers}] workers <<<')
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        for _ in range(workers):
            executor.submit(time.sleep, .1)

        all_frames_cnt = 0
        all_removed_markers_cnt = 0

        for cam_idx in cam_ids:
            t_cam = time.time()
            _, cam_chunk_files = utils.folder_contents(chunks_folder, True, files_filter_lambda=lambda x: x.startswith(
                f'cam_{utils.leading_zeros(cam_idx,3)}') and x.endswith('block_labels.json'))

            if len(cam_chunk_files) == 0:
                print(f'> Skipping cam idx {cam_idx} | No chunk files!')
                continue

            print(f'> Starting cam idx {cam_idx} | Chunks: {chunk_cnt} | Range: [{cam_first_frame}-{cam_last_frame}]')

            cam_frames_cnt = 0
            cam_removed_markers_cnt = 0
            # ffmpeg_process = None

            for chunk_idx in range(chunk_cnt):

                t_chunk = time.time()

                chunk_first_frame = (chunk_idx)*chunk_size
                chunk_last_frame = min(((chunk_idx+1)*chunk_size)-1, cam_last_frame - cam_first_frame)
                chunk_overall_first_frame = chunk_first_frame + cam_first_frame
                chunk_overall_last_frame = chunk_last_frame + cam_first_frame
                chunk_removed_markers_cnt = 0

                print(f'> Starting chunk [{chunk_idx+1}/{len(cam_chunk_files)}] | Frames: {chunk_size} | Range: [{chunk_overall_first_frame}-{chunk_overall_last_frame}]')

                with open(cam_chunk_files[chunk_idx]) as f:
                    all_markers = json.load(f)

                c_files = all_files[cam_idx][chunk_idx]

                cam_frames_cnt += len(c_files)

                futures = []

                # print(f'> Setup took {(time.time() - t_chunk):.02f}s')

                def done_refine_markers(future):
                    frame_idx, refined_markers, refined_blocks, refined_bock_labels, removed_markers_cnt = future.result()

                    all_markers[frame_idx]['refined_markers'] = refined_markers
                    all_markers[frame_idx]['refined_blocks'] = refined_blocks
                    all_markers[frame_idx]['refined_bock_labels'] = refined_bock_labels
                    nonlocal chunk_removed_markers_cnt
                    chunk_removed_markers_cnt += removed_markers_cnt

                for frame_idx in range(len(c_files)):
                    fut = executor.submit(process_frame, frame_idx, c_files[frame_idx], all_markers[frame_idx])
                    fut.add_done_callback(done_refine_markers)
                    futures.append(fut)

                for f in futures:
                    f.result()

                cam_removed_markers_cnt += chunk_removed_markers_cnt

                if output:
                    os.makedirs(output, exist_ok=True)
                    fn = os.path.join(output, f'cam_{utils.leading_zeros(cam_idx,3)}_chunk_{utils.leading_zeros(chunk_idx,3)}_[{chunk_first_frame}-{chunk_last_frame}]_refined_markers.json')
                    with open(fn, 'w+', 1) as f:
                        json_string = json.dumps(all_markers, separators=(',', ":"))  # Compact JSON structure
                        f.write(json_string)

                t_chunk = time.time() - t_chunk
                print(f'> Chunk [{chunk_idx+1}/{chunk_cnt}] | Time: {t_chunk:.02f}s (Avg {(t_chunk / len(c_files)):.02f}s / frame) | Frames: {len(c_files)} (Chunk range: [{chunk_first_frame}-{chunk_last_frame}]) | Markers removed: {chunk_removed_markers_cnt}\n')

            all_frames_cnt += cam_frames_cnt
            all_removed_markers_cnt += cam_removed_markers_cnt

            t_cam = time.time() - t_cam
            print(
                f'> Cam idx {cam_idx} took {t_cam:.02f}s (Avg: {(t_cam / cam_frames_cnt):.02f}s) | Frames: {cam_frames_cnt} (Range: [{cam_first_frame}-{cam_last_frame}]) | Markers removed: {cam_removed_markers_cnt}\n')

    t_global = time.time() - t_global
    print(
        f'> Processed {cam_ids} ({chunk_cnt} chunks / cam) in {t_global:.02f}s (Avg: {(t_global / all_frames_cnt):.02f}s) | Cam frame range: [{cam_first_frame}-{cam_last_frame}] | Frames: {all_frames_cnt}  | Markers removed: {all_removed_markers_cnt}\n')


if __name__ == '__main__':
    main()
