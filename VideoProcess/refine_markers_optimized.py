import scipy.spatial
import utils

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


def process_frame(frame_idx: int, fn: str, all_markers:list):
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

    return frame_idx, refined_markers, refined_blocks, refined_bock_labels


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

    start_frame = 0
    end_frame = 99
    workers = 12
    chunk_size = 92

    markers_chunks_folder = "./Out/synth_01/block_labels/"
    _, markers_chunks_files = utils.folder_contents(markers_chunks_folder, True, files_filter_lambda=lambda x: x.endswith('.json'))

    input_folder = "./Blender/renders/cameras/"
    input = "./Blender/renders/cameras/"
    all_files = []
    for cam in os.listdir(input):
        _, folder_files = utils.folder_contents(os.path.join(input, cam))
        all_files.extend(folder_files[start_frame:end_frame+1])

    output = "./Out/synth_01/refined_markers/"

    chunks = int(len(all_files) / chunk_size) + 1
    start_chunk = int(start_frame / chunk_size)+1

    print(f'>>> Setup took {(time.time() - t_global):.02f}s')

    print(f'>>> Running ProcessPoolExecutor() with [{workers}] workers <<<')
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        for c in range(start_chunk, chunks + 1, 1):

            # print(f'> Starting chunk [{c}/{chunks}] | Frames [{start_frame}-{end_frame}] | Total [{len(files)}]')
            t_chunk = time.time()

            c_first_frame = (c-1)*chunk_size
            c_last_frame = min(len(all_files), c*chunk_size)-1
            c_files = all_files[c_first_frame:c_last_frame+1]

            with open(markers_chunks_files[c-1]) as f:
                all_markers = json.load(f)

            def done_refine_markers(future):
                frame_idx, refined_markers, refined_blocks, refined_bock_labels = future.result()

                all_markers[frame_idx]['refined_markers'] = refined_markers
                all_markers[frame_idx]['refined_blocks'] = refined_blocks
                all_markers[frame_idx]['refined_bock_labels'] = refined_bock_labels

            futures = []

            for frame_idx in range(len(c_files)):
                fut = executor.submit(process_frame, frame_idx, c_files[frame_idx], all_markers[frame_idx])
                fut.add_done_callback(done_refine_markers)
                futures.append(fut)

            for f in futures:
                f.result()

            if output:
                os.makedirs(output, exist_ok=True)
                fn = os.path.join(output, f'refined_markers_chunk_{utils.leading_zeros(c,4)}_frames_[{c_first_frame}-{c_last_frame}].json')
                with open(fn, 'w+', 1) as f:
                    json_string = json.dumps(all_markers, separators=(',', ":"))  # Compact JSON structure
                    f.write(json_string)

            t_chunk = time.time() - t_chunk
            print(f'> Chunk [{c}/{chunks}] | Time: {t_chunk:.02f}s (Avg {(t_chunk / len(all_markers)):.02f}s / frame) | Frames: {len(c_files)} (Chunk range: [{c_first_frame}-{c_last_frame}])')

    t_global = time.time() - t_global
    print(f'>>> Processed [{c}/{chunks}] chunks | Time: {t_global:.02f}s (Avg: {(t_global / (len(all_files))):.02f}s) | Frames: {len(all_files)} (Cam range: [{start_frame}-{end_frame}] <<<')


if __name__ == '__main__':
    main()
