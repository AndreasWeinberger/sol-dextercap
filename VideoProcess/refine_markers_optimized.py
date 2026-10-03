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


def process_frame(x: list):
    t_start = time.time()
    idx, fn, m = x

    frame = skimage.io.imread(fn)
    gray = skimage.color.rgb2gray(frame)

    markers = np.asarray(m['checked_markers'])
    blocks = np.asarray([blk[1] for blk in m['blocks']])
    block_labels = m.get('block_labels', None)
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

    print(f'\t> Refine markers took {time.time() - t_start:.02f}s')

    return idx, refined_markers, refined_blocks, refined_bock_labels


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

    start_frame = 0
    end_frame = 99
    workers = 12
    chunk_size = 92

    markers_chunks_folder = "./Out/synth_01/markers/"
    _, markers_chunks_files = utils.folder_contents(markers_chunks_folder, True, files_filter_lambda=lambda x: x.endswith('.json'))

    all_markers = []


    for file in markers_chunks_files:
        import json
        with open(file) as f:
            chunk = json.load(f)
            for frame_idx in range(len(chunk['chunk_markers'])):
                all_markers.append({
                    "markers": chunk['chunk_markers'][frame_idx],
                    "confidence": chunk['chunk_confidence'][frame_idx],
                    "checked_markers": chunk['chunk_checked_markers'][frame_idx],
                    "blocks": chunk['chunk_blocks'][frame_idx],
                })

    input_folder = "./Blender/renders/cameras/"
    image_files = []
    for cam in os.listdir(input_folder):
        _, f = utils.folder_contents(os.path.join(input_folder, cam))
        image_files.extend(f[start_frame:end_frame+1])

    assert len(all_markers) == len(image_files), f"Got {len(all_markers)} frames ({len(markers_chunks_files)} chunks) but {len(image_files)} image files. They need to be equal length!"

    output = "./Out/synth_01/refined_markers/"

    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        t_global = time.time()
        print(f'> Start refine markers...')
        t_refine_markers = time.time()
        for frame_idx, refined_markers, refined_blocks, refined_bock_labels in executor.map(process_frame, [[frame_idx,image_files[frame_idx], all_markers[frame_idx]] for frame_idx in range(len(all_markers))]):
            all_markers[frame_idx]['refined_markers'] = refined_markers
            all_markers[frame_idx]['refined_blocks'] = refined_blocks
            all_markers[frame_idx]['refined_bock_labels'] = refined_bock_labels
        t_refine_markers = time.time() - t_refine_markers
        print(f'> Process refine markers took {t_refine_markers:.02f}s (Avg {(t_refine_markers / len(all_markers)):.02f}s per frame ({len(all_markers)} frames))')

        if output:
            os.makedirs(output, exist_ok=True)
            import json
            fn = os.path.join(output, f'refined_markers_[{len(all_markers)}].json')
            with open(fn, 'w+', 1) as f:
                json_string = json.dumps(all_markers, separators=(',', ":"))  # Compact JSON structure
                f.write(json_string)

    t_global = time.time() - t_global
    print(f'>>> Processed [{len(markers_chunks_files)}/{len(markers_chunks_files)}] chunks in {t_global:.02f}s (Avg: {(t_global / (len(all_markers))):.02f}s) | Frames [{start_frame}-{end_frame}] | Total [{len(all_markers)}] <<<')

if __name__ == '__main__':
    main()
