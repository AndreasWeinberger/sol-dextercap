import scipy.sparse
import scipy.spatial
from . import utils

import skimage
import imageio.v3 as iio
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import json
import argparse

import os
import time
import itertools

from typing import List, Tuple
import multiprocessing


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


def refine_markers(input: str, output: str, override: bool = False, save_video: bool = True, start_frame: int = 0, end_frame: int = -1):

    in_folder, in_file = os.path.split(input)
    out_folder, out_file = os.path.split(output)

    in_split = in_file.split('.')
    out_split = out_file.split('.')

    if out_split[0] != '':
        output = f'{out_folder}\\{out_split[0]}'
    elif in_split[0] != '':
        output = f'{out_folder}\\{in_split[0]}'

    def process_frame(frame_idx: int, frame: np.ndarray, output: str, all_markers, ffmpeg_process):

        all_marker_pos = np.concatenate([frame['checked_markers'] for frame in all_markers if len(frame['checked_markers']) > 0], axis=0)
        # print(all_marker_pos.shape)
        roi_min = np.maximum(0, np.floor(all_marker_pos.reshape(-1, 2).min(axis=0)).astype(int) - 10) // 2 * 2
        roi_max = (np.ceil(all_marker_pos.reshape(-1, 2).max(axis=0)).astype(int)) // 2 * 2 + 10
        # print(roi_min, roi_max)

        # now start to process each frames
        gray = skimage.color.rgb2gray(frame)

        markers = np.asarray(all_markers[frame_idx]['checked_markers'])
        blocks = np.asarray([blk[1] for blk in all_markers[frame_idx]['blocks']])
        block_labels = all_markers[frame_idx].get('block_labels', None)
        assert block_labels is None or len(block_labels) == len(blocks)

        markers_subpixel, blocks_refined, removed_markers, removed_blocks = subpixel_refine(gray, markers, blocks)

        block_labels_refined = block_labels
        if block_labels is not None and len(removed_blocks) > 0:
            block_labels_refined = [lbl for i, lbl in enumerate(block_labels) if not i in removed_blocks]

        if override:
            all_markers[frame_idx]['checked_markers'] = markers_subpixel.tolist()
            all_markers[frame_idx]['blocks'] = [(
                cv2.contourArea(markers_subpixel[list(blk)].reshape(-1, 1, 2).astype(np.float32)),
                blk.tolist()
            ) for blk in blocks_refined]

            if block_labels_refined is not None:
                all_markers[frame_idx]['block_labels'] = block_labels_refined
        else:
            all_markers[frame_idx]['refined_markers'] = markers_subpixel.tolist()
            all_markers[frame_idx]['refined_blocks'] = [(
                cv2.contourArea(markers_subpixel[list(blk)].reshape(-1, 1, 2).astype(np.float32)),
                blk.tolist()
            ) for blk in blocks_refined]

            if block_labels_refined is not None:
                all_markers[frame_idx]['refined_block_labels'] = block_labels_refined

        if save_video:
            frame_marker = frame[:]
            block_pts = [np.round(markers_subpixel[list(block_indices)].reshape(-1, 1, 2)).astype(int) for block_indices in blocks_refined]
            cv2.polylines(frame_marker, block_pts, True, color=[230, 250, 50], thickness=2)

            frame_marker = utils.draw_markers(frame_marker, markers, radius=3, color=(255, 255, 255))
            frame_marker = utils.draw_markers(frame_marker, markers_subpixel, radius=6, color=(20, 255, 20))

            frame_marker = frame_marker[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]][:]

            # frame_marker = skimage.color.gray2rgb(frame_marker)
            # out_img = (frame_marker*255).astype(np.uint8)
            out_img = frame_marker.astype(np.uint8)

            if ffmpeg_process is None:
                import ffmpeg
                ffmpeg_process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{frame_marker.shape[1] if frame_marker.shape[1] % 2 == 0 else frame_marker.shape[1] + 1}x{frame_marker.shape[0] if frame_marker.shape[0] % 2 == 0 else frame_marker.shape[0] + 1}').output(
                    output, pix_fmt='yuv420p', loglevel="quiet").overwrite_output().run_async(pipe_stdin=True))
            ffmpeg_process.stdin.write(out_img)

        return all_markers, ffmpeg_process

    def finish(output, ffmpeg_process, all_markers, start_time):
        if ffmpeg_process is not None:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            print(f'> Video saved to "{output}_refined_markers.mp4"')
            print('---')

        if len(all_markers) == 0:
            print(f'> No markers to save!\n')
            print('---')
            return

        if output != '':
            # End
            import json
            json_string = json.dumps(all_markers, separators=(',', ":"))  # Compact JSON structure
            open(f'{output}_refined_markers.json', "w+", 1).write(json_string)

            sum_markers = sum([len(frame['markers']) for frame in all_markers])
            sum_ref_markers = sum([len(frame['refined_markers']) for frame in all_markers])
            sum_ref_blocks = sum([len(frame['refined_blocks']) for frame in all_markers])

            avg_markers = int(sum_markers / len(all_markers))
            avg_ref_markers = int(sum_ref_markers / len(all_markers))
            avg_ref_blocks = int(sum_ref_blocks / len(all_markers))
            print(f'> Saved refined markers to "{output}_refined_markers.json"\n\n\tFrames: {len(all_markers)}\n\tMarkers: {sum_markers} (Avg: {avg_markers})\n\tRefined markers: {sum_ref_markers} (Avg: {avg_ref_markers})\n\tRefined blocks: {sum_ref_blocks} (Avg: {avg_ref_blocks})\n\tTime: {(time.time() - start_time):.2f}s\n')
            print('---')

    # Start
    t_start_global = time.time()

    if os.path.isfile(input) and in_split[1] == 'mp4':
        with open(f'{output}_markers.json') as f:
            all_markers = json.load(f)
        ffmpeg_process = None
        count = 0

        for frame_idx, frame in enumerate(iio.imiter(input, plugin="pyav"), start_frame):
            if end_frame > -1 and frame_idx > end_frame:
                break

            all_markers, ffmpeg_process = process_frame(frame_idx, frame, f'{output}_refined_markers.mp4', all_markers, ffmpeg_process)
            count += 1

        finish(output, ffmpeg_process, all_markers)

    else:
        subfolders = os.listdir(input)
        if len(subfolders) > 0:
            sorted(subfolders, key=lambda fn: fn)
            subfolders = [os.path.join(input, folder) for folder in subfolders]
        else:
            subfolders = in_folder

        base_output = output

        for folder_idx, subfolder in enumerate(subfolders):
            images = [os.path.join(subfolder, frame) for frame in os.listdir(subfolder)]
            if len(images) > 0:
                sorted(images, key=lambda fn: os.path.basename(fn))

            output = os.path.join(base_output, os.path.basename(subfolder))

            if not os.path.isdir(output):
                os.makedirs(output, exist_ok=True)

            output += f'\\{os.path.basename(subfolder)}'

            with open(f'{output}_markers.json') as f:
                all_markers = json.load(f)

            ffmpeg_process = None
            count = 0

            # Start
            t_start = time.time()

            for frame_idx, frame in enumerate(images, start_frame):
                if (end_frame > -1 and frame_idx > end_frame) or frame_idx > len(all_markers) - 1:
                    break

                if not os.path.isfile(frame):
                    print(f'> Failed to read "{frame}"')
                    break

                frame = skimage.io.imread(frame)

                all_markers, ffmpeg_process = process_frame(frame_idx, frame, f'{output}_refined_markers.mp4', all_markers, ffmpeg_process)
                count += 1

            finish(output, ffmpeg_process, all_markers, t_start)

    print(f'> Refining markers complete\n\tTotal time: {(time.time() - t_start_global):.2f}s')
    print('---')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="Images or videos")
    parser.add_argument('-o', '--output', type=str, required=True, help="Yields refined markers and video")

    parser.add_argument('--override', default=False, action='store_true', help='override current marker and block labels')
    parser.add_argument('--save-video', type=bool, default=True)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=-1, help="-1 for all frames in a folder")
    parser.add_argument('--gpu', type=int, default=0, help="Which gpu to use (if there are multiple)")

    args = parser.parse_args()

    if len(args.output) != 0:
        if not os.path.isfile(args.output) and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

    print(f'> Input: {args.input}')
    print(f'> Output: {args.output}')
    print(f'> Override: {args.override}')
    print(f'> Save video: {args.save_video}')
    print(f'> Start frame: {args.start_frame}')
    print(f'> End frame: {args.end_frame}')
    print(f'---')

    refine_markers(args.input, args.output, args.override, args.save_video, args.start_frame, args.end_frame)

    # with multiprocessing.Pool(processes=len(cam_list)) as pool:
    #    pool.map(refine, cam_list)


if __name__ == '__main__':
    main()
