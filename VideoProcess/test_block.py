import torch
from torch import nn
from torch.utils.data import DataLoader
import skimage
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

# from sklearn import metrics
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

from skimage import filters, morphology

from . import utils
from .models import BlockNet
# from models import BlockNet_1 as BlockNet
from .datasets import BlockCodeDataset

import cv2

import os
import time
import itertools

import argparse


@torch.no_grad()
def infer_image(model: nn.Module, image: np.array, markers: np.ndarray, blocks: np.ndarray, confidence_thr: float = 0.5):
    if len(blocks) == 0:
        return [], [], []

    blk_imgs = []
    for blk in blocks:
        corners = markers[blk, :]
        blk_img = dataset.extract_block(image, corners)
        blk_imgs.append(blk_img.astype(np.float32))

    # User defined grid of blocks
    # size = 6
    # for i in range(pow(size,2)):
    #     if i < len(blk_imgs):
    #         plt.subplot(size, size, i + 1)
    #         plt.imshow(blk_imgs[i])
    #     else:
    #         break
    # plt.show()

    # First block
    # plt.subplot(2, 2, 1)
    # plt.imshow(image)
    # plt.subplot(2, 2, 2)
    # plt.imshow(blk_imgs[0])
    # plt.show()

    blk_imgs = np.stack(blk_imgs, axis=0).reshape(-1, 1, dataset.image_size, dataset.image_size)
    blk_imgs = torch.from_numpy(blk_imgs).to(device)

    label_0, label_1, blk_dir = model(blk_imgs)
    label_0 = torch.softmax(label_0, dim=-1)
    label_1 = torch.softmax(label_1, dim=-1)
    blk_dir = torch.softmax(blk_dir, dim=-1)

    label_0 = label_0.cpu().numpy()
    label_1 = label_1.cpu().numpy()
    blk_dir = blk_dir.cpu().numpy()

    label_0_idx = np.argmax(label_0, axis=-1)
    label_1_idx = np.argmax(label_1, axis=-1)
    blk_dir_idx = np.argmax(blk_dir, axis=-1)

    label_0_char = [(dataset.label_characters_0[idx], label_0[i, idx]) for i, idx in enumerate(label_0_idx)]
    label_1_char = [(dataset.label_characters_1[idx], label_1[i, idx]) for i, idx in enumerate(label_1_idx)]
    blk_dir_idx = [(idx, blk_dir[i, idx]) for i, idx in enumerate(blk_dir_idx)]

    # print(label_0_char, label_1_char, blk_dir_idx)

    return label_0_char, label_1_char, blk_dir_idx


def draw_block_code(image: np.ndarray, markers: np.ndarray, blocks: np.ndarray, label_0_char, label_1_char, blk_dir, wrap_text: bool, label_confidence_thr: float = 0.5):
    for blk_idx, blk in enumerate(blocks):
        corners = markers[blk, :]
        label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
        label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]

        if label_confidence < label_confidence_thr:
            continue

        direction = blk_dir[blk_idx][0]
        dir_confidence = blk_dir[blk_idx][1]

        pt = np.round(np.mean(corners, axis=0)).astype(int)

        if not wrap_text:
            dir_color = dir_confidence * 1.0
            if direction == 0:
                image[max(pt[1]-10, 0):pt[1], pt[0]] = dir_color
            if direction == 2:
                image[pt[1]+1:pt[1]+10, pt[0]] = dir_color
            if direction == 1:
                image[pt[1], max(pt[0]-10, 0):pt[0]] = dir_color
            if direction == 3:
                image[pt[1], pt[0]+1:pt[0]+10] = dir_color

        h = 1 if not wrap_text else 0.7
        label_text = label[0] if label[0] in ['-', '*'] else label

        if wrap_text:
            text_img_size, _ = cv2.getTextSize('MW', cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_img_size = max(text_img_size) + 1
            text_img = np.zeros((text_img_size, text_img_size), dtype=image.dtype)

            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_w, text_h = text_size

            t_x = max(0, (text_img_size - text_w) // 2)
            t_y = min(max(0, (text_img_size - text_h) // 2 + text_h), text_img_size - 1)
            c = 255
            cv2.putText(text_img, label_text, (t_x, t_y), cv2.FONT_HERSHEY_PLAIN, h, c, 1, cv2.LINE_AA)

            # Set up the destination points for the perspective transform
            src = np.array([
                [0, 0],
                [text_img_size - 1, 0],
                [text_img_size - 1, text_img_size - 1],
                [0, text_img_size - 1]], dtype=np.float32)

            src = np.roll(src, shift=direction, axis=0)

            xy_min = np.floor(corners.min(axis=0)).astype(int)
            xy_max = np.ceil(corners.max(axis=0)).astype(int)
            blk_size = max(xy_max - xy_min)

            # Calculate the perspective transform matrix and apply it
            M = cv2.getPerspectiveTransform(src, (corners - xy_min).astype(np.float32).reshape(-1, 2))
            # warped = cv2.warpPerspective(text_img, M, (blk_size, blk_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped = cv2.warpPerspective(text_img, M, (blk_size, blk_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped = np.clip(warped, 0, 255)

            img_blk = image[xy_min[1]:xy_min[1]+warped.shape[0],
                            xy_min[0]:xy_min[0]+warped.shape[1]]

            text_mask = warped > 0
            img_blk[text_mask] = img_blk[text_mask]*(1-warped[text_mask]) + warped[text_mask]

        else:
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_w, text_h = text_size

            c = 255
            text_img = np.zeros((text_h+1, text_w+1), dtype=image.dtype)
            cv2.putText(text_img, label_text, (0, text_h), cv2.FONT_HERSHEY_PLAIN, h, c, 1, cv2.LINE_AA)

            img_blk = image[max(0, pt[1]-text_img.shape[0]//2):pt[1]+text_img.shape[0]//2,
                            max(0, pt[0]-text_img.shape[1]//2):pt[0]+text_img.shape[1]//2]
            text_img = text_img[:img_blk.shape[0], :img_blk.shape[1]]
            text_mask = text_img > 0

            img_blk[text_mask] = text_img[text_mask]
            img_blk[:] = text_img

    return image


def draw_block_corners(image: np.ndarray, markers: np.ndarray, blocks: np.ndarray, label_0_char, label_1_char, blk_dir):
    h = 0.5
    for blk_idx, blk in enumerate(blocks):
        corners = markers[blk, :]
        label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
        label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]

        direction = blk_dir[blk_idx][0]
        dir_confidence = blk_dir[blk_idx][1]

        code_idx = dataset.code_index(label)
        if code_idx < 0:  # wrong code, '*', or '-'
            continue

        if direction == 4:
            continue

        corners = np.roll(corners, shift=-direction, axis=0)
        center = np.mean(corners, axis=0)
        blk_img = dataset.extract_block(image, corners)
        # print(code_idx, label, direction)
        # plt.imshow(blk_img)
        # plt.show()

        utils.draw_markers(image, corners, radius=1, color=1, inplace=True)

        c = 255
        for i, pt in enumerate(corners):
            pt_id = f'{code_idx}:{i}'
            text_size, _ = cv2.getTextSize(pt_id, cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_w, text_h = text_size
            pt = pt.round().astype(int)
            rel = pt - center

            x, y = pt

            if rel[0] < 0:
                x = x + 1
            else:
                x = x - text_w

            if rel[1] > 0:
                y = y - text_h
            else:
                y = y + text_h

            cv2.putText(image, pt_id, (x, y), cv2.FONT_HERSHEY_PLAIN, h, c, 1, cv2.LINE_AA)

    return image


def infer_block_labels(block_model: nn.Module, input: str, output:str, marker_file, label_confidence_thr: float = 0.5, start_frame: int = 0, end_frame: int = -1):
    with open(marker_file) as f:
        import json
        consolidated_markers = json.load(f)
    in_folder, in_file = os.path.split(input)
    out_folder, out_file = os.path.split(output)

    in_split = in_file.split('.')
    out_split = out_file.split('.')

    if out_split[0] != '':
        output = f'{out_folder}\\{out_split[0]}'
    elif in_split[0] != '':
        output = f'{out_folder}\\{in_split[0]}'

    def process_frame(idx: int, frame: np.ndarray, output:str, all_markers, ffmpeg_process):
    
        # Start processing...
        t_process_start = time.time()

        frame = skimage.color.rgb2gray(frame)

        markers = np.asarray(all_markers[idx]['checked_markers'])
        blocks = np.asarray([blk[1] for blk in all_markers[idx]['blocks']])  # [block_num, 4]

        label_0_char, label_1_char, blk_dir = infer_image(block_model, frame, markers, blocks, confidence_thr=label_confidence_thr)

        block_labels = []
        for blk_idx, blk in enumerate(blocks):
            label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
            label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]
            direction = blk_dir[blk_idx][0]
            dir_confidence = blk_dir[blk_idx][1]

            # if label != '--':
            #    print(f'\tLabel: {label} - conf: {label_confidence}')

            block_labels.append({'label': label, 'label_confidence': float(label_confidence),
                                    'direction': int(direction), 'dir_confidence': float(dir_confidence)})
        all_markers[idx]['block_labels'] = block_labels

        if output != '':
            frame = (frame*255).astype(np.uint8)
            frame_marker = frame.copy()
            frame1_marker = frame.copy()
            frame1_marker_black = np.zeros_like(frame)
            frame2_marker = frame.copy()
            # print(len(block_candidates))
            draw_block_code(frame1_marker, markers, blocks, label_0_char, label_1_char, blk_dir, wrap_text=False, label_confidence_thr=label_confidence_thr)
            draw_block_code(frame1_marker_black, markers, blocks, label_0_char, label_1_char, blk_dir, wrap_text=True, label_confidence_thr=label_confidence_thr)
            draw_block_corners(frame2_marker, markers, blocks, label_0_char, label_1_char, blk_dir)

            block_pts = [markers[block_indices].reshape(-1, 1, 2) for block_indices in blocks]
            cv2.polylines(frame_marker, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)
            cv2.polylines(frame1_marker, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)
            cv2.polylines(frame1_marker_black, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)
            cv2.polylines(frame2_marker, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)

            frame_marker = utils.draw_markers(frame_marker, markers, color=255)
            frame1_marker = utils.draw_markers(frame1_marker, markers, color=255)
            frame1_marker_black = utils.draw_markers(frame1_marker_black, markers, color=255)

            frame_marker = np.concatenate((
                frame_marker[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]],  # [:,w//2:],
                frame1_marker_black[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]],  # [:,w//2:],
            ), axis=1)

            frame_marker = skimage.color.gray2rgb((frame_marker/255).astype(np.float64))

            out_img = (frame_marker*255).astype(np.uint8)

            if ffmpeg_process is None:
                import ffmpeg
                ffmpeg_process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{frame_marker.shape[1] if frame_marker.shape[1] % 2 == 0 else frame_marker.shape[1] + 1}x{frame_marker.shape[0] if frame_marker.shape[0] % 2 == 0 else frame_marker.shape[0] + 1}').output(
                    output, pix_fmt='yuv420p', loglevel="quiet").overwrite_output().run_async(pipe_stdin=True))
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
            print(f'> Video saved to "{output}_block_labels.mp4" in {(time.time() - t_finish_start):.2f}s')
            print('---')

        if len(all_markers) == 0:
            print(f'> No block labels to save!\n')
            print('---')
            return
            
        # End
        t_finish_start = time.time()
        import json
        with open(f'{output}_block_labels.json', 'w') as f:
            json_string = json.dumps(all_markers, separators=(',', ":"))  # Compact JSON structure
            open(f'{output}_block_labels.json', "w+", 1).write(json_string)
            
            all = sum([len(frame['blocks']) for frame in all_markers])
            avg = int(all / len(all_markers))
            print(f'> Saved block labels to "{output}_block_labels.json"\n\n\tTotal: {all}\n\tAverage: {avg}\n\tTime: {(time.time() - t_finish_start):.2f}s\n')
            print('---')
        

    # Start
    t_start_global = time.time()

    if os.path.isfile(input) and in_split[1] == 'mp4':
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

        base = output

        for idx_folder, subfolder in enumerate(subfolders):
            all_markers = consolidated_markers[os.path.basename(subfolder)]
            all_marker_pos = np.concatenate([frame['checked_markers'] for frame in all_markers if len(frame['checked_markers']) > 0], axis=0)
            roi_min = np.maximum(0, np.floor(all_marker_pos.reshape(-1, 2).min(axis=0)).astype(int) - 10)
            roi_max = np.ceil(all_marker_pos.reshape(-1, 2).max(axis=0)).astype(int) + 10
            
            images = [os.path.join(subfolder, frame) for frame in os.listdir(subfolder)]
            if len(images) > 0:
                sorted(images, key=lambda fn: os.path.basename(fn))

            ffmpeg_process = None
            count = 0

            output = os.path.join(base, os.path.basename(subfolder))

            if not os.path.isdir(output):
                os.makedirs(output, exist_ok=True)

            output += f'\\{os.path.basename(subfolder)}'

            # Start
            t_start = time.time()

            for id_img, fn in enumerate(images, start_frame):
                if (end_frame > -1 and id_img > end_frame) or id_img >= len(all_markers):
                    break

                if not os.path.isfile(fn):
                    print(f'> Failed to read "{fn}"')
                    break

                frame = skimage.io.imread(fn)

                all_markers, ffmpeg_process = process_frame(id_img, frame, f'{output}_block_labels.mp4', all_markers, ffmpeg_process)
                count += 1

            finish(output, ffmpeg_process, all_markers)

    print(f'> Processing block labels complete\n\n\tTotal time: {(time.time() - t_start_global):.2f}s')
    print('---')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-model', default='', type=str, required=True)

    parser.add_argument('--input', default='', type=str, required=True, help="Input video/images")
    parser.add_argument('--consolidated-markers', default='', type=str, required=True, help="Marker .json file from conv and edge models")
    parser.add_argument('--labels', default='', type=str, required=True, help="Dataset .json file with annotated data and image fields")
    parser.add_argument('--output', default='', type=str, required=True, help="Output file in .json format")

    parser.add_argument('--label-confidence-thr', default=0.5, type=float, help="Default: 0.5")

    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=-1)
    parser.add_argument('--gpu', type=int, default=0, help="Which gpu to use (if there are multiple)")

    args = parser.parse_args()
    args.dataset_folder, args.labels = os.path.split(args.labels)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = BlockCodeDataset(args.dataset_folder, args.labels, size=128*500, train=True)

    block_model = BlockNet(label0_chars=len(dataset.label_characters_0), label1_chars=len(dataset.label_characters_1), blk_dirs=5).to(device)

    # Load block net model from snapshot or checkpoint
    with open(args.block_model, 'rb') as f:
        data = torch.load(f, map_location=device, weights_only=True)
    m = data
    if 'model' in data:
        m = data['model']
    block_model.load_state_dict(m)
    block_model.eval()

    if len(args.output) != 0:
        if not os.path.isfile(args.output) and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

    print(f'> Input: {args.input}')
    print(f'> Output: {args.output}')
    print(f'> Start Frame: {args.start_frame}')
    print(f'> End Frame: {args.end_frame}')
    print(f'---')

    infer_block_labels(block_model, args.input, args.output, args.consolidated_markers, args.label_confidence_thr, args.start_frame, args.end_frame)
