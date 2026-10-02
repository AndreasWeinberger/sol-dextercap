import torch
from torch import nn
import skimage
import numpy as np

import utils
from models import BlockNet
from datasets import BlockCodeDataset

import cv2

import os
import time

import concurrent.futures


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


def draw_block_corners(image: np.ndarray, markers: np.ndarray, blocks: np.ndarray, dataset, label_0_char, label_1_char, blk_dir):
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


@torch.inference_mode()
def process_block_inference(all_blk_imgs: torch.Tensor, block_model: nn.Module, dataset: BlockCodeDataset):
    if len(all_blk_imgs) == 0:
        return [], [], []

    all_label_0_char = []
    all_label_1_char = []
    all_blk_dir_idx = []

    for blk_imgs in all_blk_imgs:
        t_start = time.time()

        label_0, label_1, blk_dir = block_model(blk_imgs)
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

        all_label_0_char.append(label_0_char)
        all_label_1_char.append(label_1_char)
        all_blk_dir_idx.append(blk_dir_idx)

        print(f'\t> Process block model took {(time.time() - t_start):.02f}s')

    return all_label_0_char, all_label_1_char, all_blk_dir_idx


def process_block_labels(x: list):

    t_start = time.time()

    idx, blocks, label_0_char, label_1_char, blk_dir = x

    block_labels = []
    for blk_idx, blk in enumerate(blocks):
        label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
        label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]
        direction = blk_dir[blk_idx][0]
        dir_confidence = blk_dir[blk_idx][1]

        block_labels.append({'label': label, 'label_confidence': float(label_confidence),
                             'direction': int(direction), 'dir_confidence': float(dir_confidence)})

    print(f'\t> Process block labels took {time.time() - t_start:.2f}s')

    return idx, block_labels


@torch.inference_mode()
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--block-model', default='', type=str, required=True)

    # parser.add_argument('--input', default='', type=str, required=True, help="Input video/images")
    # parser.add_argument('--labels', default='', type=str, required=True, help="Dataset .json file with annotated data and image fields")
    # parser.add_argument('--output', default='', type=str, required=True, help="Output file in .json format")

    # parser.add_argument('--label-confidence-thr', default=0.7, type=float, help="Default: 0.5")

    # parser.add_argument('--start-frame', type=int, default=0)
    # parser.add_argument('--end-frame', type=int, default=-1)
    # parser.add_argument('--gpu', type=int, default=0, help="Which gpu to use (if there are multiple)")

    # args = parser.parse_args()

    t_global = time.time()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    start_frame = 0
    end_frame = 9
    workers = 12

    labels = "./Data/synth_01/dataset/train/labels.json"
    dataset_folder, labels = os.path.split(labels)
    dataset = BlockCodeDataset(dataset_folder, labels, size=128*500, train=True)

    block_model = BlockNet(label0_chars=len(dataset.label_characters_0), label1_chars=len(dataset.label_characters_1), blk_dirs=5).to(device)
    block_model.load_state_dict(torch.load("./ckpts/0005/block/best_lr_0.001_bs_512_model_156_f1_score=0.9857.pt", weights_only=True, map_location=device))
    block_model.eval()
    # block_model.to(dtype=torch.bfloat16)

    # block_model = torch.compile(
    #     block_model,
    #     fullgraph=False,
    #     dynamic=True,
    #     mode="reduce-overhead"
    # )

    import json
    with open(os.path.join("./Out/synth_01/refined_markers/", os.listdir("./Out/synth_01/refined_markers/")[0])) as f:
        refined_markers = json.load(f)

    input_folder = "./Blender/renders/cameras/"
    image_files = []
    for cam in os.listdir(input_folder):
        _, f = utils.folder_contents(os.path.join(input_folder, cam))
        image_files.extend(f[start_frame:end_frame+1])

    output = "./Out/synth_01/block_labels/"

    frames = []
    for fn in image_files:
        frame = skimage.io.imread(fn)
        frame = skimage.color.rgb2gray(frame)
        frames.append(frame)

    all_blk_imgs = []

    for i in range(len(frames)):

        markers = np.asarray(refined_markers[i]['checked_markers'])
        blocks = np.asarray([blk[1] for blk in refined_markers[i]['blocks']])  # [block_num, 4]

        if len(blocks) == 0:
            all_blk_imgs.append([])

        blk_imgs = []
        for blk in blocks:
            corners = markers[blk, :]
            blk_img = dataset.extract_block(frames[i], corners)
            blk_imgs.append(blk_img.astype(np.float32))

        blk_imgs = np.stack(blk_imgs, axis=0).reshape(-1, 1, dataset.image_size, dataset.image_size)
        blk_imgs = torch.from_numpy(blk_imgs).to(device)
        all_blk_imgs.append(blk_imgs)

    print('> Start block inference...')
    t_block = time.time()
    all_label_0_char, all_label_1_char, all_blk_dir_idx = process_block_inference(all_blk_imgs, block_model, dataset)
    t_block = time.time() - t_block
    print(f'> Total block inference took {t_block:.02f}s (Avg {(t_block / (len(frames))):.02f}s per frame ({len(frames)} frames))')

    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        print(f'> Start process block labels...')
        t_block_labels = time.time()
        for frame_idx, block_labels in executor.map(process_block_labels, [[frame_idx, refined_markers[frame_idx]['blocks'], all_label_0_char[frame_idx], all_label_1_char[frame_idx], all_blk_dir_idx[frame_idx]] for frame_idx in range(len(refined_markers))]):
            refined_markers[frame_idx]['block_labels'] = block_labels
        t_block_labels = time.time() - t_block_labels
        print(f'> Process block labels took {t_block_labels:.02f}s (Avg {(t_block_labels / len(refined_markers)):.02f}s per frame ({len(refined_markers)} frames))')

        if output:
            os.makedirs(output, exist_ok=True)
            import json
            fn = os.path.join(output, f'block_labels_[{len(refined_markers)}].json')
            with open(fn, 'w+', 1) as f:
                json_string = json.dumps(refined_markers, separators=(',', ":"))  # Compact JSON structure
                f.write(json_string)

    print(f'> Start render video...')
    t_video = time.time()
    vid_path = os.path.join(output, f'block_labels_[{len(refined_markers)}].mp4')
    ffmpeg_process = None

    all_marker_pos = np.concatenate([frame['checked_markers'] for frame in refined_markers if len(frame['checked_markers']) > 0], axis=0)
    roi_min = np.maximum(0, np.floor(all_marker_pos.reshape(-1, 2).min(axis=0)).astype(int) - 10) // 2 * 2
    roi_max = (np.ceil(all_marker_pos.reshape(-1, 2).max(axis=0)).astype(int)) // 2 * 2 + 10

    for i, frame in enumerate(frames):
        frame = (frame*255).astype(np.uint8)
        frame_marker = frame.copy()
        frame1_marker = frame.copy()
        frame1_marker_black = np.zeros_like(frame)
        frame2_marker = frame.copy()

        markers = np.asarray(refined_markers[i]['checked_markers'])
        blocks = np.asarray([blk[1] for blk in refined_markers[i]['blocks']])

        draw_block_code(frame1_marker, markers, blocks, all_label_0_char[i], all_label_1_char[i], all_blk_dir_idx[i], wrap_text=False, label_confidence_thr=0.0)
        draw_block_code(frame1_marker_black, markers, blocks, all_label_0_char[i], all_label_1_char[i], all_blk_dir_idx[i], wrap_text=True, label_confidence_thr=0.0)
        draw_block_corners(frame2_marker, markers, blocks, dataset, all_label_0_char[i], all_label_1_char[i], all_blk_dir_idx[i])

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
                vid_path, pix_fmt='yuv420p', loglevel="quiet").overwrite_output().run_async(pipe_stdin=True))
        ffmpeg_process.stdin.write(out_img)

    if ffmpeg_process is not None:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        t_video = time.time() - t_video
        print(f'> Render video took {t_video:.02f}s Path: "{vid_path}"')

    t_global = time.time() - t_global
    print(f'> Total time for test block: {t_global:.02f}s (Avg {(t_global / (len(refined_markers))):.02f}s per frame ({len(refined_markers)} frames))')


if __name__ == '__main__':
    main()
