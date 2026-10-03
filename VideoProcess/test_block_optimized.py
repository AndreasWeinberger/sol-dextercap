import torch
from torch import nn
import skimage
import numpy as np

import utils
from models import BlockNet
from datasets import BlockCodeDataset_Base

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
@torch.no_grad()
def process_block_inference(frame_idx: int, blk_imgs: torch.Tensor, block_model: nn.Module, label_characters_0: list, label_characters_1: list):

    t_start = time.time()

    label_0, label_1, blk_dir = block_model(blk_imgs)
    label_0 = torch.softmax(label_0, dim=-1)
    label_1 = torch.softmax(label_1, dim=-1)
    blk_dir = torch.softmax(blk_dir, dim=-1)

    label_0 = label_0.float().cpu().numpy()
    label_1 = label_1.float().cpu().numpy()
    blk_dir = blk_dir.float().cpu().numpy()

    label_0_idx = np.argmax(label_0, axis=-1)
    label_1_idx = np.argmax(label_1, axis=-1)
    blk_dir_idx = np.argmax(blk_dir, axis=-1)

    label_0_char = [(label_characters_0[idx], label_0[i, idx]) for i, idx in enumerate(label_0_idx)]
    label_1_char = [(label_characters_1[idx], label_1[i, idx]) for i, idx in enumerate(label_1_idx)]
    blk_dir_idx = [(idx, blk_dir[i, idx]) for i, idx in enumerate(blk_dir_idx)]

    del label_0
    del label_1
    del blk_dir
    torch.cuda.empty_cache()

    print(f'\t> Frame: {frame_idx} | Block inference took {(time.time() - t_start):.02f}s')

    return label_0_char, label_1_char, blk_dir_idx


def process_block_labels(frame_idx, blocks, label_0_char, label_1_char, blk_dir):

    t_start = time.time()

    block_labels = []
    for blk_idx, blk in enumerate(blocks):
        label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
        label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]
        direction = blk_dir[blk_idx][0]
        dir_confidence = blk_dir[blk_idx][1]

        block_labels.append({'label': label, 'label_confidence': float(label_confidence),
                             'direction': int(direction), 'dir_confidence': float(dir_confidence)})

    print(f'\t> Frame {frame_idx} | Process block labels took {(time.time() - t_start):.2f}s')

    return frame_idx, block_labels

@torch.inference_mode()
@torch.no_grad()
def prepare_blk_imgs(frame_idx: int, fn: str, refined_markers: list, dataset: BlockCodeDataset_Base, device):
    t_start = time.time()

    frame = skimage.io.imread(fn)
    frame = skimage.color.rgb2gray(frame)

    markers = np.asarray(refined_markers['checked_markers'])
    blocks = np.asarray([blk[1] for blk in refined_markers['blocks']])  # [block_num, 4]

    if len(blocks) == 0:
        return (frame_idx, [])

    blk_imgs = []
    for blk in blocks:
        corners = markers[blk, :]
        blk_img = dataset.extract_block(frame, corners)
        blk_imgs.append(blk_img.astype(np.float32))

    blk_imgs = np.stack(blk_imgs, axis=0).reshape(-1, 1, dataset.image_size, dataset.image_size)

    blk_imgs = np.resize(blk_imgs, (300, blk_imgs.shape[1], blk_imgs.shape[2], blk_imgs.shape[3]))  # consistent shape for CUDA graph compile of block model

    blk_imgs = torch.from_numpy(blk_imgs).bfloat16().to(device)

    print(f'\t> Frame: {frame_idx} | Prepare block imgs took {(time.time() - t_start):.2f}s')

    return frame_idx, blk_imgs, len(blocks)


@torch.inference_mode()
@torch.no_grad()
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

    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    start_frame = 0
    end_frame = 99
    workers = 6
    chunk_size = 32

    labels = "./Data/synth_01/dataset/train/labels.json"
    dataset_folder, labels = os.path.split(labels)
    dataset = BlockCodeDataset_Base(dataset_folder, labels)

    block_model = BlockNet(label0_chars=len(dataset.label_characters_0), label1_chars=len(dataset.label_characters_1), blk_dirs=5).to(device)
    block_model.load_state_dict(torch.load("./ckpts/0005/block/best_lr_0.001_bs_512_model_156_f1_score=0.9857.pt", weights_only=True, map_location=device))
    block_model.to(dtype=torch.bfloat16)

    # block_model.eval()

    # block_model = torch.compile(
    #     block_model,
    #     fullgraph=True,
    #     dynamic=False,
    #     mode="reduce-overhead"
    # )

    import json
    with open(os.path.join("./Out/synth_01/refined_markers/", os.listdir("./Out/synth_01/refined_markers/")[0])) as f:
        refined_markers = json.load(f)

    input_folder = "./Blender/renders/cameras/"
    all_files = []
    for cam in os.listdir(input_folder):
        _, f = utils.folder_contents(os.path.join(input_folder, cam))
        all_files.extend(f[start_frame:end_frame+1])

    assert len(refined_markers) == len(all_files), f"Got {len(refined_markers)} frames in refined_markers.json but {len(all_files)} files. They need to be equal length!"

    output = "./Out/synth_01/block_labels/"

    chunks = int(len(all_files) / chunk_size)+1
    start_chunk = int(start_frame / chunk_size)+1

    all_blocks_count = 0

    print(f'>>> Setup took {(time.time() - t_global):.02f}s')

    print(f'>>> Running ProcessPoolExecutor() with [{workers}] workers <<<')
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        # for _ in range(workers):
        #     executor.submit(time.sleep, .1)

        for c in range(start_chunk, chunks + 1, 1):

            futures = []

            t_chunk = time.time()

            c_first_frame = (c-1)*chunk_size
            c_last_frame = min(len(all_files), c*chunk_size)-1
            c_files = all_files[c_first_frame:c_last_frame+1]

            chunk_blocks_count = 0

            def done_block_labels(future):
                frame_idx, block_labels = future.result()

                refined_markers[frame_idx]['block_labels'] = block_labels

            def done_prepare_blk_imgs(future):
                frame_idx, blk_imgs, blocks_count = future.result()

                assert len(blk_imgs) != 0, f'No blocks for frame {frame_idx}'

                nonlocal chunk_blocks_count
                chunk_blocks_count += blocks_count

                label_0_char, label_1_char, blk_dir_idx = process_block_inference(frame_idx, blk_imgs, block_model, dataset.label_characters_0, dataset.label_characters_1)

                fut = executor.submit(process_block_labels, frame_idx, refined_markers[frame_idx]['blocks'], label_0_char, label_1_char, blk_dir_idx)
                fut.add_done_callback(done_block_labels)
                futures.append(fut)

            for i in range(len(c_files)):
                fut = executor.submit(prepare_blk_imgs, i, c_files[i], refined_markers[i], dataset, device)
                fut.add_done_callback(done_prepare_blk_imgs)
                futures.append(fut)

            for f in futures:
                f.result()

            if output:
                os.makedirs(output, exist_ok=True)
                import json
                fn = os.path.join(output, f'block_labels_chunk_{utils.leading_zeros(c,4)}_frames_[{c_first_frame}-{c_last_frame}].json')
                with open(fn, 'w+', 1) as f:
                    json_string = json.dumps(refined_markers, separators=(',', ":"))  # Compact JSON structure
                    f.write(json_string)

                # print(f'>>> Start render video...')
                # t_video = time.time()
                # vid_path = os.path.join(output, f'block_labels_[{len(refined_markers)}].mp4')
                # ffmpeg_process = None

                # all_marker_pos = np.concatenate([frame['checked_markers'] for frame in refined_markers if len(frame['checked_markers']) > 0], axis=0)
                # roi_min = np.maximum(0, np.floor(all_marker_pos.reshape(-1, 2).min(axis=0)).astype(int) - 10) // 2 * 2
                # roi_max = (np.ceil(all_marker_pos.reshape(-1, 2).max(axis=0)).astype(int)) // 2 * 2 + 10

                # for i, frame in enumerate(frames):
                #     frame = (frame*255).astype(np.uint8)
                #     frame_marker = frame.copy()
                #     frame1_marker = frame.copy()
                #     frame1_marker_black = np.zeros_like(frame)
                #     frame2_marker = frame.copy()

                #     markers = np.asarray(refined_markers[i]['checked_markers'])
                #     blocks = np.asarray([blk[1] for blk in refined_markers[i]['blocks']])

                #     draw_block_code(frame1_marker, markers, blocks, all_label_0_char[i], all_label_1_char[i], all_blk_dir_idx[i], wrap_text=False, label_confidence_thr=0.0)
                #     draw_block_code(frame1_marker_black, markers, blocks, all_label_0_char[i], all_label_1_char[i], all_blk_dir_idx[i], wrap_text=True, label_confidence_thr=0.0)
                #     draw_block_corners(frame2_marker, markers, blocks, dataset, all_label_0_char[i], all_label_1_char[i], all_blk_dir_idx[i])

                #     block_pts = [markers[block_indices].reshape(-1, 1, 2) for block_indices in blocks]
                #     cv2.polylines(frame_marker, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)
                #     cv2.polylines(frame1_marker, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)
                #     cv2.polylines(frame1_marker_black, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)
                #     cv2.polylines(frame2_marker, block_pts, True, 153, thickness=1, lineType=cv2.LINE_AA)

                #     frame_marker = utils.draw_markers(frame_marker, markers, color=255)
                #     frame1_marker = utils.draw_markers(frame1_marker, markers, color=255)
                #     frame1_marker_black = utils.draw_markers(frame1_marker_black, markers, color=255)

                #     frame_marker = np.concatenate((
                #         frame_marker[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]],  # [:,w//2:],
                #         frame1_marker_black[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]],  # [:,w//2:],
                #     ), axis=1)

                #     frame_marker = skimage.color.gray2rgb((frame_marker/255).astype(np.float64))

                #     out_img = (frame_marker*255).astype(np.uint8)

                #     if ffmpeg_process is None:
                #         import ffmpeg
                #         ffmpeg_process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{frame_marker.shape[1] if frame_marker.shape[1] % 2 == 0 else frame_marker.shape[1] + 1}x{frame_marker.shape[0] if frame_marker.shape[0] % 2 == 0 else frame_marker.shape[0] + 1}').output(
                #             vid_path, pix_fmt='yuv420p', loglevel="quiet").overwrite_output().run_async(pipe_stdin=True))
                #     ffmpeg_process.stdin.write(out_img)

                # if ffmpeg_process is not None:
                #     ffmpeg_process.stdin.close()
                #     ffmpeg_process.wait()
                #     print(f'> Render video took {(time.time() - t_video):.02f}s | Output path: "{vid_path}"')

            t_chunk = time.time() - t_chunk
            print(f'> Chunk [{c}/{chunks}] | Time: {t_chunk:.02f}s (Avg: {(t_chunk / (len(c_files))):.02f}s / frame) | Blocks: {chunk_blocks_count} (Avg: {(chunk_blocks_count / (len(c_files))):.02f} / frame) | Frames: {len(c_files)} (Range: [{c_first_frame}-{c_last_frame}] / camera) <')

    t_global = time.time() - t_global
    print(f'>>> Processed [{c}/{chunks}] chunks | Time: {t_global:.02f}s (Avg: {(t_global / (len(refined_markers))):.02f}s / frame) | Blocks: {all_blocks_count} (Avg: {(all_blocks_count / (len(refined_markers))):.02f} / frame) | Frames: {len(all_files)} (Range: [{start_frame}-{end_frame}] / camera) <<<')


if __name__ == '__main__':
    main()
