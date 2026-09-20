import numpy as np
import cv2
import os
from typing import Tuple, List
import argparse


def extract_frames(video_path: str, interval: int = 20) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame.copy())
        frame_count += 1
    cap.release()
    print(f"> Extracted {len(frames)}/{frame_count} frames from {video_path}")
    return frames


def calibrate_camera(camera_name, frames: List[np.ndarray], chessboard_size: Tuple[int] | List[int], chessboard_edge_length: float, debug: bool = False):

    chessboard_coords = np.zeros((*chessboard_size, 3), np.float32).reshape(-1, 3)
    chessboard_coords[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_edge_length

    img_points = []
    obj_points = []

    cv2.namedWindow(camera_name, cv2.WINDOW_AUTOSIZE)

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ACCURACY)

        if debug:
            draw_chessboard_corners_image = cv2.drawChessboardCorners(gray, (7, 10), corners, ret)
            draw_chessboard_corners_image = cv2.resize(gray, (int(gray.shape[1]*0.5), int(gray.shape[0]*0.5)))
            cv2.imshow(camera_name, draw_chessboard_corners_image)
            cv2.setWindowTitle(camera_name, f'{camera_name} frame: {i:002}')
            cv2.waitKey(1)
        if not ret:
            continue
        corners = corners[::-1]

        img_points.append(corners)
        obj_points.append(chessboard_coords)
    cv2.destroyWindow(camera_name)

    print(f'> Calibrating with {len(img_points)}/{len(frames)} frames... ')
    ret, intrinsic_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    out = cv2.calibrationMatrixValues(intrinsic_matrix, [2448, 2048], 36, 36 * 2048/2448)
    # For the MER2 cameras we have 1/1.8'' sensor size (hor) --> 14.11mm
    # out = cv2.calibrationMatrixValues(intrinsic_matrix, [2448, 2048], 1/1.8 * 25.4, 1/1.8 * 25.4 * 2048/2448)

    print(f'> Camera parameter estimation\n\tFOV: {tuple([out[0], out[1]])}\n\tFocal length: {out[2]}\n\tPrincipal point: {out[3]}\n\tAspect ratio: {out[4]}')

    errors = []
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], intrinsic_matrix, distortion)
        error = np.sqrt((np.asarray(img_points[i] - imgpoints2).squeeze()**2).sum(axis=-1))
        errors.append(error)

    errors = np.asarray(errors)
    print(f"> Error\n\tMean: {np.mean(errors)}\n\tStd: {np.std(errors)}\n\tMax: {np.max(errors)}\n\tMin: {np.min(errors)}\n")

    return ret, intrinsic_matrix, distortion, rvecs, tvecs, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True, help="calibration video path")
    parser.add_argument('-o', '--output', type=str, help='a npz file', required=True)
    parser.add_argument('--chessboard-size', type=str, help='Number of inner corner points as tuple (n,m), note there should be no space', default='(7,10)')
    parser.add_argument('--chessboard-edge-length', type=float, help='length of chessboard edges, in meter', default=0.02)
    parser.add_argument('--save-check-image', action='store_true', default=True)
    parser.add_argument('--debug', type=int, default=False)

    args = parser.parse_args()

    chessboard_size = tuple(map(int, args.chessboard_size.strip('()"\',').split(',')))

    subfolders = os.listdir(args.input)
    subfolders = [os.path.join(args.input, f) for f in subfolders]

    for fn in subfolders:
        files = os.listdir(fn)
        files = [os.path.join(fn, f) for f in files]
        frames = []

        for frame in files:
            frames.append(cv2.imread(frame))
        name = os.path.basename(fn)
        ret, intrinsic_matrix, distortion, rvecs, tvecs, errors = calibrate_camera(name, frames, chessboard_size, args.chessboard_edge_length, args.debug)
        output = os.path.join(args.output, name)
        output += f'\\{name}'
        np.savez_compressed(f'{output}_intrinsics.npz', intrinsic_matrix=intrinsic_matrix, distortion=distortion, rvecs=rvecs, tvecs=tvecs, errors=errors)
        print(f'> Saved intrinsics to "{output}_intrinsics.npz"')
        print('---')


if __name__ == "__main__":
    main()
