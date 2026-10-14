import time
from .utils import load_label_patches
import os
import json
import numpy as np


def load_patches(patch_path):
    with open(patch_path, 'r') as f:
        patch_json = json.load(f)
    patches = patch_json  # key: patch_name, value: [row, col, label]
    inversed_patches = {}  # key: label, value: patch_name

    for patch_name, patch in patch_json.items():
        for row in patch:
            for label in row:
                if label != "**":
                    assert label not in inversed_patches, f'Label "{label}" already exists in inversed_patches. This means there are duplicate block labels!'
                    inversed_patches[label] = patch_name

    return patches, inversed_patches


def locate_label(label, patch):
    for r in range(len(patch)):
        for c in range(len(patch[r])):
            if patch[r][c] == label:
                return (r, c)
    return (-1, -1)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', default='', type=str, required=True, help="Folder with subfolders per camera")
    # parser.add_argument('--patches', default='', type=str, required=True, help="Single file containing all patches in the scene with row / col information")
    # parser.add_argument('--refined', default=True, type=bool, help='Whether to use refined markers files. Look for "_refined_markers.json" instead of "_markers.json"')
    # args = parser.parse_args()

    t_global = time.time()

    markers_3d_positions_fn = "./Out/synth_06/post_process/all_markers_3d_positions_interpolated.npy"
    patches_fn = "./Out/synth_05/patches/all.json"

    output = "./Out/synth_06/triangulation/"

    if output:
        os.makedirs(output, exist_ok=True)

    markers_3d_positions = np.load(markers_3d_positions_fn)
    _, blocks, patches = load_label_patches(patches_fn)

    all_patch_transforms = []

    for frame_idx in range(len(markers_3d_positions)):
        if frame_idx % 20 == 0:
            print(f'> Frame [{frame_idx}/{len(markers_3d_positions)}]...')
        patch_transforms = {}
        patch_edge_directions = {}
        patch_location_markers = {}
        visited = []

        # go through each block, determine label and patch, go through each edge and get their 3d markers, determine the normalized direction vector and store it
        for curr_label in blocks:
            patch_name = blocks[curr_label]['patch']
            patch = patches[patch_name]

            corners = blocks[curr_label]['markers']

            if patch_name not in patch_edge_directions:
                patch_edge_directions[patch_name] = [[], []]

            for i in range(4):   # up, right, down, left
                edge = (corners[i % 4], corners[(i + 1) % 4])
                if edge in visited:
                    continue

                axis = 1 if i == 1 or i == 3 else 0  # 0 = X | 1 = Y

                start = markers_3d_positions[frame_idx][edge[0]]
                end = markers_3d_positions[frame_idx][edge[1]]

                if start[0] == -1000 or end[0] == -1000:
                    patch_edge_directions[patch_name][axis].append([])
                    continue

                if patch_name not in patch_location_markers:
                    patch_location_markers[patch_name] = []

                if edge[0] not in patch_location_markers[patch_name]:
                    patch_location_markers[patch_name].append(edge[0])
                if edge[1] not in patch_location_markers[patch_name]:
                    patch_location_markers[patch_name].append(edge[1])

                dir = end-start
                dir_norm = dir / np.linalg.norm(dir)
                if i == 2 or i == 3:
                    dir_norm *= -1

                patch_edge_directions[patch_name][axis].append(dir_norm)

                visited.append(edge)

        # Calculate mean x- and y-axis for each row and column array
        for patch in patch_edge_directions:
            # location
            patch_transforms[patch] = [[], []]
            if patch in patch_location_markers:
                loc = np.mean(np.stack(list(filter(lambda x: x[0] != -1000, [markers_3d_positions[frame_idx][i] for i in patch_location_markers[patch]])), axis=0), axis=0, keepdims=True).tolist()[0]
                count = len(patch_location_markers[patch])
                all = (len(patches[patch])+1)*(len(patches[patch][0])+1)

                conf_loc = count / all
                patch_transforms[patch][0] = (loc, conf_loc)
                # print(f'> Mean Location for {patch} {loc} Conf: {conf_loc:.02f}')

            x = patch_edge_directions[patch][0]
            # X-Axis
            filtered_x = [i for i in x if len(i) != 0]
            if len(filtered_x) == 0:
                print(f'> Cannot determine mean X-Axis for patch {patch}!')
                continue
            conf_x = len(filtered_x) / len(x)
            mean_x: np.ndarray = np.mean(np.stack(filtered_x), axis=0, keepdims=True)
            # print(f'> Mean X-Axis for {patch} {mean_x} Conf: {conf_x:.02f}')

            y = patch_edge_directions[patch][1]
            # Y-Axis
            filtered_y = [i for i in y if len(i) != 0]
            if len(filtered_y) == 0:
                print(f'> Cannot determine mean Y-Axis for patch {patch}!')
                continue
            conf_y = len(filtered_y) / len(y)
            mean_y: np.ndarray = np.mean(np.stack(filtered_y), axis=0, keepdims=True)
            # print(f'> Mean Y-Axis for {patch} {mean_y} Conf: {conf_y:.02f}')

            # Z-Axis
            mean_z = np.cross(mean_x, mean_y)
            # print(f'> Cross Z-Axis for {patch} {mean_z} Conf: {(conf_x*conf_y):.02f}\n')

            R = np.stack([mean_x, mean_y, mean_z], axis=2).reshape(3, 3)
            patch_transforms[patch][1] = (R.tolist(), conf_x*conf_y)

            # e = transforms3d.euler.mat2euler(R, axes='sxyz') --> creates wrong rotation in blender

            # beta = -np.arcsin(R[2, 0])
            # alpha = np.arctan2(R[2, 1]/np.cos(beta), R[2, 2]/np.cos(beta))
            # gamma = np.arctan2(R[1, 0]/np.cos(beta), R[0, 0]/np.cos(beta))
            # print((alpha, beta, gamma))

        all_patch_transforms.append(patch_transforms)

    if output:
        os.makedirs(output, exist_ok=True)
        fn = os.path.join(output, f'{os.path.split(patches_fn)[1].split(".")[0]}_patches_coordinates.json')
        with open(fn, 'w+', 1) as f:
            json_string = json.dumps(all_patch_transforms, separators=(',', ":"))  # Compact JSON structure
            f.write(json_string)

    t_global = time.time() - t_global
    print(f'> Processed {len(patches)} patches in {t_global: .02f}s')


if __name__ == "__main__":
    main()
