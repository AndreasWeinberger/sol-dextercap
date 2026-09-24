import argparse
import os
from VideoProcess.utils import load_label_patches
import numpy as np

#       15 - 14 - 13 - \ Thumb
#                       \
#       3-- 2 -- 1 ----- 0 Index
#        6 -- 5 -- 4 -- / Middle
#    12 -- 11 -- 10 -- / Ring
#      9 -- 8 -- 7 -- / Pinky

mapping = {
    "base_lower": 0,
    "base_upper": 0,
    "pinky_meta": 0,
    "thumb_meta_inner": 13,
    "thumb_meta_outer": 13,
    "thumb_01": 14,
    "thumb_02": 15,
    "index_01": 1,
    "index_02": 2,
    "index_03": 3,
    "middle_01": 4,
    "middle_02": 5,
    "middle_03": 6,
    "ring_01": 10,
    "ring_02": 11,
    "ring_03": 12,
    "pinky_01": 7,
    "pinky_02": 8,
    "pinky_03": 9
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label-patches', default='', type=str)
    parser.add_argument('-o', '--output', default='', type=str)

    args = parser.parse_args()

    markers_def, blocks, _ = load_label_patches(args.label_patches)

    indices = np.full(len(markers_def), -1)

    for block in blocks:
        for i in blocks[block]['markers']:
            indices[i] = mapping[blocks[block]['patch']]

    for m in mapping:
        c = len([j for j in indices if j == mapping[m]])
        print(f'Markers corresponting to segment [{mapping[m]}] (patch "{m}"): {c}')

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)
    np.save(args.output, indices)


if __name__ == '__main__':
    main()
