import json
import numpy as np
import time
import multiprocessing
import argparse
import os


def load_patches(patch_path):
    with open(patch_path, 'r') as f:
        patch_json = json.load(f)
    patches = {} # key: patch_name, value: [row, col, label]
    inversed_patches = {}  # key: label, value: patch_name

    for name, p in patch_json.items():
        #new_p = [[p[i][j:j+2] for j in range(0, len(p[i]), 2)] for i in range(len(p))]
        patches[name] = p
        for i in p:
            if i['label'] != "**":
                inversed_patches[i['label']] = name

    return patches, inversed_patches


def dict_append(dict, key):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1


def locate_label(label, patch):
    for r in range(len(patch)):
        for c in range(len(patch[r])):
            if patch[r][c] == label:
                return (r, c)
    print(f"Warning: label {label} not found in patch")
    exit()


def dfs_vote(blk_idx, curr_direction, blocks, edge2blocks, patch, 
             inversed_patches, block_label_candidates, visited, r, c):
             # const: blocks, edge2blocks, patch, inversed_patches
    
    if visited[r][c]:
        return
    
    curr_label = patch[r][c]
    if curr_label == "**":
        dict_append(block_label_candidates[blk_idx], ("**", 4))
    else:
        dict_append(block_label_candidates[blk_idx], (curr_label, curr_direction))

    visited[r][c] = True 


    corners = blocks[blk_idx][1]

    delta = [[-1, 0], [0, 1], [1, 0], [0, -1]] # up, right, down, left

    for i in range(4):  # up, right, down, left
        next_r, next_c = r + delta[i][0], c + delta[i][1]
        if next_r < 0 or next_r >= len(patch) or next_c < 0 or next_c >= len(patch[0]):
            continue

        if visited[next_r][next_c]:
            continue

        edge = (corners[(curr_direction + i) % 4], corners[(curr_direction + i + 1) % 4])
        if edge not in edge2blocks:
            print(f"Warning: edge {edge} not in edge2blocks")
            exit()
            
        for next_blk_idx in edge2blocks[edge]:
            if next_blk_idx == blk_idx:
                continue

            """
            calculation of next_direction is a little bit difficult ...
            
            i      /     old   in_new
            0    up      d       3
            1    right   d+1     0
            2    down    d+2     1
            3    left    d+3     2            
            in_new[i] = (i+3) % 4
            
            Make corners[direction + i] == new_corners[(new_direction + (i+3) % 4) % 4]
            idx := new_corners.index(corners[direction + i])
            idx == (new_direction + i + 3) % 4
            new_direction = (idx - i + 5) % 4

            """
            
            new_corners = blocks[next_blk_idx][1]
            idx = new_corners.index(corners[(curr_direction + i) % 4])
            next_direction = (idx - i + 5) % 4

            dfs_vote(next_blk_idx, next_direction, blocks, edge2blocks, patch, 
                     inversed_patches, block_label_candidates, visited, next_r, next_c)

    return



def process_frame(frame, patches, inversed_patches):
    new_frame = {}

    for k, v in frame.items():
        if "block" in k:
            continue
        new_frame[k] = v.copy()

    block_num = len(frame["blocks"])

    blocks = frame["blocks"]
    refined_blocks = frame["refined_blocks"]
    block_labels = frame["block_labels"]

    # remove blocks not in patches
    for i in range(block_num-1, -1, -1):
        if block_labels[i]["label"] not in inversed_patches and block_labels[i]["label"] != "**":
            blocks.pop(i)
            refined_blocks.pop(i)
            block_labels.pop(i)
            block_num -= 1

    block_label_candidates = [{} for _ in range(block_num)]  # key: (label, direction), value: vote count

    # mapping: edge -> block index
    edge2blocks = {} # key: edge, value: list of block index (maybe more than 1 !)
    for i in range(block_num):
        corners = blocks[i][1]
        for j in range(4):
            e = (corners[j], corners[(j+1)%4])
            inv_e = (corners[(j+1)%4], corners[j])
            if e not in edge2blocks: edge2blocks[e] = [i]
            else:   edge2blocks[e].append(i)
            if inv_e not in edge2blocks: edge2blocks[inv_e] = [i]
            else:   edge2blocks[inv_e].append(i)

    # voting
    for i in range(block_num):
        curr_label = block_labels[i]["label"]
        curr_direction = block_labels[i]["direction"]

        if curr_label == "**":
            dict_append(block_label_candidates[i], ("**", 4))
        else:
            patch = patches[inversed_patches[curr_label]]
            visited = [[False for _ in range(len(patch[0]))] for _ in range(len(patch))]
            r,c = locate_label(curr_label, patch)
            dfs_vote(i, curr_direction, blocks, edge2blocks, patch, \
                     inversed_patches, block_label_candidates, visited, r, c)

        
    # select the best candidate
    new_blocks = []
    new_refined_blocks = []
    new_block_labels = []

    deleted_num, changed_num = 0, 0

    for i in range(block_num):
        # print(block_label_candidates[i])
        new_label_direction = max(block_label_candidates[i], key=block_label_candidates[i].get)
        voter = block_label_candidates[i][new_label_direction]
        if voter == 1 and len(block_label_candidates[i]) > 1:
            deleted_num += 1
            continue
        
        new_blocks.append(blocks[i])
        new_refined_blocks.append(refined_blocks[i])
        new_block_labels.append({
            "label": new_label_direction[0],
            "label_confidence": 1.0,
            "direction": new_label_direction[1],
            "dir_confidence": 1.0
        })

        # print(new_label_direction, voter)

        if new_label_direction[0] != block_labels[i]["label"] or \
           new_label_direction[1] != block_labels[i]["direction"]:
            changed_num += 1    

    # print(deleted_num, changed_num)
    new_frame["blocks"] = new_blocks
    new_frame["refined_blocks"] = new_refined_blocks
    new_frame["block_labels"] = new_block_labels

    return new_frame, deleted_num, changed_num


def modify_block_labels(block_path, patch_path):
    print(f"processing block labels... {block_path}")
    camera_id = os.path.basename(block_path).split("_block_labels.json")[0].split('_')[1]

    patches, inversed_patches = {}, {}

    if not os.path.basename(patch_path).endswith('.json'):
        patches_files = [os.path.join(patch_path,p) for p in os.listdir(patch_path)]
        for p in patches_files:
            p, i = load_patches(p)
            patches = {**patches, **p} # merges two dicts
            inversed_patches = {**inversed_patches, **i}
    else:
        patches, inversed_patches = load_patches(patch_path)
        
    with open(block_path, 'r') as f:
        block_json = json.load(f)
    new_block_json = []

    t0 = time.time()
    interval = 5000

    deleted_num_total  = 0
    changed_num_total = 0
    
    for i, frame in enumerate(block_json):
        if i % interval == 0:   
            t1 = time.time()
            print(f"Camera {camera_id} - {i} / {len(block_json)}: {t1 - t0:.3f}s for {interval} frames, ", end="")
            print(f"avg_deleted: {deleted_num_total / interval:.3f}, avg_changed: {changed_num_total / interval:.3f}")
            t0 = t1
            
            deleted_num_total = 0
            changed_num_total = 0


        new_frame, deleted_num, changed_num = process_frame(frame, patches, inversed_patches)

        new_block_json.append(new_frame)
        deleted_num_total += deleted_num
        changed_num_total += changed_num

    return new_block_json


def voting(input:str,output:str, patch_path:str, refined:bool = True):
    in_folder, in_file = os.path.split(input)
    out_folder, out_file = os.path.split(output)

    in_split = in_file.split('.')
    out_split = out_file.split('.')

    if out_split[0] != '':
        output = f'{out_folder}\\{out_split[0]}'
    elif in_split[0] != '':
        output = f'{out_folder}\\{in_split[0]}'
        
    subfolders = os.listdir(input)
    subfolders = [os.path.join(input,f) for f in subfolders]
    
    for folder in subfolders:
        block_path = f"{folder}\\{os.path.basename(folder)}_block_labels.json"
        new_block_json = modify_block_labels(block_path, patch_path)

        json_string = json.dumps(new_block_json, separators=(',', ":"))  # Compact JSON structure
        open(f'{folder}_voted.json', "w+", 1).write(json_string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='', type=str, required=True, help="Folder with subfolders per camera")
    parser.add_argument('--output', default='', type=str, required=True, help="Output _voted.json file")
    parser.add_argument('--patches', default='', type=str, required=True, help="Folder or single file containing patches with row / col information")
    parser.add_argument('--refined', default=True, type=bool, help='Whether to use refined markers files. Look for "_refined_markers.json" instead of "_markers.json"')
    args = parser.parse_args()

    args.input = os.path.join('', args.input)
    args.output = os.path.join('', args.output)
    args.patches = os.path.join('', args.patches)

    voting(args.input,args.output, args.patches, args.refined)

    # with multiprocessing.Pool(processes=len(subfolders)) as pool:
    #     pool.starmap(process_camera, [(folder, args.patches) for folder in subfolders])


if __name__ == "__main__":
    main()
