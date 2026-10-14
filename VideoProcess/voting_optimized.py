import time
from .utils import leading_zeros, folder_contents, get_file_names_with_cam_ids, chunk_cam_file_names
from os import makedirs
from os.path import join
import json


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
    return (-1, -1)


def dfs_vote(blk_idx, curr_direction, blocks, edge2blocks, patch,
             inversed_patches, block_label_candidates, visited, r, c, block_labels):
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

    delta = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # up, right, down, left

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
                     inversed_patches, block_label_candidates, visited, next_r, next_c, block_labels)

    return


def patch_vote(blk_idx, curr_direction, blocks, edge2blocks, inversed_patches, patch_candidates: dict[str, int], visited: list[int], block_labels):

    corners = blocks[blk_idx][1]

    for i in range(4):  # up, right, down, left
        edge = (corners[(curr_direction + i) % 4], corners[(curr_direction + i + 1) % 4])
        if edge not in edge2blocks:
            print(f"Warning: edge {edge} not in edge2blocks")
            exit()

        for next_blk_idx in edge2blocks[edge]:
            if next_blk_idx == blk_idx:
                continue

            if next_blk_idx in visited:
                continue

            new_corners = blocks[next_blk_idx][1]
            ind = corners[(curr_direction + i) % 4]
            if ind not in new_corners:
                continue
            idx = new_corners.index(ind)
            next_direction = (idx - i + 5) % 4

            curr_label = block_labels[next_blk_idx]["label"]
            curr_direction = block_labels[next_blk_idx]["direction"]

            if curr_label != '**':
                if curr_label in inversed_patches:
                    p_name = inversed_patches[curr_label]
                    if p_name not in patch_candidates:
                        patch_candidates[p_name] = 0
                    patch_candidates[p_name] += 1     # tried doing this with label confidence, not ideal, since it leads to a patch being selected in the max() later even though they are ambiguous
                    if patch_candidates[p_name] > 2:    # it is very likely that we found the patch. skip iterating through the rest to save performance on large patches
                        return

            visited.append(next_blk_idx)

            patch_vote(next_blk_idx, next_direction, blocks, edge2blocks, inversed_patches, patch_candidates, visited, block_labels)

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
    # for i in range(block_num-1, -1, -1):
    #     if block_labels[i]["label"] not in inversed_patches and block_labels[i]["label"] != "**":
    #         blocks.pop(i)
    #         refined_blocks.pop(i)
    #         block_labels.pop(i)
    #         block_num -= 1

    block_label_candidates = [{} for _ in range(block_num)]  # key: (label, direction), value: vote count

    # mapping: edge -> block index
    edge2blocks = {}  # key: edge, value: list of block index (maybe more than 1 !)
    for i in range(block_num):
        corners = blocks[i][1]
        for j in range(4):
            e = (corners[j], corners[(j+1) % 4])
            inv_e = (corners[(j+1) % 4], corners[j])
            if e not in edge2blocks:
                edge2blocks[e] = [i]
            else:
                edge2blocks[e].append(i)
            if inv_e not in edge2blocks:
                edge2blocks[inv_e] = [i]
            else:
                edge2blocks[inv_e].append(i)

    # voting
    for i in range(block_num):
        curr_label = block_labels[i]["label"]
        curr_direction = block_labels[i]["direction"]

        if curr_label == "**":
            dict_append(block_label_candidates[i], ("**", 4))
        else:
            # change this to determine patch based on surrounding block labels
            # traverse recursively through the patch in the image using edge2blocks
            # determine row / column and where it sits in the scene, if that corresponds to a patch --> we found it
            # otherwise fallback to default (retrieve patch from inversed patches)

            if curr_label not in inversed_patches:
                continue

            patch_candidates = {}
            patch_vote(i, curr_direction, blocks, edge2blocks, inversed_patches, patch_candidates, [i], block_labels)

            max_patches = [k for k, v in patch_candidates.items() if v == max(patch_candidates.values())]

            p_name = inversed_patches[curr_label]

            if len(max_patches) > 0:
                if len(patch_candidates) > 1 and p_name != max_patches[0]:
                    if p_name not in patch_candidates:
                        patch_candidates[p_name] = 1
                    patch_candidates[p_name] += 1

                    max_patches = [k for k, v in patch_candidates.items() if v == max(patch_candidates.values())]

            if len(max_patches) == 1:
                patch = patches[max_patches[0]]
            else:
                patch = patches[p_name]

            visited = [[False for _ in range(len(patch[0]))] for _ in range(len(patch))]
            r, c = locate_label(curr_label, patch)
            dfs_vote(i, curr_direction, blocks, edge2blocks, patch,
                     inversed_patches, block_label_candidates, visited, r, c, block_labels)

    # remove blocks not in patches
    for i in range(block_num-1, -1, -1):
        if block_labels[i]["label"] not in inversed_patches and block_labels[i]["label"] != "**":
            blocks.pop(i)
            refined_blocks.pop(i)
            block_labels.pop(i)
            block_label_candidates.pop(i)
            block_num -= 1

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
            print(f'> Deleted: {block_label_candidates[i]}')
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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', default='', type=str, required=True, help="Folder with subfolders per camera")
    # parser.add_argument('--patches', default='', type=str, required=True, help="Single file containing all patches in the scene with row / col information")
    # parser.add_argument('--refined', default=True, type=bool, help='Whether to use refined markers files. Look for "_refined_markers.json" instead of "_markers.json"')
    # args = parser.parse_args()

    t_global = time.time()

    # Tuple of ids e.g.: (0,1,2,3,4,5,6) or (0,) for only cam_idx 0
    cam_ids = "0,1,2,3,4,5,6,7,8,9,10,11,12"
    cam_first_frame = 0
    cam_last_frame = 199
    workers = 12
    chunk_size = 64
    render_video = False

    chunks_folder = "./Out/synth_06/refined_markers/"
    patches_fn = "./Out/synth_05/patches/all.json"
    patches, inversed_patches = load_patches(patches_fn)

    imgs_folder = "./Blender/renders/cameras_D/"
    output = "./Out/synth_06/voted/"

    if output:
        makedirs(output, exist_ok=True)

    cam_ids = tuple(int(x) for x in cam_ids.split(","))

    all_files = get_file_names_with_cam_ids(cam_ids, imgs_folder, cam_first_frame, cam_last_frame)

    if chunk_size == -1:
        chunk_cnt = 1
        chunk_size = cam_last_frame+1-cam_first_frame
    else:
        chunk_cnt = int((cam_last_frame-cam_first_frame) / chunk_size)+1

    if chunk_cnt > 1:
        all_files = chunk_cam_file_names(all_files, chunk_cnt, chunk_size)

    all_deleted_total, all_changed_total = 0, 0

    # print(f'>>> Running ProcessPoolExecutor() with [{workers}] workers <<<')
    # with concurrent.futures.ProcessPoolExecutor(workers) as executor:
    #     for _ in range(workers):
    #         executor.submit(time.sleep, .1)

    all_frames_cnt = 0

    for cam_idx in cam_ids:
        t_cam = time.time()
        _, cam_chunk_files = folder_contents(chunks_folder, True, files_filter_lambda=lambda x: x.startswith(
            f'cam_{leading_zeros(cam_idx,3)}') and x.endswith('refined_markers.json'))

        if len(cam_chunk_files) == 0:
            print(f'\n>>> Skipping cam idx {cam_idx} | No chunk files!')
            continue

        cam_frames_cnt = 0
        cam_deleted_cnt = 0
        cam_changed_cnt = 0

        for chunk_idx in range(chunk_cnt):

            t_chunk = time.time()

            chunk_first_frame = (chunk_idx)*chunk_size
            chunk_last_frame = min(((chunk_idx+1)*chunk_size)-1, cam_last_frame - cam_first_frame)
            chunk_overall_first_frame = chunk_first_frame + cam_first_frame
            chunk_overall_last_frame = chunk_last_frame + cam_first_frame

            print(f'\n>>> Starting chunk [{chunk_idx+1}/{len(cam_chunk_files)}] | Frames: {chunk_size} | Range: [{chunk_overall_first_frame}-{chunk_overall_last_frame}] <<<')

            with open(cam_chunk_files[chunk_idx]) as f:
                all_markers = json.load(f)

            c_files = all_files[cam_idx][chunk_idx]

            cam_frames_cnt += len(c_files)

            chunk_deleted_cnt, chunk_changed_cnt = 0, 0

            print(f'> Setup took {(time.time() - t_chunk):.02f}s')

            for frame_idx in range(len(c_files)):
                voted, deleted_num, changed_num = process_frame(all_markers[frame_idx], patches, inversed_patches)

                chunk_deleted_cnt += deleted_num
                chunk_changed_cnt += changed_num

                all_markers[frame_idx] = voted

            cam_deleted_cnt += chunk_deleted_cnt
            cam_changed_cnt += chunk_changed_cnt

            if output:
                makedirs(output, exist_ok=True)
                fn = join(output, f'cam_{leading_zeros(cam_idx,3)}_chunk_{leading_zeros(chunk_idx,3)}_[{chunk_first_frame}-{chunk_last_frame}]_voted.json')
                with open(fn, 'w+', 1) as f:
                    json_string = json.dumps(all_markers, separators=(',', ":"))  # Compact JSON structure
                    f.write(json_string)

            t_chunk = time.time() - t_chunk
            print(f'> Chunk [{chunk_idx+1}/{chunk_cnt}] | Time: {t_chunk:.02f}s (Avg {(t_chunk / len(c_files)):.02f}s / frame) | Deleted: {chunk_deleted_cnt} (Avg: {(chunk_deleted_cnt/len(c_files)):.02f} / frame) | Changed: {chunk_changed_cnt} (Avg: {(chunk_changed_cnt/len(c_files)):.02f} / frame) | Frames: {len(c_files)} (Chunk range: [{chunk_first_frame}-{chunk_last_frame}])')

        all_frames_cnt += cam_frames_cnt
        all_deleted_total += cam_deleted_cnt
        all_changed_total += cam_changed_cnt

        t_cam = time.time() - t_cam
        print(
            f'>>> Cam {cam_idx} took {t_cam:.02f}s (Avg: {(t_cam / cam_frames_cnt):.02f}s) | Deleted: {cam_deleted_cnt} (Avg: {(cam_deleted_cnt/cam_frames_cnt):.02f} / frame) | Changed: {cam_changed_cnt} (Avg: {(cam_changed_cnt/cam_frames_cnt):.02f} / frame) | Frames: {cam_frames_cnt} (Range: [{cam_first_frame}-{cam_last_frame}]) <<<\n')

    t_global = time.time() - t_global
    print(
        f'>>> Processed cam idx {cam_ids} ({chunk_cnt} chunks / cam) in {t_global:.02f}s (Avg: {(t_global / all_frames_cnt):.02f}s) | Deleted: {all_deleted_total} (Avg: {(all_deleted_total/all_frames_cnt):.02f} / frame) | Changed: {all_changed_total} (Avg: {(all_changed_total/all_frames_cnt):.02f} / frame) | Cam frame range: [{cam_first_frame}-{cam_last_frame}] | Frames: {all_frames_cnt} <<<\n')


if __name__ == "__main__":
    main()
