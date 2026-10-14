import torch
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from .utils import folder_contents, load_label_patches
from typing import List
import smplx
from HandReconstruction.Utility.utils_mesh import compute_vertex_normals
import trimesh
from typing import Tuple
import os

block_size = 64
stride = 32
width = 2448
height = 2048
frames = 32


def export_label_patch_definition(labels_fn, output_fn):
    marker_defs, blocks, patches = load_label_patches(labels_fn)

    out = {
        "marker_defs": marker_defs,
        "blocks": blocks,
        "patches": patches
    }

    with open(output_fn, 'w+', 1) as f:
        import json
        json_string = json.dumps(out, separators=(',', ":"))  # Compact JSON structure
        f.write(json_string)


def split_files_into_folders(folder_containing_files: str, folder_list: Tuple[str], limit: int = -1):
    _, files = folder_contents(folder_containing_files)
    for folder in folder_list:
        cnt = 0
        while cnt < len(files):
            if folder not in os.path.basename(files[cnt]):
                cnt += 1
                continue
            folder_full = os.path.join(folder_containing_files, folder)
            os.makedirs(folder_full, exist_ok=True)
            new_path = os.path.join(folder_full, os.path.basename(files[cnt]))

            print(f'> Moving "{files[cnt]}" to "{new_path}"')
            os.rename(files.pop(cnt), new_path)

            if limit > 0 and cnt == limit:
                return


def export_smplx():
    device = 'cpu'
    smplx_model = smplx.create(
        model_path="./HandReconstruction/Data/HumanModels",
        model_type="mano",
        flat_hand_mean=True,
        is_rhand=True,
        use_pca=False,
        batch_size=1,
    ).to(device)

    hand_parms = {
        "global_orient": torch.zeros(1, 3),
        "transl": torch.zeros(1, 3),
        "hand_pose": torch.zeros(1, 45),
        "betas": torch.zeros(1, 10),
    }

    smplx_output = smplx_model(**hand_parms)

    vertices = smplx_output.vertices[0].detach().cpu().numpy()
    joints = smplx_output.joints[0].detach().cpu().numpy()
    face_indices = smplx_model.faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=face_indices, vertex_normals=compute_vertex_normals(vertices, smplx_model.faces))
    mesh.export("./model.obj")


def search_and_replace_json_fields(folder: str, search: str, replace: str):
    _, marker_files = folder_contents(folder)
    for marker_file in marker_files:
        import json
        with open(marker_file) as f:
            data = json.load(f)

        new_data = {}

        for k, v in data.items():
            new_data[k.replace(search, replace)] = v

        with open(marker_file, 'w+', 1) as f:
            json_string = json.dumps(new_data, separators=(',', ":"))  # Compact JSON structure
            f.write(json_string)


def main():
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    buf = [torch.empty((block_size-1, 75, 2, block_size, block_size), device='cpu', dtype=torch.bfloat16) for _ in range(frames)]

    smm = SharedMemoryManager()
    smm.start()
    sl = smm.ShareableList(buf)
    print(sl[0].shape)


if __name__ == '__main__':
    # search_and_replace_json_fields("./Out/synth_01/markers", "chunk_", "")
    # export_smplx()
    # split_files_into_folders("./Blender/renders/cameras_D", ("cam_001","cam_002","cam_003","cam_004","cam_005","cam_006","cam_007","cam_008","cam_009","cam_010","cam_011","cam_012","cam_013"))
    export_label_patch_definition("./Out/synth_05/patches/all.json", "./Out/synth_06/patches/all_patch_definitions.json")
