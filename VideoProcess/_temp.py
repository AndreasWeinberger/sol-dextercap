import torch
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from utils import folder_contents
from typing import List

block_size = 64
stride = 32
width = 2448
height = 2048
frames = 32

def search_and_replace_json_fields(folder: str, search:str, replace:str):
    _, marker_files = folder_contents(folder)
    for marker_file in marker_files:
        import json
        with open(marker_file) as f:
            data = json.load(f)

        new_data = {}

        for k,v in data.items():
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
    search_and_replace_json_fields("./Out/synth_01/markers", "chunk_", "")