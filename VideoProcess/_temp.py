import torch
import numpy as np
from multiprocessing.managers import SharedMemoryManager

block_size = 64
stride = 32
width = 2448
height = 2048
frames = 32


def main():
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    buf = [torch.empty((block_size-1, 75, 2, block_size, block_size), device='cpu', dtype=torch.bfloat16) for _ in range(frames)]

    smm = SharedMemoryManager()
    smm.start()
    sl = smm.ShareableList(buf)
    print(sl[0].shape)


if __name__ == '__main__':
    main()
