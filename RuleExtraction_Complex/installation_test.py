import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import mmcv
import mmpose
import sys

def test_installations():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MMCV version: {mmcv.__version__}")
    print(f"MMPose version: {mmpose.__version__}")

if __name__ == "__main__":
    test_installations()