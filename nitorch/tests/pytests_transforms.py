import os
import sys

NITORCH_DIR = os.getcwd()
sys.path.insert(0, NITORCH_DIR)
from nitorch.transforms import IntensityRescale

if __name__ == "__main__":
    intensity = IntensityRescale(masked=False, on_gpu=True)
    print("hello world")
