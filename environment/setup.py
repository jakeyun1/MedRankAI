"""
setup.py

Downloads the datasets and the testbench dependencies.
"""

import os
import sys

# Make the current directory (for the subprocess) relative to the testbench program
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

python = sys.executable

# Download the dependencies
os.system(f"{python} -m pip install -r .{os.sep}dependencies.txt")

# Import after downloading dependencies
import kagglehub

# Download the datasets
# PAD-UFES-20
kagglehub.dataset_download("mahdavi1202/skin-cancer")

# CheXpert
kagglehub.dataset_download("ashery/chexpert")

# CBIS-DDSM
kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")

# ODIR-5K
kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")

# HAM10000
kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("\nTestbench setup done.")