#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Installs dependencies for open-source users.
"""

import os
import requests

DP_FILES = {
    "privacy_engine_augmented.py": "privacy_engine_augmented.py",
}

DP_UTIL_FILES = {
    "NFnet.py": "NFnet.py",
}

FIL_FILES = {
    "acountant.py": "accountant.py",
    "datasets.py": "datasets.py",
    "trainer.py": "trainer.py",
    "utils.py": "utils.py",
}

def main():
    
    for filename in DP_FILES.values():
        patch_filename = filename.replace(".py", ".patch")
        os.system(f"patch dp/tan/src/opacus_augmented/{filename} patches/{patch_filename}")

    for filename in DP_UTIL_FILES.values():
        patch_filename = filename.replace(".py", ".patch")
        os.system(f"patch dp/tan/src/models/{filename} patches/{patch_filename}")

    for filename in FIL_FILES.values():
        patch_filename = filename.replace(".py", ".patch")
        os.system(f"patch fil/bounding_data_reconstruction/{filename} patches/{patch_filename}")

    print('Done')

if __name__ == "__main__":
    main()