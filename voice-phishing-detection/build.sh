#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies (ffmpeg is bundled via imageio-ffmpeg pip package)
pip install --upgrade pip
pip install -r backend/requirements.txt
