#!/bin/bash

# Install tapnet from GitHub
pip install git+https://github.com/google-deepmind/tapnet.git

# Create necessary directories
mkdir -p tapnet/checkpoints
mkdir -p tapnet/examplar_videos

# Download causal Bootstrapped TAPIR PyTorch checkpoint
wget -P tapnet/checkpoints https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt

# Download example video
wget -P tapnet/examplar_videos https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4
