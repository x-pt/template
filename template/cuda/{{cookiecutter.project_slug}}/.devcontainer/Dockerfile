# Choose a recent, common CUDA version as a base. User can customize the tag.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install common development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git vim curl wget sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'vscode' and give it sudo access
RUN useradd -ms /bin/bash -u 1000 vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode

USER vscode
WORKDIR /home/vscode
