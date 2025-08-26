FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y \
	git \
	curl \
	python3.12 \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

# Set python3 to python3.12 for convenience
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv (universal virtualenv)
RUN pip3 install --upgrade pip && pip3 install uv

# Set working directory
WORKDIR /workspace

# Copy dependency and version files
COPY pyproject.toml uv.lock .python-version ./

# Install Python dependencies using uv sync for reproducibility
RUN uv sync --system

# Set PYTHONPATH if needed
ENV PYTHONPATH=/storage/UiT_FYS-8603/python_packages:$PYTHONPATH


