FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
	git \
	curl \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

# Install uv (universal virtualenv)
RUN pip3 install --upgrade pip && pip3 install uv

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv pip install --system --requirement pyproject.toml

# Set PYTHONPATH if needed
ENV PYTHONPATH=/storage/UiT_FYS-8603/python_packages:$PYTHONPATH


