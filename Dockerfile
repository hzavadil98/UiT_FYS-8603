

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent tzdata from prompting for timezone during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y \
	python3-pip \
	git \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Install uv (universal virtualenv) using Python 3.10
RUN pip install --upgrade pip && pip install uv

# Set working directory
WORKDIR /workspace

# Copy dependency and version files
COPY pyproject.toml uv.lock .python-version ./

# Install Python dependencies using uv sync for reproducibility
RUN uv sync

# Set PYTHONPATH if needed
ENV PYTHONPATH=/storage/UiT_FYS-8603/python_packages

CMD ["bash", "-c", "source /workspace/.venv/bin/activate && exec bash"]
