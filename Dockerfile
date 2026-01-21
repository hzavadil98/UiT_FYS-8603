#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent tzdata from prompting for timezone during build
#ENV DEBIAN_FRONTEND=noninteractive
#ENV TZ=Etc/UTC
#
## Install system dependencies and Python 3.10
#RUN apt-get update && apt-get install -y \
#	python3-pip \
#	git \
#	curl \
#	&& rm -rf /var/lib/apt/lists/*
#
## Install uv (universal virtualenv) using Python 3.10
#RUN pip install --upgrade pip && pip install uv
#
## Set working directory
#WORKDIR /workspace
#
## Copy dependency and version files
#COPY pyproject.toml uv.lock .python-version ./
#
## Install Python dependencies using uv sync for reproducibility
#RUN uv sync
#
## Set PYTHONPATH if needed
#ENV PYTHONPATH=/storage/UiT_FYS-8603/python_packages
#
#CMD ["bash", "-c", "source /workspace/.venv/bin/activate && exec bash"]


FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-venv \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install R before Python dependencies
RUN apt-get update && apt-get install -y r-base

# Make python3 point to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install uv
RUN pip install --upgrade pip && pip install uv

WORKDIR /workspace

COPY pyproject.toml uv.lock .python-version ./

# Sync dependencies (using 3.12 now)
RUN uv sync

# Use venv by default
ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["bash"]