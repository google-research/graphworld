FROM ubuntu:bionic
# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04 

# tzdata asks questions.
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"

# && apt-get install -y nvidia-container-runtime \

# Need this to add graph tool pubkey
# RUN apt-get -y update && apt-get install -y apt-transport-https gnupg ca-certificates

RUN apt-get -y update \
        && apt-get install -y apt-transport-https gnupg ca-certificates \
        && echo "deb [ arch=amd64 ] http://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list \
        && apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25 \
        && apt-get -y update \
        && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        python3-decorator \ 
        python3-graph-tool


# # Set up venv to avoid root installing/running python.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:/app:$PATH"
ENV PYTHONPATH "/app:$PYTHONPATH"

COPY ./src /app

# Used --no-cache-dir to save resources when using Docker Desktop on OSX.
# Otherwise the install failed on my local machine.
# torch MUST be installed prior to installing torch-geometric.
# Image size was causing download timeouts when using cuda versions of torch+sparse+geometric
# Switched to CPU, need to re-think this if GPU is needed.
        # && pip3 install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html \
        # && pip3 install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html \
# Note, torch-geometric and it's deps can't be installed in the same
# pip command...
RUN pip3 install --upgrade pip \
        && pip install --no-cache-dir torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
        && pip3 install --no-index --no-cache-dir torch-sparse torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html \
        && pip3 install --no-cache-dir torch-geometric \
        && pip3 install --no-cache-dir -r /app/requirements.txt \
        && pip3 cache purge

# Install apache beam sdk
# Example: https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/dataflow/gpu-workers/Dockerfile
COPY --from=apache/beam_python3.6_sdk /opt/apache/beam /opt/apache/beam

# Set the entrypoint to Apache Beam SDK worker launcher.
ENTRYPOINT [ "/opt/apache/beam/boot" ]
