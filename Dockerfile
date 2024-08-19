FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV NVIDIA_DRIVER_CAPABILITIES graphics, utility, compute
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    libglew-dev \
    build-essential \
    libopenblas-dev \
    unzip \
    wget \
    git

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--force_cuda" --install-option="--blas=openblas"

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade numpy
RUN pip3 install scikit-learn
RUN pip3 install scikit-image
RUN pip3 install tqdm
RUN pip3 install natsort
RUN pip3 install matplotlib

WORKDIR /DomainBridgingNav