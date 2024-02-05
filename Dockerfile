FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y \
    libopenblas-base \
    python3 \
    python3-pip \
    libgomp1 \
    libvips \
    libcrypto++ \
    python3-openslide \ 
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

CMD nvidia-smi 

RUN apt-get update && apt-get install -y python3 python3-pip python3.10-venv wget zip
COPY ./deeperhistreg /src/deeperhistreg/
COPY ./deeperhistreg_params /src/deeperhistreg_params/
COPY ./requirements.txt /src/requirements.txt
RUN python3 -m venv /opt/venv/DeeperHistReg
RUN source /opt/venv/DeeperHistReg/bin/activate
RUN pip3 install -r ./src/requirements.txt

ENTRYPOINT ["python3", "/src/deeperhistreg/run.py"]


