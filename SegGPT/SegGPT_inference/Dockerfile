FROM cnstark/pytorch:1.13.1-py3.9.16-cuda11.7.1-ubuntu20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6  -yq
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install numpy opencv-python timm==0.3.2 fairscale==0.4.13 git+https://github.com/facebookresearch/detectron2.git
RUN sed -i "s|from torch._six import container_abcs|import collections.abc as container_abcs|g" /usr/local/lib/python3.9/site-packages/timm/models/layers/helpers.py
