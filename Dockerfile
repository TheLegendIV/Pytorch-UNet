FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .
