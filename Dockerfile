FROM python:3.6.12-slim as builder
MAINTAINER zhanghao <zhanghao_3389@163.com>

RUN apt update \
    && apt install -y ffmpeg libsm6 libxext6

ENV EXTERNAL_PYPI_SERVER=https://mirrors.aliyun.com/pypi/simple/
ENV MIRROR=mirrors.aliyun.com
ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir /faceRecognition
WORKDIR /faceRecognition
ADD . /faceRecognition

RUN sed -i "s/deb.debian.org/$MIRROR/g" /etc/apt/sources.list \
    && sed -i "s/security.debian.org/$MIRROR/g" /etc/apt/sources.list \
    && apt update \
    && apt install -y build-essential \
    && python -m venv /opt/venv \
    && python -m pip install --upgrade pip \
    && pip install -i $EXTERNAL_PYPI_SERVER --upgrade pip poetry \
    && pip config set global.extra-index-url $EXTERNAL_PYPI_SERVER \
    && echo "[easy_install]\nindex-url=$EXTERNAL_PYPI_SERVER" > ~/.pydistutils.cfg \
    && pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]