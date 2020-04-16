FROM tensorflow/tensorflow:1.15.2-gpu-py3

WORKDIR /simclr

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH=$PWD:$PYTHONPATH

EXPOSE 6006

ENTRYPOINT bash
