# FROM python:3.7
# FROM  nvidia/cuda:10.2-runtime
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip curl git

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --set python3 /usr/bin/python3.7

# Setup the virtualenv
#ENV VIRTUAL_ENV=/opt/venv
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
#RUN [ "python", "-m", "venv", "$VIRTUAL_ENV" ]

RUN apt-get update && \
    apt-get install -y libblas-dev liblapack-dev && \
    python3 -m pip install -U pip 

# Create app directory
WORKDIR /usr/src/app

# Bundle app source
COPY requirements.txt .

# Install python dependencies
RUN [ "python3", "-m", "pip", "install", "-r", "requirements.txt" ]
RUN [ "python3", "-m", "pip", "install", "pybind11==2.2.3" ]
RUN [ "python3", "-m", "pip", "install", "git+https://github.com/deepmipt/fastText.git#egg=fastText==0.8.22" ]

RUN apt-get install -y libsm6 libxext6 libxrender-dev nano bash mlocate

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/

RUN ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2 /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.0 && \
    ln -s /usr/lib/x86_64-linux-gnu/libcublas.so /usr/lib/x86_64-linux-gnu/libcublas.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcufft.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcufft.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcurand.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcurand.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusolver.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusolver.so.10.0 && \
    ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusparse.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcusparse.so.10.0

RUN python3 -c 'import tensorflow as tf; tf.test.is_gpu_available()'

# CMD ["/bin/bash"]

# Download spacy
RUN [ "python3", "-m", "spacy", "download", "en_core_web_sm" ]

# Download DeepPavlov datasets for our model
COPY model_config.json .
COPY slotfill.json .
COPY ner.json .
COPY intents.json .
RUN [ "python3", "-m", "deeppavlov", "download", "model_config.json" ]

RUN [ "python3", "-m", "deeppavlov", "download", "intents.json" ]

COPY . .

EXPOSE 5000
CMD [ "python3", "-m", "deeppavlov", "riseapi", "model_config.json", "-p", "5000" ]
