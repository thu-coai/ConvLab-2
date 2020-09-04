FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends  software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.7 python3-pip build-essential libssl-dev libffi-dev python3.7-dev

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install setuptools wheel

RUN which python3.7
RUN which pip3

RUN ln -f -s /usr/bin/python3.7 /usr/bin/python
RUN ln -f -s /usr/bin/pip3 /usr/bin/pip
RUN python --version

RUN pip install nltk==3.4
RUN pip install tqdm==4.30
RUN pip install checksumdir==1.1
RUN pip install dataclasses
RUN pip install visdom
RUN pip install Pillow
RUN pip install future
RUN pip install torch
RUN pip install numpy==1.15.0
RUN pip install scipy
RUN pip install scikit-learn==0.20.3
RUN pip install pytorch-pretrained-bert==0.6.1
RUN pip install transformers==2.3.0
RUN pip install tensorflow==1.14
RUN pip install tensorboard==1.14.0
RUN pip install tensorboardX==1.7
RUN pip install tokenizers==0.8.0
RUN pip install allennlp==0.9.0
RUN pip install requests
RUN pip install simplejson
RUN pip install spacy
RUN pip install unidecode
RUN pip install jieba
RUN pip install embeddings
RUN pip install quadprog
RUN pip install pyyaml


RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]

WORKDIR /root

CMD ["/bin/bash"]
