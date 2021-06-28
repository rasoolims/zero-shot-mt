FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
LABEL maintainer="Mohammad Sadegh Rasooli"
LABEL repository="root_mt"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip \
                   openssh-client \
                   vim && \
    rm -rf /var/lib/apt/lists

RUN apt-get update && apt-get -y install openssh-client
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install torch==1.8.1
RUN python3 -m pip install numpy==1.18.2
RUN python3 -m pip install sacrebleu
RUN python3 -m pip install future==0.18.2
RUN python3 -m pip install pillow
RUN python3 -m pip install ipython
RUN python3 -m pip install transformers==4.5.1
RUN python3 -m pip install pandas
RUN python3 -m pip install pyarrow
RUN python3 -m pip install fire
RUN python3 -m pip install boto3
RUN python3 -m pip install pyicu
RUN python3 -m pip install  sentencepiece

WORKDIR /root
RUN cd /root
RUN mkdir /root/development
RUN mkdir /root/development/global_voices
RUN mkdir /root/development/ted_talks
RUN mkdir /root/models
ENV DATA_FOLDER=/root/development
ENV SAVE_PATH=/root/models
ENV LANG C.UTF-8
RUN echo 'alias python="python3"' >> ~/.bashrc

RUN cd /root
RUN git clone https://github.com/rasoolims/zero-shot-mt

WORKDIR /root/zero-shot-mt/
COPY dataset_prep /root/zero-shot-mt/dataset_prep
COPY mt_pret /root/zero-shot-mt/mt_pret

WORKDIR /root/

WORKDIR /root/zero-shot-mt/

CMD ["/bin/bash"]