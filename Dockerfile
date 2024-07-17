FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y tmux

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    && apt-get clean

# dockerが起動し続けるためのもの
CMD ["tail", "-f", "/dev/null"]

# ENTRYPOINT ["/bin/bash"]