FROM ubuntu:bionic

RUN echo "deb http://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list \
        && apt-get update \
        && apt-get install -y python3-pip \
        && apt-get install -y python3-graph-tool \
        && cd /usr/loca/bin \
        && ln -s /usr/bin/python3 python \
        && pip3 install --upgrade pip

# Get latest pip
RUN pip3 install --upgrade pip

COPY ./src /app

RUN pip3 install -r /app/requirements.txt

CMD ["python3", "/app/main.py"]


