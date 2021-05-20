FROM ubuntu:bionic

# tzdata asks questions
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"

RUN echo "deb [trusted=yes] http://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list \
        && apt-get --allow-unauthenticated update \
        && apt-get install -y python3-pip \
        && apt-get --allow-unauthenticated install -y python3-graph-tool \
        && cd /usr/local/bin \
        && ln -s /usr/bin/python3 python \
        && pip3 install --upgrade pip

# Get latest pip
RUN pip3 install --upgrade pip

COPY ./src /app

RUN pip3 install -r /app/requirements.txt

CMD ["python3", "/app/main.py"]


