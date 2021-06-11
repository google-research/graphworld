FROM ubuntu:bionic

# tzdata asks questions.
ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"

# && apt-get install -y nvidia-container-runtime \

# Needed at add graphtool key
RUN apt-get -y update
RUN apt-get install -y apt-transport-https gnupg ca-certificates

RUN echo "deb [ arch=amd64 ] http://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list \
        && apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25 \
        && apt-get -y update \
        && apt-get install -y python3-pip \
        && apt-get install -y python3-venv \
        && apt-get install -y python3-cairo \
        && apt-get install -y python3-matplotlib \
        && apt-get install -y python3-decorator \ 
        && apt-get install -y python3-graph-tool

# RUN echo "deb [trusted=yes] http://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list \
#         && apt-get --allow-unauthenticated update \
#         && apt-get install -y python3-pip \
#         && apt-get install -y python3-venv \
#         && apt-get install -y python3-cairo \
#         && apt-get install -y python3-matplotlib \
#         && apt-get install -y python3-decorator \ 
#         && apt-get --allow-unauthenticated install -y python3-graph-tool 


# # Set up venv to avoid root installing/running python.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages ${VIRTUAL_ENV}
# RUN source ${VIRTUAL_ENV}/bin/activate
ENV PATH="${VIRTUAL_ENV}/bin:/app:$PATH"

RUN pip3 install --upgrade pip

COPY ./src /app

# Used --no-cache-dir to save resources when using Docker Desktop on OSX.
# Otherwise the install failed on my local machine.
RUN pip3 install --no-cache-dir -r /app/requirements.txt

CMD ["python3", "/app/test_graph_tool.py"]


