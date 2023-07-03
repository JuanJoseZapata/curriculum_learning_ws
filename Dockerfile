ARG BASE_IMAGE
ARG BASE_VERSION

FROM ${BASE_IMAGE}:${BASE_VERSION} AS development

RUN apt update && apt dist-upgrade -y && DEBIAN_FRONTEND=noninteractive apt install -y \
    sudo \
    build-essential \
    bash-completion \
    git \
    nano \
    wget \
    software-properties-common \
    texlive-full

RUN useradd --create-home --groups sudo --shell /bin/bash developer \
    --uid 1002 --user-group \
    # && mkdir /etc/sudoers.d/ \
    && echo "developer ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer \
    && chmod 600 /etc/sudoers.d/developer

USER developer
RUN mkdir /home/developer/workspace
WORKDIR /home/developer/workspace

COPY --chown=developer:developer . ./
RUN sudo bash install.sh 
RUN pipenv sync
