FROM sky360/sky360lib-app:latest

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libtbb-dev \
        python3-dev \
        python3-numpy \
        wget \
        git \
        libusb-1.0-0-dev \
    && wget --no-check-certificate -O get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get autoclean && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp
