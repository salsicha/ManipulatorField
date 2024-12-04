
FROM ubuntu:noble AS build

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/venv/bin:$PATH"    

# This prevents ROS setup.bash from failing
SHELL ["/bin/bash","-c"]

RUN apt update && apt install -q -y --no-install-recommends \
    curl gnupg2 lsb-release python3-pip python3-venv vim wget \
    gnupg2 lsb-release unzip ca-certificates cmake build-essential \
    ccache wget sudo git xorg-dev libxcb-shm0 libglu1-mesa-dev python3-dev clang \
    libc++-dev libc++abi-dev libsdl2-dev ninja-build libxi-dev \
    libtbb-dev libosmesa6-dev libudev-dev autoconf libtool && \
    rm -rf /var/lib/apt/lists/*

COPY scripts /scripts
COPY notebooks /notebooks

RUN python3 -m venv /venv

RUN mkdir /workspace
WORKDIR /workspace

RUN git clone https://github.com/peterropac/Aegolius.git aegolius
RUN cd aegolius/Code/spomso && pip3 install .

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

### This is the nightly build as is already out of date, if this fails get a new URL from here:
### https://github.com/isl-org/Open3D/releases
RUN wget https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-0.18.0+c8856fc-cp312-cp312-manylinux_2_31_x86_64.whl
RUN pip3 install ./open3d-0.18.0+c8856fc-cp312-cp312-manylinux_2_31_x86_64.whl

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


#####################################################################

FROM scratch

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    TERM=xterm \
    PYTHONIOENCODING=UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/venv/bin:$PATH"    

COPY --from=build / /

WORKDIR /notebooks

ENTRYPOINT ["/entrypoint.sh"]

CMD ["jupyter-lab", "--port", "9999", "--no-browser", "--LabApp.token=''", "--allow-root"]
