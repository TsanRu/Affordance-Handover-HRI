# ROS 2 Jazzy dev image for ur_python_utilities (UR control + gz-sim bringup).
# Mirrors the parent osx_robot_env docker/dev/Dockerfile (Ubuntu 24.04 + CUDA + Jazzy).
# OpenGL for RViz2 / gz GUI comes from the NVIDIA container runtime
# (NVIDIA_DRIVER_CAPABILITIES=all in docker-compose.yml) plus the GL libs below.
#
# Build context: ur_python_utilities repo root (see docker-compose.yml).
# Source is bind-mounted at runtime; colcon build via docker/scripts/initialize-colcon-workspace.sh.

FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

RUN echo 'Asia/Tokyo' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=jazzy

################################################################################
# Locale: ROS 2 requires a UTF-8 locale.
RUN apt-get update && apt-get install -q -y --no-install-recommends locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

################################################################################
# Bootstrap tools + apt sources for ROS 2 and Gazebo.
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository universe

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" \
        | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

################################################################################
# Core ROS 2 desktop + colcon/rosdep tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-desktop \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update --rosdistro ${ROS_DISTRO}

################################################################################
# MoveIt 2, ros2_control, and Gazebo Harmonic integration.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-moveit-visual-tools \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-controller-manager \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-xacro \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*

################################################################################
# UR driver/sim, Robotiq description, kinematics, and Python/KDL deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ur \
    ros-${ROS_DISTRO}-ur-simulation-gz \
    ros-${ROS_DISTRO}-robotiq-description \
    ros-${ROS_DISTRO}-trac-ik \
    ros-${ROS_DISTRO}-trac-ik-kinematics-plugin \
    python3-pykdl \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

################################################################################
# OpenGL / GUI runtime libs (RViz2 + gz GUI).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libegl1 \
    libgles2 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

################################################################################
# Developer tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    build-essential \
    clang-format \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python-is-python3 \
    wget \
    locate \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    htop \
    inetutils-ping \
    iproute2 \
    less \
    nano \
    net-tools \
    openssh-client \
    python3-tk \
    sshpass \
    unzip \
    usbutils \
    vim \
    && rm -rf /var/lib/apt/lists/*

################################################################################
# Python runtime deps for ur_control / ur_pykdl.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages numpy-quaternion

################################################################################
# Speed up colcon/CMake builds with ccache.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache \
    && rm -rf /var/lib/apt/lists/*
RUN echo 'export PATH=/usr/lib/ccache:$PATH' >> $HOME/.bashrc

################################################################################
# Colcon workspace skeleton (sources bind-mounted at runtime).
WORKDIR /root/ws
RUN mkdir -p src

COPY ./docker/scripts/. /root/ws/docker/scripts/
RUN chmod -R +x /root/ws/docker/scripts/*

RUN echo "source /root/ws/docker/scripts/initialize-bash-shell.sh" >> /root/.bashrc
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

ARG DOCKERFILE_COMMIT_SHORT_SHA
ENV DOCKER_IMAGE_VERSION_ur_python_utilities=${DOCKERFILE_COMMIT_SHORT_SHA}

ENTRYPOINT []
CMD ["/bin/bash"]
