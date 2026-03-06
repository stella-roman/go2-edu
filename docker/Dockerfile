FROM althack/ros2:humble-full

ENV ROS_DISTRO=humble

# RUN apt-get update || true && apt-get install -y --no-install-recommends \
#     curl \
#     gnupg2 \
#     lsb-release && \
#     curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
#     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install basic requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    portaudio19-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-rosidl-generator-dds-idl \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    software-properties-common \
    ffmpeg \
    xvfb x11vnc novnc websockify fluxbox xterm 

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Install specific numpy version first
RUN pip3 install --no-cache-dir 'numpy<2.0.0'

# Create workspace
ENV COLCON_WS=/root/ros2_ws
WORKDIR ${COLCON_WS}

# Clone the repository with submodules
COPY go2_humble/ /${COLCON_WS}/src/go2_ros2_project

# Initialize git submodules (required for aioice and other dependencies)
# If .git directory exists, update submodules; otherwise skip
RUN cd src && \
    (git submodule update --init --recursive 2>/dev/null || \
     echo "Git submodules not available, assuming dependencies are included") && \
    cd ..

# # Install additional Python dependencies required by go2_driver_node
# # These were identified during TF troubleshooting (see cursor_tf.md)
# # Use standard PyPI as primary index to avoid timeout issues
# # - aiortc: WebRTC connection support
# # - wasmtime/wasmtime-py: Runtime dependency
# RUN pip install --no-cache-dir \
#     --index-url https://pypi.org/simple \
#     --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple \
#     aiortc \
#     wasmtime || pip install --no-cache-dir \
#     --index-url https://pypi.org/simple \
#     wasmtime-py

RUN pip3 install --no-cache-dir -r /${COLCON_WS}/src/go2_ros2_project/requirements.txt

# Install ROS dependencies
# rosdep: Install all system dependencies required by the source packages in 'src'
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

COPY docker/start-novnc.sh /root/start-novnc.sh

RUN chmod +x /root/start-novnc.sh
# Set environment variables
ENV ROBOT_IP="192.168.123.161"
ENV CONN_TYPE="cyclonedds"
ENV WEBRTC_SERVER_HOST="0.0.0.0"
ENV WEBRTC_SERVER_PORT="9991"

# Remove apt cache
RUN rm -rf /var/lib/apt/lists/*

EXPOSE 6080

CMD ["bash"]

# CMD ["ros2", "launch", "go2_robot_sdk", "robot.launch.py"]

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc  && \
    echo "source ${COLCON_WS}/install/setup.bash" >> ~/.bashrc  && \
    echo "alias cb='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
