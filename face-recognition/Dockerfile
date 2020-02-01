FROM ubuntu:16.04
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR
COPY ./openvino $TEMP_DIR
RUN cd $TEMP_DIR && tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa && apt update
RUN apt install python3.7 -y &&  apt-get install python3.7-dev -y
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py
ENV LD_LIBRARY_PATH /opt/intel/openvino/opencv/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/hddl/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/gna/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64:/opt/intel/openvino/openvx/lib:/opt/intel/openvino/opencv/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/hddl/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/gna/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64:/opt/intel/openvino/openvx/lib
ENV PYTHONPATH /opt/intel/openvino/python/python3.7/openvino:/opt/intel/openvino/deployment_tools/model_optimizer:/opt/intel/openvino/python/python3.7:/opt/intel/openvino/deployment_tools/model_optimizer
RUN mkdir -p /src
COPY requirements.txt /src/requirements.txt
WORKDIR /src
RUN pip3.7 install -r requirements.txt
