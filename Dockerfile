#List of supported tags:https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
ARG CUDA_VERSION=10.0
ARG CUDNN_VERSION=7
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

LABEL maintainer="Pradip Gupta <pradip.gupta@ril.com>"

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure environment
ARG USERNAME=jioai
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG PIP_EXTRA_INDEX_URL='https://pypi.org/simple'
ENV CONDA_DIR='/opt/conda'
ENV CONDA_BIN='/opt/conda/bin/conda'
ARG CONDA_ENV_NAME='base'
ENV CONDA_ENV_NAME=$CONDA_ENV_NAME
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONPATH='/app:$PYTHONPATH'

# Install system dependencies:
RUN apt-get update && \
    apt-get -y install --no-install-recommends apt-utils dialog 2>&1 && \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    apt-get -y install wget make curl unzip git vim bash-completion locales shtool iproute2 procps lsb-release && \
    apt-get -y install build-essential

# Install conda
ARG MINICONDA_VERSION=py37_4.8.2
ARG MINCONDA_MD5=87e77f097f6ebb5127c77662dfc3165e
ENV MINICONDA_VERSION=$MINICONDA_VERSION \
    MINCONDA_MD5=$MINCONDA_MD5
    
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINCONDA_MD5} *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash /Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# Create a non-root user to use if preferred
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # [Optional] Add sudo support for the non-root user
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    chown -R $USERNAME:$USER_GID /opt/conda && \
    chmod -R g+w /opt/conda && \
    #
    # Create alternate global install location that both uses have rights to access
    # && mkdir -p /usr/local/share/pip-global \
    # && chown ${USERNAME}:root /usr/local/share/pip-global \
    #  
    # Clean up
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# set Locale
ENV LANGUAGE=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
RUN sudo locale-gen en_US.UTF-8 \
    && sudo dpkg-reconfigure locales \
    && sudo localedef -i en_US -f UTF-8 en_US.UTF-8

# Copy environment.yml (if found) to a temp locaition so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment*.yml noop.txt /tmp/conda-tmp/
COPY requirements* /tmp/conda-tmp/
    
# Create Python environment based on environment-gpu.yml
RUN $CONDA_BIN env create -n $CONDA_ENV_NAME -f /tmp/conda-tmp/environment-gpu.yml && \
    #
    rm -rf /tmp/conda-tmp

COPY ./src /app/src
COPY predict.sh /app/predict.sh

RUN mkdir -p /app{data,src} && \
    chown -R $USERNAME /app

RUN ["/bin/bash", "-c", "source /opt/conda/bin/activate && \
    conda activate ${CONDA_ENV_NAME} && \  
    gdown https://drive.google.com/uc?id=1D6yagKJNuBmbzSAwODGqOPhFNifckXYD -O /app/model_file/cascaded-segmentation.h5"]

RUN chown $USERNAME:$USER_GID /app/predict.sh && \
    chmod a+x /app/predict.sh

USER $USERNAME
WORKDIR /app

EXPOSE 5000
ENTRYPOINT ["./predict.sh"]