SHELL := $(shell which bash)
LATEST_GIT_COMMIT := $(shell git log -1 --format=%h)

CONDA_BIN = $(shell which conda)
CONDA_ROOT = $(shell $(CONDA_BIN) info --base)
CONDA_ENV_NAME ?= "base"
CONDA_ENV_PREFIX = $(shell conda env list | grep $(CONDA_ENV_NAME) | sort | awk '{$$1=""; print $$0}' | tr -d '*\| ')
CONDA_ACTIVATE := source $(CONDA_ROOT)/etc/profile.d/conda.sh ; conda activate $(CONDA_ENV_NAME) && PATH=${CONDA_ENV_PREFIX}/bin:${PATH};	

STEM?=rmcnn
TAG?=latest
IMAGE := $(STEM):$(TAG)
DOCKER_FILE=Dockerfile
GPU?=all
CONTAINER_NAME?=rmcnn_predict

CUDA_VERSION?=10.0
CUDNN_VERSION?=7
STORAGE_PATH="./results"

docker_build:
	docker build -f $(DOCKER_FILE) \
				 --build-arg CUDA_VERSION=${CUDA_VERSION} \
				 --build-arg CUDNN_VERSION=${CUDNN_VERSION} \
				 --build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
				 -t $(IMAGE) .
	docker tag $(IMAGE) $(STEM):latest

docker_run:
	docker run --gpus ${GPU} -it --rm -v ${STORAGE_PATH}:${STORAGE_PATH} --name ${CONTAINER_NAME} $(IMAGE)
