#!/bin/bash

source /opt/conda/bin/activate
conda activate ${CONDA_ENV_NAME}
python src/prediction/run_prediction.py