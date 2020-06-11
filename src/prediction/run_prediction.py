import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.append(os.getcwd())

from flask import Flask, request
import logging

#to suppres future warnings
import tensorflow  as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from src.prediction.predict import Predictor

app = Flask(__name__)

logger = logging.getLogger('rmrcnn')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

predictor = Predictor()
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image_file' not in request.files:
        print('no image_file found')
        return None
    try:
        filename = predictor.process_file(rgb_path)
        print(filename)
        return filename
    except Exception as e:
        print(e)
        return None
        
app.run(host="0.0.0.0", port=5000)