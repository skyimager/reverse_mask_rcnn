## Standard imports
import os, sys, glob
sys.path.append(os.getcwd())

import buzzard as buzz
import numpy as np
from keras.models import load_model, Input, Model

## Custom imports
from src.models.custom_layers import normalization, BilinearUpSampling
from src.prediction.tile_infer import predict_from_file


class Predictor:
    def __init__(self, project_dir, wrdr):
        self.project_dir = project_dir
        model_path = os.path.join(wrdr, "models", "cascaded-segmentation.h5")
        self.model = self.model_int(model_path)
        print("Model "+ self.model_name +" loaded Successfully")

    def model_int(self, model_path):
        custom =  {'BilinearUpSampling':BilinearUpSampling, 'normalization': normalization}
        model = load_model(model_path, compile=False, custom_objects=custom)
        return model

    def predict(self, filepath):
        predicted_probamap, fp = predict_from_file(filepath, self.model,
                                                 downsampling_factor=1, tile_size=self.args.tile_size, no_of_gpu=1, 
                                                 batch_size=self.args.batch_size)
        return predicted_probamap, fp

    def process_file(self, rgb_path):
        self.rgb_path = rgb_path

        try:
            filename = os.path.join(self.project_dir, self.args.project + '_' + self.model_name[:-3] + '_probamap.npy')
            predicted_probamap, fp = self.predict(self.rgb_path)
            np.save(filename, predicted_probamap)
            return filename

        except Exception as e:
            print(e)
            sys.exit()

        
