import tensorflow as tf
from keras.engine import Layer
from keras import backend as K
import numpy as np

class BilinearUpSampling(Layer) :
    '''Keras Custom Layer for Bilinear Upsampling
    '''
    def __init__(self, factor, **kwargs) :
        super( BilinearUpSampling, self ).__init__( **kwargs )
        self.factor = factor

    def compute_output_shape(self, input_shape):
        bsize, nb_rows, nb_cols, nb_filts = input_shape
        out_nb_rows = None if (nb_rows is None) else nb_rows * self.factor
        out_nb_cols = None if (nb_cols is None) else nb_cols * self.factor
        return tuple( [bsize, out_nb_rows, out_nb_cols, nb_filts] )

    def build(self, input_shape):
        super(BilinearUpSampling, self).build(input_shape)

    def call(self, x):
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array([self.factor, self.factor]).astype('int32'))
        x = tf.cast(tf.image.resize(x, new_shape, method="bilinear"), dtype=tf.float32)        
        x.set_shape((None, original_shape[1] * self.factor if original_shape[1] is not None else None,
                     original_shape[2] * self.factor if original_shape[2] is not None else None, None))
        return x

    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(BilinearUpSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

def normalization(inputs):
    import tensorflow as tf
    result = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), inputs)   
    return result