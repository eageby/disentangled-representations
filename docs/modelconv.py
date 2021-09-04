import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path

def main():
    # source = Path('../data/models/experiment/BetaVAE/Shapes3d/HP3/RS12')
    source = Path('/home/elias/Documents/thesis/data/models/BetaTCVAE/Shapes3d/')
    dest = Path('./models/BetaTCVAE/Shapes3d/')
    model = tf.keras.models.load_model(source)

    encode = model.encode.get_concrete_function(tf.TensorSpec([None, 64,64,3], tf.float32))
    decode = model.decode.get_concrete_function(tf.TensorSpec([None, 10], tf.float32))

    signatures = {"serve_encoder" : encode,
                    "serve_decoder": decode}
    # tf.saved_model.save(model, str(dest), signatures=signatures)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[64,64,3]),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(2, activation='softmax')])

    tfjs.converters.save_keras_model(model, dest)

main()
