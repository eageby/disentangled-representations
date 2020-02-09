from disentangled.model.components import *
from disentangled.model.vae import *
import disentangled.model.loss as loss
import tensorflow as tf

def main():
    cifar = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = VAE(
        Mlp([128, 128], 32),
        Latent(),
        DeMlp([128, 128], x_train.shape[1:]),
        0 
        )

    model.compile('adam', loss=['mse', loss.GaussianKL])
    
    model.fit(x_train, [x_train, x_train])
    model.evaluate(x_test, [x_test, x_test])

if __name__ == "__main__":
    main()
