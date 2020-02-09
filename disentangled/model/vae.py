import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, encoder, latent, decoder, loss, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder
        self.loss = loss


    def call(self, x, training=False):
        x = self.encoder(x)
        z = self.latent(x, training=training)
        y = self.decoder(x)
        # self.loss(y, z)
        return y, z

