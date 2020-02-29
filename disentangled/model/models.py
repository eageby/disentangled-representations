from . import betavae

betavae_mnist = betavae.Conv_32_1(latents=32)
betavae_shapes3d = betavae.Conv_64_3(latents=32)
