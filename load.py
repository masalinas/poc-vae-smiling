import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

latent_dim = 16

def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=tf.shape(z_mean))

    return z_mean + tf.exp(0.5 * z_log_var) * eps

def sample_latent(latent_dim, n_samples=1):
    return np.random.normal(0, 1, size=(n_samples, latent_dim)).astype("float32")

def show_image(img, title="Generated Face"):
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.savefig('results/face.png', dpi=300, bbox_inches="tight", pad_inches=0)

    plt.show()

def show_grid(images, n=4):
    plt.figure(figsize=(n, n))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('results/face_grid.png', dpi=300, bbox_inches="tight", pad_inches=0)
    
    plt.show()

encoder = load_model(
    "models/encoder.keras",
    compile=False,
    custom_objects={"sampling": sampling}
)

decoder = load_model(
    "models/decoder.keras",
    compile=False
)

z = sample_latent(latent_dim)
x_gen = decoder.predict(z)

show_image(x_gen[0])

z = sample_latent(latent_dim, n_samples=16)
x_gen = decoder.predict(z)
show_grid(x_gen)