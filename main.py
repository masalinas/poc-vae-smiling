import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# image variables
DATA_DIR = "dataset/ffhq/thumbnails_128x128"
IMG_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 64

# model variables
LATENT_DIM = 32

# training model variables
BETA = 0.5
LEARNING_RATE = 2e-4
EPOCHS = 1

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
        self.beta = beta

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = self.sampling([z_mean, z_log_var])
            x_hat = self.decoder(z)

            # Reconstruction loss (MSE)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(x - x_hat), axis=(1, 2, 3)
                )
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "recon": recon_loss,
            "kl": kl_loss
        }

    def generate(self, n_samples=1):
        # Sample from standard normal latent space
        z = tf.random.normal(shape=(n_samples, self.encoder.output[0].shape[-1]))
        generated = self.decoder(z)

        return generated    

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=CHANNELS)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    return img

def show_samples(dataset, n=9):
    batch = next(iter(dataset))
    plt.figure(figsize=(6, 6))

    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(batch[i])
        plt.axis("off")
    plt.show()

def build_encoder():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

    x = inputs
    for filters in [32, 64, 128, 256, 512]:
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    # Now shape is (4, 4, 512)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)

    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)

    return tf.keras.Model(inputs, [z_mean, z_log_var], name="encoder")

def build_decoder():
    inputs = layers.Input(shape=(LATENT_DIM,))

    x = layers.Dense(4 * 4 * 512)(inputs)
    x = layers.Reshape((4, 4, 512))(x)

    for filters in [256, 128, 64, 32]:
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    outputs = layers.Conv2DTranspose(
        3, 4, strides=2, padding="same", activation="sigmoid"
    )(x)

    return tf.keras.Model(inputs, outputs, name="decoder")

def show_reconstructions(vae, dataset, n=9):
    """
    Shows original images (top row) and reconstructions (bottom row)
    """

    # Get one batch
    x = next(iter(dataset))

    # Encode (use mean for deterministic reconstructions)
    z_mean, _ = vae.encoder(x)

    # Decode
    x_hat = vae.decoder(z_mean)

    # Plot
    plt.figure(figsize=(2 * n, 4))

    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(tf.clip_by_value(x[i], 0.0, 1.0))
        plt.axis("off")

        # Reconstruction
        plt.subplot(2, n, i + n + 1)
        plt.imshow(tf.clip_by_value(x_hat[i], 0.0, 1.0))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/show_reconstructions.png")

def sample_faces(decoder, n=16):
    z = tf.random.normal(shape=(n, LATENT_DIM))
    imgs = decoder(z)

    plt.figure(figsize=(6, 6))
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(imgs[i])
        plt.axis("off")
        
    plt.savefig("results/sample_faces.png")

def load_model(encoder_name, decoder_name):
    encoder = load_model(encoder_name, compile=False)
    decoder = load_model(decoder_name, compile=False)

    return encoder, decoder

print("🟢 check GPU") 
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

print("🟢 Load dataset") 
image_paths = tf.data.Dataset.list_files(
    os.path.join(DATA_DIR, "*.png"),
    shuffle=True
)

dataset = (
    image_paths
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("🟢 VAE encoder") 
encoder = build_encoder()
encoder.summary()

print("🟢 VAE decoder") 
decoder = build_decoder()
decoder.summary()

print("🟢 VAE model") 
vae = VAE(encoder, decoder, beta=BETA)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

print("🟢 Train VAE model") 
vae.fit(dataset, epochs=EPOCHS)

print("🟢 Sample new faces from latent space")
sample_faces(decoder)

print("🟢 Visualize reconstructions model") 
show_reconstructions(vae, dataset)

print("🟢 Save model")
encoder.save("models/encoder.keras")
decoder.save("models/decoder.keras")
