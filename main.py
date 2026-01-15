import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

DATA_DIR = "dataset/ffhq/thumbnails_128x128"

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 30

BETA = 1.0  # β-VAE for disentanglement

latent_dim = 128

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, kl_warmup_steps=10_000, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        # 🔹 KL warm-up state
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_weight = tf.Variable(0.0, trainable=False)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, x):
        # 🔹 Update step counter
        self.step.assign_add(1)

        # 🔹 Linearly increase KL weight
        kl_weight = tf.minimum(
            self.beta,
            tf.cast(self.step, tf.float32) / self.kl_warmup_steps * self.beta,
        )
        self.kl_weight.assign(kl_weight)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            x_recon = self.decoder(z)

            # Reconstruction loss (pixel-wise)
            recon_loss = tf.reduce_mean(
                tf.reduce_mean(tf.square(x - x_recon), axis=(1, 2, 3))
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )

            # 🔹 Total loss with warm-up
            total_loss = recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight,
        }

    def generate(self, n_samples=1):
        # Sample from standard normal latent space
        z = tf.random.normal(shape=(n_samples, self.encoder.output[0].shape[-1]))
        generated = self.decoder(z)

        return generated
    
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    return img

def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=tf.shape(z_mean))

    return z_mean + tf.exp(0.5 * z_log_var) * eps

def show_reconstructions(model, dataset, n=8):
    x = next(iter(dataset))
    z_mean, _, _ = model.encoder(x)
    x_recon = model.decoder(z_mean)

    plt.figure(figsize=(16, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i])
        plt.axis("off")

        # Reconstruction
        plt.subplot(2, n, i + n + 1)
        plt.imshow(x_recon[i])
        plt.axis("off")

    plt.savefig("results/show_reconstructions.png")

def sample_faces(decoder, n=16):
    z = tf.random.normal(shape=(n, latent_dim))
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

print("🟢 Encoder") 
encoder_inputs = layers.Input(shape=(128, 128, 3))
# Convolutional feature extractor
x = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(encoder_inputs)   # 64×64
x = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(x)               # 32×32
x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(x)              # 16×16
x = layers.Conv2D(256, 4, strides=2, padding="same", activation="relu")(x)              # 8×8
x = layers.Conv2D(512, 4, strides=2, padding="same", activation="relu")(x)              # 4

x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)  # increased to match larger latent

# Latent space
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

print("🟢 Decoder") 
latent_inputs = layers.Input(shape=(latent_dim,))

# Project latent vector into feature map
x = layers.Dense(4 * 4 * 512, activation="relu")(latent_inputs)
x = layers.Reshape((4, 4, 512))(x)

# Upsampling decoder
x = layers.Conv2DTranspose(512, 4, strides=2, padding="same", activation="relu")(x)  # 8×8
x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", activation="relu")(x)  # 16×16
x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)  # 32×32
x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)   # 64×64
x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu")(x)   # 128×128

decoder_outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

print("🟢 VAE model") 
vae = VAE(encoder, decoder, beta=BETA)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4))

print("🟢 Train model") 
vae.fit(dataset, epochs=EPOCHS)

print("🟢 Sample new faces from the latent space")
sample_faces(decoder)

print("🟢 Visualize reconstructions model") 
show_reconstructions(vae, dataset)

print("🟢 Save model")
encoder.save("models/encoder.keras")
decoder.save("models/decoder.keras")
