import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# image configuration
DATA_DIR="dataset/ffhq/thumbnails_128x128"
IMG_SIZE=128
CHANNELS=3
BATCH_SIZE=64

# training model configuration
LATENT_DIM=32
WARMUP_EPOCHS=20
START_EPOCH=40
PATIENCE=10
LEARNING_RATE=2e-4
EPOCHS=60

class KLWarmUp(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs):
        super().__init__()

        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_weight = min(1.0, epoch / self.warmup_epochs)
        self.model.kl_weight.assign(new_weight)

class VisualEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, val_batch, start_epoch=40, patience=10, min_delta=1e-4):
        super().__init__()

        self.val_batch = val_batch
        self.start_epoch = start_epoch
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return

        # Deterministic reconstruction (use mean)
        z_mean, _ = self.model.encoder(self.val_batch)
        x_hat = self.model.decoder(z_mean)

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(self.val_batch, x_hat),
                axis=(1, 2)
            )
        ).numpy()

        if recon_loss < self.best_loss - self.min_delta:
            self.best_loss = recon_loss
            self.wait = 0
        else:
            self.wait += 1

        print(
            f"[VisualEarlyStopping] epoch={epoch} "
            f"recon_loss={recon_loss:.4f} "
            f"best={self.best_loss:.4f} "
            f"wait={self.wait}/{self.patience}"
        )

        if self.wait >= self.patience:
            print("🛑 Early stopping: no visual improvement")
            self.model.stop_training = True

class ReconstructionsLogger(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir, num_images=8):
        super().__init__()

        self.dataset = dataset
        self.num_images = num_images
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, "images")
        )

    def on_epoch_end(self, epoch, logs=None):
        x = next(iter(self.dataset))
        z_mean, _ = self.model.encoder(x)
        x_hat = self.model.decoder(z_mean)

        with self.file_writer.as_default():
            tf.summary.image(
                "Original",
                x[:self.num_images],
                step=epoch
            )

            tf.summary.image(
                "Reconstruction",
                x_hat[:self.num_images],
                step=epoch
            )

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, kl_weight=0.0, **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float32)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))
            x_hat = self.decoder(z)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, x_hat),
                    axis=(1, 2)
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

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
    img = tf.image.decode_png(img, channels=CHANNELS)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    return img

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

def show_reconstructions(vae, dataset, n=8):
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

print("🟢 configure TensorBoard") 
log_dir = os.path.join(
    "logs",
    "vae",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    write_graph=True,
    update_freq="epoch"
)

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

print("🟢 Create a fixed validation batch for Stop early") 
val_batch = next(iter(dataset))

print("🟢 Build VAE encoder") 
encoder = build_encoder()
encoder.summary()

print("🟢 Build VAE decoder") 
decoder = build_decoder()
decoder.summary()

print("🟢 Build VAE model and compile") 
vae = VAE(encoder, decoder)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

print("🟢 Train VAE model") 
vae.fit(
    dataset, 
    epochs=EPOCHS,
    callbacks=[
        KLWarmUp(warmup_epochs=WARMUP_EPOCHS),
        VisualEarlyStopping(
            val_batch=val_batch,
            start_epoch=START_EPOCH,
            patience=PATIENCE
        ),        
        tensorboard_callback,
        ReconstructionsLogger(dataset, log_dir)
    ]    
)

print("🟢 Sample new faces from latent space")
sample_faces(decoder)

print("🟢 Visualize reconstructions model") 
show_reconstructions(vae, dataset)

print("🟢 Save VAE model")
encoder.save("models/encoder.keras")
decoder.save("models/decoder.keras")
