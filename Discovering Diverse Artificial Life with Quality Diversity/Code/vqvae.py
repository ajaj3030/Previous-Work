from typing import Tuple, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import logging 

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float = 0.5

    def setup(self):
        self.embedding = self.param('embedding',
            nn.initializers.uniform(),
            (self.num_embeddings, self.embedding_dim))
        
    def squared_euclidean_distance(self, a, b):
        a_norm = jnp.sum(a**2, axis=1, keepdims=True)
        b_norm = jnp.sum(b**2, axis=1)
        dist = a_norm + b_norm - 2 * jnp.dot(a, b.T)
        return jnp.maximum(dist, 0)  # Ensure non-negative distances

    def __call__(self, inputs):
        input_shape = inputs.shape
        flat_inputs = jnp.reshape(inputs, (-1, self.embedding_dim))
        
        distances = self.squared_euclidean_distance(flat_inputs, self.embedding)

        encoding_indices = jnp.argmin(distances, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)

        quantized = jnp.dot(encodings, self.embedding)
        quantized = jnp.reshape(quantized, input_shape)

        loss = jnp.mean((jax.lax.stop_gradient(quantized) - inputs)**2) + \
        self.commitment_cost * jnp.mean((quantized - jax.lax.stop_gradient(inputs))**2)

        quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
        avg_probs = jnp.mean(encodings, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class GumbelVectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float = 0.5
    temperature: float = 1.0

    def setup(self):
        self.embedding = self.param('embedding',
            nn.initializers.uniform(),
            (self.num_embeddings, self.embedding_dim))
        
    def squared_euclidean_distance(self, a, b):
        a_norm = jnp.sum(a**2, axis=1, keepdims=True)
        b_norm = jnp.sum(b**2, axis=1)
        dist = a_norm + b_norm - 2 * jnp.dot(a, b.T)
        return jnp.maximum(dist, 0)  # Ensure non-negative distances

    def gumbel_softmax(self, logits, temperature, eps=1e-20):
        U = jax.random.uniform(jax.random.PRNGKey(0), logits.shape)
        gumbel_noise = -jnp.log(-jnp.log(U + eps) + eps)
        y = logits + gumbel_noise
        return jax.nn.softmax(y / temperature)

    def __call__(self, inputs):
        input_shape = inputs.shape
        flat_inputs = jnp.reshape(inputs, (-1, self.embedding_dim))
        
        distances = self.squared_euclidean_distance(flat_inputs, self.embedding)
        
        # Apply Gumbel-Softmax
        logits = -distances
        gumbel_output = self.gumbel_softmax(logits, self.temperature)
        
        quantized = jnp.dot(gumbel_output, self.embedding)
        quantized = jnp.reshape(quantized, input_shape)

        loss = self.commitment_cost * jnp.mean((quantized - jax.lax.stop_gradient(inputs))**2)

        # Straight-through estimator
        quantized = inputs + jax.lax.stop_gradient(quantized - inputs)

        # Calculate perplexity
        avg_probs = jnp.mean(gumbel_output, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class Encoder(nn.Module):
    latent_size: int
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))
        x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))
        x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))
        x = x.reshape(*x.shape[:-3], -1)
        x = nn.Dense(features=4 * 4 * self.features)(x)
        return nn.Dense(features=self.latent_size)(x)

class Decoder(nn.Module):
    img_shape: Tuple[int, int, int]
    features: int

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
        z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
        z = z.reshape(*z.shape[:-1], 4, 4, self.features)
        z = nn.relu(nn.ConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
        z = nn.relu(nn.ConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
        z = nn.ConvTranspose(features=self.img_shape[-1], kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z)
        return z

class RotationEquivariantConv(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x):
        pad_h, pad_w = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        padding = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)) if x.ndim == 4 else ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))

        conv = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.strides, padding="SAME")
        
        x_0 = conv(jnp.pad(x, padding, mode='constant'))
        x_90 = jnp.rot90(conv(jnp.pad(jnp.rot90(x, k=1, axes=(-3, -2)), padding, mode='constant')), k=-1, axes=(-3, -2))
        x_180 = jnp.rot90(conv(jnp.pad(jnp.rot90(x, k=2, axes=(-3, -2)), padding, mode='constant')), k=-2, axes=(-3, -2))
        x_270 = jnp.rot90(conv(jnp.pad(jnp.rot90(x, k=3, axes=(-3, -2)), padding, mode='constant')), k=-3, axes=(-3, -2))
        
        return jnp.maximum(jnp.maximum(x_0, x_90), jnp.maximum(x_180, x_270))

class RotationEquivariantConvTranspose(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)
    
    @nn.compact
    def __call__(self, x):
        pad_h, pad_w = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        padding = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)) if x.ndim == 4 else ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))

        conv_transpose = nn.ConvTranspose(features=self.features, kernel_size=self.kernel_size, strides=self.strides, padding="SAME")
        
        x_0 = conv_transpose(jnp.pad(x, padding, mode='constant'))
        x_90 = jnp.rot90(conv_transpose(jnp.pad(jnp.rot90(x, k=1, axes=(-3, -2)), padding, mode='constant')), k=-1, axes=(-3, -2))
        x_180 = jnp.rot90(conv_transpose(jnp.pad(jnp.rot90(x, k=2, axes=(-3, -2)), padding, mode='constant')), k=-2, axes=(-3, -2))
        x_270 = jnp.rot90(conv_transpose(jnp.pad(jnp.rot90(x, k=3, axes=(-3, -2)), padding, mode='constant')), k=-3, axes=(-3, -2))
        
        return (x_0 + x_90 + x_180 + x_270) / 4.0

class RotationInvariantEncoder(nn.Module):
    latent_size: int
    features: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.relu(RotationEquivariantConv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))
        x = nn.relu(RotationEquivariantConv(features=self.features * 2, kernel_size=(3, 3), strides=(2, 2))(x))
        x = nn.relu(RotationEquivariantConv(features=self.features * 4, kernel_size=(3, 3), strides=(2, 2))(x))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.relu(nn.Dense(features=self.features * 8)(x))
        x = nn.relu(nn.Dense(features=self.features * 4)(x))
        return nn.Dense(features=self.latent_size)(x)

class RotationInvariantDecoder(nn.Module):
    img_shape: Tuple[int, int, int]
    features: int
    
    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
        z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
        z = z.reshape(*z.shape[:-1], 4, 4, self.features)
        z = nn.relu(RotationEquivariantConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2))(z))
        z = nn.relu(RotationEquivariantConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2))(z))
        z = RotationEquivariantConvTranspose(features=self.img_shape[-1], kernel_size=(3, 3), strides=(2, 2))(z)
        return z

class VQVAE(nn.Module):
    img_shape: Tuple[int, int, int]
    latent_size: int
    features: int
    num_embeddings: int = 64
    commitment_cost: float = 0.25
    use_gcnn: bool = False
    quantizer_type: Literal["standard", "gumbel"] = "standard"
    gumbel_temperature: float = 1.0

    def setup(self):
        if self.use_gcnn:
            self.encoder = RotationInvariantEncoder(latent_size=self.latent_size, features=self.features)
            self.decoder = RotationInvariantDecoder(img_shape=self.img_shape, features=self.features)
        else:
            self.encoder = Encoder(latent_size=self.latent_size, features=self.features)
            self.decoder = Decoder(img_shape=self.img_shape, features=self.features)
        
        if self.quantizer_type == "standard":
            self.vector_quantizer = VectorQuantizer(
                num_embeddings=self.num_embeddings, 
                embedding_dim=self.latent_size,
                commitment_cost=self.commitment_cost
            )
        elif self.quantizer_type == "gumbel":
            self.vector_quantizer = GumbelVectorQuantizer(
                num_embeddings=self.num_embeddings, 
                embedding_dim=self.latent_size,
                commitment_cost=self.commitment_cost,
                temperature=self.gumbel_temperature
            )
        else:
            raise ValueError(f"Unknown quantizer type: {self.quantizer_type}")

    def __call__(self, x, key=None):
        z = self.encode(x)
        quantized, vq_loss, perplexity = self.quantize(z)
        reconstructed = self.decode(quantized)

        # EMA update for embedding (assuming vector_quantizer has an 'embedding' attribute)
        #ema_decay = 0.999
        #self.vector_quantizer.embedding = ema_decay * self.vector_quantizer.embedding + (1 - ema_decay) * jax.lax.stop_gradient(quantized)

        return reconstructed, z, vq_loss, perplexity

    def encode(self, x, key=None):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def quantize(self, z):
        return self.vector_quantizer(z)

    def generate(self, indices, key=None):
        quantized = self.vector_quantizer.embedding[indices]
        return nn.sigmoid(self.decode(quantized))

@jax.jit
def vae_loss(reconstructed, targets, vq_loss, perplexity, beta=1.0, gamma=1):
    #recon_loss = binary_cross_entropy_with_logits(reconstructed, targets).mean()
    recon_loss = mse(reconstructed, targets)
    diversity_loss = -gamma * jnp.log(perplexity)  # Encourage higher perplexity
    total_loss = recon_loss + beta * vq_loss + diversity_loss
    
    logger.info(f"Reconstruction loss: {recon_loss}")
    logger.info(f"VQ loss: {vq_loss}")
    logger.info(f"Perplexity: {perplexity}")
    logger.info(f"Diversity loss: {diversity_loss}")
    
    return total_loss

@jax.jit
def binary_cross_entropy_with_logits(logits, labels):
    if logits.shape != labels.shape:
        logits = jax.image.resize(logits, labels.shape, method='bilinear')
    
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))

@jax.jit
def mse(targets, predictions):
    return jnp.mean(jnp.square(targets - predictions))
