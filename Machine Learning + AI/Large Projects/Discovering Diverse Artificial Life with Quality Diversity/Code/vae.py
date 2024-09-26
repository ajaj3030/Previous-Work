from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

import logging
import random 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RotationEquivariantConv(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x):
        logger.info(f"RotationEquivariantConv input shape: {x.shape}")

        # Compute padding
        pad_h = self.kernel_size[0] - 1
        pad_w = self.kernel_size[1] - 1
        
        # Create a padding configuration that matches the input dimensions
        if x.ndim == 4:  # (batch, height, width, channels)
            padding = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
        elif x.ndim == 3:  # (height, width, channels)
            padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        logger.info(f"Padding configuration: {padding}")

        # Regular convolution with padding
        conv = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.strides, padding="SAME")
        
        # Apply convolution to input and its 90-degree rotations
        x_0 = conv(jnp.pad(x, padding, mode='constant'))
        logger.info(f"Shape after 0 degree rotation and conv: {x_0.shape}")

        x_90 = jnp.rot90(conv(jnp.pad(jnp.rot90(x, k=1, axes=(0, 1) if x.ndim == 3 else (1, 2)), padding, mode='constant')), k=-1, axes=(0, 1) if x.ndim == 3 else (1, 2))
        logger.info(f"Shape after 90 degree rotation and conv: {x_90.shape}")

        x_180 = jnp.rot90(conv(jnp.pad(jnp.rot90(x, k=2, axes=(0, 1) if x.ndim == 3 else (1, 2)), padding, mode='constant')), k=-2, axes=(0, 1) if x.ndim == 3 else (1, 2))
        logger.info(f"Shape after 180 degree rotation and conv: {x_180.shape}")

        x_270 = jnp.rot90(conv(jnp.pad(jnp.rot90(x, k=3, axes=(0, 1) if x.ndim == 3 else (1, 2)), padding, mode='constant')), k=-3, axes=(0, 1) if x.ndim == 3 else (1, 2))
        logger.info(f"Shape after 270 degree rotation and conv: {x_270.shape}")
        
        # Max pool over rotations
        result = jnp.maximum(jnp.maximum(x_0, x_90), jnp.maximum(x_180, x_270))
        logger.info(f"RotationEquivariantConv output shape: {result.shape}")
        return result

class RotationEquivariantConvTranspose(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)
    
    @nn.compact
    def __call__(self, x):
        logger.info(f"RotationEquivariantConvTranspose input shape: {x.shape}")

        # Compute padding
        pad_h = self.kernel_size[0] - 1
        pad_w = self.kernel_size[1] - 1
        
        # Create a padding configuration that matches the input dimensions
        if x.ndim == 4:  # (batch, height, width, channels)
            padding = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
        elif x.ndim == 3:  # (height, width, channels)
            padding = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        logger.info(f"Padding configuration: {padding}")

        # Regular transposed convolution with padding
        conv_transpose = nn.ConvTranspose(features=self.features, kernel_size=self.kernel_size, strides=self.strides, padding="SAME")
        
        # Apply transposed convolution to input and its 90-degree rotations
        x_0 = conv_transpose(jnp.pad(x, padding, mode='constant'))
        logger.info(f"Shape after 0 degree rotation and conv_transpose: {x_0.shape}")

        x_90 = jnp.rot90(conv_transpose(jnp.pad(jnp.rot90(x, k=1, axes=(0, 1) if x.ndim == 3 else (1, 2)), padding, mode='constant')), k=-1, axes=(0, 1) if x.ndim == 3 else (1, 2))
        logger.info(f"Shape after 90 degree rotation and conv_transpose: {x_90.shape}")

        x_180 = jnp.rot90(conv_transpose(jnp.pad(jnp.rot90(x, k=2, axes=(0, 1) if x.ndim == 3 else (1, 2)), padding, mode='constant')), k=-2, axes=(0, 1) if x.ndim == 3 else (1, 2))
        logger.info(f"Shape after 180 degree rotation and conv_transpose: {x_180.shape}")

        x_270 = jnp.rot90(conv_transpose(jnp.pad(jnp.rot90(x, k=3, axes=(0, 1) if x.ndim == 3 else (1, 2)), padding, mode='constant')), k=-3, axes=(0, 1) if x.ndim == 3 else (1, 2))
        logger.info(f"Shape after 270 degree rotation and conv_transpose: {x_270.shape}")
        
        # Average over rotations
        result = (x_0 + x_90 + x_180 + x_270) / 4.0
        logger.info(f"RotationEquivariantConvTranspose output shape: {result.shape}")
        return result

class RotationInvariantDecoder(nn.Module):
    img_shape: Tuple[int, int, int]
    features: int
    
    @nn.compact
    def __call__(self, z):
        logger.info(f"Decoder input shape: {z.shape}")

        z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
        z = nn.relu(nn.Dense(features=4 * 4 * self.features)(z))
        z = z.reshape(*z.shape[:-1], 4, 4, self.features)
        logger.info(f"Shape after reshaping: {z.shape}")

        z = nn.relu(RotationEquivariantConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2))(z))
        logger.info(f"Shape after first RotationEquivariantConvTranspose: {z.shape}")

        z = nn.relu(RotationEquivariantConvTranspose(features=self.features, kernel_size=(3, 3), strides=(2, 2))(z))
        logger.info(f"Shape after second RotationEquivariantConvTranspose: {z.shape}")

        z = RotationEquivariantConvTranspose(features=self.img_shape[-1], kernel_size=(3, 3), strides=(2, 2))(z)
        logger.info(f"Shape after third RotationEquivariantConvTranspose: {z.shape}")

        return z


class RotationInvariantEncoder(nn.Module):
    latent_size: int
    features: int
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        logger.info(f"RotationInvariantEncoder input shape: {x.shape}")

        x = RotationEquivariantConv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        logger.info(f"Shape after first RotationEquivariantConv: {x.shape}")

        x = RotationEquivariantConv(features=self.features * 2, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        logger.info(f"Shape after second RotationEquivariantConv: {x.shape}")

        x = RotationEquivariantConv(features=self.features * 4, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        logger.info(f"Shape after third RotationEquivariantConv: {x.shape}")

        x = x.reshape((x.shape[0], -1))  # Flatten
        logger.info(f"Shape after flattening: {x.shape}")

        x = nn.Dense(features=self.features * 8)(x)
        x = nn.relu(x)
        logger.info(f"Shape after first Dense layer: {x.shape}")

        x = nn.Dense(features=self.features * 4)(x)
        x = nn.relu(x)
        logger.info(f"Shape after second Dense layer: {x.shape}")

        mean = nn.Dense(features=self.latent_size)(x)
        logvar = nn.Dense(features=self.latent_size)(x)
        logger.info(f"Output shapes - mean: {mean.shape}, logvar: {logvar.shape}")

        return mean, logvar

class Encoder(nn.Module):
	latent_size: int
	features: int

	@nn.compact
	def __call__(self, x):
		x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))  # (16, 16, self.features,)
		x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))  # (8, 8, self.features,)
		x = nn.relu(nn.Conv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x))  # (4, 4, self.features,)
		x = x.reshape(*x.shape[:-3], -1)  # (4 * 4 * self.features,)
		x = nn.Dense(features=4 * 4 * self.features)(x)  # (4 * 4 * self.features,)
		mean = nn.Dense(features=self.latent_size)(x)
		logvar = nn.Dense(features=self.latent_size)(x)
		return mean, logvar

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
		z = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z)
		return z

class VAE(nn.Module):
	img_shape: Tuple[int, int, int]
	latent_size: int
	features: int
	group_cnn: bool
 
	def setup(self):
		if self.group_cnn:
			self.encoder = RotationInvariantEncoder(latent_size=self.latent_size, features=self.features)
			self.decoder = RotationInvariantDecoder(img_shape=self.img_shape, features=self.features)
		else:
			self.encoder = Encoder(latent_size=self.latent_size, features=self.features)
			self.decoder = Decoder(img_shape=self.img_shape, features=self.features)

	def reparameterize(self, random_key, mean, logvar):
		eps = jax.random.normal(random_key, shape=mean.shape)
		return eps * jnp.exp(logvar * .5) + mean

	def encode(self, x, random_key):
		mean, logvar = self.encoder(x)
		return self.reparameterize(random_key, mean, logvar)

	def encode_with_mean_logvar(self, x, random_key):
		mean, logvar = self.encoder(x)
		return self.reparameterize(random_key, mean, logvar), mean, logvar

	def decode(self, z, random_key):
		return self.decoder(z)

	def generate(self, z, random_key):
		return nn.sigmoid(self.decoder(z))

	def __call__(self, x, random_key):
		z, mean, logvar = self.encode_with_mean_logvar(x, random_key)
		logits = self.decode(z, random_key)
		return logits, mean, logvar

@jax.jit
def kl_divergence(mean, logvar):
	return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.jit
def binary_cross_entropy_with_logits(logits, labels):
    # Ensure logits and labels have the same shape
    if logits.shape != labels.shape:
        logits = jax.image.resize(logits, labels.shape, method='bilinear')
    
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))

@jax.jit
def loss(logits, targets, mean, logvar):
	bce_loss = binary_cross_entropy_with_logits(logits, targets).mean()
	kld_loss = kl_divergence(mean, logvar).mean()
	return bce_loss + kld_loss