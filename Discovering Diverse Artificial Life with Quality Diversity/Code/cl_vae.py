from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
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
def rotation_consistency_loss(means, aug_means):
    """Encourage consistent latent representations across rotations."""
    batch_size = means.shape[0]
    num_augs = aug_means.shape[0] // batch_size
    
    # Reshape aug_means to [batch_size, num_augs, latent_size]
    aug_means = aug_means.reshape(batch_size, num_augs, -1)
    
    # Compute pairwise distances between original and augmented means
    distances = jnp.sum(jnp.square(means[:, None, :] - aug_means), axis=-1)
    
    # Return the mean distance
    return jnp.mean(distances)

@jax.jit
def binary_cross_entropy_with_logits(logits, labels):
    # Ensure logits and labels have the same shape
    if logits.shape != labels.shape:
        logits = jax.image.resize(logits, labels.shape, method='bilinear')
    
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))

@jax.jit
def kl_divergence(mean, logvar):
	return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.jit
def triplet_margin_loss(means, aug_means, key, margin=0.8):
    """
    Compute the triplet margin loss comparing each mean to all its augmentations.
    
    Args:
    means: The original embeddings of shape (batch_size, embedding_dim)
    aug_means: The augmented embeddings of shape (num_augs * batch_size, embedding_dim)
    margin: The margin to enforce between positive and negative distances

    Returns:
    The triplet margin loss (scalar)
    """
    batch_size = means.shape[0]
    num_augs = aug_means.shape[0] // batch_size

    # Reshape aug_means to (batch_size, num_augs, embedding_dim)
    aug_means_reshaped = aug_means.reshape(batch_size, num_augs, -1)

    # Compute pairwise distances between means and their augmentations
    pos_dist = jnp.sum(jnp.square(means[:, None, :] - aug_means_reshaped), axis=-1)  # (batch_size, num_augs)

    # Compute pairwise distances between means and all other means
    neg_dist = jnp.sum(jnp.square(means[:, None, :] - means[None, :, :]), axis=-1)  # (batch_size, batch_size)

    # Mask out self-comparisons in negative distances
    neg_dist = neg_dist + jnp.eye(batch_size) * 1e6

    # Compute the triplet loss for all possible triplets
    loss = jnp.maximum(0, pos_dist[:, :, None] - neg_dist[:, None, :] + margin)

    # Take the mean over all valid triplets
    return jnp.mean(loss)


@jax.jit
def nt_xent_loss(z_orig, z_augs, temperature=0.8):
    batch_size, features = z_orig.shape
    num_augs = z_augs.shape[0] // batch_size

    # Reshape z_augs to separate augmentations
    z_augs = z_augs.reshape(batch_size, num_augs, features)
    
    # Normalize representations
    z_orig = z_orig / jnp.linalg.norm(z_orig, axis=1, keepdims=True)
    z_augs = z_augs / jnp.linalg.norm(z_augs, axis=2, keepdims=True)

    # Compute similarity matrix for positive pairs
    pos_sim_matrix = jnp.einsum('bf,baf->ba', z_orig, z_augs) / temperature

    # Compute similarity matrix for all pairs
    z_all = jnp.concatenate([z_orig[:, None, :], z_augs], axis=1)
    sim_matrix = jnp.einsum('bif,bjf->bij', z_all, z_all) / temperature

    # Create a mask for positive pairs
    pos_mask = jnp.eye(num_augs + 1, dtype=bool)[0]
    pos_mask = pos_mask.at[0].set(False)  # Exclude self-similarity

    # Compute NT-Xent loss
    exp_sim_matrix = jnp.exp(sim_matrix - jnp.max(sim_matrix, axis=-1, keepdims=True))
    
    # Positive similarities (only with augmentations)
    pos_sim = jnp.sum(jnp.exp(pos_sim_matrix - jnp.max(pos_sim_matrix, axis=-1, keepdims=True)), axis=-1)
    
    # Negative similarities (excluding self and own augmentations)
    neg_sim = jnp.sum(exp_sim_matrix * (1 - pos_mask[None, :, None]), axis=(1, 2))
    
    loss = -jnp.log(pos_sim / (pos_sim + neg_sim))
    
    return jnp.mean(loss)

@jax.jit
def nt_xent_loss_euclidean(z_orig, z_augs, temperature=0.8):
    batch_size, features = z_orig.shape
    num_augs = z_augs.shape[0] // batch_size

    # Reshape z_augs to separate augmentations
    z_augs = z_augs.reshape(batch_size, num_augs, features)

    # Compute pairwise Euclidean distances
    z_all = jnp.concatenate([z_orig[:, None, :], z_augs], axis=1)
    distances = jnp.sqrt(jnp.sum((z_all[:, None, :, :] - z_all[:, :, None, :]) ** 2, axis=-1))

    # Convert distances to similarities (smaller distance = higher similarity)
    sim_matrix = -distances / temperature

    # Create a mask for positive pairs
    pos_mask = jnp.eye(num_augs + 1, dtype=bool)
    pos_mask = pos_mask.at[0, 1:].set(True)
    pos_mask = pos_mask.at[1:, 0].set(True)

    # Compute NT-Xent loss
    exp_sim_matrix = jnp.exp(sim_matrix - jnp.max(sim_matrix, axis=-1, keepdims=True))
    
    # Positive similarities
    pos_sim = jnp.sum(exp_sim_matrix * pos_mask, axis=-1)
    
    # Negative similarities (excluding positive pairs)
    neg_sim = jnp.sum(exp_sim_matrix * (1 - pos_mask), axis=-1)
    
    loss = -jnp.log(pos_sim / (pos_sim + neg_sim))
    
    return jnp.mean(loss)


@jax.jit
def loss(logits, targets, mean, logvar, aug_mean, contrastive_loss_flag, key, beta=1.0, gamma=1.0, delta=0, temperature=0.8, margin=0.8):
    recon_loss = binary_cross_entropy_with_logits(logits, targets).mean()
    kld_loss = kl_divergence(mean, logvar)
    
    cont_loss = jax.lax.cond(
        contrastive_loss_flag == 0,
        lambda _: nt_xent_loss(mean, aug_mean, temperature=temperature),
        lambda _: triplet_margin_loss(mean, aug_mean, key, margin=margin),
        operand=None
    )
    consistency_loss = rotation_consistency_loss(mean, aug_mean)
    
    # clip the loss values to avoid NaNs
    recon_loss = jnp.clip(recon_loss, a_min=0, a_max=1e6)
    kld_loss = jnp.clip(kld_loss, a_min=0, a_max=1e6)
    cont_loss = jnp.clip(cont_loss, a_min=0, a_max=1e6)
    consistency_loss = jnp.clip(consistency_loss, a_min=0, a_max=1e6)
    
    total_loss = recon_loss + beta * kld_loss + gamma * cont_loss + delta * consistency_loss
    return total_loss, (recon_loss, kld_loss, cont_loss)
