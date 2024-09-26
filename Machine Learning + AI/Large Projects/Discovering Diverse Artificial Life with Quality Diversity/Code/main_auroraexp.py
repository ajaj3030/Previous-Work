import os
import time
import pickle
from functools import partial
import logging
logger = logging.getLogger(__name__)
import mediapy
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax.scipy import ndimage
import optax
import random
import json
from lenia.lenia import ConfigLenia, Lenia
from vae import VAE
from vae import loss as loss_vae
from qdax.core.aurora import AURORA
from qdax.core.emitters.mutation_operators import isoline_variation as isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from common import get_metric, repertoire_variance

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Import the patterns dictionary from patterns.py
from lenia.patterns import patterns

def load_new_pattern(file_path):
    with open(file_path, 'r') as f:
        new_patterns = json.load(f)
    
    # Select a pattern (e.g., the first one or randomly)
    pattern_id = random.choice(list(new_patterns.keys()))  # or random.choice(list(new_patterns.keys()))
    new_pattern = new_patterns[pattern_id]
    print(f"Selected pattern: {pattern_id}")
    
    # Ensure the new pattern has the required structure
    if "cells" not in new_pattern or "kernels" not in new_pattern:
        raise ValueError("New pattern does not have the required structure")
    
    return new_pattern

@hydra.main(version_base=None, config_path="configs/", config_name="aurora")
def main(config: DictConfig) -> None:
	def run_aurora(config:DictConfig, iteration: int, pattern) -> dict:
		logging.info("Starting AURORA...")

		# Retrieve Wandb API key from environment variable
		wandb_api_key = os.environ.get('WANDB_API_KEY')

		# Initialize wandb with API key
		wandb.login(key=wandb_api_key)

		# Initialise wandb
		currentdatetime = time.strftime("%Y-%m-%d_%H-%M-%S")
		wandb.init(project="Thesis", name=f"AURORA-Standard-{currentdatetime}-{config.pattern_id} -{iteration}", config=OmegaConf.to_container(config, resolve=True))
		# Init a random key
		key = jax.random.PRNGKey(config.seed)

		# Lenia initialization
		config_lenia = ConfigLenia(
			pattern_id=config.pattern_id,
			R = config.R,
			T = config.T,
			key=config.seed,
			world_size=config.world_size,
			world_scale=config.world_scale,
			n_step=config.n_step,
			n_params_size=config.n_params_size,
			n_cells_size=config.n_cells_size,
		)
		lenia = Lenia(config_lenia, custom_pattern=pattern)

		# Load pattern
		init_carry, init_genotype, other_asset = lenia.load_pattern(pattern)
	
		# VAE
		key, subkey_1, subkey_2 = jax.random.split(key, 3)
		if config.qd.group_cnn:
			phenotype_fake = jnp.zeros((config.qd.n_keep, config.phenotype_size, config.phenotype_size, lenia.n_channel))
		else:
			phenotype_fake = jnp.zeros((config.phenotype_size, config.phenotype_size, lenia.n_channel))
	
		vae = VAE(img_shape=phenotype_fake.shape, 
			latent_size=config.qd.hidden_size, 
			features=config.qd.features,
			group_cnn=config.qd.group_cnn)
	
		params = vae.init(subkey_1, phenotype_fake, subkey_2)

		params_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
		logging.info(f"VAE params count: {params_count}")

		# Create train state
		train_steps_per_epoch = config.qd.repertoire_size // config.qd.ae_batch_size
		train_steps_total = config.qd.n_generations * config.qd.train_ratio * train_steps_per_epoch
		learning_rate_fn = optax.linear_schedule(
			init_value=config.qd.lr_init_value,
			end_value=config.qd.lr_init_value,
			transition_steps=config.qd.lr_transition_steps,
			transition_begin=config.qd.lr_transition_begin,
		)
		tx = optax.adam(learning_rate_fn)
		train_state = TrainState.create(apply_fn=vae.apply, params=params, tx=tx)

		# Define the scoring function
		def latent_mean(observation, train_state, key):
			latents = vae.apply(train_state.params, observation.phenotype[-config.qd.n_keep:], key, method=vae.encode)
			return jnp.mean(latents, axis=-2)

		def latent_variance(observation, train_state, key):
			latents = vae.apply(train_state.params, observation.phenotype[-config.qd.n_keep:], key, method=vae.encode)
			latent_mean = jnp.mean(latents, axis=-2)
			return -jnp.mean(jnp.linalg.norm(latents - latent_mean[..., None, :], axis=-1), axis=-1)

		def fitness_fn(observation, train_state, key):
			if config.qd.fitness == "unsupervised":
				fitness = latent_variance(observation, train_state, key)
			else:
				fitness = get_metric(observation, config.qd.fitness, config.qd.n_keep)
				assert fitness.size == 1
				fitness = jnp.squeeze(fitness)

			if config.qd.secondary_fitness:
				secondary_fitness = get_metric(observation, config.qd.secondary_fitness, config.qd.n_keep)
				assert secondary_fitness.size == 1
				secondary_fitness = jnp.squeeze(secondary_fitness)
				fitness += config.qd.secondary_fitness_weight * secondary_fitness

			failed = jnp.logical_or(observation.stats.is_empty.any(), observation.stats.is_full.any())
			failed = jnp.logical_or(failed, observation.stats.is_spread.any())
			fitness = jnp.where(failed, -jnp.inf, fitness)
			return fitness

		def kernel_swap_mutation(genotype, key):
			# Extract the kernel parameters from the genotype
			kernel_params = genotype[:, :lenia.n_params]
			kernel_params = kernel_params.reshape((genotype.shape[0], config.n_params_size, lenia.n_kernel))

			# Generate two random indices for swapping
			key, subkey1, subkey2 = jax.random.split(key, 3)
			idx1, idx2 = jax.random.choice(subkey1, lenia.n_kernel, shape=(2,), replace=False)

			# Generate interpolation factor and squeeze the last dimension
			alpha = jax.random.uniform(subkey2, shape=(genotype.shape[0], config.n_params_size, 1), minval=0.1, maxval=0.9)
			alpha = jnp.squeeze(alpha, axis=-1)  # This will make alpha shape (128, 3)

			# Interpolate between kernel parameters instead of hard swap
			new_kernel_params = kernel_params.at[:, :, idx1].set(
				alpha * kernel_params[:, :, idx1] + (1 - alpha) * kernel_params[:, :, idx2]
			)
			new_kernel_params = new_kernel_params.at[:, :, idx2].set(
				(1 - alpha) * kernel_params[:, :, idx1] + alpha * kernel_params[:, :, idx2]
			)

			# Flatten the kernel parameters
			new_kernel_params = new_kernel_params.reshape((genotype.shape[0], -1))

			# Reconstruct the genotype with swapped kernel parameters
			new_genotype = jnp.concatenate([new_kernel_params, genotype[:, lenia.n_params:]], axis=1)

			return new_genotype, key

		def kernel_shift_mutation(genotype, key):
			# Extract the kernel parameters from the genotype
			kernel_params = genotype[:, :lenia.n_params]
			kernel_params = kernel_params.reshape((genotype.shape[0], config.n_params_size, lenia.n_kernel))

			# Generate shift amount (between 0 and 0.5 for a more conservative shift)
			key, subkey = jax.random.split(key)
			shift_amount = jax.random.uniform(subkey, shape=(genotype.shape[0], config.n_params_size, 1), minval=0, maxval=0.5)

			# Perform a partial, parameter-wise shift
			shifted = jnp.roll(kernel_params, shift=1, axis=2)
			new_kernel_params = kernel_params * (1 - shift_amount) + shifted * shift_amount

			# Flatten the kernel parameters
			new_kernel_params = new_kernel_params.reshape((genotype.shape[0], -1))

			# Reconstruct the genotype with shifted kernel parameters
			new_genotype = jnp.concatenate([new_kernel_params, genotype[:, lenia.n_params:]], axis=1)

			return new_genotype, key

		def rgb_channel_swap_mutation(genotype, key):
			# Extract the seed pattern from the genotype
			seed_pattern = genotype[:, lenia.n_params:]
			seed_shape = (genotype.shape[0], config.n_cells_size, config.n_cells_size, 3)
			seed_pattern = seed_pattern.reshape(seed_shape)

			# Generate random permutations for each individual in the batch
			def permute_channels(key):
				return jax.random.permutation(key, jnp.arange(3))

			keys = jax.random.split(key, genotype.shape[0])
			permutations = jax.vmap(permute_channels)(keys)

			# Apply the permutations to swap RGB channels
			new_seed_pattern = jnp.take_along_axis(seed_pattern, permutations[:, None, None, :], axis=-1)

			# Flatten the new seed pattern
			new_seed_pattern = new_seed_pattern.reshape((genotype.shape[0], -1))

			# Reconstruct the genotype with swapped RGB channels
			new_genotype = jnp.concatenate([genotype[:, :lenia.n_params], new_seed_pattern], axis=1)

			return new_genotype, key

		def adaptive_mutation(genotype, key):
			# Determine which mutation to apply based on random choice
			key, subkey = jax.random.split(key)
			# log what the key is
			logging.info(f"key: {key}")
			mutation_choice = jax.random.choice(subkey, jnp.array([0, 1, 2]))

			def apply_kernel_swap(g, k):
				return kernel_swap_mutation(g, k)

			def apply_kernel_shift(g, k):
				return kernel_shift_mutation(g, k)

			def apply_rgb_swap(g, k):
				return rgb_channel_swap_mutation(g, k)

			return jax.lax.switch(mutation_choice,
								[apply_kernel_swap,
								apply_kernel_shift,
								apply_rgb_swap],
								genotype, key)

		# Define mutation function based on config
		if config.qd.mutation_type == "adaptive":
			mutation_fn = adaptive_mutation
		elif config.qd.mutation_type == "kernel_swap":
			mutation_fn = kernel_swap_mutation
		elif config.qd.mutation_type == "kernel_shift":
			mutation_fn = kernel_shift_mutation
		elif config.qd.mutation_type == "color_swap":
			mutation_fn = rgb_channel_swap_mutation
		elif config.qd.mutation_type == "None":
			mutation_fn = None
		else:
			raise ValueError(f"Unknown mutation type: {config.qd.mutation_type}")

		def descriptor_fn(observation, train_state, key):
			descriptor_unsupervised = latent_mean(observation, train_state, key)
			return descriptor_unsupervised

		def evaluate(genotype, train_state, key):
			carry = lenia.express_genotype(init_carry, genotype)
			
			# Randomly generate a number to determine whether to switch the phenotype,
			# with the expected value being around half the number of generations.
			# Returns a True or False flag. Once true, it never changes again
			switch_key, fitness_key, descriptor_key = jax.random.split(key, 3)
			switch_flag = jax.random.bernoulli(switch_key, p=1 / config.qd.n_generations)
			
			# if switch flag within 10% of the half the number of generations, then switch
			flag = jnp.logical_and(
				jnp.logical_and(switch_flag, config.adaptive_phenotype),
				jnp.abs(config.qd.n_generations / 2 - config.qd.n_generations) < 0.1 * config.qd.n_generations
			)
			"""
			# if fitness function contains the word linearvelocity, kinetic energy, or momentum, then we need an uncentered phenotype for the fitness function
			if "linearvelocity" in config.qd.fitness or "kineticenergy" in config.qd.fitness or "momentum" in config.qd.fitness:
				lenia_step = partial(lenia.step, phenotype_size=config.phenotype_size,
									center_phenotype=False,
									record_phenotype=config.record_phenotype,
									adaptive_phenotype=config.adaptive_phenotype,
									switch=flag)    
				fitness_carry, fitness_accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))
			"""
			
			lenia_step = partial(lenia.step, phenotype_size=config.phenotype_size,
								center_phenotype=config.center_phenotype,
								record_phenotype=config.record_phenotype,
								adaptive_phenotype=config.adaptive_phenotype,
								switch=flag)
			
			carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))
			fitness = fitness_fn(accum, train_state, key)
			descriptor = descriptor_fn(accum, train_state, key)
			accum = jax.tree.map(lambda x: x[-config.qd.n_keep_ae:], accum)
			
			return fitness, descriptor, accum

		def scoring_fn(genotypes, train_state, key):
			batch_size = jax.tree.leaves(genotypes)[0].shape[0]
			key, *keys = jax.random.split(key, batch_size+1)
			fitnesses, descriptors, observations = jax.vmap(evaluate, in_axes=(0, None, 0))(genotypes, train_state, jnp.array(keys))

			fitnesses_nan = jnp.isnan(fitnesses)
			descriptors_nan = jnp.any(jnp.isnan(descriptors), axis=-1)
			fitnesses = jnp.where(fitnesses_nan | descriptors_nan, -jnp.inf, fitnesses)

			return fitnesses, descriptors, {"observations": observations}, key

		@jax.jit
		def calculate_rotational_diversity(repertoire):
			# Extract the phenotypes from the repertoire
			phenotypes = repertoire.observations.phenotype[:, -config.qd.n_keep:]

			def pairwise_rotational_diversity(phenotype1, phenotype2):
				def rotate_and_compare(angle):
					rotated = jax.vmap(lambda frame: rotate_image(frame, angle))(phenotype2)
					return jnp.mean(jnp.abs(phenotype1 - rotated))
				
				diversities = jax.vmap(rotate_and_compare)(jnp.arange(0, 360, 15))
				return jnp.min(diversities)

			# Calculate the full diversity matrix
			diversity_matrix = jax.vmap(lambda p1: jax.vmap(lambda p2: pairwise_rotational_diversity(p1, p2))(phenotypes))(phenotypes)

			# For each phenotype, get the average of the 5 most similar solutions
			def get_top_k_average(row):
				k=5
				sorted_row = jnp.sort(row)
				return jnp.mean(sorted_row[1:1+k]), row[0]  # Exclude the first one as it's the self-comparison (0)
				

			top_5_averages, self_diversity_scores = jax.vmap(get_top_k_average)(diversity_matrix)

			# Calculate mean diversity
			mean_diversity = jnp.mean(top_5_averages)
			mean_self_diversity = jnp.mean(self_diversity_scores)

			# Calculate unique patterns
			threshold = 0.1  # Adjust this value based on your specific needs
			unique_patterns = jnp.sum(top_5_averages > threshold)

			return mean_diversity, unique_patterns

		# Define a metrics function
		metrics_fn = partial(default_qd_metrics, qd_offset=0.)
		
		
	
		# Define emitter
		variation_fn = partial(isoline_variation, iso_sigma=config.qd.iso_sigma, line_sigma=config.qd.line_sigma)
		mixing_emitter = MixingEmitter(
			mutation_fn=mutation_fn,
			variation_fn=variation_fn,
			variation_percentage=config.qd.variation_percentage,
			batch_size=config.qd.batch_size
		)

		def affine_transform(image, matrix):
				"""Apply an affine transformation to an image."""
				h, w = image.shape[:2]
				y, x = jnp.mgrid[:h, :w]
				coords = jnp.stack([x.ravel(), y.ravel(), jnp.ones_like(x.ravel())])
				new_coords = jnp.dot(matrix, coords)
				new_x, new_y = new_coords[0] / new_coords[2], new_coords[1] / new_coords[2]
				new_x = jnp.clip(new_x, 0, w - 1).reshape(h, w)
				new_y = jnp.clip(new_y, 0, h - 1).reshape(h, w)
				coords = jnp.stack([new_y, new_x])
				
				# Handle multi-channel images
				if image.ndim == 3:
					return jnp.stack([ndimage.map_coordinates(image[..., c], coords, order=1, mode='nearest')
									for c in range(image.shape[-1])], axis=-1)
				else:
					return ndimage.map_coordinates(image, coords, order=1, mode='nearest')

		# Train
		if config.qd.use_data_augmentation:
			def data_augmentation(batch, key):
				original_batch = batch

				# Flip
				flipped_batch = jnp.flip(batch, axis=1)
				batch_1, batch_2, batch_3, batch_4 = jnp.split(batch, 4)
				batch_1 = jax.vmap(lambda x: jnp.rot90(x, k=0, axes=(0, 1)))(batch_1)
				batch_2 = jax.vmap(lambda x: jnp.rot90(x, k=1, axes=(0, 1)))(batch_2)
				batch_3 = jax.vmap(lambda x: jnp.rot90(x, k=2, axes=(0, 1)))(batch_3)
				batch_4 = jax.vmap(lambda x: jnp.rot90(x, k=3, axes=(0, 1)))(batch_4)
				rotated_batch = jnp.concatenate([batch_1, batch_2, batch_3, batch_4], axis=0)

				# Concatenate all augmentations
				aug_batch = jnp.concatenate([flipped_batch] + [rotated_batch], axis=0)

				# Shear
				shear_angles = [0.2, -0.2]  # Shear angles in radians
				sheared_batches = []
				for angle in shear_angles:
					shear_matrix = jnp.array([[1, -jnp.sin(angle), 0],
											[0, jnp.cos(angle), 0],
											[0, 0, 1]])
					sheared_batch = jax.vmap(lambda x: affine_transform(x, shear_matrix))(batch)
					sheared_batches.append(sheared_batch)

				# Squeeze (scale)
				scale_factors = [0.8, 1.2]  # Scale factors
				squeezed_batches = []
				for scale in scale_factors:
					scale_matrix = jnp.array([[scale, 0, 0],
											[0, scale, 0],
											[0, 0, 1]])
					squeezed_batch = jax.vmap(lambda x: affine_transform(x, scale_matrix))(batch)
					squeezed_batches.append(squeezed_batch)

				aug_batch = jnp.concatenate([rotated_batch] + sheared_batches + squeezed_batches, axis=0)

				return aug_batch
		else:
			def data_augmentation(batch, key):
				return batch

		@partial(jax.jit, static_argnames=("learning_rate_fn",))
		def train_step(train_state, batch, key, learning_rate_fn):
			def loss_fn(params):
				logits, mean, logvar = train_state.apply_fn(params, batch, key)
				return loss_vae(logits, batch, mean, logvar)

			loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
			train_state = train_state.apply_gradients(grads=grads)

			learning_rate = learning_rate_fn(train_state.step)

			return train_state, {"loss": loss, "learning_rate": learning_rate}

		def train_epoch(train_state, repertoire, key):
			steps_per_epoch = repertoire.size // config.qd.ae_batch_size

			key, subkey = jax.random.split(key)
			valid = repertoire.fitnesses != -jnp.inf
			indices = jax.random.choice(subkey, jnp.arange(repertoire.size), shape=(repertoire.size,), p=valid)
			indices = indices[:steps_per_epoch * config.qd.ae_batch_size]
			indices = indices.reshape((steps_per_epoch, config.qd.ae_batch_size))

			def scan_train_step(carry, x):
				train_state = carry
				batch_indices, key = x
				subkey_1, subkey_2, subkey_3 = jax.random.split(key, 3)
				step_indices = jax.random.randint(subkey_1, shape=(config.qd.ae_batch_size,), minval=0, maxval=config.qd.n_keep_ae)
				batch = repertoire.observations.phenotype[batch_indices, step_indices]
				batch = data_augmentation(batch, subkey_2)
				train_state, metrics = train_step(train_state, batch, subkey_3, learning_rate_fn)
				return train_state, metrics

			keys = jax.random.split(key, steps_per_epoch)
			train_state, metrics = jax.lax.scan(
				scan_train_step,
				train_state,
				(indices, keys),
				length=steps_per_epoch,
			)
			return train_state, metrics

		def train_fn(key, repertoire, train_state):
			def scan_train_epoch(carry, x):
				train_state = carry
				key = x
				train_state, metrics = train_epoch(train_state, repertoire, key)
				return train_state, metrics

			keys = jax.random.split(key, config.qd.train_ratio)
			train_state, metrics = jax.lax.scan(
				scan_train_epoch,
				train_state,
				keys,
				length=config.qd.train_ratio,
			)
			return train_state, metrics

		@jax.jit
		def rotate_image(image, angle):
			# Convert angle to radians
			angle = jnp.deg2rad(angle)
			
			# Create rotation matrix
			cos_theta, sin_theta = jnp.cos(angle), jnp.sin(angle)
			rotation_matrix = jnp.array([[cos_theta, -sin_theta],
										[sin_theta, cos_theta]])
			
			# Get image dimensions
			height, width = image.shape[:2]
			
			# Create coordinate grid
			x = jnp.arange(width) - width // 2
			y = jnp.arange(height) - height // 2
			X, Y = jnp.meshgrid(x, y)
			
			# Rotate coordinates
			coords = jnp.stack([X.ravel(), Y.ravel()])
			rotated_coords = jnp.dot(rotation_matrix, coords)
			
			# Reshape and shift coordinates back
			rotated_x = rotated_coords[0].reshape(height, width) + width // 2
			rotated_y = rotated_coords[1].reshape(height, width) + height // 2
			
			# Clip coordinates to image boundaries
			rotated_x = jnp.clip(rotated_x, 0, width - 1)
			rotated_y = jnp.clip(rotated_y, 0, height - 1)
			
			# Interpolate
			x0, x1 = jnp.floor(rotated_x).astype(int), jnp.ceil(rotated_x).astype(int)
			y0, y1 = jnp.floor(rotated_y).astype(int), jnp.ceil(rotated_y).astype(int)
			
			Ia = image[y0, x0]
			Ib = image[y1, x0]
			Ic = image[y0, x1]
			Id = image[y1, x1]
			
			wa = (x1 - rotated_x) * (y1 - rotated_y)
			wb = (x1 - rotated_x) * (rotated_y - y0)
			wc = (rotated_x - x0) * (y1 - rotated_y)
			wd = (rotated_x - x0) * (rotated_y - y0)
			
			return wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

		@jax.jit
		def frame_rotational_similarity_score(frame):
			def body_fun(_, angle):
				rotated = rotate_image(frame, angle)
				return _, jnp.mean(jnp.abs(frame - rotated))
			
			_, scores = jax.lax.scan(body_fun, None, jnp.arange(0, 360, 15))
			return jnp.mean(scores)

		@jax.jit
		def rotational_similarity_score(phenotype):
			# phenotype shape: (time_steps, height, width, channels)
			return jnp.mean(jax.vmap(frame_rotational_similarity_score)(phenotype))

		# Vectorize the function to apply it to a batch of phenotypes
		vectorized_rotational_similarity_score = jax.vmap(rotational_similarity_score)

		# Init AURORA
		aurora = AURORA(
			emitter=mixing_emitter,
			scoring_fn=scoring_fn,
			fitness_fn=fitness_fn,
			descriptor_fn=descriptor_fn,
			train_fn=train_fn,
			metrics_fn=metrics_fn,
		)

		# Init step of the aurora algorithm
		logging.info("Initializing AURORA...")
		key, subkey = jax.random.split(key)
		init_genotypes = init_genotype[None, ...].repeat(config.qd.batch_size, axis=0)
		init_genotypes += jax.random.normal(subkey, shape=(config.qd.batch_size, lenia.n_gene)) * config.qd.iso_sigma
		repertoire, emitter_state, key = aurora.init(
			init_genotypes,
			train_state,
			config.qd.repertoire_size,
			key,
		)

		metrics = dict.fromkeys([
			"generation", "qd_score", "coverage", "max_fitness", "loss", 
			"learning_rate", "n_elites", "variance", "time", 
			"Rotational Similarity Score", "Rotational Diversity Score", "Unique Patterns"
		], jnp.array([]))
		csv_logger = CSVLogger("./log.csv", header=list(metrics.keys()))

		# Main loop
		logging.info("Starting main loop...")

		def aurora_scan(carry, unused):
			repertoire, train_state, key = carry

			# AURORA update
			repertoire, _, metrics, key = aurora.update(
				repertoire,
				None,
				key,
				train_state,
			)

			# AE training
			key, subkey = jax.random.split(key)
			repertoire, train_state, metrics_ae = aurora.train(
				repertoire, train_state, subkey
			)

			# Calculate rotational similarity score
			rot_similarity_scores = vectorized_rotational_similarity_score(repertoire.observations.phenotype[:, -config.qd.n_keep:])
			avg_rot_similarity = jnp.mean(rot_similarity_scores)

			mean_diversity, unique_patterns = calculate_rotational_diversity(repertoire)
			return (repertoire, train_state, key), (metrics, metrics_ae, avg_rot_similarity, mean_diversity, unique_patterns)

		for generation in range(0, config.qd.n_generations, config.qd.log_interval):
			start_time = time.time()
			(repertoire, train_state, key), (current_metrics, current_metrics_ae, avg_rot_similarity, mean_diversity, unique_patterns) = jax.lax.scan(
				aurora_scan,
				(repertoire, train_state, key),
				(),
				length=config.qd.log_interval,
			)
			timelapse = time.time() - start_time

			# Metrics
			current_metrics["generation"] = jnp.arange(1+generation, 1+generation+config.qd.log_interval, dtype=jnp.int32)
			current_metrics["n_elites"] = jnp.sum(current_metrics["is_offspring_added"], axis=-1)
			del current_metrics["is_offspring_added"]
			variance = repertoire_variance(repertoire)
			current_metrics["variance"] = jnp.repeat(variance, config.qd.log_interval)
			current_metrics["time"] = jnp.repeat(timelapse, config.qd.log_interval)
			current_metrics["Rotational Similarity Score"] = jnp.repeat(avg_rot_similarity, config.qd.log_interval)
			current_metrics["Rotational Diversity Score"] = jnp.repeat(mean_diversity, config.qd.log_interval)
			current_metrics["Unique Patterns"] = jnp.repeat(unique_patterns, config.qd.log_interval)
			current_metrics_ae = jax.tree_util.tree_map(lambda metric: jnp.repeat(metric[-1], config.qd.log_interval), current_metrics_ae)
			current_metrics |= current_metrics_ae

			metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

			# Log
			log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)  # log last value
			csv_logger.log(log_metrics)
			wandb.log(log_metrics)
			logging.info(log_metrics)
	
		# Metrics
		logging.info("Saving metrics...")
		with open(f"./metrics_iteration{iteration}.pickle", "wb") as metrics_file:
			pickle.dump(metrics, metrics_file)

		# Repertoire
		logging.info("Saving repertoire...")
		os.mkdir(f"./repertoire_iteration{iteration}/")
		repertoire.replace(observations=jnp.nan).save(path=f"./repertoire_iteration{iteration}/")

		# Autoencoder
		logging.info("Saving autoencoder params...")
		with open(f"./params.pickle_iteration{iteration}", "wb") as params_file:
			pickle.dump(train_state.params, params_file)
   
		# Save images of the patterns to a directory
		patterns_dir = f'./patterns_iteration{iteration}/'
		os.makedirs(patterns_dir, exist_ok=True)
  
		num_genotypes = len(repertoire.genotypes)
		batch_size = 10  # Process genotypes in batches of 10

		# Select random genotypes
		valid_indices = jnp.where(repertoire.fitnesses != -jnp.inf)[0]
		selected_genotypes = repertoire.genotypes[valid_indices]

		def evaluate(genotype):
			lenia_step = partial(lenia.step, phenotype_size=config.world_size, center_phenotype=True, record_phenotype=True, adaptive_phenotype=False, switch=False)
			carry = lenia.express_genotype(init_carry, genotype)
			carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))
			return accum

		# Evaluate all genotypes
		all_accum = jax.vmap(evaluate)(selected_genotypes)

		# Generate all titles
		all_titles = [f"Genotype {index}" for index in valid_indices]

		# Create a random permutation of indices
		key = jax.random.PRNGKey(0)  # You might want to use a different seed
		permutation = jax.random.permutation(key, len(valid_indices))

		# Use the permutation to shuffle images and titles
		shuffled_images = all_accum.phenotype[:, -1][permutation]
		shuffled_titles = [all_titles[int(i)] for i in permutation]

		# Generate HTML content
		html = ""
		for i in range(0, len(shuffled_images), batch_size):
			batch_images = shuffled_images[i:i+batch_size]
			batch_titles = shuffled_titles[i:i+batch_size]
			
			# Generate HTML for the batch
			batch_html = mediapy.show_images(batch_images, titles=batch_titles, width=128, height=128, return_html=True)
			html += batch_html

		# Save HTML content to a file
		output_file = f"./images_iteration{iteration}.html"
		with open(output_file, "w") as file:
			file.write(html)

		patterns = {}
		for i, genotype in enumerate(repertoire.genotypes):
			# if fitness is -inf, skip
			if jnp.isnan(repertoire.fitnesses[i]):
				continue
			carry = lenia.express_genotype(init_carry, genotype)
			lenia_step = partial(lenia.step, phenotype_size=config.phenotype_size,
									center_phenotype=config.center_phenotype,
									record_phenotype=config.record_phenotype,
									adaptive_phenotype=config.adaptive_phenotype,
									switch=config.switch)
			
			carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))

			pattern = lenia.save_pattern(carry, accum, i)
			patterns[i] = pattern
	
		# save the pattern as a json
		import json
		with open(f'./pattern_iteration{iteration}.json', 'w') as f:
			json.dump(patterns, f)
   
		print(f"HTML file saved as: {output_file}")
		print(f"Individual images saved in: {patterns_dir}")
		# Finish the wandb run
		wandb.finish()
	
		return patterns

	def select_new_pattern(patterns: dict) -> str:
		pattern_id = random.choice(list(patterns.keys()))
		return patterns[pattern_id]

	# Main loop for continuous runs
	iteration = 0
	current_pattern = patterns[config.pattern_id]  # Start with the initial pattern from config

	fitnesses = ['pos_mass_var', 'neg_angle_var', 'unsupervised', 'pos_linearvelocity_avg']
	while True:
		if iteration == 0:
			logging.info("Starting the first iteration")
			# Use the initial pattern for the first run
			new_patterns = run_aurora(config, iteration, current_pattern)
		else:
			# Load a new pattern from the file generated in the previous iteration
			file_path = f'./pattern_iteration{iteration-1}.json'
			current_pattern = load_new_pattern(file_path)
			
			# Update the config with the new pattern id
			config.pattern_id = f"custom_pattern_{iteration}"
   
			if config.qd.adaptive_fitness:
				config.qd.fitness = random.choice(fitnesses)

			if config.adaptive_phenotype:
				config.phenotype_size += config.phenotype_size_increment
				config.n_cells_size += config.phenotype_size_increment

				# limit them both to world size
				config.phenotype_size = min(config.phenotype_size, config.world_size)
				config.n_cells_size = min(config.n_cells_size, config.world_size)
			
			config.seed += 1
			# Run AURORA with the new pattern
			new_patterns = run_aurora(config, iteration, current_pattern)
		
		iteration += 1

		# Optional: Add a condition to stop after a certain number of iterations
		if iteration >= config.max_iterations:
			break
  
if __name__ == "__main__":
	main()
