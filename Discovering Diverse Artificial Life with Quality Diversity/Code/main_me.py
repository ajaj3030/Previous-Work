import os
import time
import pickle
from functools import partial
import logging
logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp

from lenia.lenia import ConfigLenia, Lenia
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.map_elites import MAPElites
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from common import get_metric, repertoire_variance
import wandb
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="me")
def main(config: DictConfig) -> None:
	logging.info("Starting MAP-Elites...")
	
	# Retrieve Wandb API key from environment variable
	wandb_api_key = os.environ.get('WANDB_API_KEY')
	if not wandb_api_key:
		raise ValueError("WANDB_API_KEY not found in environment variables")
	
	# Initialize wandb with API key
	wandb.login(key=wandb_api_key)

	# Init a random key
	key = jax.random.PRNGKey(config.seed)

	# Lenia
	logging.info("Initializing Lenia...")
	config_lenia = ConfigLenia(
		# Init pattern
		pattern_id=config.pattern_id,
		key=config.seed,

		# Simulation
		world_size=config.world_size,
		world_scale=config.world_scale,
		n_step=config.n_step,

		# Genotype
		n_params_size=config.n_params_size,
		n_cells_size=config.n_cells_size,
	)
	lenia = Lenia(config_lenia)

	# Load pattern
	init_carry, init_genotype, other_asset = lenia.load_pattern(lenia.pattern)

	# Define the scoring function
	def fitness_fn(observation):
		fitness = get_metric(observation, config.qd.fitness, config.qd.n_keep)
		assert fitness.size == 1
		fitness = jnp.squeeze(fitness)

		failed = jnp.logical_or(observation.stats.is_empty.any(), observation.stats.is_full.any())
		failed = jnp.logical_or(failed, observation.stats.is_spread.any())
		fitness = jnp.where(failed, -jnp.inf, fitness)
		return fitness

	def descriptor_fn(observation):
		descriptor = jnp.concatenate([get_metric(observation, descriptor, config.qd.n_keep) for descriptor in config.qd.descriptor])
		return descriptor

	def evaluate(genotype):
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
		
		lenia_step = partial(lenia.step, phenotype_size=config.phenotype_size,
							center_phenotype=config.center_phenotype,
							record_phenotype=config.record_phenotype,
							adaptive_phenotype=config.adaptive_phenotype,
							switch=flag)
		
		carry, accum = jax.lax.scan(lenia_step, init=carry, xs=jnp.arange(lenia._config.n_step))

		fitness = fitness_fn(accum)
		descriptor = descriptor_fn(accum)
		accum = jax.tree.map(lambda x: x[-1:], accum)  # to compute variance
		return fitness, descriptor, accum

	def scoring_fn(genotypes, key):
		fitnesses, descriptors, observations = jax.vmap(evaluate)(genotypes)

		fitnesses_nan = jnp.isnan(fitnesses)
		descriptors_nan = jnp.any(jnp.isnan(descriptors), axis=-1)
		fitnesses = jnp.where(fitnesses_nan | descriptors_nan, -jnp.inf, fitnesses)

		return fitnesses, descriptors, {"observations": observations}, key

	# Compute the centroids
	descriptor_min = jnp.array(config.qd.descriptor_min)
	descriptor_max = jnp.array(config.qd.descriptor_max)
	centroids, key = compute_cvt_centroids(
		num_descriptors=descriptor_min.size,
		num_init_cvt_samples=config.qd.n_init_cvt_samples,
		num_centroids=config.qd.repertoire_size,
		minval=descriptor_min,
		maxval=descriptor_max,
		random_key=key,
	)
 
	def kernel_swap_mutation(genotype, key):
		# Print debugging information
		print(f"Genotype shape: {genotype.shape}")
		print(f"lenia.n_params: {lenia.n_params}")
		print(f"config.n_params_size: {config.n_params_size}")
		print(f"lenia.n_kernel: {lenia.n_kernel}")

		# Extract the kernel parameters from the genotype
		kernel_params = genotype[:, :lenia.n_params]
		print(f"Extracted kernel_params shape: {kernel_params.shape}")

		# Reshape the kernel parameters
		try:
			kernel_params = kernel_params.reshape((genotype.shape[0], config.n_params_size, lenia.n_kernel))
			print(f"Reshaped kernel_params shape: {kernel_params.shape}")
		except ValueError as e:
			print(f"Reshape failed: {e}")
			return genotype, key  # Return original genotype and key if reshape fails

		# Generate two random indices for swapping
		key, subkey1, subkey2 = jax.random.split(key, 3)
		idx1, idx2 = jax.random.choice(subkey1, lenia.n_kernel, shape=(2,), replace=False)

		# Generate interpolation factor
		alpha = jax.random.uniform(subkey2, shape=(genotype.shape[0], config.n_params_size, 1), minval=0.1, maxval=0.9)

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

	# Define a metrics function
	metrics_fn = partial(default_qd_metrics, qd_offset=0.)

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

	# Define emitter
	variation_fn = partial(isoline_variation, iso_sigma=config.qd.iso_sigma, line_sigma=config.qd.line_sigma)
	mixing_emitter = MixingEmitter(
		mutation_fn=mutation_fn,
		variation_fn=variation_fn,
		variation_percentage=1.0,
		batch_size=config.qd.batch_size
	)

	# Instantiate MAP-Elites
	me = MAPElites(
		scoring_function=scoring_fn,
		emitter=mixing_emitter,
		metrics_function=metrics_fn,
	)

	# Compute initial repertoire and emitter state
	logging.info("Initializing MAP-Elites...")
	key, subkey = jax.random.split(key)
	init_genotypes = init_genotype[None, ...].repeat(config.qd.batch_size, axis=0)
	init_genotypes += jax.random.normal(subkey, shape=(config.qd.batch_size, lenia.n_gene)) * config.qd.iso_sigma
	repertoire, emitter_state, key = me.init(
		init_genotypes,
		centroids,
		key,
	)

	metrics = dict.fromkeys(["generation", "qd_score", "coverage", "max_fitness", "n_elites", "variance", "time"], jnp.array([]))
	csv_logger = CSVLogger("./log.csv", header=list(metrics.keys()))

	# Main loop
	logging.info("Starting main loop...")

	def me_scan(carry, unused):
		repertoire, key = carry

		# ME update
		(repertoire, _, metrics, key,) = me.update(
			repertoire,
			None,
			key,
		)

		return (repertoire, key), metrics

	for generation in range(0, config.qd.n_generations, config.qd.log_interval):
		start_time = time.time()
		(repertoire, key,), current_metrics = jax.lax.scan(
			me_scan,
			(repertoire, key),
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
		metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

		# Log
		log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
		csv_logger.log(log_metrics)
		logging.info(log_metrics)

	# Metrics
	logging.info("Saving metrics...")
	with open("./metrics.pickle", "wb") as metrics_file:
		pickle.dump(metrics, metrics_file)

	# Repertoire
	logging.info("Saving repertoire...")
	os.mkdir("./repertoire/")
	repertoire.replace(observations=jnp.nan).save(path="./repertoire/")


if __name__ == "__main__":
	main()
