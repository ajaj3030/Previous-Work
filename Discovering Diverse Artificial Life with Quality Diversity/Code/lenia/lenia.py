from typing import Any
from functools import partial
from dataclasses import dataclass
from collections import namedtuple
import os
import jax
import jax.numpy as jnp
from lenia.reintegration_tracking import ReintegrationTracking
import matplotlib.pyplot as plt
from lenia.patterns import patterns
from lenia.patterns import generate_random_pattern, generate_random_variant
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import logging
logger = logging.getLogger(__name__)
from jax import random
from datetime import datetime
import string

Carry = namedtuple('Carry', [ 'world', 'param', 'asset', 'temp' ])
Accum = namedtuple('Accum', [ 'phenotype', 'stats' ])
Param = namedtuple('Params', [ 'm', 's', 'h' ])
Asset = namedtuple('Asset', [ 'fK', 'X', 'reshape_c_k', 'reshape_k_c', 'R', 'T' ])
Temp = namedtuple('Temp', [ 'last_center', 'last_shift', 'total_shift', 'last_angle' ])
Stats = namedtuple('Stats', [
    'mass',
    'center_x',
    'center_y',
    'linear_velocity',
    'angle',
    'angular_velocity',
    'is_empty',
    'is_full',
    'is_spread',
    'size',
    'linmomentum',
    'kinetic'
])
Others = namedtuple('Others', [ 'D', 'K', 'cells', 'init_cells' ])


bell = lambda x, mean, stdev: jnp.exp(-((x-mean)/stdev)**2 / 2)
growth = lambda x, mean, stdev: 2 * bell(x, mean, stdev) - 1


@dataclass
class ConfigLenia:
    # Init pattern
    pattern_id: str = "5N7KKM"
    R: int = 18
    T: int = 10
    key: int = 0
 
    # World
    world_size: int = 128
    world_scale: int = 1

    # Simulation
    n_step: int = 200

    # Genotype
    n_params_size: int = 3
    n_cells_size: int = 32

    # New fields for obstacles
    use_obstacles: bool = False
    obstacle_types: List[str] = field(default_factory=lambda: ['wall', 'food'])
    obstacle_positions: List[Tuple[int, int]] = field(default_factory=lambda: [(16, 16), (100, 100)])
    obstacle_properties: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: {
		'wall': {'values': [1.0, 1.0, 1.0], 'size': 10},
		'food': {'values': [0.5, 0.3, 0.7], 'size': 8}
	})
    
    # Flow Lenia specific parameters
    dt: float = 0.2
    dd: int = 5
    sigma: float = 0.65
    border: str = "wall"
    has_hidden: bool = False
    mix: str = "stoch"



class Lenia:
	def __init__(self, config: ConfigLenia, custom_pattern=None):
		self._config = config
		self.pattern = None
		self.n_kernel = None
		self.n_channel = None
		self.n_params = None
		self.n_cells = None
		self.n_gene = None

		if custom_pattern is not None:
			self.pattern = custom_pattern
		elif self._config.pattern_id == "random":
			self.pattern = self.generate_viable_random_pattern()
		elif self._config.pattern_id == "random_variant":
			self.pattern = generate_random_variant(self._config.key)
		elif self._config.pattern_id in patterns:
			self.pattern = patterns[self._config.pattern_id]
		else:
			raise ValueError(f"Unknown pattern_id: {self._config.pattern_id}")

		self.initialize_attributes()    

		self.obstacles = None
		if self._config.use_obstacles:
			self.initialize_obstacles()
   
	def save_world_image(self, filename="world"):
		"""
		Creates and saves an image of the current world state, including obstacles.
		
		Args:
		filename (str): The name of the file to save the image (without extension).
		"""
		# Ensure we have a world state to visualize
		if not hasattr(self, 'original_world'):
			init_carry, _, _ = self.load_pattern(self.pattern)
			self.original_world = init_carry.world

		world_state = self.original_world

		fig, axes = plt.subplots(1, self.n_channel + 2, figsize=(5*(self.n_channel + 2), 5))
		
		# Visualize each channel
		for i in range(self.n_channel):
			axes[i].imshow(world_state[:,:,i], cmap='viridis', vmin=0, vmax=1)
			axes[i].set_title(f'Channel {i}')
			axes[i].axis('off')
		
		# Visualize combined channels
		combined = jnp.sum(world_state, axis=-1)
		axes[-2].imshow(combined, cmap='viridis')
		axes[-2].set_title('Combined')
		axes[-2].axis('off')
		
		# Visualize obstacles
		if self._config.use_obstacles and hasattr(self, 'obstacles'):
			obstacle_vis = jnp.sum(self.obstacles, axis=-1)
			axes[-1].imshow(obstacle_vis, cmap='gray', vmin=0, vmax=1)
			axes[-1].set_title('Obstacles')
			axes[-1].axis('off')
		else:
			axes[-1].axis('off')
			axes[-1].set_title('No Obstacles')
		
		plt.tight_layout()
		
		# Create 'world' directory if it doesn't exist
		os.makedirs('world', exist_ok=True)
		
		# Save the figure
		plt.savefig(f'world/{filename}.png')
		plt.close(fig)  # Close the figure to free up memory
   
	def initialize_obstacles(self):
		self.obstacles = jnp.zeros((self._config.world_size, self._config.world_size, self.n_channel))
		for obs_type, pos in zip(self._config.obstacle_types, self._config.obstacle_positions):
			properties = self._config.obstacle_properties[obs_type]
			# Make obstacles larger
			size = properties.get('size', 5)  # Default size of 5x5
			x, y = pos
			x_slice = slice(max(0, x-size//2), min(self._config.world_size, x+size//2+1))
			y_slice = slice(max(0, y-size//2), min(self._config.world_size, y+size//2+1))
			for channel, value in enumerate(properties['values']):
				self.obstacles = self.obstacles.at[y_slice, x_slice, channel].set(value)
		
		# Print some debug information
		print(f"Obstacle types: {self._config.obstacle_types}")
		print(f"Obstacle positions: {self._config.obstacle_positions}")
		print(f"Obstacle properties: {self._config.obstacle_properties}")
		print(f"Max obstacle value: {jnp.max(self.obstacles)}")
		print(f"Min obstacle value: {jnp.min(self.obstacles)}")

	def initialize_attributes(self):
		# Genotype
		self.n_kernel = len(self.pattern["kernels"])  # k, number of kernels
		self.n_channel = len(self.pattern["cells"])  # c, number of channels
		self.n_params = self._config.n_params_size * self.n_kernel  # p*k, number of parameters inside genotype
		self.n_cells = self._config.n_cells_size * self._config.n_cells_size * self.n_channel  # e*e*c, number of embryo cells inside genotype
		self.n_gene = self.n_params + self.n_cells  # size of genotype

	def generate_viable_random_pattern(self):
		max_attempts = 10000
		for attempt in range(max_attempts):
			if (attempt + 1) % 20 == 0:
				print(f"Attempt number {attempt + 1}")
			pattern = generate_random_pattern(self._config.R, self._config.T)
			self.pattern = pattern
			self.initialize_attributes()
			
			# Simulate for a longer time
			n_steps = 300
			init_carry, init_genotype, _ = self.load_pattern(pattern)
			carry = self.express_genotype(init_carry, init_genotype)
			
			def step(carry, _):
				return self.step(carry, None, self._config.world_size, False, False)

			final_carry, accum = jax.lax.scan(step, carry, None, length=n_steps)
			
			# Check if the pattern is still alive and evolving at the end
			final_mass = accum.stats.mass[-1]
			mass_change = jnp.abs(accum.stats.mass[-1] - accum.stats.mass[0]) / accum.stats.mass[0]
			
			if final_mass > 10:
				print(f"Viable pattern found after {attempt + 1} attempts")

				# Save an image of the pattern
				final_state = final_carry.world
				fig, ax = plt.subplots(figsize=(10, 10))
				
				# If the world is multi-channel, we'll sum across channels
				if final_state.ndim == 3:
					final_state = jnp.sum(final_state, axis=-1)
				
				im = ax.imshow(final_state, cmap='viridis', interpolation='nearest')
				plt.colorbar(im)
				ax.set_title(f"Viable Pattern (Attempt {attempt + 1})")
				plt.savefig(f"viable_pattern_attempt_{attempt + 1}.png")
				plt.close(fig)

				return pattern

		raise ValueError(f"Failed to generate a viable pattern after {max_attempts} attempts")

	def simulate_pattern(self, pattern, n_steps=200):
		init_carry, init_genotype, _ = self.load_pattern(pattern)
		carry = self.express_genotype(init_carry, init_genotype)
		
		def step(carry, _):
			return self.step(carry, None, self._config.world_size, False, False)

		final_carry, accum = jax.lax.scan(step, carry, None, length=n_steps)
		return not accum.stats.is_empty[-1] and not accum.stats.is_full[-1] and not accum.stats.is_spread[-1]

	def create_world_from_cells(self, cells):
		mid = self._config.world_size // 2

		# scale cells
		scaled_cells = cells.repeat(self._config.world_scale, axis=-3).repeat(self._config.world_scale, axis=-2)
		cy, cx = scaled_cells.shape[0], scaled_cells.shape[1]

		# create empty world and place cells
		A = jnp.zeros((self._config.world_size, self._config.world_size, self.n_channel))  # (y, x, c,)
		A = A.at[mid-cx//2:mid+cx-cx//2, mid-cy//2:mid+cy-cy//2, :].set(scaled_cells)
  
		if self._config.use_obstacles:
			# Ensure obstacles are initialized
			if self.obstacles is None:
				self.initialize_obstacles()
			# Apply obstacles to the world
			A = A * (1 - self.obstacles) + self.obstacles
			print(f"World max after applying obstacles: {jnp.max(A)}")
			print(f"World min after applying obstacles: {jnp.min(A)}")

		return A

	def create_cells_from_world(self, world):
		mid = self._config.world_size // 2
		n_cells_size = self._config.n_cells_size  # Assuming this is 32

		# Calculate the size of the scaled cells in the world
		scaled_size = n_cells_size * self._config.world_scale

		# Extract the scaled cells from the center of the world
		start = mid - scaled_size // 2
		end = start + scaled_size
		scaled_cells = world[start:end, start:end, :]

		# Downscale the cells
		cells = scaled_cells[::self._config.world_scale, ::self._config.world_scale, :]

		return cells
	
	def load_pattern(self, pattern):
		# unpack pattern data
		cells = jnp.transpose(jnp.asarray(pattern['cells']), axes=[1, 2, 0])  # (y, x, c,)
		kernels = pattern['kernels']
		R = pattern['R'] * self._config.world_scale
		T = pattern['T']

		# get params from pattern (to be put in genotype)
		m = jnp.array([k['m'] for k in kernels])  # (k,)
		s = jnp.array([k['s'] for k in kernels])  # (k,)
		h = jnp.array([k['h'] for k in kernels])  # (k,)
		init_params = jnp.vstack([m, s, h])  # (p, k,)

		# get reshaping arrays (unfold and fold)
		reshape_c_k = jnp.zeros(shape=(self.n_channel, self.n_kernel))  # (c, k,)
		reshape_k_c = jnp.zeros(shape=(self.n_kernel, self.n_channel))  # (k, c,)
  
		# define self.reshape_c_k and self.reshape_k_c, self.R, self.T for save_pattern
		self.R = R
		self.T = T
		self.reshape_c_k = reshape_c_k
		self.reshape_k_c = reshape_k_c
		# get list of b values for each kernel
		self.b = [k['b'] for k in kernels]
		self.r = [k['r'] for k in kernels]
  
		for i, k in enumerate(kernels):
			reshape_c_k = reshape_c_k.at[k['c0'], i].set(1.0)
			reshape_k_c = reshape_k_c.at[i, k['c1']].set(1.0)

		# calculate kernels and related stuff
		mid = self._config.world_size // 2
		X = jnp.mgrid[-mid:mid, -mid:mid] / R  # (d, y, x,), coordinates
		D = jnp.linalg.norm(X, axis=0)  # (y, x,), distance from origin
		Ds = [D * len(k['b']) / k['r'] for k in kernels]  # (y, x,)*k
		Ks = [(D<len(k['b'])) * jnp.asarray(k['b'])[jnp.minimum(D.astype(int),len(k['b'])-1)] * bell(D%1, 0.5, 0.15) for D,k in zip(Ds, kernels)]  # (x, y,)*k
		K = jnp.dstack(Ks)  # (y, x, k,), kernels
		nK = K / jnp.sum(K, axis=(0, 1), keepdims=True)  # (y, x, k,), normalized kernels
		fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0, 1)), axes=(0, 1))  # (y, x, k,), FFT of kernels

		# pad pattern cells into initial cells (to be put in genotype)
		try:
			cy, cx = cells.shape[0], cells.shape[1]
			py, px = self._config.n_cells_size - cy, self._config.n_cells_size - cx
			init_cells = jnp.pad(cells, pad_width=((py//2, py-py//2), (px//2, px-px//2), (0,0)), mode='constant')  # (e, e, c,)
		except:
			print(cy, cx)
   
		# create world from initial cells
		A = self.create_world_from_cells(init_cells)

		# pack initial data
		init_carry = Carry(
			world = A,
			param = Param(m, s, h),
			asset = Asset(fK, X, reshape_c_k, reshape_k_c, R, T),
			temp  = Temp(jnp.zeros(2), jnp.zeros(2, dtype=int), jnp.zeros(2, dtype=int), 0.0),
		)
		init_genotype = jnp.concatenate([init_params.flatten(), init_cells.flatten()])
		other_asset = Others(D, K, cells, init_cells)
		return init_carry, init_genotype, other_asset

	def save_pattern(self, carry, accum, i):
		# Extract necessary data from carry
		world = carry.world
		param = carry.param
		asset = carry.asset

		# Unpack parameters
		m, s, h = param.m, param.s, param.h

		# Extract necessary data from asset
		fK, X, reshape_c_k, reshape_k_c, R, T = asset.fK, asset.X, asset.reshape_c_k, asset.reshape_k_c, asset.R, asset.T

		# Generate a random pattern number
		pattern_id = i

		# Create kernels list
		kernels = []
		for i in range(self.n_kernel):
			kernel = {
				"b": self.b[i],
				"m": m[i].item(),
				"s": s[i].item(),
				"h": h[i].item(),
				"r": self.r[i],  # Note: You might want to store this value if it varies
				"c0": int(jnp.argmax(reshape_c_k[:, i]).item()),
				"c1": int(jnp.argmax(reshape_k_c[i, :]).item())
			}
			kernels.append(kernel)

		# Extract cells from world
		cells = self.create_cells_from_world(world)

		# Create pattern dictionary
		pattern = {
			"name": pattern_id,
			"R": float(R / self._config.world_scale),  # Convert back to original scale
			"T": float(T),
			"kernels": kernels,
			"cells": jax.device_get(cells).transpose(2, 0, 1).tolist()
		}

		# Optionally, clip cell values between 0 and 1
		# Uncomment the following line if needed:
		# pattern["cells"] = jnp.clip(jnp.array(pattern["cells"]), 0, 1).tolist()
		return pattern

	def express_genotype(self, carry, genotype):
		params = genotype[:self.n_params].reshape((self._config.n_params_size, self.n_kernel))
		cells = genotype[self.n_params:].reshape((self._config.n_cells_size, self._config.n_cells_size, self.n_channel))

		m, s, h = params
		A = self.create_world_from_cells(cells)

		carry = carry._replace(world=A)
		carry = carry._replace(param=Param(m, s, h))
		return carry

	def calculate_stability(self,observation, n_keep):
		mass_history = observation.stats.mass[-n_keep:]
		stability = -jnp.var(mass_history)
		return stability

	def calculate_energy_efficiency(self,observation, size, n_keep):
		mass_history = observation.stats.mass[-n_keep:]
		energy_efficiency = jnp.mean(mass_history) / size
		return energy_efficiency

	@partial(jax.jit, static_argnames=("self", "phenotype_size", "center_phenotype", "record_phenotype",))
	def step(self, carry: Carry, unused: Any, phenotype_size, center_phenotype, record_phenotype, adaptive_phenotype, switch):
		# Calculate mid at the beginning of the method
		mid = self._config.world_size // 2
		half_size = phenotype_size // 2
  
		# unpack data from last step
		A = carry.world
		m, s, h = carry.param
		fK, X, reshape_c_k, reshape_k_c, R, T = carry.asset
		last_center, last_shift, total_shift, last_angle = carry.temp
		m = m[None, None, ...]  # (1, 1, k,)
		s = s[None, None, ...]  # (1, 1, k,)
		h = h[None, None, ...]  # (1, 1, k,)

		# center world for accurate calculation of center and velocity
		A = jnp.roll(A, -last_shift, axis=(-3, -2))  # (y, x, c,)

		# Lenia step
		fA = jnp.fft.fft2(A, axes=(-3, -2))  # (y, x, c,)
		fA_k = jnp.dot(fA, reshape_c_k)  # (y, x, k,)
		U_k = jnp.real(jnp.fft.ifft2(fK * fA_k, axes=(-3, -2)))  # (y, x, k,)
		G_k = growth(U_k, m, s) * h  # (y, x, k,)
		G = jnp.dot(G_k, reshape_k_c)  # (y, x, c,)

		if self._config.use_obstacles:
			G = G * (1-self.obstacles) # Obstacles block growth channel wise
  
		next_A = jnp.clip(A + 1/T * G, 0, 1)  # (y, x, c,)

		# calculate center
		m00 = A.sum()
		AX = next_A.sum(axis=-1)[None, ...] * X  # (d, y, x,)
		center = AX.sum(axis=(-2, -1)) / m00  # (d,)
		shift = (center * R).astype(int)
		total_shift += shift
  
		if self._config.use_obstacles:
			next_A = next_A * (1-self.obstacles) + self.obstacles # Ensure that obstacles persist

		# get phenotype
		if record_phenotype:
			if center_phenotype:
				phenotype = next_A
			else:
				phenotype = jnp.roll(next_A, total_shift - shift, axis=(0, 1))
			mid = self._config.world_size // 2
			half_size = phenotype_size // 2
			phenotype = phenotype[mid-half_size:mid+half_size, mid-half_size:mid+half_size]
		else:
			phenotype = None

		# calculate mass and velocity
		mass = m00 / R / R
		actual_center = center + total_shift / R
		center_diff = center - last_center + last_shift / R
		linear_velocity = jnp.linalg.norm(center_diff) * T

		# calculate angular velocity
		angle = jnp.arctan2(center_diff[1], center_diff[0]) / jnp.pi  # angle = [-1.0, 1.0]
		angle_diff = (angle - last_angle + 3) % 2 - 1
		angle_diff = jax.lax.cond(linear_velocity > 0.01, lambda: angle_diff, lambda: 0.0)
		angular_velocity = angle_diff * T
  
		# size
		size = jnp.sum(next_A > 0.1)

		# check if world is empty or full
		is_empty = (next_A < 0.1).all(axis=(-3, -2)).any()
		borders = next_A[..., 0, :, :].sum() + next_A[..., -1, :, :].sum() + next_A[..., :, 0, :].sum() + next_A[..., :, -1, :].sum()
		is_full = borders > 0.1
		is_spread = A[mid-half_size:mid+half_size, mid-half_size:mid+half_size].sum()/m00 < 0.9
  
		# Calculate new stats
		linmomentum = linear_velocity * mass
		kinetic = 0.5 * mass * linear_velocity ** 2

		# Pack data for next step
		carry = carry._replace(world=next_A)
		carry = carry._replace(temp=Temp(center, shift, total_shift, angle))
		stats = Stats(mass, actual_center[1], -actual_center[0], linear_velocity, angle, angular_velocity, is_empty, is_full, is_spread,
					 size, linmomentum, kinetic)
		accum = Accum(phenotype, stats)
		return carry, accum

class FlowLenia:

	def __init__(self, config: ConfigLenia):
		self._config = config
		self.pattern = patterns[self._config.pattern_id]

		# Genotype
		self.n_kernel = len(self.pattern["kernels"])  # k, number of kernels
		self.n_channel = len(self.pattern["cells"])  # c, number of channels
		self.n_params = self._config.n_params_size * self.n_kernel  # p*k, number of parameters inside genotype
		self.n_cells = self._config.n_cells_size * self._config.n_cells_size * self.n_channel  # e*e*c, number of embryo cells inside genotype
		self.n_gene = self.n_params + self.n_cells  # size of genotype
  
		self.RT = ReintegrationTracking(SX=self._config.world_size, SY=self._config.world_size, 
										dt=self._config.dt, dd=self._config.dd, sigma=self._config.sigma, 
										border=self._config.border, has_hidden=self._config.has_hidden, 
										mix=self._config.mix)
		self.sobel_x = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		self.sobel_y = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

	def create_world_from_cells(self, cells):
		mid = self._config.world_size // 2

		# scale cells
		scaled_cells = cells.repeat(self._config.world_scale, axis=-3).repeat(self._config.world_scale, axis=-2)
		cy, cx = scaled_cells.shape[0], scaled_cells.shape[1]

		# create empty world and place cells
		A = jnp.zeros((self._config.world_size, self._config.world_size, self.n_channel))  # (y, x, c,)
		A = A.at[mid-cx//2:mid+cx-cx//2, mid-cy//2:mid+cy-cy//2, :].set(scaled_cells)
		return A

	def load_pattern(self, pattern):
		# unpack pattern data
		cells = jnp.transpose(jnp.asarray(pattern['cells']), axes=[1, 2, 0])  # (y, x, c,)
		kernels = pattern['kernels']
		R = pattern['R'] * self._config.world_scale
		T = pattern['T']

		# get params from pattern (to be put in genotype)
		m = jnp.array([k['m'] for k in kernels])  # (k,)
		s = jnp.array([k['s'] for k in kernels])  # (k,)
		h = jnp.array([k['h'] for k in kernels])  # (k,)
		init_params = jnp.vstack([m, s, h])  # (p, k,)

		# get reshaping arrays (unfold and fold)
		reshape_c_k = jnp.zeros(shape=(self.n_channel, self.n_kernel))  # (c, k,)
		reshape_k_c = jnp.zeros(shape=(self.n_kernel, self.n_channel))  # (k, c,)
		for i, k in enumerate(kernels):
			reshape_c_k = reshape_c_k.at[k['c0'], i].set(1.0)
			reshape_k_c = reshape_k_c.at[i, k['c1']].set(1.0)

		# calculate kernels and related stuff
		mid = self._config.world_size // 2
		X = jnp.mgrid[-mid:mid, -mid:mid] / R  # (d, y, x,), coordinates
		D = jnp.linalg.norm(X, axis=0)  # (y, x,), distance from origin
		Ds = [D * len(k['b']) / k['r'] for k in kernels]  # (y, x,)*k
		Ks = [(D<len(k['b'])) * jnp.asarray(k['b'])[jnp.minimum(D.astype(int),len(k['b'])-1)] * bell(D%1, 0.5, 0.15) for D,k in zip(Ds, kernels)]  # (x, y,)*k
		K = jnp.dstack(Ks)  # (y, x, k,), kernels
		nK = K / jnp.sum(K, axis=(0, 1), keepdims=True)  # (y, x, k,), normalized kernels
		fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0, 1)), axes=(0, 1))  # (y, x, k,), FFT of kernels

		# pad pattern cells into initial cells (to be put in genotype)
		cy, cx = cells.shape[0], cells.shape[1]
		py, px = self._config.n_cells_size - cy, self._config.n_cells_size - cx
		init_cells = jnp.pad(cells, pad_width=((py//2, py-py//2), (px//2, px-px//2), (0,0)), mode='constant')  # (e, e, c,)

		# create world from initial cells
		A = self.create_world_from_cells(init_cells)

		# pack initial data
		init_carry = Carry(
			world = A,
			param = Param(m, s, h),
			asset = Asset(fK, X, reshape_c_k, reshape_k_c, R, T),
			temp  = Temp(jnp.zeros(2), jnp.zeros(2, dtype=int), jnp.zeros(2, dtype=int), 0.0),
		)
		init_genotype = jnp.concatenate([init_params.flatten(), init_cells.flatten()])
		other_asset = Others(D, K, cells, init_cells)
		return init_carry, init_genotype, other_asset

	def express_genotype(self, carry, genotype):
		params = genotype[:self.n_params].reshape((self._config.n_params_size, self.n_kernel))
		cells = genotype[self.n_params:].reshape((self._config.n_cells_size, self._config.n_cells_size, self.n_channel))

		m, s, h = params
		A = self.create_world_from_cells(cells)

		carry = carry._replace(world=A)
		carry = carry._replace(param=Param(m, s, h))
		return carry

	@partial(jax.jit, static_argnums=(0,))
	def sobel(self, image):
		grad_x = jax.scipy.signal.convolve(image, self.sobel_x[..., None], mode='same')
		grad_y = jax.scipy.signal.convolve(image, self.sobel_y[..., None], mode='same')
		return jnp.stack((grad_x, grad_y), axis=-2)

	@partial(jax.jit, static_argnames=("self", "phenotype_size", "center_phenotype", "record_phenotype"))
	def step(self, carry: Carry, unused: Any, phenotype_size, center_phenotype, record_phenotype, adaptive_phenotype=False):
		# Unpack data from last step
		A = carry.world
		initial_mass = jnp.sum(A)
		m, s, h = carry.param
		fK, X, reshape_c_k, reshape_k_c, R, T = carry.asset
		last_center, last_shift, total_shift, last_angle = carry.temp

		m = m[None, None, ...]  # (1, 1, k,)
		s = s[None, None, ...]  # (1, 1, k,)
		h = h[None, None, ...]  # (1, 1, k,)

		# Center world for accurate calculation of center and velocity
		A = jnp.roll(A, -last_shift, axis=(-3, -2))  # (y, x, c,)

		# Lenia step
		fA = jnp.fft.fft2(A, axes=(-3, -2))  # (y, x, c,)
		fA_k = jnp.dot(fA, reshape_c_k)  # (y, x, k,)
		U_k = jnp.real(jnp.fft.ifft2(fK * fA_k, axes=(-3, -2)))  # (y, x, k,)

		# Calculate flow
		nabla_U = self.sobel(U_k)  # (y, x, 2, k)
		nabla_A = self.sobel(A.sum(axis=-1, keepdims=True))  # (y, x, 2, 1)
		alpha = jnp.clip((A[:, :, None, :] / self.n_channel) ** 2, 0.0, 1.0)  # (y, x, 1, c)

		nabla_U_expanded = nabla_U[..., None]  # Shape: (y, x, 2, k, 1)
		alpha_expanded = alpha[:, :, None, :, :]  # Shape: (y, x, 1, 1, c)
		nabla_A_expanded = nabla_A[:, :, :, None, :]  # Shape: (y, x, 2, 1, 1)

		F = nabla_U_expanded * (1 - alpha_expanded) - nabla_A_expanded * alpha_expanded
		F = F.sum(axis=3)  # Sum over the kernel dimension, resulting shape: (y, x, 2, c)

		# Apply flow
		nA = self.RT(A, F)  # Apply the RT function

		# Normalize to conserve mass
		nA = nA * (initial_mass / jnp.sum(nA))

		# Apply Lenia update
		G_k = growth(U_k, m, s) * h  # (y, x, k,)
		G = jnp.dot(G_k, reshape_k_c)  # (y, x, c,)
		next_A = jnp.clip(nA + 1/T * G, 0, 1)  # (y, x, c,)

		# Calculate center
		m00 = next_A.sum()
		AX = next_A.sum(axis=-1)[None, ...] * X  # (d, y, x,)
		center = AX.sum(axis=(-2, -1)) / m00  # (d,)
		shift = (center * R).astype(int)
		total_shift += shift

		# Get phenotype
		if record_phenotype:
			if center_phenotype:
				phenotype = next_A
			else:
				phenotype = jnp.roll(next_A, total_shift - shift, axis=(0, 1))
			mid = self._config.world_size // 2
			half_size = phenotype_size // 2
			phenotype = phenotype[mid-half_size:mid+half_size, mid-half_size:mid+half_size]
		else:
			phenotype = None

		# Calculate mass and velocity
		mass = m00 / R / R
		actual_center = center + total_shift / R
		center_diff = center - last_center + last_shift / R
		linear_velocity = jnp.linalg.norm(center_diff) * T

		# Calculate angular velocity
		angle = jnp.arctan2(center_diff[1], center_diff[0]) / jnp.pi  # angle = [-1.0, 1.0]
		angle_diff = (angle - last_angle + 3) % 2 - 1
		angle_diff = jax.lax.cond(linear_velocity > 0.01, lambda: angle_diff, lambda: 0.0)
		angular_velocity = angle_diff * T

		# size
		size = jnp.sum(next_A > 0.1)

		# check if world is empty or full
		is_empty = (next_A < 0.1).all(axis=(-3, -2)).any()
		borders = next_A[..., 0, :, :].sum() + next_A[..., -1, :, :].sum() + next_A[..., :, 0, :].sum() + next_A[..., :, -1, :].sum()
		is_full = borders > 0.1
		mid = self._config.world_size // 2
		half_size = phenotype_size // 2
		is_spread = next_A[mid-half_size:mid+half_size, mid-half_size:mid+half_size].sum()/m00 < 0.9

		# Calculate new stats
		linmomentum = linear_velocity * mass
		kinetic = 0.5 * mass * linear_velocity ** 2

		# Pack data for next step
		carry = carry._replace(world=next_A)
		carry = carry._replace(temp=Temp(center, shift, total_shift, angle))
		stats = Stats(mass, actual_center[1], -actual_center[0], linear_velocity, angle, angular_velocity, is_empty, is_full, is_spread,
					size, linmomentum, kinetic)
		accum = Accum(phenotype, stats)

		return carry, accum