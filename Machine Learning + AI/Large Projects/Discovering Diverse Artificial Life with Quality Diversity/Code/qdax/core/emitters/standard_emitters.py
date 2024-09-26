import jax
from jax import lax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Optional, Tuple, Any
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Genotype, Metrics, RNGKey
from qdax.core.containers.repertoire import Repertoire

class MixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the repertoire,
        copied and cross-over to obtain new offsprings. One batch of
        (1.0 - variation_percentage) * batch_size genotypes are sampled in the
        repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            x1, random_key = repertoire.sample(random_key, n_variation)
            x2, random_key = repertoire.sample(random_key, n_variation)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, random_key




    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size

class GaussianMutationEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.1,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size
        self._mutation_rate = mutation_rate
        self._mutation_std = mutation_std
        
    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        genotypes, random_key = repertoire.sample(random_key, self._batch_size)
        
        mutation_key, mask_key, random_key = jax.random.split(random_key, 3)
        mutations = jax.random.normal(
            mutation_key, 
            shape=jax.tree_util.tree_map(lambda x: x.shape, genotypes)
        ) * self._mutation_std
        
        mutation_mask = jax.random.bernoulli(
            mask_key, 
            p=self._mutation_rate, 
            shape=jax.tree_util.tree_map(lambda x: x.shape, genotypes)
        )
        
        mutated_genotypes = jax.tree_util.tree_map(
            lambda g, m, mask: g + m * mask,
            genotypes, mutations, mutation_mask
        )
        
        return mutated_genotypes, random_key
    
class AdaptiveMixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        initial_batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._initial_batch_size = initial_batch_size
        self._min_batch_size = min_batch_size
        self._max_batch_size = max_batch_size
        self._current_batch_size = jnp.array(initial_batch_size, dtype=jnp.float32)

    @property
    def batch_size(self) -> int:
        return self._current_batch_size

    @partial(jax.jit,static_argnames=("self",),)
    def emit(
        self,
        repertoire: UnstructuredRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        # Adaptive batch size calculation
        coverage = jnp.sum(repertoire.fitnesses != -jnp.inf) / repertoire.fitnesses.size
        target_batch_size = jnp.clip(
            jnp.floor(self._max_batch_size * (1 - coverage) + self._min_batch_size * coverage),
            self._min_batch_size,
            self._max_batch_size
        )
        self._current_batch_size = jnp.maximum(self._current_batch_size * 0.9, target_batch_size)

        # Use lax.round and lax.convert_element_type instead of int
        n_variation = lax.convert_element_type(
            lax.round(self._current_batch_size * self._variation_percentage),
            jnp.int32
        )
        n_mutation = lax.convert_element_type(
            lax.round(self._current_batch_size - n_variation),
            jnp.int32
        )

        if n_variation > 0:
            x1, random_key = repertoire.sample(random_key, n_variation)
            x2, random_key = repertoire.sample(random_key, n_variation)
            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, random_key
    
class TMVOMixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
        momentum_strength: float,
        noise_strength: float,
        history_length: int,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size
        self._momentum_strength = momentum_strength
        self._noise_strength = noise_strength
        self._history_length = history_length
        self._historical_changes = []

    def _temporal_momentum_variation(
        self,
        x: Genotype,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        def compute_momentum(changes):
            weights = jnp.arange(1, len(changes) + 1)
            weighted_changes = jax.tree_map(lambda *args: jnp.stack(args) * weights[:, None], *changes)
            return jax.tree_map(lambda x: jnp.sum(x, axis=0) / jnp.sum(weights), weighted_changes)
        
        def apply_variation(x, momentum, key):
            noise = jax.random.normal(key, shape=x.shape) * self._noise_strength
            return x + self._momentum_strength * momentum + noise
        
        momentum = compute_momentum(self._historical_changes[-self._history_length:])
        
        random_key, subkey = jax.random.split(random_key)
        new_x = jax.vmap(apply_variation, in_axes=(0, 0, 0))(
            x, momentum, jax.random.split(subkey, x.shape[0])
        )
        
        self._historical_changes.append(jax.tree_map(lambda a, b: a - b, new_x, x))
        if len(self._historical_changes) > self._history_length:
            self._historical_changes = self._historical_changes[-self._history_length:]
        
        return new_x, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            x1, random_key = repertoire.sample(random_key, n_variation)
            x2, random_key = repertoire.sample(random_key, n_variation)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)
            x_variation, random_key = self._temporal_momentum_variation(x_variation, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)
            x_mutation, random_key = self._temporal_momentum_variation(x_mutation, random_key)

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, random_key
