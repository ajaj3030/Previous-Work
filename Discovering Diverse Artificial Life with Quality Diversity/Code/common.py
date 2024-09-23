import pickle
import jax.numpy as jnp
import jax.lax
import pandas as pd
from omegaconf import OmegaConf

def get_metric(observation, metric, n_keep):
    sign, *metric, operator = metric.split("_")
    metric = "_".join(metric)

    if operator == "avg":
        operator = jnp.mean
    elif operator == "var":
        operator = jnp.var
    elif operator == "max":
        operator = jnp.max
    elif operator == "min":
        operator = jnp.min
    else:
        raise NotImplementedError

    if sign == "pos":
        sign = 1.
    elif sign == "neg":
        sign = -1.
    else:
        raise NotImplementedError

    if metric == "mass":
        return sign * operator(observation.stats.mass[-n_keep:], keepdims=True)
    elif metric == "linearvelocity":
        return sign * jnp.sqrt(jnp.square(observation.stats.center_x[-1:] - observation.stats.center_x[-n_keep]) + jnp.square(observation.stats.center_y[-1:] - observation.stats.center_y[-n_keep]))
    elif metric == "momentum":
        linear_velocity = jnp.sqrt(jnp.square(observation.stats.center_x[-1:] - observation.stats.center_x[-n_keep]) + 
                                   jnp.square(observation.stats.center_y[-1:] - observation.stats.center_y[-n_keep]))
        mass = observation.stats.mass[-n_keep:]
        return sign * operator(linear_velocity * mass, keepdims=True)
    elif metric == "verticalmomentum":
        linear_velocity = observation.stats.center_y[-1:] - observation.stats.center_y[-n_keep]
        mass = observation.stats.mass[-n_keep:]
        return sign * operator(linear_velocity * mass, keepdims=True)
    elif metric == "kineticenergy":
        linear_velocity = jnp.sqrt(jnp.square(observation.stats.center_x[-1:] - observation.stats.center_x[-n_keep]) + 
                                   jnp.square(observation.stats.center_y[-1:] - observation.stats.center_y[-n_keep]))
        mass = observation.stats.mass[-n_keep:]
        return sign * operator(0.5 * mass * linear_velocity ** 2, keepdims=True)
    elif metric == "stability":
        mass_values = observation.stats.mass[-n_keep:]
        mass_mean = jnp.mean(mass_values)
        mass_variance = jnp.var(mass_values)
        mass_threshold = 1.0
        return jax.lax.cond(
            mass_mean >= mass_threshold,
            lambda _: -mass_variance,
            lambda _: -mass_variance - (jnp.exp(mass_threshold - mass_mean) - 1),
            operand=None
        )
    elif metric == "efficiency":
        mass = jnp.mean(observation.stats.mass[-n_keep:])
        return sign * operator(mass / observation.stats.size[-1:], keepdims=True)
    elif metric == "angularvelocity":
        return sign * operator(observation.stats.angular_velocity[-n_keep:], keepdims=True)
    elif metric == "angle":
        return sign * operator(observation.stats.angle[-n_keep:], keepdims=True)
    elif metric == "center_x":
        return sign * operator(observation.stats.center_x[-n_keep:], keepdims=True)
    elif metric == "center_y":
        return sign * operator(observation.stats.center_y[-n_keep:], keepdims=True)
    elif metric == "color":
        return sign * operator(observation.phenotype[-n_keep:, ...], axis=(0, 1, 2))
    elif metric == "colorfitness":
        phenotypes = observation.phenotype[-n_keep:, ...]
        color_vectors = jnp.mean(phenotypes, axis=(1, 2))  # Average color for each phenotype
        fitnesses = jax.vmap(color_fitness)(color_vectors)
        return sign * operator(fitnesses, keepdims=True)
    else:
        raise NotImplementedError

def color_fitness(color_vector):
    weights = jnp.array([0.299, 0.587, 0.114])  # Human perception weights for R, G, B
    return jnp.dot(color_vector, weights) * jnp.std(color_vector)

def get_config(run_dir):
    config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return config

def color_fitness2(color_vector):
    mean = jnp.mean(color_vector)
    variance = jnp.mean((color_vector - mean) ** 2)
    return variance

def get_metrics(run_dir):
    with open(run_dir / "metrics.pickle", "rb") as metrics_file:
        metrics = pickle.load(metrics_file)
    try:
        del metrics["loss"]
        del metrics["learning_rate"]
    except:
        pass
    return pd.DataFrame.from_dict(metrics)

def get_df(results_dir):
    metrics_list = []
    for fitness_dir in results_dir.iterdir():
        if fitness_dir.is_file():
            continue
        for run_dir in fitness_dir.iterdir():
            # Get config and metrics
            config = get_config(run_dir)
            metrics = get_metrics(run_dir)

            # Fitness
            try:
                metrics["fitness"] = config.qd.fitness
            except:
                metrics["fitness"] = "none"

            # Run
            metrics["run"] = run_dir.name

            # Number of Evaluations
            metrics["n_evaluations"] = metrics["generation"] * config.qd.batch_size

            # Coverage
            metrics["coverage"]

            metrics_list.append(metrics)
    return pd.concat(metrics_list, ignore_index=True)

def repertoire_variance(repertoire):
    is_occupied = (repertoire.fitnesses != -jnp.inf)
    var = jnp.var(repertoire.observations.phenotype[:, -1], axis=0, where=is_occupied[:, None, None, None])
    return jnp.mean(var)