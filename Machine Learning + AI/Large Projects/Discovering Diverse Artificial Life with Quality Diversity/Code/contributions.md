# Contributions

This project extended an existing method. This markdown file aims to make clear new contributions. It is very likely I have forgotten absolutely everything I added to this project, but this file should make the important contributions

### Lenia Directory
##### animals.json 
A set of creatures previously discovered by Bert Chan. Need a way to properly 
##### lenia.py
generate_viable_random_pattern() -> Designed to initialise from random. Theoretically operation, takes a loooong time to find something though
generate_random_variant -> Loads in the predefined pattern, then add a lot of noise to the cells and parameters.
custom_pattern -> used for the automatic Leniabreeder process 

save_world_image() -> Takes a screenshot of the current world. Function should work, used to evaluate the random world generation

initialize_obstacles() -> We added obstacles to the Lenia world. However, these obstacles do emit and grow, basically just always leading to exploding creatures. This could be super interesting when combined with Flow Lenia, as the mass conservation would allow creatures to treat obstacles like food.

create_cells_from_world() -> Reverses the create_world_from_cells function. This is required to save genotype data to the JSON file.

save_pattern() -> Saves individual patterns to the same json file. This allows them to be reloaded into the automatic Leniabreeder.

calculate_stability() -> calculates the stability of a creature (negative mass variance)

calculate_energy_efficiency() -> Calculates the creature density overtime. Poorly named function

FlowLenia (class) -> Attempt at implementing Flow Lenia. This will run, although I do not know if it is succesful. I think the mass being added by the isolineDD operator messes with the mass conservation causing erratic behaviours. Would want to investigate using it with just mutation operator. Promising start though

##### patterns.py
We add 3 new patterns (bizarre cells, mothership and self-replicator)

A variety of functions designed for random world generation.

##### Reintegration_tracking.py
Required component of Flow Lenia. Not as relevant for the work in this project, but could be a useful starting point.

### QDAX
standard_emitters.py -> Added a couple mutators here (not the Lenia specific ones). They're not as good as isoLineDD though, so redundant.

##### cl_vae.py
Implement Triplet Margin and NT-Xent Loss functions. Both of these have parameters that can be adjusted from the config (be it the weighting of the CL loss, or the temperature/margin value)

We also have the option of the standard VAE architecture or the GCNN VAE architecture. We include euclidean and cosine NT-Xent implementations. The euclidean approach isn't that effective, and sometimes does not work. The literature favours the cosine approach, so we recommend that anyway.

##### vae.py
We added the new GCNN encoder and decoder

##### vqvae.py
Again we have the GCNN encoder and decoder

VectorQuantizer -> Class to embed latent vectors to discrete latent space. This calculates the commitment_loss and perplexity values as well.

GumbelVectorQUantizer -> We had a crack at implementing this, but I do not belive it works so I would not recommend using it
MSE loss -> We use MSE loss here instead of BCE, I found it just works better here for whatever reason.

##### common.py
We implement a few new fitness functions:
Momentum -> mass*velocity

LinearMomentum -> Momentum, but only rewarded in the vertical direction

kineticenergy -> 0.5mv^2

stability -> Measurement of how much the mass varies over the full update loop

efficiency -> Think of this as the density

colorfitness -> Two different fitness functions to optimase for colour values. One of them is based on the human perception weights for RBG, the other one is the variance of the colour channels

### Main.py files
##### Files to run
main_aurora_cl.py -> Main file used to run the contrastive learning variant of AURORA

main_aurora_vq -> Main file used to run the vq-vae implementation of AURORA

main_auroraexp.py -> The automated Leniabreeder discovery framework.

##### New Implementations that exist in all variants.
GCNN Layers (and transposed layers) for Encoder-Decoder structures. Can be used in standard AURORA, VQ-VAE AURORA and Contrastive Learning AURORA

Kernel_swap_mutation() -> More accurately is kernel parameter interpolation. Used to mutate the genotype parameters

Kernel_shift_mutation() -> Randomly shifts the values of the kernel parameters.

RGB_channel_swap_mutation() -> Randomly swaps the pixel-wise values between the colour channels.

adaptive_mutation() -> Randomly select one of the previously define mutators at each generation

calculate_rotational_diversity() -> Calculates the mean diversity of the repertoire. Compares creatures at different angles and timesteps to get the closest match. For more details see the paper

affline_transform() -> Used in the data_augmentation

data_augmentation() -> We extend the original data_ augmentation to include sheers and scaling

rotate_image() -> Used to rotate the image at a series of small amgles. We use billinear interpolation for smooth rotations at small angles

rotational_similarity_score -> Calculates a creatures' self-pairwise difference. This is called rotational_similarity_score in the code, but rotational self-diversity in the paper.

### Visualiser files 
Note: To use it, you must enter the results directory found in output. It should be clear what this is based on standard_run dir. We enhance the original notebook in the original Leniabreeder.

Interactive t-SNE plot: We create interactive t-SNE plotting to both visualise the embedding and trajectories of our soliton. Click on the embedding points in the t-sne embeddings to see the corresponding soliton.

We also have uninteractive t-sne plots, but they are not as useful.

We alter the html video generation for Fitness, Random and (introduce) Novelty preferences. This realistically just used the existing code

We add something to save just the final images of all the solitons in the repertoire, we select them randomly.

We add a PCA fitness heatmap, but this is a bit random as to when it works. It isn't that useful for now though, but maybe it could be in the future

Finally we add something to interpolate between two genotypes. This is not that interesting though, even when it works.

##### Deeper dive into main_auroraexp.py
Functionality to find and load a pattern from a json file from the previous iteration randomly.

Functionality such that Leniabreeder saves all the creature genotypes to a reachable location in a json format (shockingly difficult to do). It also saves all of these creatures to a html image format so that we do not need to visualise them manually.

We have added the dynamic configuration, which can be enabled and controlled from the original config. The wrapper is specifically around main_aurora.py.

##### Experimental
main_aurora_group.py designed to be
