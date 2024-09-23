# Leniabreeder

Repository for Discovering Artificial Life using Quality Diversity, an extension of "Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity" (ALIFE 2024).

## Installation

```bash
git https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2023-2024/aidan_holmes/lenia.git && cd Leniabreeder
```

### Using virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

### Using virtual environment

At the root of the repository, execute:
```bash
source venv/bin/activate
```

### WANDB
Leniabreeder expects a wandb API key. It takes this from an environment variable. wandb_api_key = os.environ.get('WANDB_API_KEY'.

### Commands

To run an experiment with the default configuration, execute the following command:
```bash
python main.py seed=$RANDOM qd=<algo>
```
with `<algo>` replaced with 
`me` For MAP-Elites
`aurora` For AURORA
`aurora_cl` For contrastive Learning AURORA (qd.contrastive_loss_fn=0 for NT-Xent, qd.contrastive_loss_fn=1 for Triplet Margin)
`aurora_vq` For vq-vae aurora
`aurora_exp` For the automatic leniabreeder process

All hyperparameters are available in the `configs/` directory and can be overridden via the command line. For example:
```
python main.py qd=aurora_exp qd.kernel_mutate=True n_step=300 qd.mutation_type=color_swap qd.fitness=pos_mass_var qd.iso_sigma=0.05 qd.variation_percentage=0.5 max_iterations=20
```

## Analyze Experiments

Running an experiment (see previous section) creates a directory in `output`. To analyze the results, you can use run the jupyter notebooks:

visualiser_aurora.ipynb
visualiser_aurora_cl.ipynb
visualiser_aurora_vq.ipynb

## Automatic Leniabreeder
Automatic Leniabreeder automatically generate html outputs of every discovered creature. For more in depth analysis, we recommend using visualiser_aurora.ipynb. Changes will be needed to make it run, but they should be simple. That being said, the extended notebook functionality are designed for evaluating setups, while the automatic process is designed to maximise the gain from known effective setups.

## Citation for the original paper we build from

```
@article{faldor2024leniabreeder,
	author    = {Faldor, Maxence and Cully, Antoine},
	title     = {Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity},
	journal   = {ALIFE},
	year      = {2024},
}
```
