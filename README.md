# Iterated Population Based Training

**TL;DR** A novel Population Based Training variant for hyperparameter optimization that efficiently adjusts its step size via task-agnostic restarts

## Setup

Create and activate the conda environment:

```
conda env create --file env_ipbt.yml
conda activate ipbt
```

Patch Ray Tune to ensure checkpointing is done before the learning rate scheduler restarts (applies only to random search and ASHA):

```
python patch_ray_tune.py --expect-ray 2.10.0
```

Install [our fork of SMAC3](https://github.com/AwesomeLemon/SMAC3):

```
cd ../SMAC3
make install-dev
cd -
```

Export ``PYTHONPATH``: ``
export PYTHONPATH=/path/to/this/code/dir
``

For Brax RL tasks, to manage VRAM when running several Jax processes on the same GPU, execute ``export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05``.

Start Ray:

```
ray start --head
```

## Running experiments

All commands used for running the experiments in the paper are listed in ``commands_ipbt.sh``.

The ``run.py`` script should be used for IPBT and other PBT variants, ``run_raytune.py`` for random search and ASHA, and ``run_raytune.py`` for SMAC. 

The Hydra configs are in the ``config`` directory.

# Acknowledgments

The code is based on [PBT-Zoo](https://github.com/AwesomeLemon/PBT-Zoo).