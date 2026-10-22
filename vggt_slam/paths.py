"""Where the dataset and the pipeline's output live. The one place that knows the cluster paths.

Each location can be overridden with an environment variable of the same name (the sbatch files
pass them through); scripts also take them as command-line flags.
"""
import os

DATASET_ROOT = os.environ.get("DATASET_ROOT", "/capstor/store/cscs/swissai/ab036/datasets/natix")
TRAJ_ROOT = os.environ.get("TRAJ_ROOT", "/capstor/store/cscs/swissai/ab036/processed_datasets/natix/trajectories")
SAMPLE_MANIFEST = os.environ.get(
    "SAMPLE_MANIFEST", "/capstor/store/cscs/swissai/ab036/processed_datasets/natix/manifest/samples_24fps.parquet")

# durable records of the post-processing stages (qc csv, depth-step manifest, integration log, logs, backups)
PIPELINE_DIR = os.path.join(TRAJ_ROOT, "_pipeline")
