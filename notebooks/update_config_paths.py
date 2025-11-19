import json
from pathlib import Path

nb_path = Path('/home/sbialek/ONC/selfsupervision_anomalies_onc/notebooks/interactive_hydrophone_anomaly_workshop_local.ipynb')
with open(nb_path, 'r') as f:
    nb = json.load(f)

cells = nb['cells']

def find_cell_index(cells, snippet):
    for i, cell in enumerate(cells):
        if snippet in ''.join(cell['source']):
            return i
    return -1

# Find the global config cell
idx_config = find_cell_index(cells, "## Global paths and config")
if idx_config == -1:
    print("Could not find global config cell")
    exit()

# The new config code
new_config_code = [
    "## Global paths and config\n",
    "\n",
    "from pathlib import Path\n",
    "import os\n",
    "import sys\n",
    "import subprocess\n",
    "\n",
    "\n",
    "def find_repo_root(start: Path = Path.cwd()) -> Path:\n",
    "    for parent in [start] + list(start.parents):\n",
    "        if (parent / '.git').exists():\n",
    "            return parent\n",
    "    try:\n",
    "        root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()\n",
    "        return Path(root)\n",
    "    except Exception:\n",
    "        return start\n",
    "\n",
    "\n",
    "REPO_DIR = find_repo_root()\n",
    "\n",
    "# Where you create things during the workshop\n",
    "USER_DATA_DIR = REPO_DIR / 'data'\n",
    "\n",
    "# Where the server already has stuff waiting for you (datasets + trained models)\n",
    "SERVER_DATA_DIR = Path.home() / 'data'\n",
    "# SERVER_DATA_DIR = REPO_DIR / 'data'\n",
    "SERVER_DATASETS_DIR = SERVER_DATA_DIR / 'datasets'\n",
    "SERVER_MODELS_DIR = SERVER_DATA_DIR / 'trained_models'\n",
    "SERVER_PRETRAIN_DIR = SERVER_MODELS_DIR / 'pretrain'\n",
    "SERVER_FINETUNE_DIR = SERVER_MODELS_DIR / 'finetune'\n",
    "\n",
    "# ---------------------------------------------------------------------------\n",
    "# Main FULL dataset (always provided for you)\n",
    "# ---------------------------------------------------------------------------\n",
    "FULL_NAME = 'different_locations_incl_backgroundpipelinenormals_multilabel.h5'\n",
    "DATASET_FULL_USER = USER_DATA_DIR / FULL_NAME\n",
    "DATASET_FULL_SERVER = SERVER_DATASETS_DIR / FULL_NAME\n",
    "\n",
    "# We always expect the FULL H5 to be provided on the server, but if you\n",
    "# dropped a copy into the repo we'll happily use that instead.\n",
    "# We will only be using the one pre-downloaded in the SERVER_DATA_DIR\n",
    "DATASET_H5 = DATASET_FULL_SERVER\n",
    "\n",
    "# ---------------------------------------------------------------------------\n",
    "# SMALL dataset (you can create your own; server copy is a backup)\n",
    "# ---------------------------------------------------------------------------\n",
    "SMALL_NAME = 'different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5'\n",
    "SMALL_H5_USER = USER_DATA_DIR / SMALL_NAME\n",
    "SMALL_H5_SERVER = SERVER_DATASETS_DIR / SMALL_NAME\n",
    "\n",
    "if SMALL_H5_USER.exists():\n",
    "    SMALL_H5 = SMALL_H5_USER\n",
    "elif SMALL_H5_SERVER.exists():\n",
    "    SMALL_H5 = SMALL_H5_SERVER\n",
    "else:\n",
    "    SMALL_H5 = SMALL_H5_USER  # default target if you create it later\n",
    "\n",
    "# ONC API token and data directory\n",
    "ONC_TOKEN = os.environ.get('ONC_TOKEN', '')\n",
    "DATA_DIR = USER_DATA_DIR\n",
    "\n",
    "# ---------------------------------------------------------------------------\n",
    "# CNN paths (your experiments live in the repo; optional backup on server)\n",
    "# ---------------------------------------------------------------------------\n",
    "CNN_EXP_DIR = REPO_DIR / 'cnn_experiments'\n",
    "CNN_BEST = CNN_EXP_DIR / 'cnn_best.pt'\n",
    "\n",
    "# Optional prepared CNN checkpoint under $HOME/data if you decide to ship one\n",
    "CNN_PREP_CKPT = SERVER_MODELS_DIR / 'cnn_baseline' / 'cnn_best.pt'\n",
    "\n",
    "# ---------------------------------------------------------------------------\n",
    "# SSL paths (pretrained backbone + finetune checkpoints)\n",
    "# ---------------------------------------------------------------------------\n",
    "SSL_EXP_DIR = REPO_DIR / 'ssamba_experiments_small'\n",
    "\n",
    "# Pretrained model: prefer user copy (if they downloaded it), else server copy\n",
    "# Note: The download script puts it directly in SERVER_PRETRAIN_DIR\n",
    "PRETRAIN_CKPT_NAME = 'pretrain-joint_best_checkpoint.pth'\n",
    "PRETRAIN_ARGS_NAME = 'args.pkl'\n",
    "\n",
    "if (USER_DATA_DIR / 'trained_models' / 'pretrain' / PRETRAIN_CKPT_NAME).exists():\n",
    "    SSL_PRETRAINED = USER_DATA_DIR / 'trained_models' / 'pretrain' / PRETRAIN_CKPT_NAME\n",
    "    SSL_PRETRAIN_ARGS = USER_DATA_DIR / 'trained_models' / 'pretrain' / PRETRAIN_ARGS_NAME\n",
    "else:\n",
    "    SSL_PRETRAINED = SERVER_PRETRAIN_DIR / PRETRAIN_CKPT_NAME\n",
    "    SSL_PRETRAIN_ARGS = SERVER_PRETRAIN_DIR / PRETRAIN_ARGS_NAME\n",
    "\n",
    "# Finetuned SSL checkpoint: prefer your run in the repo, fall back to server\n",
    "USER_FT_CKPT = SSL_EXP_DIR / 'models' / 'ft-avgtok_best_checkpoint.pth'\n",
    "\n",
    "# The server copy is downloaded by the script to SERVER_FINETUNE_DIR\n",
    "PREP_FT_CKPT = SERVER_FINETUNE_DIR / 'ft-cls_best_checkpoint.pth'\n",
    "\n",
    "if USER_FT_CKPT.exists():\n",
    "    SSL_FT_CKPT = USER_FT_CKPT\n",
    "elif PREP_FT_CKPT.exists():\n",
    "    SSL_FT_CKPT = PREP_FT_CKPT\n",
    "else:\n",
    "    SSL_FT_CKPT = USER_FT_CKPT  # placeholder; later cells will complain if missing\n",
    "\n",
    "# Audio eval defaults\n",
    "AUDIO_PATH = Path('')\n",
    "CHECKPOINT_PATH = SSL_FT_CKPT\n",
    "\n",
    "print('REPO_DIR =', REPO_DIR)\n",
    "print('USER_DATA_DIR =', USER_DATA_DIR)\n",
    "print('SERVER_DATA_DIR =', SERVER_DATA_DIR)\n",
    "print('DATASET_H5 =', DATASET_H5)\n",
    "print('SMALL_H5 (active) =', SMALL_H5)\n",
    "print('CNN_EXP_DIR =', CNN_EXP_DIR)\n",
    "print('SSL_EXP_DIR =', SSL_EXP_DIR)\n",
    "print('SSL_PRETRAINED =', SSL_PRETRAINED)\n",
    "print('SSL_FT_CKPT =', SSL_FT_CKPT)\n",
    "print('ONC_TOKEN set:', bool(ONC_TOKEN))\n",
    "print('DATA_DIR (where you write stuff) =', DATA_DIR)\n"
]

cells[idx_config]['source'] = new_config_code

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
