# config.py
from gotenNet import GotenNetConfig

TRAIN_CFG = {
    # data
    #"dataset_path": "/work3/khepe/data/dataset_ipa.pckl", # voronoi
    #"dataset_path": "data/dataset_cutoff_ipa.pckl", # cutoff
    "data_df_path": "data/data.pckl",
    "train_frac":   0.8,
    "val_frac":     0.1,
    "batch_size":           32,
    "num_workers":          0,
    "seed":                 0,
    "train_seed":           0,
    "split_seed":           0,

    # device
    "device":               "cuda",

    # training
    "max_epochs":           1000,
    "lr":                   1e-4, 
    "min_lr":               1e-8,
    "weight_decay":         0.01, 
    "gradient_clipping":    10,

    # loss function
    "loss_fn":              "mae",   # "mae" | "mse" | "sc" | "inverse_huber"
    "loss_percentile":      0.8,     # only used by inverse_huber

    # learning rate scheduler (ReduceLROnPlateau)
    "alpha":                0.9,       # EMA smoothing factor for val loss
    "patience":             150,        # early stopping patience

    # experiment
    "experiment_number":    40, 
    "run_name":             "exp40_voro_small_no_vr", 

    # wandb
    "use_wandb":            True,
    "wandb_project":        "dialectric",
    "wandb_entity":         "gnn_experiments",

    # offset and standardization
    "target_log_transform": False,   # applies log(y + offset) before training
    "target_log_offset": 131,
    "target_standardize":   True,   # applies (y - mean) / std after optional log step

    # Gaussianization
    #"use_gaussianization":  True,
}

GotenNet_Model_CFG = GotenNetConfig(
    dne=256,
    ded=256,
    dxpd=256,
    num_layers=6,
    num_rbfs=64,
    cutoff=6.0,
    Lmax=2,
    dropout=0.1,
    num_heads=8,
    readout_depth = 2,
    readout_n_hidden = 256,
    plain_last = True,
)