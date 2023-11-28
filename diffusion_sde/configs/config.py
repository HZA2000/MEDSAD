import torch

CFGS = {
    "training":{
        "batch_size":32,
        "val_frac":0.2,
        "n_iters":1001,
        "sde":"vpsde",
        "continuous":True,
        "reduce_mean":True,
        "ckpt_freq":100,
        "ckpt_dir":"./weight/",
    },
    "sampling":{
        "n_steps":1500,
        "sample_dir":"./samples/",
    },
    "model":{
        "in_channel":2,
        "out_channel":1,
        "inner_channel":64,
        "channel_mults":[1,2,4,8],
        "attn_res":[16],
        "num_head_channels":32,
        "res_blocks":2,
        "dropout":0.2,
        "image_size":128,
    },
    "optim":{
        "weight_decay":0,
        "optimizer":'Adam',
        "lr":1e-4,
        "beta1":0.9,
        "eps":1e-8,
        "warmup":1000,
        "grad_clip":1,
        "seed":42,
    },  
    "ema": {
        "ema_rate":0.9999,
    },
    "device" : 'cuda' if torch.cuda.is_available() else 'cpu',
}