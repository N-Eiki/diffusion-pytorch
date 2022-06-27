from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
from tqdm import tqdm

from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import DiffusionDataset
from utils import get_paths, train_fn, save
from config import CFG

def main():
    model = Unet(
        dim = CFG.dim,
        dim_mults = (1, 2, 4),
        channels=3
    ).cuda()


    diffusion = GaussianDiffusion(
        model,
        image_size=CFG.image_size,
        timesteps = 1000,   # number of steps
        loss_type = 'l1' ,   # L1 or L2
        channels=3
    ).cuda()

    trainer = Trainer(
        diffusion,
        CFG.image_path,
        image_size = CFG.image_size,
        train_batch_size = 64,
        train_lr = 2e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 1000,
    )

    trainer.train()

if __name__=="__main__":
    main()
