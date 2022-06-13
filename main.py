from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
from tqdm import tqdm

from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import DiffusionDataset
from utils import get_paths, train_fn, save
from config import CFG


# def main():
#     model = Unet(
#         dim=16,
#         dim_mults=(1, 2, 4)
#     )

#     diffusion = GaussianDiffusion(
#         model,
#         image_size = 32,
#         timesteps = 1000,
#         loss_type = 'l1'
#     )
    
#     diffusion.train()    
#     paths = get_paths(CFG.image_path)

#     dataset = DiffusionDataset(paths)
#     DataLoader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    
#     if(not os.path.exists("trained_sample")):
#         os.mkdir("trained_sample")

#     for step in tqdm(range(CFG.steps)):
#         avg_loss = train_fn(diffusion, DataLoader)
#         if step%CFG.verbose==0:
#             diffusion.eval()
#             save(step, diffusion, CFG.result, f"step{step}")
#             images_tensor = diffusion.sample(3)
#             for i,image_tensor in enumerate(images_tensor):
#                 image = transforms.ToPILImage(mode='RGB')(image_tensor)
#                 if not os.path.exists(f"{CFG.result}/images/{step}_image"):
#                     os.makedirs(f"{CFG.result}/images/{step}_image/", exist_ok=True)
#                 image.save(f"{CFG.result}/images/{step}_image/sample_{i}.png")
#         tqdm.write(f"train loss: {avg_loss}")


def main():
    model = Unet(
        dim = CFG.dim,
        dim_mults = (1, 2, 4),
        channels=3
    ).cuda()


    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps = 1000,   # number of steps
        loss_type = 'l1' ,   # L1 or L2
        channels=3
    ).cuda()

    trainer = Trainer(
        diffusion,
        CFG.image_path,
        image_size = 32,
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
