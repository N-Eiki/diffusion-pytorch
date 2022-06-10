from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os.path
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset


from dataset import DiffusionDataset
from config import CFG


if __name__ == "__main__":
    model = Unet(
        dim=16,
        dim_mults=(1, 2, 4)
    )

    model = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 1000,
        loss_type = 'l1'
    )
    
    model.train()    
    trainer = Trainer(
        model1,
        CFG.image_path,
        train_batch_size = CFG.batch_size,
        train_lr = CFG.lr,
        train_num_steps = CFG.steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True                        # turn on mixed precision
    )
    # diffusion.load_state_dict(data['model'])
    # diffusion.eval()
    images_tensor = diffusion.sample(5)

    if(not os.path.exists("trained_sample")):
        os.mkdir("trained_sample")

    for i,image_tensor in enumerate(images_tensor):
        image = transforms.ToPILImage(mode='RGB')(image_tensor)
        image.save(f"trained_sample/sample_{i}.png")