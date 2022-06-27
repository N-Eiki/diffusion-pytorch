from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os.path
from torchvision import transforms
import torch

from config import CFG
if __name__ == "__main__":
    model = Unet(
        dim=CFG.dim,
        dim_mults=(1, 2, 4),
        channels=3
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=CFG.image_size,
        timesteps=1000,  # number of steps
        loss_type='l1'  # L1 or L2
    )

    data = torch.load("results/model-699.pt",map_location=torch.device('cpu'))
    diffusion.load_state_dict(data['model'])
    diffusion.eval()
    images_tensor = diffusion.sample(3)

    if(not os.path.exists("samples")):
        os.mkdir("samples")

    for i,image_tensor in enumerate(images_tensor):
        image = transforms.ToPILImage(mode='RGB')(image_tensor)
        image.save(f"samples/sample_{i}.png")
