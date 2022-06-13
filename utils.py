import os
import torch

def get_paths(image_path):
    images = os.listdir(image_path)
    images = [os.path.join(image_path, img) for img in images if img[-3:] in ["jpg", "png"]]
    return images

def train_fn(diffusion, loader):
    diffusion.train()
    losses = 0
    for i, images in enumerate(loader):
        loss = diffusion(images)
        losses += loss
        loss.backward()
    return losses/len(loader)


def save(step, model, results_folder, milestone):
        data = {
            'step': step,
            'model': model.state_dict(),
            # 'ema': ema_model.state_dict(),
            # 'scaler': scaler.state_dict()
        }

        if not os.path.exists(os.path.join(results_folder, "checkpoints")):
            os.makedirs(os.path.join(results_folder, "checkpoints"),exist_ok=True)
        torch.save(data, os.path.join(results_folder, "checkpoints", f'model-{milestone}.pt'))