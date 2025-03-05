from monai_generative.models import DiffusionModelUNet, DDPM
import torch
from monai.networks.nets import UNet
from monai_generative.losses import PatchAdversarialLoss

def get_ldm_model():
    # 3D latent diffusion model
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=2,
        num_channels=(32, 64, 128, 256),
        attention_levels=(False, True, True, True),
        norm_num_groups=16,
        with_conditioning=False
    )
    
    # 初始化扩散模型
    diffusion = DDPM(
        model=model,
        timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        objective="pred_noise"
    )
    return diffusion