"""
ADM Wrapper using Guided Diffusion UNet
Loads pretrained Guided Diffusion / ADM model for use as mu_real in DMD2 distributional matching
"""

import os
import sys
import torch
import torch.nn as nn

# Add guided-diffusion to path
guided_diffusion_path = os.path.join(os.path.dirname(__file__), '../../guided-diffusion')
if os.path.exists(guided_diffusion_path):
    sys.path.insert(0, guided_diffusion_path)

from guided_diffusion.script_util import create_model
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule


def get_x0_from_noise(noisy_image, noise_pred, alphas_cumprod, timesteps):
    """
    Recover x0 (clean image) from noisy image and predicted noise
    Using DDPM formula: x0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)

    Args:
        noisy_image: [B, C, H, W] noisy image (x_t)
        noise_pred: [B, C, H, W] predicted noise
        alphas_cumprod: [T] cumulative alphas
        timesteps: [B] timestep indices

    Returns:
        x0_pred: [B, C, H, W] predicted clean image
    """
    alpha_prod_t = alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    x0_pred = (noisy_image - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

    return x0_pred


class ADMWrapper(nn.Module):
    """
    Wrapper for pretrained Guided Diffusion / ADM model (mu_real in DMD2)
    """
    def __init__(
        self,
        model_path=None,
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_train_timesteps=1000,
        channel_mult="",
        attention_resolutions="16,8",
        num_heads=4,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        dropout=0.0,
        learn_sigma=False,
        class_cond=False,
    ):
        super().__init__()
        self._model_path = model_path
        self.num_train_timesteps = num_train_timesteps
        self.image_size = image_size

        # Create Guided Diffusion UNet
        self.unet = create_model(
            image_size=image_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            learn_sigma=learn_sigma,
            class_cond=class_cond,
            use_checkpoint=False,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=use_scale_shift_norm,
            dropout=dropout,
            resblock_updown=resblock_updown,
            use_fp16=False,
            use_new_attention_order=False,
        )

        # Freeze all parameters
        self.freeze_parameters()

        # Setup beta schedule for noise computation
        betas = get_named_beta_schedule("linear", num_train_timesteps)
        betas = torch.from_numpy(betas).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('betas', betas.float())

        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)

    def freeze_parameters(self):
        """Freeze all model parameters (teacher model should not be trained)"""
        self.unet.requires_grad_(False)
        for param in self.unet.parameters():
            param.requires_grad = False

    def predict_noise(self, noisy_image, timesteps):
        """
        Predict noise from noisy image at given timesteps

        Args:
            noisy_image: [B, C, H, W] noisy image
            timesteps: [B] timestep indices

        Returns:
            predicted_noise: [B, C, H, W] noise prediction
        """
        with torch.no_grad():
            # Ensure timesteps are long tensor
            if not isinstance(timesteps, torch.LongTensor) and timesteps.dtype != torch.long:
                timesteps = timesteps.long()

            # Guided Diffusion UNet forward: (x, timesteps) -> noise prediction
            model_output = self.unet(noisy_image, timesteps)

            # If learn_sigma=True, output is [B, 6, H, W] (noise + sigma)
            # We only need noise, so take first 3 channels
            C = noisy_image.shape[1]
            noise_pred = model_output[:, :C]

        return noise_pred

    def forward(self, noisy_image, timesteps):
        """Forward pass - just calls predict_noise"""
        return self.predict_noise(noisy_image, timesteps)

    def add_noise(self, clean_image, noise, timesteps):
        """
        Add noise to clean image according to DDPM schedule

        Args:
            clean_image: [B, C, H, W] clean image
            noise: [B, C, H, W] noise tensor
            timesteps: [B] timestep indices

        Returns:
            noisy_image: [B, C, H, W] noisy image
        """
        alphas_cumprod = self.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1).float()
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        noisy_image = sqrt_alphas_cumprod * clean_image + sqrt_one_minus_alphas_cumprod * noise

        return noisy_image

    def load_checkpoint(self, checkpoint_path):
        """Load pretrained checkpoint"""
        print(f"Loading ADM checkpoint from {checkpoint_path}")

        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'ema' in state_dict:
                state_dict = state_dict['ema']

        # Load with strict=False to handle missing keys
        missing_keys, unexpected_keys = self.unet.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

        self.freeze_parameters()
        print(f"Successfully loaded ADM checkpoint")
