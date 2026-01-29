from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class SpectralPatchedLoss(nn.Module):
    def __init__(self,
                 smooth_l1_loss = False # Flag to use SmoothL1Loss instead of MSELoss.
                 ):
        """Spectral loss for patch-based EEG models"""
        super().__init__()
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def forward(self, x_target: Tensor, recon_magnitude: Tensor, recon_phase: Tensor):
        """Computes spectral loss from magnitude and phase"""
        x_fft = torch.fft.fft(x_target, dim=-1)
        target_amplitude = _std_norm(torch.abs(x_fft))
        target_phase = _std_norm(torch.angle(x_fft))
        amplitude_loss = self.calculate_recon_loss(recon_magnitude, target_amplitude)
        phase_loss = self.calculate_recon_loss(recon_phase, target_phase)
        return amplitude_loss, phase_loss

    def calculate_recon_loss(self, recon: Tensor, target: Tensor)-> Tensor:
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(recon, target)
        return rec_loss

def _std_norm(x):
    mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
    std = torch.std(x, dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / std
    return x