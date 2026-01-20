from typing import Dict, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor, nn as nn
import torch.nn.functional as F

class SpectralPatchedLoss(nn.Module):
    def __init__(self,
                 smooth_l1_loss = False # Flag to use SmoothL1Loss instead of MSELoss.
                 ):
        """Spectral loss for patch-based EEG models"""
        super().__init__()
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def forward(self, x_target: Tensor, recon_magnitude: Tensor, recon_phase: Tensor)-> Tuple[Tensor, Tensor]:
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


def get_vqnsp_losses(x_target: Tensor,
                     decoder_out: Dict[str, Tensor],
                     encoder_out: Dict[str, Tensor],
                     recon_loss: Optional[nn.Module] = None) -> Dict[str, Tensor]:
    if recon_loss is None:
        recon_loss = SpectralPatchedLoss()
    quantize_loss = encoder_out['quantize_loss']
    recon_amplitude = decoder_out['recon_amplitude']
    recon_phase = decoder_out['recon_phase']
    rec_amplitude_loss, rec_phase_loss = recon_loss(x_target=x_target,
                                                    recon_magnitude=recon_amplitude,
                                                    recon_phase=recon_phase)
    total_loss = rec_amplitude_loss + 0.1*rec_phase_loss + quantize_loss
    losses_out = {"total_loss": total_loss,
                  "rec_amplitude_loss": rec_amplitude_loss,
                  "rec_phase_loss": rec_phase_loss,
                  "quantize_loss": quantize_loss}
    return losses_out
