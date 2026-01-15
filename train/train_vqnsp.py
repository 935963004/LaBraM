# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
import sys
from typing import Iterable
import torch

from data import eeg_consts
from models.vqnsp_model import VQNSP
from train.logers import MetricLogger, SmoothedValue
from utils import dist_utils
from train.losses import SpectralPatchedLoss


def train_one_epoch(model: VQNSP,
                    data_loader_list: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    clip_grad: float = 0,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    ch_names_list=None,
                    args=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    recon_loss = SpectralPatchedLoss()
        
    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before each epoch")
        except:
            pass
    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        input_chans = eeg_consts.get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
            batch = batch.float().to(device, non_blocking=True) / 100

            with torch.amp.autocast(device):
                recon_out, encoder_out = model(batch, input_chans=input_chans)
                quantize_loss = encoder_out['quantize_loss']
                recon_amplitude = recon_out['recon_amplitude']
                recon_phase = recon_out['recon_phase']
                rec_amplitude_loss, rec_phase_loss = recon_loss(recon_amplitude, recon_phase)
                total_loss = rec_amplitude_loss + rec_phase_loss + quantize_loss
            loss_value = total_loss.item()

            log_loss = {}
            split = "train" if model.training else "val"
            log_loss[f'{split}/quant_loss'] = quantize_loss.detach().mean()
            log_loss[f'{split}/rec_loss'] = rec_amplitude_loss.detach().mean()
            log_loss[f'{split}/rec_angle_loss'] = rec_phase_loss.detach().mean()
            log_loss[f'{split}/total_loss'] = total_loss.detach().mean()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), force=True)
                raise NotImplementedError
                # utils.save_nan_model(args, model)
                sys.exit(1)

            optimizer.zero_grad()
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(total_loss, optimizer, clip_grad=clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            
            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            
            new_log_loss = {k.split('/')[-1]:v for k, v in log_loss.items() if k not in ['total_loss']}
            metric_logger.update(**new_log_loss)

            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(**new_log_loss, head="train/loss")

                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.update(loss_scale=loss_scale_value, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # stat the codebook usage information
    if hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['Unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return train_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader_list, model, device, log_writer=None, epoch=None, ch_names_list=None, args=None):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Validation:'

    # switch to evaluation mode
    model.eval()

    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantizer.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before testing")
        except:
            pass
    
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        input_chans = eeg_consts.get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, 10, header)):

            images = batch.float().to(device, non_blocking=True) / 100
            loss, log_loss = model(images, input_chans=input_chans)

            metric_logger.update(loss=loss.item())

            new_log_loss = {k.split('/')[-1]:v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # stat the codebook usage information
    if hasattr(model, 'module') and hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantizer._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantizer.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return test_stat

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def calculate_codebook_usage(data_loader, model, device, log_writer=None, epoch=None, args=None):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Calculating codebook usage:'

    # switch to evaluation mode
    model.eval()
    model_module = _get_torch_model(model)
    
    codebook_num = args.codebook_n_emd
    codebook_cnt = torch.zeros(codebook_num, dtype=torch.float64).to(device)

    for step, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.float().to(device, non_blocking=True) / 100

        outputs = model_module.get_tokens(images)['token'].view(-1)

        outputs_gather_list = [torch.zeros_like(outputs) for _ in range(dist_utils.get_world_size())]
        torch.distributed.all_gather(outputs_gather_list, outputs)
        all_tokens = torch.cat(outputs_gather_list, dim=0).view(-1) # [B * N * Ngpu, ]
        
        codebook_cnt += torch.bincount(all_tokens, minlength=codebook_num)

    # statistic
    zero_cnt = (codebook_cnt == 0).sum() # 0
    print(f"STAT:  {zero_cnt} tokens ({(zero_cnt / codebook_num) * 100}%) never are used in this codebook.")


def _get_torch_model(model) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model should be torch.nn.Module, but got {type(model)}")
        return model
