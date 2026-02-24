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
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics
import torch
from einops import rearrange
from timm.utils import ModelEma
from torch import nn, Tensor
from tqdm import tqdm

from configs.config_train import LossesWeightsConfig
from data import eeg_consts
from models.classifier_model import NeurolCodebookClassifier
from train.evaluation import get_metrics
from train.logers import MetricLogger
from train.losses import get_vqnsp_losses, SpectralPatchedLoss
from train.optimizers import get_loss_scale_for_deepspeed
from train.training_utils import SmoothedValue


def train_class_batch(classify_model: NeurolCodebookClassifier,

                      samples: Tensor,
                      target: Tensor,
                      classif_loss: nn.Module,
                      recon_loss: nn.Module,
                      ch_names: List[str],
                      losses_weights: LossesWeightsConfig) -> tuple[Dict[str, Tensor], Tensor]:
    batch_size, n_channels, n_samples, times = samples.shape
    pred_class, decoder_out, encoder_out = classify_model(samples, ch_names)
    loss_finetune_class = classif_loss(pred_class, target)

    losses = get_vqnsp_losses(x_target=samples,
                              decoder_out=decoder_out,
                              encoder_out=encoder_out,
                              recon_loss=recon_loss)

    losses["classifier"] = loss_finetune_class
    losses["total_loss"] = torch.cat([losses_weights[key] * val[None] for key, val in losses.items()]).sum()

    return losses, pred_class


def train_one_epoch(classify_model: NeurolCodebookClassifier,
                    classifier_loss: torch.nn.Module,
                    recon_loss: torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    encoder_model_ema: Optional[ModelEma] = None,
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None,
                    ch_names=None,
                    is_binary=True,
                    losses_weights: LossesWeightsConfig = None
                    ):
    input_chans = None
    if ch_names is not None:
        input_chans = eeg_consts.get_input_chans(ch_names)
    classify_model.train(True, wo_codebook=True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        classify_model.zero_grad()
        classify_model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data_batch in tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header)),
                                           total=len(data_loader), desc=header):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        eeg_data = data_batch['data'].float().to(device, non_blocking=True) / 100
        eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', T=200)

        targets = data_batch['label'].to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            eeg_data = eeg_data.half()
            losses, output = train_class_batch(classify_model, eeg_data, targets,
                                               classif_loss=classifier_loss,
                                               ch_names=input_chans,
                                               recon_loss=recon_loss,
                                               losses_weights=losses_weights)
        else:
            with torch.amp.autocast(device_type=device.type):
                losses, output = train_class_batch(classify_model, eeg_data, targets,
                                                   classif_loss=classifier_loss,
                                                   ch_names=input_chans,
                                                   recon_loss=recon_loss,
                                                   losses_weights=losses_weights)

        loss_total = losses["total_loss"]
        loss_value = loss_total.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss_total /= update_freq
            classify_model.backward(loss_total)
            classify_model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if encoder_model_ema is not None:
                    encoder_model_ema.update(classify_model.vqnsp.encoder)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(classify_model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_total /= update_freq
            grad_norm = loss_scaler(loss_total, optimizer, clip_grad=max_norm,
                                    parameters=classify_model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if encoder_model_ema is not None:
                    encoder_model_ema.update(classify_model.vqnsp.encoder)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'cpu':
            torch.cpu.synchronize()
        else:
            raise Exception(f"Invalid device: device={device}")

        if is_binary:
            class_acc = get_metrics(torch.sigmoid(output).detach().cpu().float().numpy(),
                                    targets.detach().cpu().float().numpy(), ["accuracy"], is_binary)[
                "accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()

        # metric_logger.update(loss=loss_value)
        for key, loss_value in losses.items():
            metric_logger.meters[key].update(loss_value.item())

        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
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
            # log_writer.update(loss=loss_value, head="loss")
            for key, loss_value in losses.items():
                log_writer.update(loss=loss_value.item(), head=f"loss/{key}")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_classifier(data_loader: torch.utils.data.DataLoader,
                        model: NeurolCodebookClassifier,
                        device: torch.device,
                        header: str = 'Test:',
                        ch_names: Optional[List[str]] = None,
                        metrics: Optional[List[str]] = None,
                        is_binary: bool = True,
                        threshold: float = 0.5,
                        losses_weights: LossesWeightsConfig = None) -> [Dict[str, float], pd.DataFrame, np.ndarray]:
    if metrics is None:
        metrics = ['acc']
    input_chans = None
    if ch_names is not None:
        input_chans = eeg_consts.get_input_chans(ch_names)
    if is_binary:
        classif_loss = torch.nn.BCEWithLogitsLoss()
    else:
        classif_loss = torch.nn.CrossEntropyLoss()
    recon_loss = SpectralPatchedLoss(freq_cutoff=losses_weights['frequency_cutoff'])
    recon_loss.to(device)
    metric_logger = MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred_label = []
    true_label = []
    id_key = []
    id_interval = []
    print(f"Run eval in mode {header}...")
    for step, batch in tqdm(enumerate(metric_logger.log_every(data_loader, 10, header)),
                            total=len(data_loader),
                            desc=f"Eval-{header}"):
        x_eeg_batch = batch['data']
        target_class_batch = batch['label']
        id_key_batch = batch['id_key']
        id_interval_batch = batch['id_interval']

        x_eeg_batch = x_eeg_batch.float().to(device, non_blocking=True) / 100
        x_eeg_batch = rearrange(x_eeg_batch, 'B N (A T) -> B N A T', T=200)
        target_class_batch = target_class_batch.to(device, non_blocking=True)
        if is_binary:
            target_class_batch = target_class_batch.float().unsqueeze(-1)

        # compute output
        with torch.amp.autocast(device_type=device.type):
            pred_out, decoder_out, encoder_out = model(x_eeg_batch, input_chans=input_chans)
            loss_classif = classif_loss(pred_out, target_class_batch)
            losses_batch = get_vqnsp_losses(x_eeg_batch, decoder_out, encoder_out, recon_loss)
            losses_batch['classifier'] = loss_classif
            losses_batch["total_loss"] = torch.cat(
                [losses_weights[key] * val[None] for key, val in losses_batch.items()]).sum()
        # losses_batch["total_loss"] = loss_total

        if is_binary:
            prob_class_batch = torch.sigmoid(pred_out).cpu()
        else:
            prob_class_batch = prob_class_batch.cpu()
        target_class_batch = target_class_batch.cpu()

        results_batch = get_metrics(prob_class_batch.numpy(), target_class_batch.numpy(), metrics, is_binary)
        pred_label.append(prob_class_batch)
        true_label.append(target_class_batch)
        id_key.append(id_key_batch)
        id_interval.append(id_interval_batch)

        batch_size = x_eeg_batch.shape[0]
        # metric_logger.update(loss=loss_classif.item())
        for key, value in results_batch.items():
            metric_logger.meters[key].update(value, n=batch_size)

        for key, loss_value in losses_batch.items():
            metric_logger.meters[key].update(loss_value.item())

        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for key, val in metric_logger.meters.items():
        print(f"***{key}: {val.global_avg:.3f}***")

    pred_label = torch.cat(pred_label, dim=0).numpy()
    true_label = torch.cat(true_label, dim=0).numpy()
    id_key = np.concatenate(id_key)
    id_interval = np.concatenate(id_interval)
    results_df = pd.DataFrame({'id_key': id_key,
                               'id_interval': id_interval,
                               'pred_label': pred_label[:, 0],
                               'true_label': true_label[:, 0]})

    eval_metrics = get_metrics(pred_label, true_label, metrics, is_binary, threshold=threshold)

    for key in losses_batch.keys():
        eval_metrics[key] = metric_logger.meters[key].global_avg

    prob_class_batch = (pred_label > threshold).astype(float)
    conf_matrix = sklearn_metrics.confusion_matrix(true_label, prob_class_batch, normalize='true')
    if is_binary:
        tn, fp, fn, tp = conf_matrix.ravel().tolist()
        eval_metrics['tn'] = tn
        eval_metrics['fp'] = fp
        eval_metrics['fn'] = fn
        eval_metrics['tp'] = tp

    return eval_metrics, results_df, conf_matrix
