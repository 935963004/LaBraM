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
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange

def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            
        metric_logger.update(loss=loss_value)
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
            log_writer.update(loss=loss_value, head="loss")
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
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        
        # compute output
        with torch.cuda.amp.autocast():
            output = model(EEG, input_chans=input_chans)
            loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret
