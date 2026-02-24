import glob
import io
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from timm.utils import ModelEma
from timm.utils import get_state_dict
from torch import Tensor
from torch.optim.optimizer import Optimizer

from configs import ConfigRunClassifierModel
from train.optimizers import NativeScalerWithGradNormCount as NativeScaler
from utils.dist_utils import save_on_master


def load_state_dict(model: nn.Module,
                    state_dict: Dict[str, Tensor],
                    prefix: str = '',
                    ignore_missing: str ="relative_position_index"):
    """
    Loads a given state dictionary into a model, handling submodules and providing detailed logs of
    missing, unexpected, and ignored keys. This function modifies the given state dictionary to ensure
    compatibility with the `_load_from_state_dict` function and filters missing keys based on provided
    criteria.

    Parameters:
    - model: nn.Module
      The model into which the state dictionary will be loaded.
    - state_dict: Dict[str, Tensor]
      A dictionary containing the states to be loaded. It maps parameter names to their corresponding
      tensors.
    - prefix: str, optional
      A prefix to be applied to the parameter names in the state dictionary. Default is an empty string.
    - ignore_missing: str, optional
      A pipe-separated string of substrings used to filter out certain missing keys. Keys containing
      these substrings are ignored when loading the state dictionary. Default is "relative_position_index".

    Raises:
    - None

    """
    def _load_submodule(module_: nn.Module, prefix_: str = ''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix_[:-1], {})
        module_._load_from_state_dict(
            state_dict, prefix_, local_metadata,
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs)
        for name, child in module_._modules.items():
            if child is not None:
                _load_submodule(child, prefix_ + name + '.')

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    _load_submodule(model, prefix_=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(output_dir: str,
               cfg: ConfigRunClassifierModel,
               epoch: int,
               model: nn.Module,
               model_without_ddp: nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_scaler,
               model_ema: Optional[torch.nn.Module] = None,
               optimizer_disc=None,
               save_ckpt_freq: int = 1):
    # TODO complite docs for save_model after configs finishing
    output_dir = Path(output_dir, 'models')
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_name = str(epoch)

    if not cfg.train.enable_deepspeed:
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name), ]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'cfg': cfg.as_dict(),
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)


    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=output_dir,
                              tag="checkpoint-%s" % epoch_name,
                              client_state=client_state)


def auto_load_models(cfg: ConfigRunClassifierModel,
                     model: nn.Module,
                     model_without_ddp: nn.Module,
                     optimizer: Optimizer,
                     loss_scaler: NativeScaler,
                     model_ema: Optional[ModelEma] = None,
                     optimizer_disc=None):
    load_dir = cfg.train.resume_ckpt_path
    file_path_ckpt = None
    # cfg.train.enable_deepspeed
    if not cfg.train.enable_deepspeed:
        # torch.amp
        if cfg.train.auto_resume and load_dir is not None:
            all_checkpoints = glob.glob(os.path.join(load_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                file_path_ckpt = os.path.join(load_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(load_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    file_path_ckpt = os.path.join(load_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % file_path_ckpt)

            if file_path_ckpt:
                if file_path_ckpt.startswith('https'):
                    checkpoint = torch.hub.load_state_dict_from_url(
                        file_path_ckpt, weights_only=False, check_hash=True)
                else:
                    checkpoint = torch.load(file_path_ckpt, weights_only=False)
                model_without_ddp.load_state_dict(checkpoint['model'])  # strict: bool=True, , strict=False
                print("Resume checkpoint %s" % file_path_ckpt)
                if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                    cfg.train.start_epoch = 1  # checkpoint['epoch'] + 1
                    if cfg.train.model_ema:
                        _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                    if 'scaler' in checkpoint:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    print("With optim & sched!")
                if 'optimizer_disc' in checkpoint:
                    optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if cfg.train.auto_resume:
            all_checkpoints = glob.glob(os.path.join(load_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                file_path_ckpt = os.path.join(load_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(file_path_ckpt, tag='checkpoint-%d' % latest_ckpt)
                cfg.train.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if cfg.train.use_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

