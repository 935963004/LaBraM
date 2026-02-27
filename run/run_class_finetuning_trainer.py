# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
from argparse import Namespace
import datetime
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, OrderedDict, Dict

# from deepspeed import DeepSpeedConfig
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model
from timm.utils import ModelEma

from configs import FeaturesType, ConfigVQNSP, ConfigRunClassifierModel, ClassifierTypes, ConfigProcEEGDataset, \
    ConfigEEGClassifier
from configs.config_optimizer import OptimizerTypes
from data import patch_datasets
from models import models_io
from models.classifier_model import NeurolCodebookClassifier
from models.neural_transformer import NeuralTransformer
from models.vqnsp_model import VQNSP
from train import optimizers, logers
from train.logers import TensorboardLogger
from train.losses import SpectralPatchedLoss
from train.optimizers import create_optimizer, get_parameter_groups, LayerDecayValueAssigner, \
    NativeScalerWithGradNormCount as NativeScaler
from train.train_finetuning_classifier import train_one_epoch, evaluate_classifier
from utils import dist_utils
from models.registry_finetune_classifiers import *
from models.registry_vqnsp_models import *

def get_default_cfg(labram_asis: bool = False) -> ConfigRunClassifierModel:
    cfg = ConfigRunClassifierModel()
    cfg.train.epochs = 30
    cfg.train.warmup_epochs = 5
    cfg.train.update_freq = 1
    cfg.train.save_ckpt_freq = 5
    cfg.train.seed = 1984
    cfg.train.resume_ckpt_path = None
    cfg.train.start_epoch = 0
    cfg.train.num_workers = 16
    cfg.train.pin_mem = True
    cfg.train.device = 'cuda'

    # cfg.log.ckpt_dir = "./checkpoints/"
    # cfg.log.log_dir = "./logs/"
    cfg.log.experiment = "finetune_dim64_2lastlayers_CLS_Bags_PATCH"
    cfg.train.losses_weights.classifier = 5.0
    if labram_asis:
        cfg.log.experiment = "labram_asis"
        cfg.train.losses_weights.classifier=1.0
        cfg.train.losses_weights.magnitude_recon = 0.0
        cfg.train.losses_weights.phase_recon =0.0
        cfg.train.losses_weights.quantize_err =0.0



    cfg.data.batch_size_train = 128
    cfg.data.batch_size_val = 128
    cfg.data.ds_name = "INTERNAL"
    cfg.data.data_split = [0.8, 0.2, 0.0]
    cfg.data.fold_split_path = None #"./checkpoints/finetune_dim64_2lastlayers_CLS_Bags_PATCH_INTERNAL_20260226-191127/fold_split_ids.yaml"

    cfg.model.name_encoder = 'labram_base_patch200_200'
    cfg.model.name_vqnsp = 'vqnsp_encoder_base_decoder_3x200x12'
    cfg.model.weights_vqnsp_path = "./checkpoints/vqnsp.pth"
    cfg.model.weights_classifier_path = "./checkpoints/labram-base.pth"
    cfg.model.feature_space = [FeaturesType.CLS_TOKEN.value,
                               FeaturesType.BAG_OF_CODES.value,
                               FeaturesType.PATCH_TOKENS.value]

    cfg.model.classifier_type = ClassifierTypes.MLP.value
    cfg.model.num_classes = 1
    cfg.model.features_emb_dim = 64
    cfg.model.linear_embedding = False

    cfg.model.encoder.qkv_bias = True
    cfg.model.encoder.use_rel_pos_bias = True
    cfg.model.encoder.use_abs_pos_emb = True
    cfg.model.encoder.init_values = 0.1
    cfg.model.encoder.drop_rate = 0.0
    cfg.model.encoder.attn_drop_rate = 0.0
    cfg.model.encoder.drop_path_rate = 0.1
    cfg.model.encoder.norm_layer = "LayerNorm"
    cfg.model.encoder.qk_norm = "LayerNorm"
    cfg.model.encoder.num_classes = 0
    cfg.model.vqnsp = ConfigVQNSP(in_chans=1, num_tokens=8192, embed_dim=64)

    if labram_asis:
        cfg.model.classifier_type = ClassifierTypes.LINEAR.value
        cfg.model.features_emb_dim = cfg.model.encoder.embed_dim
        cfg.model.feature_space = [FeaturesType.PATCH_TOKENS.value]
        cfg.model.linear_embedding = True

    cfg.optim.optimizer_type = OptimizerTypes.ADAMW.value
    cfg.optim.eps = 1e-8
    cfg.optim.opt_betas = None
    cfg.optim.clip_grad = 1.0
    cfg.optim.momentum = 0.9
    cfg.optim.weight_decay = 0.05
    cfg.optim.weight_decay_end = None
    cfg.optim.lr = 5e-4
    cfg.optim.layer_decay = 0.65
    cfg.optim.warmup_lr = 1e-6
    cfg.optim.min_lr = 1e-6
    cfg.optim.warmup_epochs = 5
    cfg.optim.warmup_steps = -1
    cfg.optim.smoothing = 0.1
    cfg.optim.disable_weight_decay_on_rel_pos_bias = True

    decoder_layers = ["quantizer", "decoder",
                  "encode_task_layer", "decode_task_layer"]
    train_last_layers = 2
    blocks_filter = [f"encoder.blocks.{i}." for i in range(cfg.model.encoder.depth - train_last_layers -1)]
    filter_opt = ["cls_token", "patch_embed", "pos_embed", "time_embed"] + decoder_layers + \
                 blocks_filter
    cfg.optim.filter_layers=filter_opt
    if labram_asis:
        cfg.optim.filter_layers = decoder_layers
        cfg.optim.clip_grad = None

    return cfg


def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)

    parser.add_argument('--debug', action='store_true', help='run in debug mode')
    # run
    parser.add_argument('--run_config', default=None, type=str, help='Path to run configuration file(yaml/json)')
    parser.add_argument('--cross_valid', action='store_true', help='run in cross validation mode')
    parser.add_argument('--labram_asis', action='store_true', help='run in labram asis mode')
    # parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--epochs', default=30, type=int)
    # parser.add_argument('--update_freq', default=1, type=int)
    # parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation

    # Model parameters
    # parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    # parser.add_argument('--qkv_bias', action='store_true')
    # parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    # parser.set_defaults(qkv_bias=True)
    # parser.add_argument('--rel_pos_bias', action='store_true')
    # parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    # parser.set_defaults(rel_pos_bias=True)
    # parser.add_argument('--abs_pos_emb', action='store_true')
    # parser.set_defaults(abs_pos_emb=False)
    # parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
    #                     help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    #
    # parser.add_argument('--input_size', default=200, type=int,
    #                     help='EEG input size')
    #
    # parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
    #                     help='Dropout rate (default: 0.)')
    # parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
    #                     help='Attention dropout rate (default: 0.)')
    # parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
    #                     help='Drop path rate (default: 0.1)')
    #
    # parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    #
    # parser.add_argument('--model_ema', action='store_true', default=False)
    # parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    # parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    #
    # # classifer parameters
    # parser.add_argument('--classifier_type', default='linear', type=str, metavar='CLASSIFIER_HEAD',
    #                     help='Type of classification head to use linear/MLP(3 layers) (default: "linear")')
    # parser.add_argument('--nb_classes', default=0, type=int,
    #                     help='number of the classification types')
    # parser.add_argument('--features_classif_dim', default=120, type=int,
    #                     help='dimension of the embedding features for classification head')
    #
    # # tokenizer settings
    # # parser.add_argument("--use_tokenizer", action="store_true", help="Use tokenizer or not.")
    # parser.add_argument("--tokenizer_weight", type=str, help="Path to tokenizer weight")
    # parser.add_argument("--tokenizer_model", type=str, default="vqnsp_encoder_base_decoder_3x200x12")
    #
    # # Tokenizer parameters
    # parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    # parser.add_argument('--codebook_dim', default=64, type=int, help='number of codebook')
    #
    # # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
    #                     help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--weight_decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')
    # parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    #     weight decay. We use a cosine schedule for WD and using a larger decay by
    #     the end of training improves performance for ViTs.""")
    #
    # parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
    #                     help='learning rate (default: 5e-4)')
    # parser.add_argument('--layer_decay', type=float, default=0.9)
    #
    # parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6)')
    # parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    #
    # parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
    #                     help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    #
    # parser.add_argument('--smoothing', type=float, default=0.1,
    #                     help='Label smoothing (default: 0.1)')
    #
    # # * Random Erase params
    # parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
    #                     help='Random erase prob (default: 0.25)')
    # parser.add_argument('--remode', type=str, default='pixel',
    #                     help='Random erase mode (default: "pixel")')
    # parser.add_argument('--recount', type=int, default=1,
    #                     help='Random erase count (default: 1)')
    # parser.add_argument('--resplit', action='store_true', default=False,
    #                     help='Do not random erase first (clean) augmentation split')
    #
    # # * Finetuning params
    # # parser.add_argument('--use_cls', action='store_true', help='use mean pooling or cls token', default=False)
    # # parser.set_defaults(use_cls=True)
    # parser.add_argument('--finetune', default='',
    #                     help='finetune from checkpoint')
    # parser.add_argument('--model_key', default='model|module', type=str)
    # parser.add_argument('--model_prefix', default='', type=str)
    # parser.add_argument('--model_filter_name', default='gzp', type=str)
    # parser.add_argument('--init_scale', default=0.001, type=float)
    # parser.add_argument('--use_mean_pooling', action='store_true')
    # parser.set_defaults(use_mean_pooling=True)
    #
    # parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true')
    #
    # # Dataset parameters
    #
    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default=None,
    #                     help='path where to tensorboard log')
    # parser.add_argument('--device', default='cuda',
    #                     help='device to use for training / testing')
    # parser.add_argument('--seed', default=1984, type=int)
    # parser.add_argument('--resume', default='',
    #                     help='resume from checkpoint')
    # parser.add_argument('--auto_resume', action='store_true')
    # parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    # parser.set_defaults(auto_resume=False)
    #
    # parser.add_argument('--save_ckpt', action='store_true')
    # parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    # parser.set_defaults(save_ckpt=True)
    #
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--eval', action='store_true',
    #                     help='Perform evaluation only')
    # parser.add_argument('--dist_eval', action='store_true', default=False,
    #                     help='Enabling distributed evaluation')
    # parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    #
    # # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    #
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    # parser.add_argument('--dataset', default='TUAB', type=str,
    #                     help='dataset: TUAB | TUEV')
    #
    # # experiment
    # parser.add_argument('--experiment', default='class_finetune', type=str)

    known_args, args_ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            # from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def run_classifier_training(ds_init, cfg: str = None, debug: bool = False, labram_asis: bool = False):
    # args: argparse.Namespace
    # dist_utils.init_distributed_mode(args)

    # if ds_init is not None:
    #     dist_utils.create_ds_config(args)
    #
    # print(args)

    if cfg is None or cfg == '':
        cfg = get_default_cfg(labram_asis=labram_asis)
    elif isinstance(cfg, str):
        if not os.path.isfile(cfg):
            raise ValueError(f"Config file {cfg} does not exist!")
        cfg = ConfigRunClassifierModel.load_config(cfg)
    if cfg.train.enable_deepspeed:
        raise NotImplementedError("DeepSpeed is not supported yet.")
        # dist_utils.init_distributed_mode(args)
        # parser = deepspeed.add_config_arguments(parser)
        ds_init = deepspeed.initialize
    device = torch.device(cfg.train.device) if torch.cuda.is_available() else torch.device('cpu')
    # fix the seed for reproducibility
    seed = cfg.train.seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    time_stamp = f"{time.strftime('%Y%m%d-%H%M%S')}"
    run_name = f"{cfg.log.experiment}_{cfg.data.ds_name}_{time_stamp}"
    if cfg.log.ckpt_dir:
        output_dir = cfg.log.ckpt_dir if not debug else os.path.join(cfg.log.ckpt_dir, 'DBG')
        output_dir = Path(output_dir, run_name)
        output_dir.mkdir(parents=True, exist_ok=True)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_val, dataset_test, ch_names, metrics = get_dataset(cfg.data, work_dir=output_dir)
    cfg.model.num_classes = cfg.data.num_classes

    # if args.disable_eval_during_finetuning:
    #     dataset_val = None
    #     dataset_test = None
    global_rank = 0
    if cfg.train.distributed:
        num_tasks = dist_utils.get_world_size()
        global_rank = dist_utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if cfg.train.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            elif dataset_test is not None :
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_test = None
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None

    if global_rank == 0 and cfg.log.log_dir is not None:
        log_dir = Path(cfg.log.log_dir, 'tb_logs') if not debug else Path(cfg.log.log_dir, 'DBG', 'tb_logs')
        log_dir.mkdir(exist_ok=True, parents=True)
        log_writer = logers.TensorboardLogger(log_dir=log_dir, experiment=run_name)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=cfg.data.batch_size_train,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=cfg.data.batch_size_inference,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=cfg.data.batch_size_inference,
                num_workers=cfg.train.num_workers,
                pin_memory=cfg.train.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        elif dataset_test is not None:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=cfg.data.batch_size_inference,
                num_workers=cfg.train.num_workers,
                pin_memory=cfg.train.pin_mem,
                drop_last=False
            )
        else:
            data_loader_test = None
    else:
        data_loader_val = None
        data_loader_test = None

    classifier_model = build_classifier_model(cfg, device)

    patch_size = classifier_model.patch_size
    print("Patch size = %s" % str(patch_size))
    # cfg.data.patch_size = patch_size
    # args.window_size = (1, args.input_size // patch_size)
    # args.patch_size = patch_size

    # encoder_model.to(device)
    # if vqnsp_model is not None:
    #     vqnsp_model.to(device)

    model_encoder_ema = None
    if cfg.train.use_ema:
        # Important to create EMA encoder_model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_encoder_ema = ModelEma(
            classifier_model.encoder,
            decay=cfg.train.ema_decay,
            device='cpu' if cfg.train.ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % cfg.train.ema_decay)

    model_without_ddp = classifier_model
    num_layers = classifier_model.encoder.num_layers
    n_parameters = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    # total_batch_size = args.batch_size * args.update_freq * dist_utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // cfg.data.batch_size_train
    # print("LR = %.8f" % args.lr)
    # print("Batch size = %d" % total_batch_size)
    # print("Update frequent = %d" % args.update_freq)
    # print("Number of training examples = %d" % len(dataset_train))
    # print("Number of training training per epoch = %d" % num_training_steps_per_epoch)
    # print("Use cls token = %s" % str(use_cls_token))
    # print("Use tokenizer: %s" % str(args.use_tokenizer))
    # num_layers = model_without_ddp.encoder.get_num_layers()
    if cfg.optim.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(cfg.optim.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = classifier_model.no_weight_decay()
    if cfg.optim.disable_weight_decay_on_rel_pos_bias:
        for i in range(cfg.model.encoder.depth):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if cfg.train.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            classifier_model,
            cfg.optim.weight_decay,
            skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        classifier_model, optimizer, _, _ = ds_init(
            args=cfg,
            model=classifier_model,
            model_parameters=optimizer_params,
            dist_init_required=not cfg.train.distributed,
            filter_name=cfg.filter_layers
        )

        print("encoder_model.gradient_accumulation_steps() = %d" % classifier_model.gradient_accumulation_steps())
        # assert classifier_model.gradient_accumulation_steps() == args.update_freq
    else:
        if cfg.train.distributed:
            classifier_model = torch.nn.parallel.DistributedDataParallel(classifier_model,
                                                                         device_ids=[cfg.train.device == 'cuda'],
                                                                         find_unused_parameters=True)
            model_without_ddp = classifier_model.module

        # blocks_filter = [f"encoder.blocks.{i}." for i in range(cfg.model.encoder.depth - 2)]
        # filter_opt = ["cls_token", "patch_embed", "pos_embed", "time_embed", "quantizer", "decoder",
        #               "encode_task_layer", "decode_task_layer"] +\
        #              blocks_filter
        optimizer = create_optimizer(
            cfg.optim, model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
            # filter_name=filter_opt
        )
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = optimizers.cosine_scheduler(
        cfg.optim.lr, cfg.optim.min_lr, cfg.train.epochs, num_training_steps_per_epoch,
        warmup_epochs=cfg.optim.warmup_epochs,
        warmup_steps=cfg.optim.warmup_steps,
    )
    if cfg.optim.weight_decay_end is None:
        cfg.optim.weight_decay_end = cfg.optim.weight_decay
    wd_schedule_values = optimizers.cosine_scheduler(
        cfg.optim.weight_decay, cfg.optim.weight_decay_end, cfg.train.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if cfg.model.num_classes == 1:
        classifier_loss = torch.nn.BCEWithLogitsLoss()
    elif cfg.train.label_smoothing > 0.:
        classifier_loss = LabelSmoothingCrossEntropy(smoothing=cfg.train.label_smoothing)
    else:
        classifier_loss = torch.nn.CrossEntropyLoss()

    recon_loss = SpectralPatchedLoss(freq_cutoff=cfg.train.losses_weights.frequency_cutoff)

    print("classifier_loss = %s" % str(classifier_loss))
    # output_dir
    models_io.auto_load_models(
        cfg=cfg,
        model=classifier_model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_encoder_ema)

    # if args.eval:
    #     print("Start evaluating for starting...")
    #     balanced_accuracy = []
    #     accuracy = []
    #
    #     if type(dataset_test) == list:
    #         for data_loader in data_loader_test:
    #             test_metrics,  test_results_df, test_conf_matrix = evaluate_classifier(data_loader,
    #                                                                        classifier_model,
    #                                                                        device, header='Test:',
    #                                                                        ch_names=ch_names,
    #                                              metrics=metrics, is_binary=(args.nb_classes == 1))
    #             accuracy.append(test_metrics['accuracy'])
    #             balanced_accuracy.append(test_metrics['balanced_accuracy'])
    #     else:
    #         test_metrics, test_results_df, test_conf_matrix = evaluate_classifier(data_loader_test, classifier_model, device, header='Test:', ch_names=ch_names,
    #                                          metrics=metrics,
    #                                          is_binary=(args.nb_classes == 1))
    #         accuracy.append(test_metrics['accuracy'])
    #         balanced_accuracy.append(test_metrics['balanced_accuracy'])
    #     print(
    #         f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
    #
    #     save_eval_results(test_conf_matrix, test_results_df, test_metrics, output_dir, mode='test')
    #     exit(0)

    print(f"Start training for {cfg.train.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    is_binary = (cfg.model.num_classes == 1)
    if dist_utils.is_main_process():
        cfg_yaml_path = Path(output_dir, 'run_cfg.yaml')
        print(f"Save cfg to {cfg_yaml_path}")
        cfg.save_to(cfg_yaml_path)
        ConfigRunClassifierModel.load_config(str(cfg_yaml_path))

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        if cfg.train.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * cfg.log.update_freq)
            log_writer.writer.add_text('tarin', f"EPOCH {epoch}", global_step=epoch)
        print(f"Epoch {epoch} starting ...")

        train_stats = train_one_epoch(
            classify_model=classifier_model,
            classifier_loss=classifier_loss,
            recon_loss=recon_loss,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=cfg.optim.clip_grad,
            encoder_model_ema=model_encoder_ema,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=cfg.log.update_freq,
            ch_names=ch_names,
            is_binary=is_binary,
            losses_weights=cfg.train.losses_weights
        )
        print(f"Epoch {epoch} training finished.")
        if output_dir and cfg.log.ckpt_dir:
            models_io.save_model(output_dir=output_dir,
                                 cfg=cfg,
                                 model=classifier_model,
                                 model_without_ddp=model_without_ddp,
                                 optimizer=optimizer,
                                 loss_scaler=loss_scaler,
                                 epoch=epoch,
                                 model_ema=model_encoder_ema,
                                 save_ckpt_freq=cfg.log.save_ckpt_freq)

        if data_loader_val is not None:
            val_metrics, valid_results_df, valid_conf_matrix = evaluate_classifier(data_loader_val,
                                                                                   classifier_model,
                                                                                   device,
                                                                                   header='Val:',
                                                                                   ch_names=ch_names,
                                                                                   metrics=metrics,
                                                                                   is_binary=is_binary,
                                                                                   losses_weights=cfg.train.losses_weights)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_metrics['accuracy']:.2f}%")
        if data_loader_test is not None:
            test_metrics, test_results_df, test_conf_matrix = evaluate_classifier(data_loader_test,
                                                                                  classifier_model,
                                                                                  device,
                                                                                  header='Test:',
                                                                                  ch_names=ch_names,
                                                                                  metrics=metrics,
                                                                                  is_binary=is_binary,
                                                                                  losses_weights=cfg.train.losses_weights)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_metrics['accuracy']:.2f}%")
        else:
            test_metrics = None
            test_results_df = None
            test_conf_matrix = None


        if max_accuracy < val_metrics["accuracy"]:
            max_accuracy = val_metrics["accuracy"]
            if output_dir:
                models_io.save_model(output_dir=output_dir,
                                     cfg=cfg,
                                     model=classifier_model,
                                     model_without_ddp=model_without_ddp,
                                     optimizer=optimizer,
                                     loss_scaler=loss_scaler,
                                     epoch="best",
                                     model_ema=model_encoder_ema)


            if log_writer is not None:
                update_tb_logger(log_writer, val_metrics, test_metrics, epoch)

            if test_conf_matrix is not None:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_metrics.items()},
                             **{f'test_{k}': v for k, v in test_metrics.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
                print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {test_metrics["accuracy"]:.2f}%')
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_metrics.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}
                print(f'Max accuracy val: {max_accuracy:.2f}%')

        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if output_dir and dist_utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(Path(output_dir, f"log_epoch_{epoch}.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if test_conf_matrix is not None:
                save_eval_results(test_conf_matrix, test_results_df, test_metrics,
                                  output_dir,
                                  mode='test', epoch=epoch)
            save_eval_results(valid_conf_matrix, valid_results_df, val_metrics,
                              output_dir,
                              mode='valid', epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def save_eval_results(conf_matrix: np.ndarray,
                      results_df: pd.DataFrame,
                      stats_metrics: Dict[str, float],
                      output_dir: Path,
                      mode: str = 'test',
                      epoch: int = 0):
    results_file = Path(output_dir, f'{mode}_results_epoch_{epoch}.csv')
    results_df.to_csv(results_file, index=False)
    conf_matrix_file = Path(output_dir, f'{mode}_conf_matrix_epoch_{epoch}.csv')
    pd.DataFrame(conf_matrix).to_csv(conf_matrix_file, index=False)
    stats_file = Path(output_dir, f'{mode}_stats_epoch_{epoch}.json')
    try:
        with open(stats_file, 'w') as json_file:
            json.dump(stats_metrics, json_file, indent=4)
    except Exception:
        pass  # Silently ignore any exception


def update_tb_logger(log_writer: TensorboardLogger, val_stats, test_stats=None, epoch: int =0):
    for key, value in val_stats.items():
        if key == 'accuracy':
            log_writer.update(accuracy=value, head="val", step=epoch)
        elif key == 'balanced_accuracy':
            log_writer.update(balanced_accuracy=value, head="val", step=epoch)
        elif key == 'f1_weighted':
            log_writer.update(f1_weighted=value, head="val", step=epoch)
        elif key == 'pr_auc':
            log_writer.update(pr_auc=value, head="val", step=epoch)
        elif key == 'roc_auc':
            log_writer.update(roc_auc=value, head="val", step=epoch)
        elif key == 'cohen_kappa':
            log_writer.update(cohen_kappa=value, head="val", step=epoch)
        elif key == 'loss':
            log_writer.update(loss=value, head="val", step=epoch)
        else:
            log_writer.update(**{key: value}, head="val", step=epoch)
    if test_stats is not None:
        for key, value in test_stats.items():
            if key == 'accuracy':
                log_writer.update(accuracy=value, head="test", step=epoch)
            elif key == 'balanced_accuracy':
                log_writer.update(balanced_accuracy=value, head="test", step=epoch)
            elif key == 'f1_weighted':
                log_writer.update(f1_weighted=value, head="test", step=epoch)
            elif key == 'pr_auc':
                log_writer.update(pr_auc=value, head="test", step=epoch)
            elif key == 'roc_auc':
                log_writer.update(roc_auc=value, head="test", step=epoch)
            elif key == 'cohen_kappa':
                log_writer.update(cohen_kappa=value, head="test", step=epoch)
            elif key == 'loss':
                log_writer.update(loss=value, head="test", step=epoch)
            else:
                log_writer.update(**{key: value}, head="test", step=epoch)


def build_classifier_model(cfg: ConfigRunClassifierModel,
                           device: torch.device,
                           labram_asis: bool = False) -> NeurolCodebookClassifier:
    encoder_model = get_encoder_models(cfg.model)
    vqnsp_model = get_visual_tokenizer(cfg.model)
    ckpt_encoder_path = cfg.model.weights_encoder_path
    if ckpt_encoder_path:
        if not os.path.isfile(ckpt_encoder_path):
            raise FileNotFoundError(f"Checkpoint of encoder {ckpt_encoder_path} not found!")
        loaded_ckpt_encoder = load_encoder_ckpt(ckpt_path=ckpt_encoder_path,
                                                # model_key=args.model_key,
                                                # model_filter_name=args.model_filter_name,
                                                device_name=device)
        loaded_ckpt_encoder = preproc_ckpt_encode(encoder_model.state_dict(), loaded_ckpt_encoder)
        models_io.load_state_dict(encoder_model, loaded_ckpt_encoder)  # , prefix=args.model_prefix)

    # feature_space = [FeaturesType.CLS_TOKEN.value, FeaturesType.BAG_OF_CODES.value]
    classifier_model = NeurolCodebookClassifier(encoder=encoder_model,
                                                vqnsp=vqnsp_model,
                                                cfg=cfg.model)


    # num_classes=args.nb_classes,
    # classifier_type=args.classifier_type,
    # features_emb_dim=args.features_classif_dim,
    # feature_space= feature_space)
    classifier_model.to(device)
    return classifier_model


def preproc_ckpt_encode(encoder_state_dict: Dict[str, Any], loaded_ckpt_encoder: OrderedDict) -> Dict[Any, Any]:
    for k in ['head.weight', 'head.bias']:
        if k in loaded_ckpt_encoder and loaded_ckpt_encoder[k].shape != encoder_state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del loaded_ckpt_encoder[k]

    all_keys = list(loaded_ckpt_encoder.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            loaded_ckpt_encoder.pop(key)
    # redundant norm layer from legacy model
    loaded_ckpt_encoder = {k.replace(".fc_norm.", ".norm."): v for k, v in loaded_ckpt_encoder.items()}
    return loaded_ckpt_encoder


def load_encoder_ckpt(ckpt_path: str,
                      model_filter_name: str = "gzp",
                      model_key: str = "model|module",
                      device_name: torch.device = torch.device("cpu")) -> OrderedDict:
    if ckpt_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            ckpt_path, map_location=device_name, check_hash=True)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device_name, weights_only=False)

    print("Loaded ckpt from %s" % ckpt_path)
    checkpoint_model = None
    for model_key_ in model_key.split('|'):
        if model_key_ in checkpoint:
            checkpoint_model = checkpoint[model_key_]
            print("Load state_dict by model_key = %s" % model_key_)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    if (checkpoint_model is not None) and (model_filter_name != ''):
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('student.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                pass
        checkpoint_model = new_dict
    return checkpoint_model


def get_encoder_models(cfg: ConfigEEGClassifier) -> NeuralTransformer:
    # use_mem_pooling = args.use_mean_pooling if not args.use_cls else False
    model = create_model(cfg.name_encoder, cfg=cfg.encoder)
    # model = create_model(
    #     args.model,
    #     pretrained=False,
    #     num_classes=0, #args.nb_classes,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     attn_drop_rate=args.attn_drop_rate,
    #     drop_block_rate=None,
    #     # use_mean_pooling=use_mem_pooling,
    #     init_scale=args.init_scale,
    #     use_rel_pos_bias=args.rel_pos_bias,
    #     use_abs_pos_emb=args.abs_pos_emb,
    #     init_values=args.layer_scale_init_value,
    #     qkv_bias=args.qkv_bias,
    #     classifier_type=args.classifier_type)

    return model


def get_visual_tokenizer(cfg: ConfigEEGClassifier) -> VQNSP:
    model = create_model(cfg.name_vqnsp,
                         cfg=cfg.vqnsp,
                         as_tokenizer=True,
                         weights_path=cfg.weights_vqnsp_path)
    # model = create_model(
    #     args.tokenizer_model,
    #     weights_path=args.tokenizer_weight,
    #     as_tokenzer=True,
    #     num_codebook_tokens=args.codebook_size,
    #     codebook_dim=args.codebook_dim,
    # )
    return model


def get_dataset(cfg: ConfigProcEEGDataset, work_dir: str):
    if cfg.ds_name == 'TUAB':
        cfg.dataset_path = Path("/home/leong/data/EEG/TAUB/TUH_Abnormal/v3.0.1/edf/processed/")
        train_dataset, test_dataset, val_dataset = patch_datasets.prepare_TUAB_dataset(cfg.dataset_path)
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        cfg.num_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif cfg.ds_name == 'TUEV':
        train_dataset, test_dataset, val_dataset = patch_datasets.prepare_TUEV_dataset("path/to/TUEV")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        cfg.num_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif cfg.ds_name == 'HMC':
        ch_names = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
        cfg.num_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        dataset_dir = Path("/Users/leon/Data/neurolm_downstream", 'HMC')
        train_dataset, test_dataset, val_dataset = patch_datasets.prepare_HMC_dataset(dataset_dir)
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    elif cfg.ds_name == 'INTERNAL':
        cfg.num_classes = 1  # 3
        cfg.is_normal_abnormal = False
        cfg.dataset_path = "/home/leong/data/EEG/INTER_DATA/lesion_control_processed_10sec"
        cfg.metadata_csv_path = "/home/leong/data/EEG/INTER_DATA/all_labels_int20K_eeg.csv"
        cfg.is_binary_label = True
        # label_names = ['is_normal', 'is_epileptiform', 'is_gen_slowing']
        cfg.label_names = ['is_control', 'is_lesion']
        ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
                    'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
        # metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        # - f1: f1
        # score
        # - precision: precision
        # score
        # - recall: recall
        # score
        metrics = ["pr_auc",
                   "roc_auc",
                   "accuracy",
                   "balanced_accuracy",
                   "f1",
                   "recall",
                   "precision"]  # binary classification
        train_dataset, val_dataset, test_dataset  = patch_datasets.prepare_internal_dataset(cfg=cfg, work_dir=work_dir)
    else:
        raise ValueError("Unknown dataset: %s" % cfg.ds_name)

    return train_dataset, val_dataset, test_dataset,  ch_names, metrics


if __name__ == '__main__':
    args, ds_init = get_args()
    dist_utils.init_distributed_mode(args)
    if ds_init is not None:
        dist_utils.create_ds_config(args)
    # if opts.output_dir:
    #     Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cross_valid:
        pass
    else:
        run_classifier_training(ds_init,
                            cfg=args.run_config,
                            debug=args.debug,
                            labram_asis=args.labram_asis)
