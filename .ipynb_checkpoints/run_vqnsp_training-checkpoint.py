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
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.models import create_model
# from optim_factory import create_optimizer

from engine_for_vqnsp import evaluate, train_one_epoch, calculate_codebook_usage
from utils import NativeScalerWithGradNormCount as NativeScaler
import modeling_vqnsp
import utils


def get_args():
    parser = argparse.ArgumentParser('LaBraM pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    # Model parameters
    parser.add_argument('--model', default='vqnsp_encoder_base_decoder_3x200x12', type=str, metavar='MODEL',  help='Name of model to train')  

    parser.add_argument('--codebook_n_emd', default=8192, type=int, metavar='MODEL',
                        help='number of codebook')
    parser.add_argument('--codebook_emd_dim', default=32, type=int, metavar='MODEL',
                        help='number of codebook')
    parser.add_argument('--ema_decay', default=0.99, type=float, metavar='MODEL', help='ema decay for quantizer')
    parser.add_argument('--quantize_kmeans_init', action='store_true', help='enable kmeans_init for quantizer')

    parser.add_argument('--input_size', default=1600, type=int, help='EEG input size for backbone')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False, help="Perform evaluation only")
    parser.add_argument('--pretrained-weight', type=str, default=None, help="Path to pretrained weights")
    parser.add_argument('--calculate_codebook_usage', action='store_true', default=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--dataset-config-path', type=str,
                        help='path to dataset config')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args, **kwargs):

    model = create_model(
        args.model,
        pretrained=args.eval,
        pretrained_weight=args.pretrained_weight,
        as_tokenzer=args.eval,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        EEG_size=args.input_size,
        decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init
    )
    return model


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)

    # get dataset
    # datasets with the same montage can be packed within a sublist
    with open(args.dataset_config_path, 'r') as f:
        dataset_config = json.load(f)
    datasets_train = dataset_config['datasets_train']
    # time window for each sublist in dataset_train
    # to ensure the total sequence length be around 256 for each dataset
    time_window = dataset_config['train_time_window']
    dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(datasets_train, time_window, stride_size=200)

    datasets_val = dataset_config['datasets_val']
    if args.disable_eval:
        dataset_val_list, val_ch_names_list = None, None
    else:
        dataset_val_list, val_ch_names_list = utils.build_pretraining_dataset(datasets_val, [4])

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = sum([len(dataset) for dataset in dataset_train_list]) // args.batch_size // num_tasks

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)

        print("Sampler_train = %s" % str(sampler_train))
        sampler_eval_list = []
        if args.dist_eval:
            # if len(dataset_val) % num_tasks != 0:
            #     print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
            #           'This will slightly alter validation results as extra duplicate entries are added to achieve '
            #           'equal num of samples per-process.')
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                sampler_eval_list.append(sampler_val)
        else:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.SequentialSampler(dataset)
                sampler_eval_list.append(sampler_val)
    else:
        sampler_train_list = [torch.utils.data.RandomSampler(dataset) for dataset in dataset_train_list]
        sampler_eval_list = [torch.utils.data.SequentialSampler(dataset) for dataset in dataset_val_list]
        global_rank = 0

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_train_list.append(data_loader_train)

    if dataset_val_list is not None:
        data_loader_val_list = []
        for dataset, sampler in zip(dataset_val_list, sampler_eval_list):
            data_loader_val = torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
            data_loader_val_list.append(data_loader_val)
    else:
        data_loader_val_list = None

    model.to(device)
    model_without_ddp = model
    if not args.eval:
        print("Model = %s" % str(model_without_ddp))
    for part in ['encoder', 'decoder']:
        model_part = eval(f"model.{part}")
        n_learnable_parameters = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        n_fix_parameters = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
        print(f'number of learnable params in model.{part}: {n_learnable_parameters / 1e6} M')
        print(f'number of fixed params in model.{part}: {n_fix_parameters / 1e6} M')

    if args.eval:
        test_stats = evaluate(data_loader_val_list, model, device, log_writer, 0, ch_names_list=val_ch_names_list, args=args)
        log_writer.update(**test_stats, head="val/loss")
        log_writer.flush()
        exit(0)

    if args.calculate_codebook_usage:
        test_stats = calculate_codebook_usage(data_loader_val_list, model, device, log_writer, 0, args=args)
        exit(0)

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    print(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = total_batch_size / 128 * args.lr
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weigth Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    def create_optimizer(_, model):
        return torch.optim.AdamW(model.parameters())

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
            
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train_list,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            ch_names_list=train_ch_names_list,
            args=args
        )
        if args.output_dir:
            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val_list is not None:
            test_stats = evaluate(data_loader_val_list, model, device, log_writer, epoch, ch_names_list=val_ch_names_list, args=args)
            print(f"Validation loss of the network on the {sum([len(dataset) for dataset in dataset_val_list])} test EEG: {test_stats['loss']:.4f}")

            if log_writer is not None:
                log_writer.update(**test_stats, head="val/loss")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch, 'n_parameters': n_learnable_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, 'n_parameters': n_learnable_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
