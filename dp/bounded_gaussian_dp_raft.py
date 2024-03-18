#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from math import ceil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from opacus.validators import ModuleValidator
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import Subset
from opacus import GradSampleModule
# from EMA import EMA
import pickle
from math import ceil
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.accountants.analysis.rdp import compute_rdp

import sys
sys.path.append("tan/")
from src.opacus_augmented.privacy_engine_augmented import PrivacyEngineAugmented, CooldownSchedule, WarmupSchedule
from src.models.EMA_without_class import create_ema, update
from src.utils.utils import (init_distributed_mode,initialize_exp,bool_flag,accuracy,get_noise_from_bs,get_epochs_from_bs,print_params,)
from src.models.prepare_models import prepare_augmult_cifar

from utils.bounded_noise_rdp import get_ex_post_dp_tn_accounting_dict, get_ex_post_dp_rn_accounting_dict,compute_epsilon
from utils.dp_raft_utils import get_ds, ARCH_TO_NUM_FEATURES, DATASET_TO_SIZE, DATASET_TO_CLASSES

import warnings

warnings.simplefilter("ignore")

def get_num_params(net):
    return sum([p.numel() for p in net.parameters() if p.requires_grad])

def train(
    model,
    ema,
    train_loader,
    optimizer,
    epoch,
    max_nb_steps,
    device,
    privacy_engine,
    K,
    logger,
    losses,
    train_acc,
    epsilons,
    grad_sample_gradients_norms_per_epoch,
    test_loader,
    is_main_worker,
    args,
    norms2_before_sigma,
    nb_steps,
    prev_rdp,
    rdp_dict,
    c_range,
    orders
):
    """
    Trains the model for one epoch. If it is the last epoch, it will stop at max_nb_steps iterations.
    If the model is being shadowed for EMA, we update the model at every step.
    """
    # nb_steps = nb_steps
    num_params = get_num_params(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    if is_main_worker:print(f"steps_per_epoch:{steps_per_epoch}")
    losses_epoch, train_acc_epoch, grad_sample_norms = [], [], []
    nb_examples_epoch = 0
    max_physical_batch_size_with_augnentation = (args.max_physical_batch_size if K == 0 else args.max_physical_batch_size // K)
    test_acc_ema = 0
    train_acc_ema = 0
    with BatchMemoryManager(data_loader=train_loader,max_physical_batch_size=max_physical_batch_size_with_augnentation,optimizer=optimizer) as memory_safe_data_loader:
        for i, (images, target,_) in enumerate(memory_safe_data_loader):
            nb_examples_epoch+=len(images)
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            target = target.to(device)
            assert K == args.transform
            l = len(images)
            ##Using Augmentation multiplicity

            # compute output
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)
            losses_epoch.append(loss.item())
            train_acc_epoch.append(acc)

            loss.backward()
            is_updated = not (optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch

            ## Logging gradient statistics on the main worker
            if is_main_worker:
                per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if g.requires_grad]
                per_sample_norms = (torch.stack(per_param_norms, dim=1).norm(2, dim=1).cpu().tolist())
                grad_sample_norms += per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s

            ## Compute rdp budget for this step
            if args.rectification or args.truncation:
                rdp, _ = optimizer.step(rdp_dict,c_range,orders,random_project=args.random_project)
                if is_main_worker and is_updated:
                    prev_rdp += rdp
            else:
                optimizer.step(rdp_dict,c_range,orders,random_project=args.random_project)
            if is_updated:
                nb_steps += 1
                if ema:
                    update(model, ema, nb_steps)
                if is_main_worker:
                    losses.append(np.mean(losses_epoch))
                    train_acc.append(np.mean(train_acc_epoch))
                    grad_sample_gradients_norms_per_epoch.append(np.mean(grad_sample_norms))
                    losses_epoch, train_acc_epoch = [],[]
                    if nb_steps % args.freq_log == 0:
                        print(f"epoch:{epoch},step:{nb_steps}")
                        m2 = max(np.mean(norms2_before_sigma)-1/args.batch_size,0)
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "nb_steps":nb_steps,
                                    "train_acc": np.mean(train_acc[-args.freq_log :]),
                                    "loss": np.mean(losses[-args.freq_log :]),
                                    "grad_sample_gradients_norms": np.mean(grad_sample_norms),
                                    "grad_sample_gradients_norms_lowerC": np.mean(np.array(grad_sample_norms)<args.max_per_sample_grad_norm),
                                    #"norms2_before_sigma":list(norms2_before_sigma),
                                   # "grad_sample_gradients_norms_hist":list(np.histogram(grad_sample_norms,bins=np.arange(100), density=True)[0]),
                                }
                            )
                        )
                        norms2_before_sigma=[]
                        grad_sample_norms = []
                    if nb_steps % args.freq_log_val == 0:
                        test_acc_ema, train_acc_ema = (
                            test(ema, test_loader, train_loader, device)
                            if ema
                            else test(model, test_loader, train_loader, device)
                        )
                        print(f"epoch:{epoch},step:{nb_steps}")
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "test_acc_ema": test_acc_ema,
                                    "train_acc_ema": train_acc_ema,
                                }
                            )
                        )
                nb_examples_epoch=0
                if nb_steps >= max_nb_steps:
                    break
        if args.truncation or args.rectification:
            ## Compute epsilon for bounded gaussian mechanism
            epsilon,_ = compute_epsilon(prev_rdp, orders, len(train_loader.dataset))
        else:
            ## Compute epsilon for vanilla gaussian mechanism. Note that the calculation
            ## is tight since q=1.
            rdps = compute_rdp(
                    q=1,
                    noise_multiplier=args.ref_noise/(num_params**0.5),
                    steps=nb_steps,
                    orders=orders,
                )
            epsilon,_ = compute_epsilon(rdps, orders, len(train_loader.dataset))
        if is_main_worker:
            logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "epsilon":epsilon,
                                    "rdp":prev_rdp[6],
                                }
                            )
                        )
            epsilons.append(epsilon)
            if nb_steps % args.freq_log_val == 0:
                np.savetxt(f'results/cifar_vit_dp_raft_correct_new_3/ cifar_orig_dpraft_accurate_sigma_{args.ref_noise: .4f}_lr_{args.lr: .2f}_bound_{args.bound: .2f}_norm_{args.max_per_sample_grad_norm: .5f}_nb_steps_{nb_steps}.txt', np.asarray([epsilons[-1], train_acc_ema, test_acc_ema]))
            return nb_steps, norms2_before_sigma,prev_rdp,test_acc_ema,train_acc_ema
        else:
            return nb_steps, norms2_before_sigma,prev_rdp,0,0


def test(model, test_loader, train_loader, device):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    test_top1_acc = []
    train_top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            test_top1_acc.append(acc)

    test_top1_avg = np.mean(test_top1_acc)

    with torch.no_grad():
        for images, target, _ in train_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            # losses.append(loss.item())
            train_top1_acc.append(acc)
    train_top1_avg = np.mean(train_top1_acc)
    # print(f"\tTest set:"f"Loss: {np.mean(losses):.6f} "f"Acc: {top1_avg * 100:.6f} ")
    return (test_top1_avg, train_top1_avg)


def main():  ## for non poisson, divide bs by world size

    args = parse_args()
    init_distributed_mode(args)
    logger = initialize_exp(args)
    use_bias = False
    if args.dataset == "cifar10":
        model = nn.Linear(ARCH_TO_NUM_FEATURES[args.arch], 10, bias=use_bias).cuda()
    elif args.dataset.lower() == "cifar100":
        model = nn.Linear(ARCH_TO_NUM_FEATURES[args.arch], 100, bias=use_bias).cuda()
    else:
        model = nn.Linear(ARCH_TO_NUM_FEATURES[args.arch], 37, bias=use_bias).cuda()

    model.cuda()
    model.weight.data.zero_()
    if use_bias:
        model.bias.data.add(-10.)
    print_params(model)

    if args.multi_gpu:
        print("using multi GPU DPDDP")
        model = DPDDP(model)
    rank = args.global_rank
    is_main_worker = rank == 0
    weights = model.module.parameters() if args.multi_gpu else model.parameters()
    train_dataset, train_loader, test_loader = get_ds(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(weights, lr=args.lr, momentum=args.momentum, nesterov=False)
    # Creating the privacy engine
    privacy_engine = PrivacyEngineAugmented(
        GradSampleModule.GRAD_SAMPLERS, 
        rectification=args.rectification,
        truncation=args.truncation,
        bound=args.bound,
    )
    sigma = get_noise_from_bs(args.batch_size, args.ref_noise, args.ref_B)

    c_range = np.arange(-100, 100, 0.1)
    orders = list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    prev_rdp = np.zeros(len(orders))
    rdp_dict = None


    if args.rectification:
        rdp_dict = get_ex_post_dp_rn_accounting_dict(
            args.batch_size, 
            len(train_dataset), 
            [-args.bound, args.bound], 
            c_range, 
            sigma*args.max_per_sample_grad_norm, 
            args.max_per_sample_grad_norm,
        )
    elif args.truncation:
        rdp_dict = get_ex_post_dp_tn_accounting_dict(
            args.batch_size, 
            len(train_dataset), 
            [-args.bound, args.bound], 
            c_range, 
            sigma*args.max_per_sample_grad_norm, 
            args.max_per_sample_grad_norm,
        )

    ##We use our PrivacyEngine Augmented to take into accoung the eventual augmentation multiplicity
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        poisson_sampling=args.poisson_sampling,
        K=args.transform
    )
    ## Changes the grad samplers to work with augmentation multiplicity
    prepare_augmult_cifar(model,args.transform)
    ema = None
    # we create a shadow model
    print("shadowing de model with EMA")
    ema = create_ema(model)
    train_acc,test_acc,epsilons,losses,top1_accs,grad_sample_gradients_norms_per_step = (0, 0, [], [], [], [])
    norms2_before_sigma = []

    sched = None
    if args.sched == 2:
        sched = CooldownSchedule(optimizer, decay_step=40, decay_factor=2, lr=args.lr)
    elif args.sched == 1:
        sched = WarmupSchedule(optimizer, warmup_step=40, warmup_factor=2, lr=args.lr)

    E = get_epochs_from_bs(args.batch_size, args.ref_nb_steps, len(train_dataset))
    if is_main_worker: print(f"E:{E},sigma:{sigma}, BATCH_SIZE:{args.batch_size}, noise_multiplier:{sigma}, EPOCHS:{E}")
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    device = rank - gpus_per_node * (rank // gpus_per_node)
    nb_steps = 0
    for epoch in range(E):
        if sched is not None:
            sched.step()
        if nb_steps >= args.ref_nb_steps:
            break
        # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        # device = rank - gpus_per_node * (rank // gpus_per_node)
        nb_steps, norms2_before_sigma, prev_rdp, test_acc_ema, train_acc_ema = train(
            model,
            ema,
            train_loader,
            optimizer,
            epoch,
            args.ref_nb_steps,
            device,
            privacy_engine,
            args.transform,
            logger,
            losses,
            top1_accs,
            epsilons,
            grad_sample_gradients_norms_per_step,
            test_loader,
            is_main_worker,
            args,
            norms2_before_sigma,
            nb_steps,
            prev_rdp,
            rdp_dict,
            c_range,
            orders,
        )
        if is_main_worker:
            if epoch == E-1:
                test_acc_ema, train_acc_ema = (
                    test(ema, test_loader, train_loader, rank)
                    if ema
                    else test(model, test_loader, train_loader, rank)
                )
            print(f"epoch:{epoch}, Current loss:{losses[-1]:.2f},nb_steps:{nb_steps}, top1_acc of model (not ema){top1_accs[-1]:.2f},average gradient norm:{grad_sample_gradients_norms_per_step[-1]:.2f}, epsilon:{epsilons[-1]:.2f}")



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")

    parser.add_argument("--batch_size",default=256,type=int,help="Batch size for simulated training. It will automatically set the noise $\sigma$ s.t. $B/\sigma = B_{ref}/sigma_{ref}$")
    parser.add_argument("--max_physical_batch_size",default=256,type=int,help="max_physical_batch_size for BatchMemoryManager",)
    parser.add_argument("--WRN_depth",default=16,type=int)
    parser.add_argument("--WRN_k",default=4,type=int,help="k of resnet block",)

    parser.add_argument("--lr","--learning_rate",default=4,type=float,metavar="LR",help="initial learning rate",dest="lr",)

    parser.add_argument("--momentum",default=0,type=float,help="SGD momentum",)
    parser.add_argument("--experiment",default=0,type=int,help="seed for initializing training. ")
    parser.add_argument("-c","--max_per_sample_grad_norm",type=float,default=1,metavar="C",help="Clip per-sample gradients to this norm (default 1.0)",)
    parser.add_argument("--delta",type=float,default=1e-5,metavar="D",help="Target delta (default: 1e-5)",)
    parser.add_argument("--ref_noise",type=float,default=3,help="reference noise used with reference batch size and number of steps to create our physical constant",)
    parser.add_argument("--ref_B",type=int,default=4096,help="reference batch size used with reference noise and number of steps to create our physical constant",)
    parser.add_argument("--nb_groups",type=int,default=16,help="number of groups for the group norms",)
    parser.add_argument("--ref_nb_steps",default=2500,type=int,help="reference number of steps used with reference noise and batch size to create our physical constant",)
    parser.add_argument("--data_root",type=str,default="",help="Where CIFAR10 is/will be stored",)
    parser.add_argument("--dump_path",type=str,default="",help="Where results will be stored",)
    parser.add_argument("--transform",type=int,default=0,help="using augmentation multiplicity",)

    parser.add_argument("--freq_log", type=int, default=10, help="every each freq_log steps, we log",)

    parser.add_argument("--freq_log_val",type=int,default=5,help="every each freq_log steps, we log val and ema acc",)

    parser.add_argument("--poisson_sampling",type=bool_flag,default=True,help="using Poisson sampling",)


    parser.add_argument("--proportion",default=1,type=float,help="proportion of the training set to use for training",)

    parser.add_argument("--exp_name", type=str, default="bypass")

    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--order1", type=int, default=0)
    parser.add_argument("--order2", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)

    parser.add_argument("--rectification", type=bool_flag, default=False)
    parser.add_argument("--truncation", type=bool_flag, default=False)
    parser.add_argument("--bound", type=float, default=1.)

    parser.add_argument("--pretrain", type=bool_flag, default=False)
    parser.add_argument("--random_project", type=bool_flag, default=False)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--sched", type=int, default=1)

    parser.add_argument(
        "--arch",
        type=str,
        choices=list(ARCH_TO_NUM_FEATURES.keys()),
        default=list(ARCH_TO_NUM_FEATURES.keys())[0],
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="src/utils/extracted_features/",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()