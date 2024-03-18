#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import jax
import numpy as np


import jax.numpy as jnp
import jax.random as jnr

from jax import grad
from jax.example_libraries import optimizers
from jax.tree_util import tree_flatten, tree_unflatten

import math

import sys
sys.path.append("bounding_data_reconstruction/")

import accountant
import datasets
import trainer
import utils
import time


def batch_predict(predict, params, images, batch_size):
    num_images = images.shape[0]
    num_batches = int(math.ceil(float(num_images) / float(batch_size)))
    predictions = []
    for i in range(num_batches):
        lower = i * batch_size
        upper = min((i+1) * batch_size, num_images)
        predictions.append(predict(params, images[lower:upper]))
    return jnp.concatenate(predictions)

def main(cfg):

    # set up random number generator:
    logging.info(f"Running using JAX {jax.__version__}...")
    rng = jnr.PRNGKey(int(time.time()))

    # create dataloader for MNIST dataset:
    if cfg.dataset.startswith("cifar100"):
        num_channels = 3
        image_size = 32
        train_images, train_labels, test_images, test_labels = datasets.cifar100_extractor(cfg.data_path)
    else:
        num_channels = 3
        image_size = 32
        train_images, train_labels, test_images, test_labels = datasets.cifar_extractor(cfg.data_path)
    logging.info(f"Training set max variance: %.4f" % train_images.var(0).max())
    
    num_samples, d = train_images.shape
    num_labels = train_labels.shape[1]
    
    data_stream, num_batches = datasets.get_datastream(
        train_images, train_labels, cfg.batch_size
    )
    batches = data_stream()

    # set up model:

    input_shape = (-1, d)
    init_params, predict = utils.get_model(rng, cfg.model, input_shape, num_labels)
    num_params = sum(p.size for p in tree_flatten(init_params)[0])

    # create optimizer:
    if cfg.optimizer == "sgd":
        opt_init, opt_update, get_params = optimizers.momentum(
            cfg.step_size, cfg.momentum_mass
        )
    elif cfg.optimizer == "adam":
        opt_init, opt_update, get_params = optimizers.adam(cfg.step_size)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    opt_state = opt_init(init_params)

    # get loss function and update functions:
    loss = trainer.get_loss_func(predict)
    grad_func = trainer.get_grad_func(loss, norm_clip=cfg.norm_clip, soft_clip=True, linf_clip=cfg.linf_clip, linf_norm_clip=cfg.norm_clip)
    update = trainer.get_update_func(
        get_params, grad_func, opt_update, norm_clip=cfg.norm_clip,
        reshape=cfg.model.startswith("cnn"), rectification=cfg.rectification,
        quantization=cfg.quantization, truncation=cfg.truncation,lb=cfg.lb, ub=cfg.ub,
        num_of_bits = cfg.num_of_bits
    )

    norm_clip_for_fil = cfg.norm_clip if (cfg.norm_clip > 0) else 1
    # get function that computes the Jacobian norms for privacy accounting:
    gelu_approx = 1.115
    fil_accountant = accountant.get_grad_jacobian_trace_func(
        grad_func, get_params, reshape=cfg.model.startswith("cnn"),
        label_privacy=cfg.label_privacy, rectification=cfg.rectification,
        quantization=cfg.quantization, truncation=cfg.truncation,lb=cfg.lb, ub=cfg.ub,
        num_of_bits = cfg.num_of_bits,sigma=cfg.sigma*norm_clip_for_fil
    )
    
    # compute subsampling factor
    if cfg.sigma > 0:
        eps = math.sqrt(2 * math.log(1.25 / cfg.delta)) * 2 * gelu_approx / cfg.sigma
        q = float(cfg.batch_size) / num_samples
        subsampling_factor = q / (q + (1-q) * math.exp(-eps))
    else:
        subsampling_factor = 0
    logging.info(f"Subsampling rate is {q:.4f}")
    logging.info(f"Subsampling factor is {subsampling_factor:.4f}")

    # train the model:
    logging.info(f"Training {cfg.model} model with {num_params} parameters using {cfg.optimizer}...")
    etas_squared = jnp.zeros((cfg.num_epochs, train_images.shape[0]))
    train_accs = jnp.zeros(cfg.num_epochs)
    test_accs = jnp.zeros(cfg.num_epochs)
    num_iters = 0
    for epoch in range(cfg.num_epochs):

        # perform full training sweep through the data:
        itercount = itertools.count()
        if epoch > 0:
            etas_squared = etas_squared.at[epoch].set(etas_squared[epoch-1])

        for batch_counter in range(num_batches):

            # get next batch:
            num_iters += 1
            i = next(itercount)
            rng = jnr.fold_in(rng, i)
            images, labels, batch_idx = next(batches)
            batch = (images, labels)

            # update privacy loss:
            if cfg.sigma > 0 and cfg.do_accounting:
                if cfg.truncation or cfg.rectification:
                    etas_batch = fil_accountant(rng, opt_state, batch)
                else:
                    etas_batch = fil_accountant(rng, opt_state, batch) / cfg.sigma / norm_clip_for_fil
                etas_squared = etas_squared.at[epoch, batch_idx].add(
                    subsampling_factor * jnp.power(etas_batch, 2), unique_indices=True
                )

            # perform private parameter update:
            opt_state, noisy_grad_unflatten = update(i, rng, opt_state, batch, cfg.sigma, cfg.weight_decay)
            

        # measure training and test accuracy, and average privacy loss:
        params = get_params(opt_state)
        spectral_norm = utils.estimate_spectral_norm(lambda x: predict(params, x), input_shape)
        train_predictions = batch_predict(predict, params, train_images, cfg.batch_size)
        test_predictions = batch_predict(predict, params, test_images, cfg.batch_size)
        train_accuracy = utils.accuracy(train_predictions, train_labels)
        test_accuracy = utils.accuracy(test_predictions, test_labels)
        train_accs = train_accs.at[epoch].set(train_accuracy)
        test_accs = test_accs.at[epoch].set(test_accuracy)
        params, _ = tree_flatten(params)
        params_norm = math.sqrt(sum([jnp.power(p, 2).sum() for p in params]))
        if cfg.sigma > 0 and cfg.do_accounting:
            median_eta = jnp.median(jnp.sqrt(etas_squared[epoch]))
            max_eta = jnp.sqrt(etas_squared[epoch]).max()
            delta = 1e-5

        # print out progress:
        logging.info(f"Epoch {epoch + 1}:")
        logging.info(f" -> training accuracy = {train_accuracy:.4f}")
        logging.info(f" -> test accuracy = {test_accuracy:.4f}")
        logging.info(f" -> parameter norm = {params_norm:.4f}, spectral norm = {spectral_norm:.4f}")
        if cfg.sigma > 0 and cfg.do_accounting:
            logging.info(f" -> Median FIL privacy loss = {median_eta:.4f}")
            logging.info(f" -> Max FIL privacy loss = {max_eta:.4f}")

    etas = jnp.sqrt(etas_squared) if cfg.sigma > 0 and cfg.do_accounting else float("inf")
    np.savetxt(f'results/cifar_linf_rectification_subsample_beit_1/ cifar_rec_sigma_{cfg.sigma: .4f}_lr_{cfg.step_size: .2f}_bound_{cfg.ub: .2f}_norm_{cfg.norm_clip: .4f}.txt', np.asarray([median_eta, max_eta, train_accuracy, test_accuracy]))
    return etas, train_accs, test_accs


# run all the things:
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MNIST training with FIL.")
    parser.add_argument("--dataset", default="cifar10", type=str,
        help="name of dataset")
    parser.add_argument("--data_path", type=str,
        help="folder in which to store data")
    parser.add_argument("--model", default="linear", type=str,
        help="model used")
    parser.add_argument("--do_accounting", action='store_false',
        help="whether we do accounting or not")
    parser.add_argument("--label_privacy", action='store_true',
        help="whether we do label privacy or not")
    parser.add_argument("--binary", action='store_true',
        help="whether the task is binary or not")
    parser.add_argument("--pca_dims", default=0, type=int,
        help="pca dim")
    parser.add_argument("--batch_size", default=200, type=int,
        help="batch size")
    parser.add_argument("--momentum_mass", default=0.9, type=float,
        help="momentum mass")
    parser.add_argument("--num_epochs", default=150, type=int,
        help="num epochs")
    parser.add_argument("--optimizer", default="sgd", type=str,
        help="optimizer")
    parser.add_argument("--step_size", default=0.03, type=float,
        help="step size")
    parser.add_argument("--weight_decay", default=0, type=float,
        help="weight decay")
    parser.add_argument("--sigma", default=0.5, type=float,
        help="Gaussian noise multiplier")
    parser.add_argument("--norm_clip", default=2, type=float,
        help="clipping norm")
    parser.add_argument("--delta", default=1e-10, type=float,
        help="delta")
    parser.add_argument("--lb", default=-0.5, type=float,
        help="lower bound for bounded interval")
    parser.add_argument("--ub", default=0.5, type=float,
        help="upper bound for bounded interval")
    parser.add_argument("--rectification", action='store_true',
        help="whether we add rectification to gaussian")
    parser.add_argument("--quantization", action='store_true',
        help="whether we discretize the gradient")
    parser.add_argument("--num_of_bits", default=1, type=int,
        help="number of quantization bits")
    parser.add_argument("--truncation", action='store_true',
        help="whether we apply truncated normal")
    parser.add_argument("--linf_clip", action='store_true',
        help="whether we apply linf clipping")
    args = parser.parse_args()

    logging.info(f"Stats for Rec {args.rectification}, Quant {args.quantization} with {args.num_of_bits: .1f}, Trunc {args.truncation} on sigma = {args.sigma: .4f}, norm = {args.norm_clip: .4f}, bound = {args.ub: .4f}, and step size = {args.step_size: .4f}")

    main(args)

    logging.info(f"Stats for Rec {args.rectification}, Quant {args.quantization} with {args.num_of_bits: .1f}, Trunc {args.truncation} on sigma = {args.sigma: .4f}, norm = {args.norm_clip: .4f}, bound = {args.ub: .4f}, and step size = {args.step_size: .4f}")
