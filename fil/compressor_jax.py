#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import jax.scipy.stats.norm as jnorm

def rectify(alphabets_range, theta):
    a,b = alphabets_range
    return jnp.clip(theta, a_min=a, a_max=b)

def generate_alphabets_evenly(alphabets_range, num_of_bits):
    a,b = alphabets_range
    return jnp.linspace(a,b,2 ** num_of_bits)

def quantize(alphabets, theta):
    theta_ext = jnp.expand_dims(theta, axis=0).repeat(len(alphabets),0)
    theta_dim = len(theta.shape)
    alphabets_ext = jnp.broadcast_to(jnp.expand_dims(alphabets, axis=np.linspace(1,theta_dim,theta_dim).astype(int)),theta_ext.shape)
    diff = jnp.abs(theta_ext - alphabets_ext)
    diff_min = jnp.argmin(diff, axis=0)
    return alphabets[diff_min]

def truncate(alphabets_range, rng, theta, sigma):
    a,b = alphabets_range
    return jnr.truncated_normal(rng,(a-theta) / sigma, (b-theta) / sigma) * sigma + theta

def calculate_rectification(bounds,theta, sigma=1):
    a,b = bounds
    dist = jnorm
    fisher = 0
    fisher += jnp.nan_to_num((dist.pdf((theta - b) / sigma) / sigma) ** 2 / dist.cdf((theta - b) / sigma))
    fisher += jnp.nan_to_num((dist.pdf((a - theta) / sigma) / sigma) ** 2 / dist.cdf((a - theta) / sigma))
    fisher += ((dist.cdf((b - theta) / sigma) - dist.cdf((a - theta) / sigma))) / sigma ** 2
    fisher += (dist.pdf((a - theta) / sigma) * (a - theta) / sigma - dist.pdf((b-theta) / sigma) * (b-theta) / sigma) / sigma ** 2
    # if (fisher * sigma ** 2 > 1).sum() > 0:
    #     print('Warning! Rectification has no amplification. Incorrect calculation of Fisher Information might encountered.')
    return fisher

def calculate_quantization(alphabets, theta, sigma=1):
    if len(alphabets) < 2:
        raise Exception("Alphabets length has to be at least 2.")
    fisher = 0
    
    dist = jnorm
    
    p_first = dist.cdf(((alphabets[0]+alphabets[1])/2 - theta) / sigma)
    dp_dtheta_first = dist.pdf(((alphabets[0]+alphabets[1])/2 - theta) / sigma) / sigma
#     if p_first > 1e-20:
    fisher += jnp.nan_to_num(1 / p_first * dp_dtheta_first ** 2)
    
    p_last = dist.cdf((theta - (alphabets[-2]+alphabets[-1])/2) / sigma)
    dp_dtheta_last = dist.pdf((theta - (alphabets[-2]+alphabets[-1])/2) / sigma) / sigma
#     if p_last > 1e-20:
    fisher += jnp.nan_to_num(1 / p_last * dp_dtheta_last ** 2)
    
    for i,a in enumerate(alphabets[1:-1]):
        p = dist.cdf(((alphabets[i+1]+alphabets[i+2])/2 - theta) / sigma) -  dist.cdf(((alphabets[i]+alphabets[i+1])/2 - theta) / sigma)
        dp_dtheta = (dist.pdf(((alphabets[i]+alphabets[i+1])/2 - theta) / sigma) -  dist.pdf(((alphabets[i+1]+alphabets[i+2])/2 - theta) / sigma)) / sigma
#         if p > 1e-20:
        fisher += jnp.nan_to_num(1 / p * dp_dtheta ** 2, nan=(((alphabets[i]+alphabets[i+1])/2 - theta) / sigma)**2*dist.pdf(((alphabets[i]+alphabets[i+1])/2 - theta) / sigma)/sigma**2)

#     print(fisher)    
    return fisher

def calculate_truncation(bounds,theta,sigma=1):
    a,b = bounds
    dist = jnorm
    fisher = 0
    fisher += 1
    fisher += jnp.nan_to_num((dist.pdf((a - theta) / sigma) * (a - theta) / sigma - dist.pdf((b-theta) / sigma) * (b-theta) / sigma) / (dist.cdf((b - theta) / sigma) - dist.cdf((a - theta) / sigma)), nan=((a - theta) / sigma)**2-1)
    fisher -= jnp.nan_to_num(((dist.pdf((b - theta) / sigma) - dist.pdf((a - theta) / sigma)) / (dist.cdf((b - theta) / sigma) - dist.cdf((a - theta) / sigma))) ** 2,nan=((a - theta) / sigma)**2)
    fisher /= sigma ** 2
    return jnp.clip(fisher,a_min=0)
