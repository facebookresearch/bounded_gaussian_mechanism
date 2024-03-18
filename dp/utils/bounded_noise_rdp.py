#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy import special
from scipy.stats import norm
from opacus.accountants.analysis.rdp import get_privacy_spent

def _compute_rdp_rn_no_subsampling_exact_left_to_right(alpha,a,c,mu,sigma):
    """
        Calculate D_{alpha}(N^R(c,sigma^2,[-a,a])||N^R(c+mu,sigma^2,[-a,a]))
    """
    dist = norm(0,1)
    rec_rdp = np.exp((alpha**2 - alpha) * (mu ** 2) / (2 * (sigma ** 2)))
    rec_rdp *= dist.cdf((a-c-(1-alpha)*mu)/sigma) - dist.cdf((-a-c-(1-alpha)*mu)/sigma)
    rec_rdp += ((dist.cdf((-a-c)/sigma)/dist.cdf((-a-c-mu)/sigma)) ** alpha) * dist.cdf((-a-c-mu)/sigma)
    rec_rdp += ((dist.cdf((c-a)/sigma)/dist.cdf((c+mu-a)/sigma)) ** alpha) * dist.cdf((c+mu-a)/sigma)
    rec_rdp = 1/(alpha-1)*np.log(rec_rdp)
    return np.clip(np.nan_to_num(rec_rdp),a_min=1e-20,a_max=None)

def _compute_rdp_rn_no_subsampling_exact_right_to_left(alpha,a,c,mu,sigma):
    """
        Calculate D_{alpha}(N^R(c+mu,sigma^2,[-a,a])||N^R(c,sigma^2,[-a,a]))
    """
    dist = norm(0,1)
    rec_rdp = np.exp((alpha**2 - alpha) * (mu ** 2) / (2 * (sigma ** 2)))
    rec_rdp *= dist.cdf((a-c-(alpha)*mu)/sigma) - dist.cdf((-a-c-(alpha)*mu)/sigma)
    rec_rdp += dist.cdf((-a-c)/sigma) * ((dist.cdf((-a-c-mu)/sigma)/dist.cdf((-a-c)/sigma)) ** alpha)
    rec_rdp += dist.cdf((c-a)/sigma) * ((dist.cdf((c+mu-a)/sigma)/dist.cdf((c-a)/sigma)) ** alpha)
    rec_rdp = 1/(alpha-1)*np.log(rec_rdp)
    return np.clip(np.nan_to_num(rec_rdp),a_min=1e-20,a_max=None)

def _compute_rdp_rn_no_subsampling_exact(alpha,a,c,mu,sigma):
    left_to_right = _compute_rdp_rn_no_subsampling_exact_left_to_right(alpha,a,c,mu,sigma)
    right_to_left = _compute_rdp_rn_no_subsampling_exact_right_to_left(alpha,a,c,mu,sigma)
    return np.maximum(left_to_right, right_to_left)

def _compute_rdp_tn_no_subsampling_exact_left_to_right(alpha,a,c,mu,sigma):
    """
        Calculate D_{alpha}(N^T(c,sigma^2,[-a,a])||N^T(c+mu,sigma^2,[-a,a]))
    """
    dist = norm.cdf
    out = alpha*(mu**2) / (2*(sigma**2))
    out += np.log(dist((a-c-mu)/sigma)-dist((-a-c-mu)/sigma))
    out -= np.log(dist((a-c)/sigma)-dist((-a-c)/sigma))
    out += 1 / (alpha - 1) * np.log(dist((a-c-(1-alpha)*mu)/sigma)-dist((-a-c-(1-alpha)*mu)/sigma))
    out -= 1 / (alpha - 1) * np.log(dist((a-c)/sigma)-dist((-a-c)/sigma))
    out = np.clip(np.nan_to_num(out),a_min=1e-20,a_max=None)
    return out

def _compute_rdp_tn_no_subsampling_exact_right_to_left(alpha,a,c,mu,sigma):
    """
        Calculate D_{alpha}(N^T(c+mu,sigma^2,[-a,a])||N^T(c,sigma^2,[-a,a]))
    """
    dist = norm.cdf
    out = alpha*(mu**2) / (2*(sigma**2))
    out -= np.log(dist((a-c-mu)/sigma)-dist((-a-c-mu)/sigma))
    out += np.log(dist((a-c)/sigma)-dist((-a-c)/sigma))
    out += 1 / (alpha - 1) * np.log(dist((a-c-alpha*mu)/sigma)-dist((-a-c-alpha*mu)/sigma))
    out -= 1 / (alpha - 1) * np.log(dist((a-c-mu)/sigma)-dist((-a-c-mu)/sigma))
    out = np.clip(np.nan_to_num(out),a_min=1e-20,a_max=None)
    return out

def _compute_rdp_no_subsampling_exact(alpha,a,c,mu,sigma):
    left_to_right = _compute_rdp_tn_no_subsampling_exact_left_to_right(alpha,a,c,mu,sigma)
    right_to_left = _compute_rdp_tn_no_subsampling_exact_right_to_left(alpha,a,c,mu,sigma)
    return np.maximum(left_to_right, right_to_left)

def _compute_rdp_sign_exact_left_to_right(alpha,a,c,mu,sigma):
    dist = norm(0,1)
    sign_rdp = 0
    sign_rdp += (dist.cdf(c/sigma) ** alpha) * (dist.cdf((c+mu)/sigma) ** (1-alpha))
    sign_rdp += (dist.cdf(-c/sigma) ** alpha) * (dist.cdf(-(c+mu)/sigma) ** (1-alpha))
    sign_rdp = 1/(alpha-1)*np.log(sign_rdp)
    return np.clip(np.nan_to_num(sign_rdp),a_min=1e-20,a_max=None)

def _compute_rdp_sign_exact_right_to_left(alpha,a,c,mu,sigma):
    dist = norm(0,1)
    sign_rdp = 0
    sign_rdp += (dist.cdf(c/sigma) ** (1-alpha)) * (dist.cdf((c+mu)/sigma) ** (alpha))
    sign_rdp += (dist.cdf(-c/sigma) ** (1-alpha)) * (dist.cdf(-(c+mu)/sigma) ** (alpha))
    sign_rdp = 1/(alpha-1)*np.log(sign_rdp)
    return np.clip(np.nan_to_num(sign_rdp),a_min=1e-20,a_max=None)


def _compute_rdp_sign_exact(alpha,a,c,mu,sigma):
    left_to_right = _compute_rdp_sign_exact_left_to_right(alpha,a,c,mu,sigma)
    right_to_left = _compute_rdp_sign_exact_right_to_left(alpha,a,c,mu,sigma)
    return np.maximum(left_to_right, right_to_left)


def compute_rdp_tn_one_step_approx(q, noise_multiplier, norm_clip, c_range, bound, orders):
    if isinstance(orders, float):
        rdp = _compute_rdp_no_subsampling_exact(orders, bound, c_range, norm_clip, noise_multiplier)
    else:
        rdp = np.array([_compute_rdp_no_subsampling_exact(order, bound, c_range, norm_clip, noise_multiplier) for order in orders])
    return np.clip(rdp,a_min=1e-15,a_max=None)

def compute_rdp_rn_one_step_approx(q, noise_multiplier, norm_clip, c_range, bound, orders):
    if isinstance(orders, float):
        rdp = _compute_rdp_rn_no_subsampling_exact(orders, bound, c_range, norm_clip, noise_multiplier)
    else:
        rdp = np.array([_compute_rdp_rn_no_subsampling_exact(order, bound, c_range, norm_clip, noise_multiplier) for order in orders])
    return np.clip(rdp,a_min=1e-15,a_max=None)

def compute_rdp_sign_one_step_approx(q, noise_multiplier, norm_clip, c_range, bound, orders):
    if isinstance(orders, float):
        rdp = _compute_rdp_sign_exact(orders, bound, c_range, norm_clip, noise_multiplier)
    else:
        rdp = np.array([_compute_rdp_sign_exact(order, bound, c_range, norm_clip, noise_multiplier) for order in orders])
    return np.clip(rdp,a_min=1e-15,a_max=None)

def get_ex_post_dp_tn_accounting_dict(batch_size, num_examples, alphabets_range, c_range, sigma, linf_norm_clip):
    q = 1
    # theta = np.asarray(jnp.concatenate([p.flatten() for p in theta]).flatten())
    neg_bound, bound = alphabets_range
    orders = list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64))

    return compute_rdp_tn_one_step_approx(q, sigma, linf_norm_clip, c_range, bound, orders)

def get_ex_post_dp_rn_accounting_dict(batch_size, num_examples, alphabets_range, c_range, sigma, linf_norm_clip):
    q = 1
    # theta = np.asarray(jnp.concatenate([p.flatten() for p in theta]).flatten())
    neg_bound, bound = alphabets_range
    orders = list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64))

    return compute_rdp_rn_one_step_approx(q, sigma, linf_norm_clip, c_range, bound, orders)

def get_ex_post_dp_sign_accounting_dict(batch_size, num_examples, alphabets_range, c_range, sigma, linf_norm_clip):
    q = 1
    # theta = np.asarray(jnp.concatenate([p.flatten() for p in theta]).flatten())
    neg_bound, bound = alphabets_range
    orders = list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64))

    return compute_rdp_sign_one_step_approx(q, sigma, linf_norm_clip, c_range, bound, orders)

def compute_epsilon(rdp_const, orders, num_examples, target_delta=1e-5):
    if num_examples * target_delta > 1.:
        warnings.warn('Your delta might be too high.')
    eps, _ = get_privacy_spent(orders=orders, rdp=rdp_const, delta=target_delta)
    return eps, rdp_const
