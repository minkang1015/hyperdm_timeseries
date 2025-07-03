"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th

def nll_loss(x, mean, var):
    """
    Negative log-likelihood loss for Gaussian.
    :param x: [B, ...] target
    :param mean: [B, ...] predicted mean
    :param var: [B, ...] predicted variance (must be >0)
    :return: [B, ...] NLL loss
    """
    # Add small epsilon for numerical stability
    eps = 1e-8
    var = var.clamp(min=eps)
    return 0.5 * (th.log(2 * th.pi * var) + ((x - mean) ** 2) / var)


def crps_loss(x, means, log_scales):
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian.
    :param x: [B, ...] target
    :param means: [B, ...] predicted mean
    :param log_scales: [B, ...] predicted log std
    :return: [B, ...] CRPS loss
    """
    # See: https://www.stat.berkeley.edu/~aldous/157/Papers/crps.pdf
    # For Gaussian: CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    # where z = (x - mu) / sigma, Phi = cdf, phi = pdf
    std = th.exp(log_scales)
    z = (x - means) / std
    sqrt_pi_inv = 1.0 / th.sqrt(th.tensor(th.pi, device=x.device, dtype=x.dtype))
    # Standard normal cdf and pdf
    Phi = 0.5 * (1 + th.erf(z / th.sqrt(th.tensor(2.0, device=x.device, dtype=x.dtype))))
    phi = th.exp(-0.5 * z ** 2) / th.sqrt(th.tensor(2 * th.pi, device=x.device, dtype=x.dtype))
    crps = std * (z * (2 * Phi - 1) + 2 * phi - sqrt_pi_inv)
    return crps


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
