from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List

from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from model.hypernetwork import HyperNetwork


class HyperDM(nn.Module):

    def __init__(self, primary_net: nn.Module, hyper_net_dims: List[int],
                 diffusion: GaussianDiffusion):
        """
        Initialize the hyper-diffusion model class.

        :param primary_net: diffusion model
        :param hyper_net_dims: hyper-network layer dimensions
        :param diffusion: Gaussian diffusion process
        """
        super().__init__()
        self.primary_net = primary_net
        self.primary_params = sum(p.numel()
                                  for p in self.primary_net.parameters())
        # Freeze primary network weights
        for param in primary_net.parameters():
            param.requires_grad = False

        hyper_net_dims.append(self.primary_params)
        self.hyper_net_input_dim = hyper_net_dims[0]
        self.hyper_net = HyperNetwork(hyper_net_dims)
        self.hyper_net = self.hyper_net
        self.hyper_params = sum(p.numel() for p in self.hyper_net.parameters())

        self.diffusion = diffusion

    def print_stats(self):
        print("# of params (primary):", self.primary_params)
        print("# of params (hyper):", self.hyper_params)

    def get_mean_variance(self,
                          M: int,
                          N: int,
                          condition: torch.Tensor,
                          device=None,
                          progress=False):
        """
        Sample the predictive distribution mean and variance. In the paper this is \mathbb{E}_{\hat{x} \sim p(x|y,\phi)}\[\hat{x}\] and \text{Var}_{\hat{x} \sim p(x|y,\phi)}\[\hat{x}\].

        :param M: number of network weights to sample
        :param N: number of predictions to sample per network weight
        :param condition: input condition to sample with
        :param device: device to run on
        :return: mean and variance of the predictive distribution
        """
        _, C, H, W = condition.shape
        mean = torch.zeros([M, C, H, W])
        var = torch.zeros([M, C, H, W])
        Ms = tqdm(range(M)) if progress else range(M)
        for i in Ms:
            net = self.sample_network(device)

            # y = condition.repeat(N, 1, 1, 1).to(device)
            # with torch.no_grad():
            #     preds = self.diffusion.ddim_sample_loop(net,
            #                                             y.shape,
            #                                             model_kwargs={"y": y},
            #                                             device=device)
            
            preds_list = []
            for _ in range(N):
                with torch.no_grad():
                    pred_sample = self.diffusion.ddim_sample_loop(
                        net,
                        condition.shape, # y.shape -> condition.shape
                        model_kwargs={"y": condition}, # y -> condition
                        device=device
                    )
                    preds_list.append(pred_sample)
                    
            preds = torch.cat(preds_list, dim=0)        
            mean[i] = preds.mean(dim=0)
            var[i] = preds.var(dim=0)
        return mean, var

    def sample_network(self, device=None):
        """
        Sample a network with weights from a Bayesian hyper-network.

        :param device: device to run on
        :return: callable primary network with weights sampled from the hyper-network
        """
        # Sample noise
        z = torch.randn(self.hyper_net_input_dim).to(device)

        # Compute weights
        weights = self.hyper_net(z)
        weights = weights.ravel()
        assert (
            len(weights) == self.primary_params
        ), f"# of generated weights {len(weights)} must match # of parameters {self.primary_params}!"
        # Format weights
        i = 0
        weight_dict = dict()
        for k, v in self.primary_net.state_dict().items():
            weight_dict[k] = weights[i:i + v.numel()].view(v.shape)
            i += v.numel()

        return partial(torch.func.functional_call, self.primary_net, weight_dict)
