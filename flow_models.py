import torch
from math import log
from torch.distributions import Normal
from utils import squeeze, unsqueeze
from blocks import FlowBlock
from utils import FinalScale
import logging

"""This package contains the flow models, e.g RealNVP and GLOW.
"""

class RealNVP(torch.nn.Module):
    def __init__(
        self,
        n_bins=256,
        base_input_shape=[3, 32, 32],
        num_scales=2,  # L
        num_step_of_flow=1,  # K
        num_resnet_blocks=8,
        
    ) -> None:
        super().__init__()
        c, h, w = base_input_shape
        self.flow_blocks = torch.nn.ModuleList(
            [
                FlowBlock(
                    base_input_shape=[
                        c * 2**i,
                        h // 2**i,
                        w // 2**i,
                    ],  # after each scale h and w area halved and added to channel, half of the channel is truncated before the next scale hence c * 2^i rather (c * 2 ^(2i))
                    num_step_of_flow=num_step_of_flow,
                    num_resnet_blocks=num_resnet_blocks,
                )
                for i in range(num_scales)
            ]
        )
        self.final_scale = FinalScale(
            base_input_shape=[
                c * 2 ** (num_scales),
                h // 2 ** (num_scales),
                w // 2 ** (num_scales),
            ],
        )  # the shape of this depends on howmany splits we are going to have

        self.normal = Normal(0.0, 1.0)
        self.n_pixels = c * h * w
        self.n_bins = n_bins

    def forward(self, x):
        x_splits = []
        total_logdet = 0

        for scale in self.flow_blocks:
            x, logdet = scale(
                x
            )  # x will be squeezed as the outcome of this e.g 3, 32, 32 --> 12, 16, 16
            total_logdet += logdet

            # split
            x, z = torch.chunk(
                x, chunks=2, dim=1
            )  # chunks x on channel dim e.g 12, 16, 16, --> (6, 6), 16, 16
            x_splits.append(z)

        x, logdet = self.final_scale(x)
        total_logdet += logdet
        # unsqueeze to get back to original shape
        # print("-----")
        for split in x_splits[::-1]:
            x = torch.concat([x, split], dim=1)
            x = unsqueeze(x)

        z_log_prob = torch.sum(self.normal.log_prob(x), dim=[1, 2, 3])
        log_prob = z_log_prob + total_logdet
        
        
        loss = -log(self.n_bins) * self.n_pixels
        loss = log_prob + loss
        final_loss = (-loss / (log(2.0) * self.n_pixels)).mean()
        return x, final_loss

    @torch.no_grad()
    def reverse(self, z, device):
        z = self.final_scale.reverse(z)
        for scale in self.flow_blocks[::-1]:
            # sample from a normal
            x = self.normal.sample(sample_shape=z.shape).to(device)
            z = torch.concat([z, x], dim=1)
            z = scale.reverse(z)
        return z

    def sample(self, device, num_samples=1, z_base_sample=None):
        """
        if z_samples are provided, use that as the base tensor, if not sample from normal.
        z_samples can be provided in order to see how the model generates images over the course of training.
        """
        if z_base_sample is None:
            num_channels = self.final_scale.base_input_shape[0]
            height, width = self.final_scale.base_input_shape[1:]
            sample_size = [num_samples, num_channels, height, width]
            z_base_sample = self.normal.sample(sample_shape=sample_size)
        else:
            # z_sample zero'th dimention is the sample size
            assert (
                z_base_sample.shape[1] == self.final_scale.base_input_shape[0]
                and z_base_sample.shape[2] == self.final_scale.base_input_shape[1]
                and z_base_sample.shape[3] == self.final_scale.base_input_shape[2]
            )
        z_base_sample = z_base_sample.to(device)
        generated_image = self.reverse(z_base_sample, device=device)
        return generated_image
