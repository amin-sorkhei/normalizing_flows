import torch
from math import log
from torch.distributions import Normal
from normalizing_flows.utils import squeeze, unsqueeze
from normalizing_flows.blocks import RealNVPBlock, GlowBlock
from normalizing_flows.utils import FinalScale, StepOfGlow
import logging


"""This package contains the flow models, e.g RealNVP and GLOW.
"""


class RealNVP(torch.nn.Module):
    def __init__(
        self,
        n_bits=8,
        base_input_shape=[3, 32, 32],
        num_scales=2,  # L
        num_step_of_flow=1,  # K
        num_resnet_blocks=8,
    ) -> None:
        super().__init__()
        c, h, w = base_input_shape
        self.flow_blocks = torch.nn.ModuleList(
            [
                RealNVPBlock(
                    base_input_shape=[
                        c * 2**i,
                        h // 2**i,
                        w // 2**i,
                    ],  # after each scale h and w area halved and added to channel, half of the channel is truncated before the next scale hence c * 2^i
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
        self.output_shape = self.final_scale.base_input_shape
        self.normal = Normal(0.0, 1.0)
        self.n_pixels = c * h * w
        self.n_bins = 2**n_bits

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
        for split in x_splits[::-1]:
            x = torch.concat([x, split], dim=1)
            x = unsqueeze(x)

        z_log_prob = torch.sum(self.normal.log_prob(x), dim=[1, 2, 3])
        log_prob = z_log_prob + total_logdet

        loss = (
            -log(self.n_bins) * self.n_pixels
        )  # log det implicit transformation that converts [0, 255] values to [0, 1]
        loss = log_prob + loss
        final_loss = (-loss / (log(2.0) * self.n_pixels)).mean()  # per pixed loss
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

    @torch.no_grad()
    def sample(self, device, num_samples=1, z_base_sample=None):
        """
        if z_samples are provided, use that as the base tensor, if not sample from normal.
        z_samples can be provided in order to see how the model generates images over the course of training.
        """
        if z_base_sample is None:
            num_channels, height, width = self.output_shape
            sample_size = [num_samples, num_channels, height, width]
            z_base_sample = self.normal.sample(sample_shape=sample_size)
        else:
            # z_sample zero'th dimention is the sample size
            assert list(z_base_sample.shape[1:]) == self.output_shape
        z_base_sample = z_base_sample.to(device)
        generated_image = self.reverse(z_base_sample, device=device)
        return generated_image


class Glow(torch.nn.Module):
    """implements a GLOW model.
    this is composed of L-1 blocks
    a block is composed of **K** steps of flow, where each step of flow is
    composed of
    1. Actnorm 2. Invertible 1x1 Convolution 3. Affine Coupling
    """

    def __init__(
        self,
        K,
        L,
        base_input_shape,
        num_resnet_filters=512,
        n_bits=256,
    ) -> None:
        """
        K: number of steps of flow
        L: number of blocks
        """
        super().__init__()
        self.normal = Normal(0.0, 1.0)
        c, h, w = base_input_shape
        self.n_pixels = c * h * w
        self.n_bins = 2**n_bits
        self.blocks = torch.nn.ModuleList()
        input_channels = c
        self.output_shape = [c * 2 ** (L + 1), h // 2**L, w // 2**L]
        for i in range(L - 1):
            self.blocks.append(GlowBlock(K=K, num_channels=input_channels))
            input_channels *= 2

        self.blocks.append(
            GlowBlock(
                is_final_block=True,
                K=K,
                num_channels=input_channels,
                num_filters=num_resnet_filters,
            )
        )

    def forward(self, x):
        total_logdet = 0
        total_log_prob = 0
        splits = []
        for block in self.blocks:
            x, logdet, z, z_log_prob = block(x)
            total_logdet += logdet
            splits.append(z if z is not None else x)

            total_log_prob += (
                z_log_prob
            )

        log_prob = total_log_prob + total_logdet
        loss = -log(self.n_bins) * self.n_pixels
        loss = log_prob + loss
        final_loss = (-loss / (log(2.0) * self.n_pixels)).mean()
        return (
            x,
            final_loss,
            (total_log_prob / (log(2) * self.n_pixels)).mean(),
            (total_logdet / (log(2) * self.n_pixels)).mean(),
        )

    @torch.no_grad()
    def reverse(self, z_l, device, fixed_sample, seed, T=1.0):
        for block in self.blocks[::-1]:
            z_l = block.reverse(
                T=T, z_l=z_l, device=device, fixed_sample=fixed_sample, seed=seed
            )
        return z_l

    @torch.no_grad()
    def sample(self, device, fixed_sample: bool, seed=None, num_samples=1, T=1.0):
        """
        fixed_sample (bool): fixed sample helps with reproducibiliy of images during training. is fixed_sample is enabled
        all normal distributions will use the same seed for sampling.
        if z_samples are provided, use that as the base tensor, if not sample from normal.
        z_samples can be provided in order to see how the model generates images over the course of training.
        """
        logging.info(f"using temperature {T} for sampling")
        num_channels, height, width = self.output_shape
        sample_size = [num_samples, num_channels, height, width]
        if fixed_sample is True:
            assert seed, "when fixed sample is required, seed needs to be provided"
            torch.manual_seed(seed)
        elif fixed_sample is False and seed is not None:
            logging.info(f"fixed_sample is set to False, seed {seed} will be ignored")
        else:
            raise ValueError("fixed sample needs to be either True or False")

        z_base_sample = self.normal.sample(sample_shape=sample_size)
        torch.random.seed()  # to make sure seed is reset
        z_base_sample = z_base_sample * T  #  apply temperature
        z_base_sample = z_base_sample.to(device)
        generated_image = self.reverse(
            z_base_sample, T=T, device=device, fixed_sample=fixed_sample, seed=seed
        )
        return generated_image
