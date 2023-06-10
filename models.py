import torch
from math import log
from torch.distributions import Normal
from utils import squeeze, unsqueeze
from blocks import RealNVPBlock, GlowBlock
from utils import FinalScale, StepOfGlow
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
        for split in x_splits[::-1]:
            x = torch.concat([x, split], dim=1)
            x = unsqueeze(x)

        z_log_prob = torch.sum(self.normal.log_prob(x), dim=[1, 2, 3])
        log_prob = z_log_prob + total_logdet

        loss = (
            -log(self.n_bins) * self.n_pixels
        )  # implicit transformation that converts [0, 255] values to [0, 1]
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
        num_resnet_blocks=8,
        num_resnet_filters=64,
        n_bins=256,
    ) -> None:
        """
        K: number of steps of flow
        L: number of blocks
        """
        super().__init__()

        c, h, w = base_input_shape

        self.normal = Normal(0.0, 1.0)
        self.n_pixels = c * h * w
        self.n_bins = n_bins
        self.blocks = torch.nn.ModuleList(
            [
                GlowBlock(
                    K=K,
                    base_input_shape=[  # *2 for squeezing, *2 for channel split
                        # when split happens channel is divided by 2 and hence
                        # there is only one *2
                        c * 2 ** (i),
                        # i+1 because we start by squeezing the input and i
                        # starts from 0
                        h // 2 ** (i),
                        w // 2 ** (i),
                    ],
                    num_resnet_blocks=num_resnet_blocks,
                )
                for i in range(L)
            ]
        )
        self.final_step_of_flow = torch.nn.ModuleList(
            [
                StepOfGlow(
                    base_input_shape=[
                        c * 2 ** (L) * 2 * 2,  # to account for squeezing, this
                        # is taken care of in GlowBlock though we are not using that.
                        h // 2 ** (L + 1),  # to account for squeezing
                        w // 2 ** (L + 1),  # to account for squeezing
                    ],
                    num_resnet_blocks=num_resnet_blocks,
                    num_filters=num_resnet_filters,
                )
                for _ in range(K)
            ]
        )
        self.output_shape = self.final_step_of_flow[-1].base_input_shape

    def forward(self, x):
        original_input = x
        total_logdet = 0
        splits = []
        log_prob_from_splits = 0
        for block in self.blocks:
            x, logdet, split = block(x)
            total_logdet += logdet

            splits.append(split)
            log_prob_from_splits += self.normal.log_prob(split).sum(dim=(1, 2, 3))

        # squeeze before the final block
        x = squeeze(x)
        for step in self.final_step_of_flow:
            x, logdet = step(x)
            total_logdet += logdet

        log_prob_from_transformed_x = self.normal.log_prob(x).sum(dim=(1, 2, 3))
        log_prob_from_splits += self.normal.log_prob(x).sum(dim=(1, 2, 3))
        total_log_prob = log_prob_from_splits + log_prob_from_transformed_x

        # TODO instead of sanity checks I need to create proper tests!
        # sanity checks
        # unsqueeze to get back to original shape(we dont' need this, just for sanity check)
        z = unsqueeze(x)
        for split in splits[::-1]:
            z = torch.concat([z, split], dim=1)
            z = unsqueeze(z)
        # assert log probas are ok (we calculate them for each z independently
        # rather than zs assembled. contrary to RealNVP)
        torch.testing.assert_close(
            self.normal.log_prob(z).sum(dim=(1, 2, 3)),
            log_prob_from_splits,
            msg="log_prob does not have the same shape",
        )
        # assert z and original input have the same shape, we should not be
        # altering shape/dimesions due to transformations
        torch.testing.assert_close(
            z.shape,
            original_input.shape,
            msg=f"z and original_input should have the same shape found {list(z.shape)} vs {list(original_input.shape)}",
        )
        # assert log probas are ok, that is no implicit broadcasting is hapenning
        torch.testing.assert_close(log_prob_from_splits.shape, total_logdet.shape)

        log_prob = total_log_prob + total_logdet
        loss = -log(self.n_bins) * self.n_pixels
        loss = log_prob + loss
        final_loss = (-loss / (log(2.0) * self.n_pixels)).mean()
        return x, final_loss

    @torch.no_grad()
    def reverse(self, z_l, device):
        for step in self.final_step_of_flow[::-1]:
            z_l = step.reverse(z_l)
        z_l = unsqueeze(z_l)
        for block in self.blocks[::-1]:
            z_i = self.normal.sample(sample_shape=z_l.shape).to(device)
            z_l = block.reverse(z_l=z_l, z_i=z_i)
        return z_l

    @torch.no_grad()
    def sample(self, device, num_samples=1, z_base_sample=None):
        # TODO refactor this
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
            assert (
                list(z_base_sample.shape[1:]) == self.output_shape
            ), f"{z_base_sample.shape[1:]} vs {self.output_shape}"
        z_base_sample = z_base_sample.to(device)
        generated_image = self.reverse(z_base_sample, device=device)
        return generated_image
