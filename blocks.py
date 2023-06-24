import torch
from utils import squeeze, unsqueeze
from utils import Scale, StepOfGlow, ZeroConv2d
import torch.distributions as td


class BaseBlock(torch.nn.Module):
    """base class for all blocks in the flow models"""

    num_step_of_flow: int
    base_input_shape: list
    num_resnet_blocks: int

    def forward(self, x):
        """peforms the forward operation of the flow block"""
        raise NotImplementedError("forward method not implemented")

    def reverse(self, z):
        """performs the reverse operation of the flow block"""
        raise NotImplementedError("reverse method not implemented")


class RealNVPBlock(BaseBlock):
    """implements the RealNVP block
    a real NVP block is composed Scale operations
    where each scale operation consists of 3 Affine(checkeboard alternating
    orientation) + BN + 3 Affine (channel mask with alternating orientation) + BN layers
    """

    def __init__(
        self,
        num_step_of_flow=1,
        base_input_shape=[3, 32, 32],
        num_resnet_blocks=8,
    ) -> None:
        # each num_step_of_flow consists of 6 AffineCoupling layers (1 Scale operation)
        super().__init__()
        self.scales = torch.nn.ModuleList(
            [
                Scale(
                    base_input_shape=base_input_shape,
                    mask_orientation=i % 2,
                    num_resnet_blocks=num_resnet_blocks,
                )
                for i in range(num_step_of_flow)
            ]
        )

    def forward(self, x):
        total_logdet = 0
        for scale in self.scales:
            x, log_det = scale(x)

            total_logdet += log_det

        x = squeeze(x)
        return x, total_logdet

    @torch.no_grad()
    def reverse(self, z):
        z = unsqueeze(z)
        for scale in self.scales[::-1]:
            z = scale.reverse(z)

        return z


class GlowBlock(torch.nn.Module):
    def __init__(
        self, K: int, num_channels, num_filters=512, is_final_block=False
    ) -> None:
        """composed of 3 steps:
        1. squeeze
        2. K steps of the flow
        3. split
        is_final_block defines if the is the last block in the model or not.
        if it is the last block, no split happens
        Args:
            K (int): number of steps of the flow in each block
        """
        super().__init__()
        self.num_channels = num_channels
        
        self.steps = torch.nn.ModuleList()
        for _ in range(K):
            self.steps.append(
                StepOfGlow(num_channels=num_channels * 4, num_filters=num_filters)
            )

        self.is_final_block = is_final_block
        if self.is_final_block is False:
            # prior, we learn the prior as a zero convolution
            # number of input channels half of the last step due to split
            # number of output channels is 2 times the input channels to
            # accommodate for mu and sigma
            self.prior = ZeroConv2d(
                in_channels=self.num_channels * 2,
                out_channels=self.num_channels * 2 * 2,
            )

    def forward(self, x):
        """forward operation
        first squeeze then repeat K steps of the flow and at the end split
        Args:
            x (_type_): _description_

        Returns:
            x_i (_type_): goes for further transformations
            z_i (_type_): comes from the split and remains unchanged
        """
        x = squeeze(x)
        total_logdet = 0
        for step in self.steps:
            x, log_det = step(x)
            total_logdet += log_det

        if self.is_final_block is False:
            # split
            x_i, z_i = torch.chunk(
                x, chunks=2, dim=1
            )  # chunks x on channel dim e.g 12, 16, 16, --> (6, 6), 16, 16

            # learn the prior
            prior_params = self.prior(x_i)
            mu, log_sigma = torch.chunk(prior_params, chunks=2, dim=1)
            sigma = torch.exp(log_sigma)
            z_log_prob = td.Normal(mu, sigma).log_prob(z_i).sum([1, 2, 3])

        else:
            # this is the final block, no need to split, no prior will be learnt
            x_i = x
            z_log_prob = td.Normal(0.0, 1.0).log_prob(x_i).sum([1, 2, 3])
            z_i = None  # bc there is no split

        return x_i, total_logdet, z_i, z_log_prob

    def reverse(self, z_l, device=torch.device("cpu")):
        """reverse operation

        Args:
            z_l (_type_): tensorf coming previous transformations

        """
        if self.is_final_block is False:
            prior_params = self.prior(z_l)
            mu, log_sigma = torch.chunk(prior_params, chunks=2, dim=1)
            sigma = torch.exp(log_sigma)
            torch.manual_seed(42)
            z_i = td.Normal(0, 1).sample(sample_shape=z_l.shape).to(device) * sigma + mu
            z = torch.concat([z_l, z_i], dim=1)

        else:
            # this is the final block. No input is expected
            z = z_l

        for reverse_step in self.steps[::-1]:
            z = reverse_step.reverse(z)

        z = unsqueeze(z)
        return z
