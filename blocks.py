import torch
from utils import squeeze, unsqueeze
from utils import Scale, StepOfGlow


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
    def __init__(self, K: int, base_input_shape, num_resnet_blocks, num_filters=64) -> None:
        """composed of 3 steps:
        1. squeeze
        2. K steps of the flow
        3. split

        Args:
            K (int): number of steps of the flow in each block
        """
        super().__init__()
        c, h, w = base_input_shape
        self.steps = torch.nn.ModuleList(
            [  # we do squeeze before calling the first step
                # hence the input shape shoud be c *2 *2, h//2, w//2
                StepOfGlow(
                    base_input_shape=[c * 2 * 2, h // 2, w // 2],
                    num_resnet_blocks=num_resnet_blocks,
                    num_filters=num_filters,
                )
                for i in range(0, K)
            ]
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

        # split
        x_i, z_i = torch.chunk(
            x, chunks=2, dim=1
        )  # chunks x on channel dim e.g 12, 16, 16, --> (6, 6), 16, 16
        return x_i, total_logdet, z_i

    def reverse(self, z_l, z_i):
        """reverse operation

        Args:
            z_l (_type_): tensorf coming previous transformations
            z_i (_type_): tensor coming from the split
        """
        z = torch.concat([z_l, z_i], dim=1)
        for reverse_step in self.steps[::-1]:
            z = reverse_step.reverse(z)

        z = unsqueeze(z)
        return z
