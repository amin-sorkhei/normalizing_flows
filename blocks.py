import torch
from utils import squeeze, unsqueeze
from utils import Scale


class FlowBlock(torch.nn.Module):
    def __init__(
        self,
        num_step_of_flow=1,
        base_input_shape=[3, 32, 32],
        num_resnet_blocks=8,
    ) -> None:
        super().__init__()
        # each num_step_of_flow consists of 6 AffineCoupling layers (1 Scale operation)
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

class GlowBlock():
    """equivalent to 1 step of the flow in the GLOW paper
    """
    
    