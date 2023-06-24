import torch
import os
import glob
from torch.distributions.normal import Normal
from math import log
from types import SimpleNamespace
from typing import Any
import matplotlib.pyplot as plt
import logging
import datetime
from torch.nn import functional as F

logger = logging.getLogger()


def time_now_to_str():
    time_now = datetime.datetime.utcnow()
    time_now_str = datetime.datetime.strftime(time_now, format="%Y%m%d_%H%M%S")
    return time_now_str


def add_noise(batch, n_bits):
    """adds noise to the batch in order to avoid
    de-generation of the learnt distribution"""

    batch = batch * 255  # move back to original space for simplicity --> int([0:255])
    batch = torch.floor(
        (batch / 256) * 2**n_bits
    )  # move to [0:2**n_bits) e.g n_bits=5 --> int([0:31])
    batch = batch / 2**n_bits  # move back to [0:1)
    batch = batch - .5  # move to [-.5:.5) this should be more friendly to the prior of N(0, 1)
    return batch + torch.rand_like(batch) / 2**n_bits  # add noise


class RecNameSpace(SimpleNamespace):
    def __init__(self, **kwargs: Any) -> None:
        nameSpaced = {
            k: RecNameSpace(**v) if isinstance(v, dict) else v
            for k, v in kwargs.items()
        }
        super().__init__(**nameSpaced)


def check_point_model(model, epoch, optimizer, loss, path):
    ckpt_path = os.path.join(path, f"best_model_{epoch}_{time_now_to_str()}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        ckpt_path,
    )


def save_plot(n_row, n_column, generated_image, epoch, path, fixed_image=False):
    logging.info(
        f"min_value {generated_image.min()} and max_value {generated_image.max()} in the generated image"
    )
    generated_image = torch.clamp(generated_image, min=0, max=1)
    fig, ax = plt.subplots(n_row, n_column, squeeze=True)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    for i in range(0, n_row):
        for j in range(0, n_column):
            ax[i, j].axis("off")
            image = generated_image[i, j].permute([1, 2, 0]).detach().cpu().numpy()
            _ = ax[i, j].imshow(image, aspect="auto")
    plt.suptitle(f"generated_plot epoch {epoch}")
    # TODO: refactor the following lines
    if fixed_image is True:
        fig.savefig(os.path.join(path, f"fixed_generated_image_epoch_{epoch}.png"))
    else:
        fig.savefig(os.path.join(path, f"generated_image_epoch_{epoch}.png"))
    # logger.setLevel(logging.INFO)
    plt.close("all")  # close all figures to avoid memory consumption


def create_deterministic_sample(sampling_params, input_shape, seed=42):
    n_row, n_column = (
        sampling_params.num_samples_nrow,
        sampling_params.num_samples_ncols,
    )
    # n_channels, h, w = realnvp.final_scale.base_input_shape
    n_channels, h, w = input_shape
    z_base_sample_shape = [n_row * n_column, n_channels, h, w]

    # if sampling_params.generate_fixed_images is True:
    torch.manual_seed(seed)
    z_base_sample = torch.distributions.Normal(0, 1).sample(
        sample_shape=z_base_sample_shape
    )
    torch.random.seed()
    return z_base_sample


class ZeroConv2d(torch.nn.Module):
    """conv2d initialized with zero weights as described in Glow paper"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = torch.nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        out = F.pad(x, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out *= torch.exp(self.scale * 3)
        return out


class GlowAffineCouplingLayer(torch.nn.Module):
    """Affine coupling layer from Glow paper"""

    def __init__(self, num_channels, filters=512) -> None:
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.first_layer = torch.nn.Conv2d(
            in_channels=num_channels
            // 2,  # only half of the channels impact scale and sigma
            out_channels=filters,
            kernel_size=3,
            padding="same",
        )
        torch.manual_seed(41)
        self.first_layer.weight.data.normal_(0, 0.05)
        self.first_layer.bias.data.zero_()

        self.second_layer = torch.nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            padding="same",
            kernel_size=1,
        )
        torch.manual_seed(41)
        self.second_layer.weight.data.normal_(0, 0.05)
        self.second_layer.bias.data.zero_()

        self.third_layer = ZeroConv2d(
            in_channels=filters,
            out_channels=num_channels,
        )

        self.net = torch.nn.Sequential(
            self.first_layer, self.relu, self.second_layer, self.relu, self.third_layer
        )

    def forward(self, x):
        x_a, x_b = torch.chunk(x, chunks=2, dim=1)
        shift_logscale = self.net(x_a)
        shift, log_scale = torch.chunk(shift_logscale, chunks=2, dim=1)
        scale = F.sigmoid(log_scale + 2)
        x_b_transformed = x_b * scale + shift
        y = torch.cat([x_a, x_b_transformed], dim=1)
        log_det = torch.sum(torch.log(scale), dim=[1, 2, 3])
        return y, log_det

    def reverse(self, z):
        z_a, z_b = torch.chunk(z, chunks=2, dim=1)
        shift_logscale = self.net(z_a)
        shift, log_scale = torch.chunk(shift_logscale, chunks=2, dim=1)
        scale = F.sigmoid(log_scale + 2)
        z_b_transformed = (z_b - shift) / scale
        x = torch.cat([z_a, z_b_transformed], dim=1)
        return x


class ConvResNet(torch.nn.Module):
    def __init__(self, input_shape, filters, num_blocks=8) -> None:
        self.input_shape = input_shape
        self.filters = filters
        super().__init__()
        self.relu_activation = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.first_layer = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=filters,
                kernel_size=3,
                padding="same",
            )
        )

        self.second_layer = torch.nn.BatchNorm2d(num_features=filters)

        self.third_layer = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=filters,
                out_channels=input_shape[0],
                kernel_size=3,
                padding="same",
            )
        )
        self.fourth_layer = torch.nn.BatchNorm2d(num_features=input_shape[0])

        self.sixth_layer = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=filters,
                kernel_size=3,
                bias=False,
                padding="same",
            )
        )
        self.seventh_layer = torch.nn.BatchNorm2d(num_features=filters)

        self.eights_layer = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=filters,
                out_channels=input_shape[0],
                kernel_size=3,
                bias=False,
                padding="same",
            )
        )
        self.ninth_layer = torch.nn.BatchNorm2d(num_features=input_shape[0])

        self.h2 = torch.nn.utils.weight_norm(
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=2 * input_shape[0],
                kernel_size=3,
                padding="same",
            )
        )

        self.blocks = torch.nn.ModuleList(
            [
                self.create_block(filters=input_shape[0], out_channels=input_shape[0])
                for _ in range(num_blocks)
            ]
        )

    def create_block(self, filters, out_channels):
        block = torch.nn.Sequential(
            self.relu_activation,
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=filters,
                    out_channels=out_channels,
                    kernel_size=3,
                    bias=False,
                    padding="same",
                )
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            self.relu_activation,
            torch.nn.utils.weight_norm(
                torch.nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding="same",
                )
            ),
        )
        return block

    def forward(self, x):
        y = self.first_layer(x)  # conv
        y = self.relu_activation(y)  # relu
        y = self.second_layer(y)  # BN
        y = self.third_layer(y)  # conv
        y = self.relu_activation(y)  # relu
        y = self.fourth_layer(y)  # BN

        y = y + x

        for block in self.blocks:
            y = y + block(y)

        z = self.sixth_layer(y)
        z = self.relu_activation(z)
        z = self.seventh_layer(z)
        z = self.eights_layer(z)
        z = self.relu_activation(z)
        z = self.ninth_layer(z)
        z = y + z

        z = self.h2(z)
        shift, log_scale = torch.chunk(z, chunks=2, dim=1)
        log_scale = self.tanh(log_scale)

        return shift, log_scale


def checkerboard_binary_mask(shape, orientation=0):
    height, width = shape[1], shape[2]
    height_range = torch.arange(height)
    width_range = torch.arange(width)
    height_odd_inx = height_range % 2 == 1.0
    width_odd_inx = width_range % 2 == 1.0
    odd_rows = torch.tile(torch.unsqueeze(height_odd_inx, -1), [1, width])
    odd_cols = torch.tile(torch.unsqueeze(width_odd_inx, 0), [height, 1])
    checkerboard_mask = torch.logical_xor(odd_rows, odd_cols)
    if orientation == 1:
        checkerboard_mask = torch.logical_not(checkerboard_mask)
    return torch.unsqueeze(checkerboard_mask, 0).float().requires_grad_(False)


def channel_binary_mask(num_channels, orientation=0):
    """
    This function takes an integer num_channels and orientation (0 or 1) as
    arguments. It should create a channel-wise binary mask with
    float, according to the above specification.
    The function should then return the binary mask.
    """
    first_half = num_channels // 2
    second_half = num_channels - first_half
    first = torch.zeros(size=[first_half, 1, 1]).bool()
    second = torch.ones(size=[second_half, 1, 1]).bool()
    channel_binary_mask = torch.concat([first, second], axis=0)
    if orientation != 0:
        channel_binary_mask = torch.logical_not(channel_binary_mask)
    return channel_binary_mask.float()


def squeeze(x):
    # squeeze
    b_size, c, h, w = x.shape
    x = x.view(b_size, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.contiguous().view(b_size, c * 4, h // 2, w // 2)
    return x


def unsqueeze(z):
    # unsqueeze
    b_size, c, h, w = z.shape
    z = z.view(b_size, c // 4, 2, 2, h, w)
    z = z.permute(0, 1, 4, 2, 5, 3)
    z = z.contiguous().view(b_size, c // 4, h * 2, w * 2)
    return z


class AffineCouplingLayer(torch.nn.Module):
    def __init__(
        self,
        mask_type,
        mask_orientation,
        input_shape,
        num_filters,
        num_resnet_blocks=8,
        conv_class=ConvResNet,
    ) -> None:
        super().__init__()
        self.mask_type = mask_type
        self.mask_orientation = mask_orientation
        self.input_shape = input_shape
        self.num_filters = num_filters
        if self.mask_type == "checkerboard_binary_mask":
            mask = checkerboard_binary_mask(
                self.input_shape, orientation=self.mask_orientation
            )
        elif self.mask_type == "channel_binary_mask":
            mask = channel_binary_mask(
                num_channels=self.input_shape[0], orientation=self.mask_orientation
            )
        else:
            raise NotImplementedError(
                f"requested mask {self.mask_type} not implemented, allowed values are [checkerboard_binary_mask, channel_binary_mask]"
            )

        # self.mask = self.mask.to(device)
        self.register_buffer("mask", mask)
        conv_class_params = {
            "input_shape": self.input_shape,
            "filters": self.num_filters,
        }
        if conv_class == ConvResNet:
            conv_class_params["num_blocks"] = num_resnet_blocks

        self.nn = conv_class(**conv_class_params)

    def forward(self, x):
        masked_x = self.mask * x
        shift, logscale = self.nn(masked_x)
        y = masked_x + (1 - self.mask) * (x * torch.exp(logscale) + shift)
        logdet = torch.sum((1 - self.mask) * logscale, dim=[1, 2, 3])
        return y, logdet

    @torch.no_grad()
    def reverse(self, z):
        masked_z = self.mask * z
        shift, logscale = self.nn(masked_z)
        # assert torch.allclose(self.y, z)
        # assert torch.allclose(shift, self.forward_shift)
        # assert torch.allclose(logscale, self.forward_logscale)
        x = masked_z + (1 - self.mask) * (z - shift) * torch.exp(-logscale)
        return x


class FinalScale(torch.nn.Module):
    def __init__(self, base_input_shape=[6, 16, 16], num_resnet_blocks=8) -> None:
        self.base_input_shape = base_input_shape
        super().__init__()
        self.aff_1 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=0,
            input_shape=self.base_input_shape,
            num_filters=128,
            num_resnet_blocks=num_resnet_blocks,
        )

        self.aff_2 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1,
            input_shape=self.base_input_shape,
            num_filters=128,
            num_resnet_blocks=num_resnet_blocks,
        )

        self.aff_3 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=0,
            input_shape=self.base_input_shape,
            num_filters=128,
        )
        self.aff_4 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1,
            input_shape=self.base_input_shape,
            num_filters=128,
            num_resnet_blocks=num_resnet_blocks,
        )
        self.aff_layers = torch.nn.ModuleList(
            [self.aff_1, self.aff_2, self.aff_3, self.aff_4]
        )

    def forward(self, x):
        total_logdet = 0
        for aff in self.aff_layers:
            x, logdet = aff(x)
            total_logdet += logdet

        return x, total_logdet

    @torch.no_grad()
    def reverse(self, z):
        for aff in self.aff_layers[::-1]:
            z = aff.reverse(z)
        return z


class BatchNormalizationBijector(torch.nn.Module):
    def __init__(self, input_shape, decay=0.999) -> None:
        super().__init__()
        c, h, w = input_shape
        self.decay = decay
        self.register_buffer("running_mean", torch.zeros(size=[1, c, h, w]))
        self.register_buffer("running_std", torch.ones(size=[1, c, h, w]))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            with torch.no_grad():
                self.running_mean = (
                    self.decay * self.running_mean + (1 - self.decay) * mean
                )
                self.running_std = (
                    self.decay * self.running_std + (1 - self.decay) * std
                )
        else:
            mean = self.runnin_mean
            std = self.running_std

        y = (x - mean) / std
        logdet = -(torch.log(std)).sum(dim=[1, 2, 3])

        return y, logdet

    @torch.no_grad()
    def reverse(self, y):
        x = self.running_std * y + self.running_mean
        return x


class Scale(torch.nn.Module):
    def __init__(
        self,
        base_input_shape=[3, 32, 32],
        mask_orientation=1,
        num_resnet_blocks=8,
    ) -> None:
        super().__init__()
        self.base_input_shape = base_input_shape
        self.output_shape = [
            base_input_shape[0] * 4,
            base_input_shape[1] // 2,
            base_input_shape[2] // 2,
        ]

        self.aff_1 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1 - mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=64,
            num_resnet_blocks=num_resnet_blocks,
        )

        self.aff_2 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=64,
            num_resnet_blocks=num_resnet_blocks,
        )

        self.aff_3 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1 - mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=64,
            num_resnet_blocks=num_resnet_blocks,
        )
        self.batch_normalization_bijector_1 = BatchNormalizationBijector(
            input_shape=base_input_shape
        )
        self.aff_4 = AffineCouplingLayer(
            mask_type="channel_binary_mask",
            mask_orientation=1 - mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=128,
            num_resnet_blocks=num_resnet_blocks,
        )

        self.aff_5 = AffineCouplingLayer(
            mask_type="channel_binary_mask",
            mask_orientation=mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=128,
            num_resnet_blocks=num_resnet_blocks,
        )

        self.aff_6 = AffineCouplingLayer(
            mask_type="channel_binary_mask",
            mask_orientation=1 - mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=128,
            num_resnet_blocks=num_resnet_blocks,
        )
        self.batch_normalization_bijector_2 = BatchNormalizationBijector(
            input_shape=self.base_input_shape
        )

        self.first_half_transformers = torch.nn.ModuleList(
            [self.aff_1, self.aff_2, self.aff_3, self.batch_normalization_bijector_1]
        )
        self.second_half_transformers = torch.nn.ModuleList(
            [self.aff_4, self.aff_5, self.aff_6, self.batch_normalization_bijector_2]
        )

    def forward(self, x):
        total_logdet = 0
        for transformer in self.first_half_transformers:
            x, logdet = transformer(x)
            total_logdet += logdet

        for transformer in self.second_half_transformers:
            x, log_det = transformer(x)
            total_logdet += log_det

        return x, total_logdet

    @torch.no_grad()
    def reverse(self, z):
        for transformer in self.second_half_transformers[::-1]:
            z = transformer.reverse(z)

        for transformer in self.first_half_transformers[::-1]:
            z = transformer.reverse(z)

        return z


class Invertibe1by1Convolution(torch.nn.Module):
    """Invertible 1x1 Convolution layer from Glow paper"""

    def __init__(self, num_channels) -> None:
        super().__init__()
        self.num_channels = num_channels
        torch.manual_seed(41)
        random_weight = torch.randn(size=[self.num_channels, self.num_channels])
        rotation_w, _ = torch.linalg.qr(random_weight)
        rotation_w = rotation_w.unsqueeze(2).unsqueeze(3)
        self.weight = torch.nn.Parameter(rotation_w)

    def forward(self, x):
        _, _, h, w = x.shape
        y = F.conv2d(input=x, weight=self.weight, padding="same", groups=1)
        log_det = h * w * torch.log(torch.abs(torch.linalg.det(self.weight.squeeze())))
        return y, log_det

    def reverse(self, x):
        w_inv = self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        z = F.conv2d(input=x, weight=w_inv, padding="same", groups=1)
        return z


class ActNorm(torch.nn.Module):
    """ActNorm layer from Glow paper"""

    def __init__(self, num_channels) -> None:
        super().__init__()
        self.batch_mean = torch.nn.Parameter(torch.zeros([1, num_channels, 1, 1]))
        self.batch_std = torch.nn.Parameter(torch.ones([1, num_channels, 1, 1]))

        self.register_buffer("initialized", torch.tensor(False))

    def initialize_layer(self, x):
        with torch.no_grad():
            batch_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            batch_std = x.std(dim=[0, 2, 3], keepdim=True)

            self.batch_mean.data.copy_(batch_mean)
            self.batch_std.data.copy_(batch_std)
            self.initialized.fill_(True)

    def forward(self, x):
        if self.initialized.item() is False:
            self.initialize_layer(x)

        y = (x - self.batch_mean) / self.batch_std  # torch.sqrt(self.batch_variance)

        # caluclate log det
        h, w = y.shape[2:]
        log_det = -h * w * torch.log(torch.abs(self.batch_std)).sum()
        # the above is logdet for one sample in the batch, we to one for each sample in the batch_size
        log_det = torch.tile(log_det, dims=[y.shape[0]])
        return y, log_det

    def reverse(self, z):
        assert (
            self.initialized.item() is True
        ), "ActNorm needs to be initialized before calling reverse"
        x = self.batch_mean + z * self.batch_std  # torch.sqrt(self.batch_variance) +
        return x


class StepOfGlow(torch.nn.Module):
    """implementes one step of the flow in the GLOW paper"""

    def __init__(self, num_channels, num_filters=512) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.act_norm = ActNorm(self.num_channels)
        self.invertible_1by1_conv = Invertibe1by1Convolution(
            num_channels=self.num_channels
        )

        self.mask_type = "channel_binary_mask"
        self.aff = GlowAffineCouplingLayer(
            num_channels=self.num_channels, filters=num_filters
        )

    def forward(self, x):
        total_log_det = 0
        x, log_det = self.act_norm(x)
        total_log_det += log_det
        x, log_det = self.invertible_1by1_conv(x)
        total_log_det += log_det
        x, log_det = self.aff(x)
        total_log_det += log_det
        return x, total_log_det

    def reverse(self, z):
        x = self.aff.reverse(z)
        x = self.invertible_1by1_conv.reverse(x)
        x = self.act_norm.reverse(x)
        return x
