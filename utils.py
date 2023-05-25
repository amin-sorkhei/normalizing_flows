from torch.utils.data import Dataset, DataLoader
import torch
import os
import glob
from PIL import Image
from torchvision.transforms import Resize, RandomHorizontalFlip, ToTensor, Compose
from torch.distributions.normal import Normal
from math import log

device = torch.device("cuda")


def files_with_suffix(directory, suffix, pure=False):
    # files = [os.path.abspath(path) for path in glob.glob(f'{directory}/**/*{suffix}', recursive=True)]  # full paths
    files = [
        os.path.abspath(path)
        for path in glob.glob(
            os.path.join(directory, "**", f"*{suffix}"), recursive=True
        )
    ]  # full paths
    if pure:
        files = [os.path.split(file)[-1] for file in files]
    return files


class ImageDataset(Dataset):
    def __init__(self, root_dir, suffix="jpg") -> None:
        self.root_dir = root_dir
        self.files_list = files_with_suffix(self.root_dir, suffix=suffix)
        super().__init__()

    def get_trans_list(self):
        transform_list = []
        transform_list.append(Resize((32, 32)))
        transform_list.append(RandomHorizontalFlip(p=0.5))
        transform_list.append(ToTensor())
        return Compose(transforms=transform_list)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        image = Image.open(self.files_list[idx])
        tensor_image = self.get_trans_list()(img=image)
        return tensor_image


def init_dataloader(image_dataset, batch_size=64, num_workers=8):
    image_loader = DataLoader(
        image_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return image_loader


class ConvResNet(torch.nn.Module):
    def __init__(self, input_shape, filters, num_blocks=8) -> None:
        self.input_shape = input_shape
        self.filters = filters
        super().__init__()
        self.relu_activation = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.first_layer = torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=filters,
            kernel_size=3,
            padding="same",
        ))

        self.second_layer = torch.nn.BatchNorm2d(num_features=filters)

        self.third_layer = torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels=filters,
            out_channels=input_shape[0],
            kernel_size=3,
            padding="same",
        ))
        self.fourth_layer = torch.nn.BatchNorm2d(num_features=input_shape[0])

        self.sixth_layer = torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=filters,
            kernel_size=3,
            padding="same",
        ))
        self.seventh_layer = torch.nn.BatchNorm2d(num_features=filters)

        self.eights_layer = torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels=filters,
            out_channels=input_shape[0],
            kernel_size=3,
            padding="same",
        ))
        self.ninth_layer = torch.nn.BatchNorm2d(num_features=input_shape[0])

        self.h2 = torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=2 * input_shape[0],
            kernel_size=3,
            padding="same",
        ))
        
        self.blocks = torch.nn.ModuleList([self.create_block(filters=input_shape[0],out_channels=input_shape[0])
                                           for _ in range(num_blocks)])
        

    def create_block(self, filters, out_channels):
        block =\
        torch.nn.Sequential(
            self.relu_activation,
            
            torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels=filters,
            out_channels=out_channels,
            kernel_size=3,
            padding="same")),
            
            torch.nn.BatchNorm2d(num_features=out_channels),
            
            self.relu_activation,
            
            torch.nn.utils.weight_norm(torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same")),
            
        )
        return block
        
    def forward(self, x):
        y = self.first_layer(x) # conv
        y = self.relu_activation(y) # relu
        y = self.second_layer(y) # BN
        y = self.third_layer(y) # conv
        y = self.relu_activation(y) # relu
        y = self.fourth_layer(y) # BN

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
    # checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    # mask = torch.tensor(checkerboard, dtype=torch.float32, device=device, requires_grad=False)
    # if orientation == 1:
    #     mask = 1 - mask
    # return mask


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
    def __init__(self,  mask_type, mask_orientation, input_shape, num_filters, num_resnet_blocks=8 ) -> None:
        super().__init__()
        self.mask_type = mask_type
        self.mask_orientation = mask_orientation
        self.input_shape = input_shape
        self.num_filters = num_filters
        if self.mask_type == "checkerboard_binary_mask":
            self.mask = checkerboard_binary_mask(
                self.input_shape, orientation=self.mask_orientation
            )
        elif self.mask_type == "channel_binary_mask":
            self.mask = channel_binary_mask(
                num_channels=self.input_shape[0], orientation=self.mask_orientation
            )
        else:
            raise NotImplementedError(
                f"requested mask {self.mask_type} not implemented, allowed values are [checkerboard_binary_mask, channel_binary_mask]"
            )

        self.mask = self.mask.to(device)
        self.nn = ConvResNet(input_shape=self.input_shape, filters=self.num_filters, num_blocks=num_resnet_blocks)

    def forward(self, x):
        masked_x = self.mask * x
        # self.x = x
        # self.masked_x = masked_x
        shift, logscale = self.nn(masked_x)
        # self.forward_shift, self.forward_logscale = shift, logscale
        y = masked_x + (1 - self.mask) * (x * torch.exp(logscale) + shift)
        logdet = torch.sum((1 - self.mask) * logscale, dim=[1, 2, 3])
        # self.y = y
        return y, logdet

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
            num_filters=128, num_resnet_blocks= num_resnet_blocks
        )

        self.aff_2 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1,
            input_shape=self.base_input_shape,
            num_filters=128, num_resnet_blocks= num_resnet_blocks
        )

        self.aff_3 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=0,
            input_shape=self.base_input_shape,
            num_filters=128, num_resnet_blocks= num_resnet_blocks
        )
        self.aff_4 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1,
            input_shape=self.base_input_shape,
            num_filters=128, num_resnet_blocks= num_resnet_blocks
        )
        self.aff_layers = torch.nn.ModuleList([self.aff_1, self.aff_2, self.aff_3, self.aff_4])

    def forward(self, x):
        total_logdet = 0
        for aff in self.aff_layers:
            x, logdet = aff(x)
            total_logdet += logdet

        return x, total_logdet

    def reverse(self, z):
        for aff in self.aff_layers[::-1]:
            z = aff.reverse(z)
        return z

class BatchNormalizationBijector(torch.nn.Module):
    def __init__(self,input_shape, decay=.999) -> None:
        super().__init__()
        self.decay = decay
        self.register_buffer("running_mean", torch.zeros(size=input_shape))
        self.register_buffer("running_std", torch.ones(size=input_shape))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            with torch.no_grad():
                self.running_mean = self.decay * self.running_mean + (1 - self.decay) * mean
                self.running_std = self.decay * self.running_std + (1 - self.decay) * std
        else:
            mean = self.runnin_mean
            std = self.running_std
        
        y = (x - mean)/std
        logdet =  -(torch.log(std)).sum(dim=[1, 2, 3])
        
        return y, logdet
    
    def reverse(self, y):
        x = self.running_std *  y + self.running_mean
        return x 


class Scale(torch.nn.Module):
    def __init__(self, base_input_shape=[3, 32, 32], mask_orientation=1,  num_resnet_blocks=8) -> None:
        super().__init__()
        self.base_input_shape = base_input_shape
        self.output_shape = [
            base_input_shape[0] * 4,
            base_input_shape[1] // 2,
            base_input_shape[2] // 2,
        ]

        self.aff_1 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1-mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=64, num_resnet_blocks=num_resnet_blocks
        )

        self.aff_2 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=64,num_resnet_blocks=num_resnet_blocks
        )

        self.aff_3 = AffineCouplingLayer(
            mask_type="checkerboard_binary_mask",
            mask_orientation=1-mask_orientation,
            input_shape=self.base_input_shape,
            num_filters=64,num_resnet_blocks=num_resnet_blocks
        )
        self.batch_normalization_bijector_1 = BatchNormalizationBijector(input_shape=base_input_shape)
        self.aff_4 = AffineCouplingLayer(
            mask_type="channel_binary_mask",
            mask_orientation=1-mask_orientation,
            input_shape=self.output_shape,
            num_filters=128,num_resnet_blocks=num_resnet_blocks
        )

        self.aff_5 = AffineCouplingLayer(
            mask_type="channel_binary_mask",
            mask_orientation=mask_orientation,
            input_shape=self.output_shape,
            num_filters=128,num_resnet_blocks=num_resnet_blocks
        )

        self.aff_6 = AffineCouplingLayer(
            mask_type="channel_binary_mask",
            mask_orientation=1-mask_orientation,
            input_shape=self.output_shape,
            num_filters=128,num_resnet_blocks=num_resnet_blocks
        )
        self.batch_normalization_bijector_2 = BatchNormalizationBijector(input_shape=self.output_shape)
        
        self.first_half_transformers = torch.nn.ModuleList([self.aff_1, self.aff_2, self.aff_3, self.batch_normalization_bijector_1])
        self.second_half_transformers = torch.nn.ModuleList([self.aff_4, self.aff_5, self.aff_6, self.batch_normalization_bijector_2])

    def forward(self, x):
        total_logdet = 0
        for transformer in self.first_half_transformers:
            x, logdet = transformer(x)
            # print("done first")
            total_logdet += logdet

        x = squeeze(x)

        for transformer in self.second_half_transformers:
            # print( x.shape, self.output_shape)
            x, log_det = transformer(x)
            # print("done second", x.shape, self.output_shape)
            total_logdet += log_det

        return x, total_logdet

    def reverse(self, z):
        for transformer in self.second_half_transformers[::-1]:
            z = transformer.reverse(z)

        z = unsqueeze(z)

        for transformer in self.first_half_transformers[::-1]:
            z = transformer.reverse(z)

        return z


class RealNVP(torch.nn.Module):
    def __init__(self, n_bins=256,base_input_shape=[3, 32, 32],
                 num_scales=2,
                 num_resnet_blocks=8) -> None:
        super().__init__()
        c, h, w = base_input_shape
        self.scales = torch.nn.ModuleList([Scale(base_input_shape=[c* 2**i, h//2**i, w//2**i], # after each scale h and w area halved and added to channel, half of the channel is truncated before the next scale hence c * 2^i rather (c * 2 ^(2i))
                                                 mask_orientation= i%2,
                                                 num_resnet_blocks=num_resnet_blocks) 
                                           for i in range(num_scales)])
        self.final_scale = FinalScale(base_input_shape=[c* 2**(num_scales),
                                                        h//2**(num_scales),
                                                        w//2**(num_scales)]) # the shape of this depends on howmany splits we are going to have

        self.normal = Normal(0.0, 1.0)
        self.n_pixels = c * h * w
        self.n_bins = n_bins
    
    def forward(self, x):
        x_splits = []
        total_logdet = 0
        
        for scale in self.scales:
            x, logdet = scale(x) # x will be squeezed as the outcome of this e.g 3, 32, 32 --> 12, 16, 16 
            total_logdet += logdet
            
            # split
            x, z = torch.chunk(x, chunks=2, dim=1) # chunks x on channel dim e.g 12, 16, 16, --> (6, 6), 16, 16
            x_splits.append(z)
         
        x, logdet = self.final_scale(x)
        total_logdet += logdet
        # unsqueeze to get back to original shape
        #print("-----")
        for split in x_splits[::-1]:
            x = torch.concat([x, split], dim=1)
            x = unsqueeze(x)
        
        z_log_prob = torch.sum(self.normal.log_prob(x), dim=[1, 2, 3])
        log_prob = z_log_prob + total_logdet

        loss = -log(self.n_bins) * self.n_pixels
        loss = log_prob + loss
        final_loss = (-loss / (log(2.0) * self.n_pixels)).mean()
        return x, final_loss

    def reverse(self, z):
        z = self.final_scale.reverse(z)
        for scale in self.scales[::-1]:
            # sample from a normal
            x = self.normal.sample(sample_shape=z.shape).to(device)
            z = torch.concat([z, x], dim=1)
            z = scale.reverse(z)
        return z

    def sample(self, num_samples=1):
        num_channels = self.final_scale.base_input_shape[0] 
        height, width = self.final_scale.base_input_shape[1:]
        sample_size = [num_samples, num_channels, height, width]
        z_samples = self.normal.sample(sample_shape=sample_size).to(device)
        generated_image = self.reverse(z_samples)
        return generated_image