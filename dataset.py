import os
import glob
import torchvision.transforms as T
import torchvision
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import torch


# def files_with_suffix(directory, suffix, pure=False):
#     files = [
#         os.path.abspath(path)
#         for path in glob.glob(
#             os.path.join(directory, "**", f"*{suffix}"), recursive=True
#         )
#     ]  # full paths
#     if pure:
#         files = [os.path.split(file)[-1] for file in files]
#     return files


def create_dataset(
    dataset_name: str,
    num_samples: int = -1,
    bathc_size: int = 128,
    num_workers: int = 1,
):
    if dataset_name not in ["lsun_bedroom", "CelebA"]:
        raise NotImplementedError(f"dataset {dataset_name} is not suppeorted")

    if dataset_name == "lsun_bedroom":
        path = "~/lsun_bedroom_subsample"
        transformers_list = [
            T.Resize(size=96),
            T.RandomCrop(size=(64, 64)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ]
        dataset = ImageFolder(root=path, transform=T.Compose(transformers_list))

    else:  # CelebA
        path = "~/real_nvp_data_celebA"
        transformers_list = [
            T.CenterCrop(size=(148, 148)),
            T.Resize(size=(64, 64)),
            T.RandomHorizontalFlip(p=0.05),
            T.ToTensor(),
        ]
        dataset = torchvision.datasets.CelebA(
            root=path,
            split="train",
            download=True,
            transform=T.Compose(transformers_list),
        )
    if num_samples != -1:
        logging.info(f"down sampling data to {num_samples}")
        indices = torch.randint(low=0, high=len(dataset), size=[num_samples])
        dataset = Subset(dataset=dataset, indices=indices)
    return init_dataloader(dataset, batch_size=bathc_size, num_workers=num_workers)


def init_dataloader(image_dataset, batch_size=64, num_workers=8):
    image_loader = DataLoader(
        image_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return image_loader
# class LSUNDataset(Dataset):
#     def __init__(self, root_dir, suffix="jpg") -> None:
#         self.root_dir = root_dir
#         self.files_list = files_with_suffix(self.root_dir, suffix=suffix)
#         super().__init__()

#     def get_trans_list(self):
#         transform_list = []
#         transform_list.append(T.Resize((32, 32)))
#         transform_list.append(T.RandomHorizontalFlip(p=0.5))
#         transform_list.append(T.ToTensor())
#         return T.Compose(transforms=transform_list)

#     def __len__(self):
#         return len(self.files_list)

#     def __getitem__(self, idx):
#         image = Image.open(self.files_list[idx])
#         tensor_image = self.get_trans_list()(img=image)
#         return tensor_image


# class CelebAImageDataset(ImageDataset):
#     def __init__(self, center_crop_dim, resize_dim, root_dir, suffix="jpg") -> None:
#         self.center_crop_dim = center_crop_dim
#         self.reresize_dim = resize_dim
#         super().__init__(root_dir, suffix)

#     def get_trans_list(self):
#         transform_list = []
#         transform_list.append(T.CenterCrop(size=self.center_crop_dim))
#         transform_list.append(T.Resize(self.reresize_dim))
#         transform_list.append(T.ToTensor())
#         return T.Compose(transforms=transform_list)



