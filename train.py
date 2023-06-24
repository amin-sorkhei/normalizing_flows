import argparse
import yaml
from dataset import create_dataset
from utils import (
    RecNameSpace,
    save_plot,
    check_point_model,
    create_deterministic_sample,
    time_now_to_str,
    add_noise,
)
from models import RealNVP, Glow
import torch
import os
import logging
import datetime
import time
import glob
from types import SimpleNamespace


logging.basicConfig(level=logging.INFO)


def train(
    model_params: SimpleNamespace,
    train_params: SimpleNamespace,
    sampling_params: SimpleNamespace,
    data_params: SimpleNamespace,
    device: torch.device,
    task: str,
):
    time_now_str = time_now_to_str()

    if task == "train":
        params_to_save = {}
        params_to_save["model_params"] = model_params.__dict__
        params_to_save["train_params"] = train_params.__dict__
        params_to_save["data_params"] = data_params.__dict__

        # in order to avoid overriding add a timestamp if model is being trained with the same config
        sampling_params.samples_output_path = os.path.join(
            sampling_params.samples_output_path, time_now_str
        )
        train_params.model_checkpoint_path = os.path.join(
            train_params.model_checkpoint_path, time_now_str
        )
        for path in [
            train_params.model_checkpoint_path,
            sampling_params.samples_output_path,
        ]:
            logging.info(f"creating path  {path}")
            os.makedirs(path)
            with open(os.path.join(path, "params.json"), "w") as f:
                yaml.safe_dump(
                    params_to_save,
                    f,
                )
    else:
        logging.info(
            f"this is a continuation run, model will be loaded from {train_params.model_checkpoint_path} \n and samples will be written to {sampling_params.samples_output_path}"
        )

    # dataset = ImageDataset(root_dir=train_params.dataset_path)
    # dataloader = init_dataloader(dataset, batch_size=train_params.batch_size)
    dataset = create_dataset(
        dataset_name=data_params.dataset_name,
        num_samples=data_params.num_samples,
        bathc_size=data_params.batch_size,
        num_workers=data_params.num_workers,
    )
    if model_params.model_type == "realnvp":
        model = RealNVP(
            base_input_shape=[3, 64, 64],
            num_scales=model_params.num_scales,
            num_step_of_flow=model_params.num_step_of_flow,
            num_resnet_blocks=model_params.num_resnet_blocks,
            n_bits=model_params.n_bits,
        ).to(device)
    elif model_params.model_type == "glow":
        model = Glow(
            base_input_shape=[3, 64, 64],
            L=model_params.num_scales,
            K=model_params.num_step_of_flow,
            n_bits=model_params.n_bits,
        ).to(device)

    else:
        raise NotImplementedError(
            f"model type {model_params.model_type} not implemented"
        )

    optimizer = torch.optim.Adam(lr=train_params.lr, params=model.parameters())
    logging.info(
        f"number of parameters in the model {sum([p.numel() for p in model.parameters()])}"
    )

    if task == "train":
        best_loss = torch.inf
        epoch_start = 0
        # initialize the model
        logging.info("initializing the model")
        model.train()
        warm_up_batch = next(iter(dataset))
        with torch.no_grad():
            z_L, _ = model(warm_up_batch[0].to(device))
        model.output_shape = z_L.shape[1:]
    else:
        logging.info("loading checkpoint")
        best_model_ckpt_path = sorted(
            glob.glob(os.path.join(train_params.model_checkpoint_path, "best_model*"))
        )[-1]
        logging.info(f"most recent checkpoint {best_model_ckpt_path}")
        ckpt = torch.load(best_model_ckpt_path)

        epoch_start = ckpt["epoch"]
        best_loss = ckpt["loss"]
        model.load_state_dict(ckpt["model_state_dict"])
        model.train()
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        logging.info(
            f"resuming model training from epoch {epoch_start} with loss {best_loss}"
        )

    epoch_end = epoch_start + train_params.epochs
    # sampling parameters
    # TODO move realNVP params to config, that is based input shape and ...

    if sampling_params.generate_fixed_images is True:
        z_base_sample = create_deterministic_sample(
            sampling_params=sampling_params,
            input_shape=model.output_shape,
        )
    loss_per_epoch = []
    for e in range(epoch_start, epoch_end):
        loss_per_bath = []
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            image_batch = data[0]  # we only need the image, hence [0]
            image_batch = add_noise(image_batch, n_bits=model_params.n_bits).to(device)
            optimizer.zero_grad()
            z_L, loss = model(image_batch)
            loss.backward()
            optimizer.step()
            if i % 99 == 0:
                print(f"epcoh {e}, batch {i}, loss {loss}")
            loss_per_bath.append(loss)

        epoch_loss = torch.Tensor(loss_per_bath).mean()
        epoch_end_time = time.time()
        logging.info(
            f"epoch {e}, took: {epoch_end_time - epoch_start_time} loss: {epoch_loss}"
        )
        loss_per_epoch.append(epoch_loss)

        if epoch_loss < best_loss:
            logging.info(
                f"epcoh loss : {epoch_loss} is better than best loss {best_loss}"
            )
            best_loss = epoch_loss
            logging.info("check pointing the model")
            check_point_model(
                model=model,
                epoch=e,
                optimizer=optimizer,
                loss=epoch_loss,
                path=train_params.model_checkpoint_path,
            )

        model.eval()
        n_row, n_column = (
            sampling_params.num_samples_nrow,
            sampling_params.num_samples_ncols,
        )
        generated_image = model.sample(
            num_samples=n_row * n_column, z_base_sample=None, device=device
        ).view(n_row, n_column, 3, 64, 64)
        save_plot(
            n_row=n_row,
            n_column=n_column,
            path=sampling_params.samples_output_path,
            generated_image=generated_image,
            epoch=e,
            fixed_image=False,
        )
        if sampling_params.generate_fixed_images is True:
            generated_image_fixed = model.sample(
                num_samples=n_row * n_column, z_base_sample=z_base_sample, device=device
            ).view(n_row, n_column, 3, 64, 64)
            save_plot(
                n_row=n_row,
                n_column=n_column,
                path=sampling_params.samples_output_path,
                generated_image=generated_image_fixed,
                epoch=e,
                fixed_image=True,
            )

        model.train()

        loss_per_epoch.append(epoch_loss)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--device", default="cpu", type=str, help="device to be used for trainig"
    )
    args.add_argument("--config_path", help="path to config yaml file", required=True)
    args.add_argument(
        "--samples_output_path",
        default="./tmp/samples",
        help="output path for the sample files, if provided it will override the one provided through the config file",
    )
    args.add_argument(
        "--task",
        help="train or resume training",
        default="train",
        choices=["train", "resume_training"],
    )
    parsed = args.parse_args()
    config = yaml.safe_load(open(parsed.config_path))

    task = parsed.task
    device = torch.device(parsed.device)
    if "samples_output_path" not in config["sampling_params"]:
        config["sampling_params"]["samples_output_path"] = parsed.samples_output_path

    # TODO create a proper dataclass config
    config = RecNameSpace(**config)
    train(
        model_params=config.model_params,
        train_params=config.train_params,
        sampling_params=config.sampling_params,
        data_params=config.data_params,
        device=device,
        task=task,
    )
