import argparse
import yaml
from utils import (
    RecNameSpace,
    ImageDataset,
    init_dataloader,
    RealNVP,
    save_plot,
    check_point_model,
)
import torch
import os
import logging
import datetime
import time
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)


def train(
    model_params: SimpleNamespace,
    train_params: SimpleNamespace,
    sampling_params: SimpleNamespace,
    device: torch.device,
    task: str,
):
    time_now = datetime.datetime.utcnow()
    time_now_str = datetime.datetime.strftime(time_now, format="%Y%m%d_%H%M%S")

    if task == "train":
        params_to_save = {}
        params_to_save["model_params"] = model_params.__dict__
        params_to_save["train_params"] = train_params.__dict__

        # in order to avoid overriding add a timestamp if model is being trained with the same config
        sampling_params.samples_output_path = os.path.join(
            sampling_params.samples_output_path, time_now_str
        )
        train_params.model_checkpoint_path = os.path.join(
            train_params.model_checkpoint_path,time_now_str
        )
        for path in [
            train_params.model_checkpoint_path,
            sampling_params.samples_output_path,
        ]:
            logging.info(f"creating path  {path}")
            os.makedirs(path)
            yaml.safe_dump(
                params_to_save,
                open(os.path.join(path, "params.json"), "w"),
            )

    dataset = ImageDataset(root_dir=train_params.dataset_path)
    dataloader = init_dataloader(dataset, batch_size=train_params.batch_size)

    realnvp = RealNVP(
        device=device,
        num_scales=model_params.num_scales,
        num_step_of_flow=model_params.num_step_of_flow,
        num_resnet_blocks=model_params.num_resnet_blocks,
    ).to(device)
    optimizer = torch.optim.Adam(lr=train_params.lr, params=realnvp.parameters())
    logging.info(
        f"number of parameters in the model {sum([p.numel() for p in realnvp.parameters()])}"
    )
    
    best_loss = torch.inf
    # sampling parameters
    # TODO move realNVP params to config, that is based input shape and ...
    n_row, n_column = (
        sampling_params.num_samples_nrow,
        sampling_params.num_samples_ncols,
    )
    n_channels, h, w = realnvp.final_scale.base_input_shape
    z_base_sample_shape = [n_row * n_column, n_channels, h, w]

    if sampling_params.generate_fixed_images is True:
        z_base_sample = torch.distributions.Normal(0, 1).sample(
            sample_shape=z_base_sample_shape
        )

    loss_per_epoch = []
    for e in range(0, train_params.epochs):
        loss_per_bath = []
        epoch_start_time = time.time()
        for i, data in enumerate(dataloader):
            image_batch = data.to(device)
            optimizer.zero_grad()
            z, loss = realnvp(image_batch)
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
                model=realnvp,
                epoch=e,
                optimizer=optimizer,
                loss=epoch_loss,
                path=train_params.model_checkpoint_path,
            )

        realnvp.eval()

        generated_image = realnvp.sample(n_row * n_column, z_base_sample=None).view(
            n_row, n_column, 3, 32, 32
        )
        save_plot(
            n_row=n_row,
            n_column=n_column,
            path=sampling_params.samples_output_path,
            generated_image=generated_image,
            epoch=e,
            fixed_image=False,
        )
        if sampling_params.generate_fixed_images is True:
            generated_image_fixed = realnvp.sample(
                n_row * n_column, z_base_sample=z_base_sample
            ).view(n_row, n_column, 3, 32, 32)
            save_plot(
                n_row=n_row,
                n_column=n_column,
                path=sampling_params.samples_output_path,
                generated_image=generated_image_fixed,
                epoch=e,
                fixed_image=True,
            )

        realnvp.train()

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
    args.add_argument("--task", help="train or resume", default="train")
    parsed = args.parse_args()
    config = yaml.safe_load(open(parsed.config_path))
    if parsed.task == "resume":
        # TODO: in order to resume, make sure that the correct sample path and model checkpoints are provided
        raise NotImplementedError(
            "resume not implemented"
        )  # TODO: fix this later when we suppoer resume

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
        device=device,
        task=task,
    )
