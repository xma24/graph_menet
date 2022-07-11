import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins import DDPPlugin
from rich import print
from termcolor import colored, cprint
import numpy as np

from expr_setting import ExprSetting

matplotlib.use("Agg")

plt.style.use("ggplot")

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument(
        "--config", default="config_default.yaml", help="train config file path"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # set up the experiment setting:
    #   - read config from the config_*.yaml files;
    #   - load model class according to the model name;
    #   - set up the checkpoing and logger;
    train_args = parse_args()
    expr_setting = ExprSetting(train_args.config)

    lr_logger, model_checkpoint, early_stop, model_class, dataloader_class, config = (
        expr_setting.lr_logger,
        expr_setting.model_checkpoint,
        expr_setting.early_stop,
        expr_setting.model_class,
        expr_setting.dataloader_class,
        expr_setting.config,
    )

    config["config_path"] = train_args.config
    os.makedirs(config["DATASET"]["WORK_DIRS"], exist_ok=True)

    seed_everything(config["SETTING"]["RANDOM_SEED"])

    if config["DATASET"]["CLASSES"] == "None":
        config["DATASET"]["CLASSES"] = np.arange(config["DATASET"]["NUM_CLASSES"])

    # modify the config if needed:
    if isinstance(config["TRAIN"]["NUM_GPUS"], int):
        num_gpus = config["TRAIN"]["NUM_GPUS"]
    elif config["TRAIN"]["NUM_GPUS"] == "autocount":
        config["TRAIN"]["NUM_GPUS"] = torch.cuda.device_count()
        num_gpus = config["TRAIN"]["NUM_GPUS"]
    else:
        gpu_list = config["TRAIN"]["NUM_GPUS"].split(",")
        num_gpus = len(gpu_list)

    if config["TRAIN"]["LOGGER"] == "neptune":
        print("Not implemented")
        exit(0)
    elif config["TRAIN"]["LOGGER"] == "csv":
        own_logger = CSVLogger(config["DATASET"]["LOGDIR"])
    else:
        own_logger = CSVLogger(config["DATASET"]["LOGDIR"])

    # prepare dataset:
    #   - the dataset name is encoded in config file;
    #       - when geting the dataset, we need the config data;

    # train_dataloader = dataloader_class.get_train_dataloader()

    train_dataloader_list, val_dataloader_list = dataloader_class.get_cv_dataloader()

    for i in range(config["DATASET"]["NUM_FOLD"]):

        train_dataloader = train_dataloader_list[i]
        val_dataloader = val_dataloader_list[i]

        print("train_dataloader: {}".format(len(train_dataloader)))
        config["TRAIN"]["BATCHES"] = len(train_dataloader) // num_gpus
        if (len(train_dataloader) // num_gpus) // 6 >= 10:
            config["TRAIN"]["NUM_SAVED_BATCHES"] = 10
        else:
            config["TRAIN"]["NUM_SAVED_BATCHES"] = (
                int((len(train_dataloader) // num_gpus) // 6) - 1
            )

        # val_dataloader = dataloader_class.get_val_dataloader()
        print("val_dataloader: {}".format(len(val_dataloader)))
        config["VAL"]["BATCHES"] = len(val_dataloader) // num_gpus
        if len(val_dataloader) % num_gpus == 0:
            config["NUM_BATCH_EACH_GPU"] = (len(val_dataloader) // num_gpus) - 1
        else:
            config["NUM_BATCH_EACH_GPU"] = len(val_dataloader) // num_gpus

        model = model_class(config, own_logger)

        print(">>> config: {}".format(config))

        if config["TRAIN"]["RESUME"] != "None":
            model = model_class.load_from_checkpoint(
                config["TRAIN"]["RESUME"], config=config, logger=own_logger
            )
            print(">>> Using checkpoint from pretrained models")
            # model = model.load_state_dict(torch.load(config["TRAIN"]["RESUME"]))

        if config["TRAIN"]["PRETRAIN"]:
            trainer = pl.Trainer(
                devices=config["TRAIN"]["NUM_GPUS"],
                # gpus=torch.cuda.device_count(),  ### let the code to detect the number of gpus to use
                num_nodes=config["TRAIN"]["NUM_NODES"],
                precision=config["TRAIN"]["PRECISION"],
                accelerator=config["TRAIN"]["ACCELERATOR"],
                logger=own_logger,
                callbacks=[lr_logger, model_checkpoint, early_stop],
                log_every_n_steps=1,
                # track_grad_norm=1,
                progress_bar_refresh_rate=config["TRAIN"]["PROGRESS_BAR_REFRESH_RATE"],
                max_epochs=config["TRAIN"]["MAX_EPOCHS"],
                resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
                # sync_batchnorm=True if num_gpus > 1 else False,
                plugins=DDPPlugin(find_unused_parameters=True),
                check_val_every_n_epoch=config["VAL"]["VAL_INTERVAL"],
            )
        else:
            # The setting of pytorch lightning Trainer:
            #   - https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/trainer/trainer.py
            if config["TRAIN"]["CPUS"]:
                print("using CPUs to do experiments ... ")
                trainer = pl.Trainer(
                    num_nodes=config["TRAIN"]["NUM_NODES"],
                    # precision=config["TRAIN"]["PRECISION"],
                    accelerator="cpu",
                    # strategy=config["TRAIN"]["STRATEGY"],
                    profiler=None,
                    logger=own_logger,
                    callbacks=[lr_logger, model_checkpoint, early_stop],
                    log_every_n_steps=1,
                    # track_grad_norm=1,
                    progress_bar_refresh_rate=config["TRAIN"][
                        "PROGRESS_BAR_REFRESH_RATE"
                    ],
                    max_epochs=config["TRAIN"]["MAX_EPOCHS"],
                    # resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
                    # sync_batchnorm=True if num_gpus > 1 else False,
                    plugins=DDPPlugin(find_unused_parameters=True),
                    check_val_every_n_epoch=config["VAL"]["VAL_INTERVAL"],
                    auto_scale_batch_size="binsearch",
                )
            else:
                print("using GPUs to do experiments ... ")
                trainer = pl.Trainer(
                    devices=config["TRAIN"]["NUM_GPUS"],
                    # gpus=torch.cuda.device_count(),  ### let the code to detect the number of gpus to use
                    num_nodes=config["TRAIN"]["NUM_NODES"],
                    precision=config["TRAIN"]["PRECISION"],
                    accelerator=config["TRAIN"]["ACCELERATOR"],
                    # strategy=config["TRAIN"]["STRATEGY"],
                    profiler=None,
                    logger=own_logger,
                    callbacks=[lr_logger, model_checkpoint, early_stop],
                    log_every_n_steps=1,
                    # track_grad_norm=1,
                    progress_bar_refresh_rate=config["TRAIN"][
                        "PROGRESS_BAR_REFRESH_RATE"
                    ],
                    max_epochs=config["TRAIN"]["MAX_EPOCHS"],
                    # resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
                    # sync_batchnorm=True if num_gpus > 1 else False,
                    plugins=DDPPlugin(find_unused_parameters=True),
                    check_val_every_n_epoch=config["VAL"]["VAL_INTERVAL"],
                    auto_scale_batch_size="binsearch",
                )

        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        # trainer.test(model, dataloaders=test_dataloader)

    #     # trainer.finetune(
    #     #     model,
    #     #     train_dataloader=train_dataloader,
    #     #     val_dataloaders=val_dataloader,
    #     #     strategy="freeze",
    #     # )

    # # Kill the training program ...
    # #   - kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')
