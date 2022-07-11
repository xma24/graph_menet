import matplotlib
import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

EPS = 1e-8


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CustomizeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, transform=None, using_PIL=None):

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.data = data_dict["data"]
        self.labels = data_dict["labels"]

        self.using_PIL = using_PIL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_ret = self.data[index]
        labels_ret = self.labels[index]

        if self.transform is not None:
            if self.using_PIL:
                # if the input data has already been normalized, times 255;
                if np.max(self.data[index]) < 1.1:
                    data_ret = self.transform(
                        Image.fromarray(np.uint8(self.data[index] * 255))
                    )
                else:
                    data_ret = self.transform(
                        Image.fromarray(np.uint8(self.data[index]))
                    )
            else:
                data_ret = self.transform(self.data[index])

        if not isinstance(data_ret, (dict)):
            data_ret = np.array(data_ret, dtype=np.float32)
            labels_ret = np.array(labels_ret, dtype=np.long)
        else:
            data_ret = data_ret
            labels_ret = np.array(labels_ret, dtype=np.long)

        return data_ret, labels_ret


class UnNormalize(object):
    @classmethod
    def denorm(self, config, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, config["mean"], config["std"]):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0, 1, 2), config["mean"], config["std"]):
                image[:, t, :, :].mul_(s).add_(m)

        return image


# Copyright (c) OpenMMLab. All rights reserved.
import logging

import torch.distributed as dist

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            f'"silent" or None, but got {type(logger)}'
        )


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """

    logger = get_logger(name="DeepModel", log_file=log_file, log_level=log_level)

    return logger


import os
import time


def csvlogger_start(config):
    # set up to log out the information from console
    #   - check the folder of csvlogger and get the version index and then find the path to save console logger
    os.makedirs(config["DATASET"]["LOGDIR"], exist_ok=True)
    # csv_root_default_folder = os.path.join(config["DATASET"]["LOGDIR"], "default")

    csv_root_default_folder = os.path.join(
        config["DATASET"]["LOGDIR"], "lightning_logs"
    )
    os.makedirs(csv_root_default_folder, exist_ok=True)

    if os.path.exists(os.path.join(csv_root_default_folder, "version_0")):
        subfolder_index_list = [
            int(f.split("_")[-1])
            for f in os.listdir(csv_root_default_folder)
            if os.path.isdir(os.path.join(csv_root_default_folder, f))
        ]
        print("subfolder_index_list: {}".format(subfolder_index_list))
        current_csv_version = max(subfolder_index_list)
        print("current_csv_version: {}".format(current_csv_version))
    else:
        current_csv_version = 0

    config["current_csv_version"] = current_csv_version

    current_csv_logger_folder = os.path.join(
        csv_root_default_folder, "version_" + str(current_csv_version)
    )
    # os.makedirs(current_csv_logger_folder, exist_ok=True)
    # if torch.cuda.current_device() == 0:
    # print(
    #     "torch.cuda.current_device(): {}, {}".format(
    #         torch.cuda.current_device(), type(torch.cuda.current_device())
    #     )
    # )
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(current_csv_logger_folder, f"{timestamp}.log")
    console_logger = get_root_logger(log_file=log_file, log_level=logging.INFO)

    return console_logger, current_csv_logger_folder, config
