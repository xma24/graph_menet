import csv
import math
import random
from os import listdir
from os.path import isfile, join

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
import torch
from numpy import linalg as LA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import *


class ADNI_UNC(object):
    def __init__(self, config):
        self.config = config
        self.data_folder = os.path.join(
            self.config["DATASET"]["DATA_ROOT"], "ADNI", "AD_Data"
        )
        self.data_matrix_folder = os.path.join(self.data_folder, "AD-Data")
        self.unc_labels_file_path = os.path.join(self.data_folder, "DataTS.csv")
        self.unc_data, self.unc_labels = self.get_unc_matrix_data_labels()
        self.norm_unc_data = self.normalize_data()
        self.norm_unc_laplacian = self.laplacian_data()
        self.eigen_val_vec_array = self.matrix2eigen()
        self.unique_labels = None
        self.one_G_index, self.zero_G_index, self.CN_index, self.EMCI_index, self.LMCI_index, self.AD_index = (
            self.merge5to2groups()
        )

    def list_filenames(self):

        filenames = [
            f
            for f in listdir(self.data_matrix_folder)
            if isfile(join(self.data_matrix_folder, f))
        ]

        return filenames

    def get_unc_id_labels(self):

        unc_labels_path = self.unc_labels_file_path

        reader = csv.reader(open(unc_labels_path, "r"), delimiter=",")
        x = list(reader)
        data_labels_matrix = np.array(x)

        subject_id_col = np.array(data_labels_matrix[1:, 0]).reshape((-1, 1))
        label_col = np.array(data_labels_matrix[1:, 4]).reshape((-1, 1))

        id_labels = np.append(subject_id_col, label_col, axis=1)

        return id_labels

    def load_each_unc_matrix(self, filename):

        brain_matrix = np.loadtxt(filename)
        brain_matrix_min = np.min(brain_matrix)
        brain_matrix_max = np.max(brain_matrix)
        brain_matrix_mean = np.mean(brain_matrix)

        thre_50 = 0.5 * brain_matrix_max

        brain_matrix_bin = 1.0 * (brain_matrix > thre_50)
        inv_brain_matrix_bin = 1 - brain_matrix_bin
        fake_brain_matrix = thre_50 * np.ones_like(brain_matrix)

        brain_matrix = (
            brain_matrix_bin * fake_brain_matrix + inv_brain_matrix_bin * brain_matrix
        )

        # interval = brain_matrix_max / 10
        # brain_matrix_flatten = brain_matrix.reshape((-1,))
        # for k in range(10):
        #     each_interval_start = k * interval
        #     each_interval_end = (k + 1) * interval - 1
        #     count = 0

        #     for j in range(brain_matrix_flatten.shape[0]):
        #         if (
        #             brain_matrix_flatten[j] >= each_interval_start
        #             and brain_matrix_flatten[j] <= each_interval_end
        #         ):
        #             count += 1
        #     print("interval: {}, count: {}".format(k, count))

        # print(
        #     "Each brain matrix info: min {}, max {}, mean {}".format(
        #         brain_matrix_min, brain_matrix_max, brain_matrix_mean
        #     )
        # )

        brain_matrix_update = brain_matrix + brain_matrix.T

        return brain_matrix_update

    def get_unc_matrix_data_labels(self):

        matrix_dir = self.data_matrix_folder

        id_labels_alignment = self.get_unc_id_labels()
        class_list = np.unique(id_labels_alignment[:, 1])

        self.unique_labels = class_list

        brain_matrix_filenames = self.list_filenames()

        init_file = brain_matrix_filenames[0]
        init_matrix = np.expand_dims(
            self.load_each_unc_matrix(os.path.join(matrix_dir, init_file)), axis=0
        )

        unc_labels = np.zeros((1, 1))

        for file_index in range(len(brain_matrix_filenames)):

            file_name = brain_matrix_filenames[file_index]
            barin_matrix_each = self.load_each_unc_matrix(
                os.path.join(matrix_dir, file_name)
            )
            barin_matrix_each_ext = np.expand_dims(barin_matrix_each, axis=0)
            init_matrix = np.append(init_matrix, barin_matrix_each_ext, axis=0)

            file_name_list = file_name.split("_")
            subject_id = file_name_list[0]

            for id_label_index in range(id_labels_alignment.shape[0]):
                if id_labels_alignment[id_label_index, 0] == subject_id:
                    unc_label_each = np.argwhere(
                        class_list == id_labels_alignment[id_label_index, 1]
                    )
                    unc_labels = np.append(unc_labels, unc_label_each, axis=0)
                    break

            if file_index % 100 == 0:
                print("{}, {}, {}".format(file_index, file_name, unc_label_each[0, 0]))

        unc_data_update = init_matrix[1:]
        unc_labels_update = unc_labels[1:].reshape(-1).astype(np.int_)

        unc_raw_dict = {}
        unc_raw_dict["data"] = unc_data_update
        unc_raw_dict["labels"] = unc_labels_update
        unc_raw_dict["class"] = class_list

        print("unc_data_update: {}".format(unc_data_update.shape))
        print("unc_labels: {}".format(unc_labels_update.shape))

        return unc_data_update, unc_labels_update

    def normalize_data(self):
        unc_data_size = self.unc_data.shape

        unc_data_2d = self.unc_data.reshape(self.unc_data.shape[0], -1)

        unc_normalized_data = unc_data_2d / np.max(unc_data_2d, axis=1).reshape(-1, 1)

        unc_normalized_data_reshape = unc_normalized_data.reshape(unc_data_size)

        return unc_normalized_data_reshape

    def laplacian_data(self):

        normal_unc_laplacian = np.zeros(
            (1, self.unc_data.shape[1], self.unc_data.shape[2])
        )

        for i in range(self.unc_data.shape[0]):

            inner_raw_G = nx.from_numpy_matrix(self.unc_data[i, :].reshape((148, 148)))
            inner_norm_L = nx.normalized_laplacian_matrix(inner_raw_G).todense()

            inner_norm_L_array = np.squeeze(np.asarray(inner_norm_L))

            normal_unc_laplacian = np.append(
                normal_unc_laplacian,
                inner_norm_L_array.reshape(
                    (1, self.unc_data.shape[1], self.unc_data.shape[2])
                ),
                axis=0,
            )

        return normal_unc_laplacian[1:]

    @classmethod
    def test_eigen_function(self):
        input = np.diag((1, 2, 3))

        w, v = LA.eig(input)

        w = np.array(w).reshape(1, 3)
        v = np.array(v)

        w_v = np.concatenate([w, v], axis=0)

        x = np.matmul(
            (np.array(w_v[1:, 0]) * w_v[0, 0]).reshape(3, 1),
            np.array(w_v[1:, 0]).reshape(1, 3),
        )
        for i in range(1, w_v.shape[1]):
            x += np.matmul(
                (np.array(w_v[1:, i]) * w_v[0, i]).reshape(3, 1),
                np.array(w_v[1:, i]).reshape(1, 3),
            )

        output = x

        diff = output - input
        print("diff: {}".format(diff))

    def matrix2eigen(self):

        eigen_value_vec = np.zeros((1, 148 + 1, 148), dtype=np.float32)

        for data_i in range(self.norm_unc_laplacian.shape[0]):

            lap_matrix = self.norm_unc_laplacian[data_i]

            eigen_value, eigen_vector = LA.eig(lap_matrix)

            each_eigen_value = np.array(eigen_value).reshape(1, 148)
            each_eigen_vec = np.array(eigen_vector)

            each_eigen_value_vec = np.concatenate(
                [each_eigen_value, each_eigen_vec], axis=0
            ).reshape((1, 148 + 1, 148))

            eigen_value_vec = np.append(eigen_value_vec, each_eigen_value_vec, axis=0)

        return eigen_value_vec[1:]

    def merge5to2groups(self):
        # class_list:  ['AD' 'CN' 'EMCI' 'LMCI' 'SMC']

        AD_index = np.where(self.unc_labels == 0)[0]
        LMCI_index = np.where(self.unc_labels == 3)[0]

        CN_index = np.where(self.unc_labels == 1)[0]
        EMCI_index = np.where(self.unc_labels == 2)[0]

        one_group_index = np.append(AD_index, LMCI_index, axis=0)
        zero_group_index = np.append(CN_index, EMCI_index, axis=0)

        return (
            one_group_index,
            zero_group_index,
            CN_index,
            EMCI_index,
            LMCI_index,
            AD_index,
        )

    def batch_image_show(self, batch_images, file_name=""):
        figure_number = batch_images.shape[0]

        figure_number_H = int(np.sqrt(figure_number)) + 1

        fig = plt.figure()
        for i in range(figure_number):
            ax = fig.add_subplot(figure_number_H, figure_number_H, i + 1)

            each_image = batch_images[i]
            if len(batch_images.shape) == 3:
                each_image = np.expand_dims(each_image, axis=-1)

                if np.max(each_image) == 1:
                    each_image *= 255
                    each_image = 255 - each_image

                plt.axis("off")
                plt.imshow(each_image.astype(np.uint8))

        plt.savefig(file_name + ".pdf")
        plt.close(fig)

    def data_visualization(self, unc_dict):
        unc_data = unc_dict["data"]
        unc_labels = unc_dict["labels"]

        unc_class = unc_dict["class"]

        for i in range(len(unc_class)):
            each_class_index = np.where(unc_labels == i)

            each_class_data = unc_data[each_class_index]

            self.batch_image_show(each_class_data, file_name=unc_class[i])


class UniDataset:
    def __init__(self, config):
        self.config = config
        print("==>> self.config: ", self.config)

        if self.config["DATASET"]["CV"]:
            self.num_folds = self.config["DATASET"]["NUM_FOLD"]

        self.train_transform = transforms.Compose([transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToTensor()])

    def get_datasets(self):

        adni_class = ADNI_UNC(self.config)
        adni_matrix = adni_class.unc_data
        adni_norm_matrix = adni_class.norm_unc_data
        adni_norm_laplacian = adni_class.norm_unc_laplacian
        adni_labels = adni_class.unc_labels
        adni_eigen_val_vec = adni_class.eigen_val_vec_array

        train_dict = {}
        data_with_one_type = adni_eigen_val_vec

        if self.config["DATASET"]["NUM_CLASSES"] == 2:
            data_CN = np.array(data_with_one_type)[adni_class.CN_index]
            data_EMCI = np.array(data_with_one_type)[adni_class.EMCI_index]
            data_LMCI = np.array(data_with_one_type)[adni_class.LMCI_index]
            data_AD = np.array(data_with_one_type)[adni_class.AD_index]

            data_dict = {}
            data_dict["CN"] = data_CN
            data_dict["EMCI"] = data_EMCI
            data_dict["LMCI"] = data_LMCI
            data_dict["AD"] = data_AD

            # data_np_one_group = np.array(data_with_one_type)[adni_class.one_G_index]
            # data_np_zero_group = np.array(data_with_one_type)[adni_class.zero_G_index]

            # data_np = np.append(data_np_one_group, data_np_zero_group, axis=0)

            # labels_np_one_group = np.ones(
            #     adni_class.one_G_index.shape[0], dtype=np.int32
            # )
            # labels_np_zero_group = np.zeros(
            #     adni_class.zero_G_index.shape[0], dtype=np.int32
            # )

            # labels_np = np.append(labels_np_one_group, labels_np_zero_group, axis=0)

        # else:
        #     data_np = np.array(data_with_one_type)
        #     labels_np = np.array(adni_labels)

        # data_labels = np.append(
        #     data_np.reshape(data_np.shape[0], -1), labels_np.reshape(-1, 1), axis=1
        # )
        # # np.random.shuffle(data_labels)

        # data_np_update = data_labels[:, :-1].reshape(data_np.shape)
        # labels_np_update = np.array(
        #     data_labels[:, -1].reshape(labels_np.shape), dtype=np.int32
        # )

        train_dict["data"] = data_dict
        # train_dict["labels"] = labels_np_update

        # train_data_number = int(0.85 * (data_np_update.shape[0]))

        # train_data = data_np_update[:train_data_number]
        # train_labels = labels_np_update[:train_data_number]
        # test_data = data_np_update[train_data_number:]
        # test_labels = labels_np_update[train_data_number:]

        # train_val_dataset_dict = {}
        # train_val_dataset_dict["data"] = np.array(train_data)
        # train_val_dataset_dict["labels"] = np.array(train_labels)

        # test_dataset_dict = {}
        # test_dataset_dict["data"] = np.array(test_data)
        # test_dataset_dict["labels"] = np.array(test_labels)

        # train_val_dataset = CustomizeDataset(
        #     train_val_dataset_dict, transform=self.train_transform
        # )

        # test_dataset = CustomizeDataset(
        #     test_dataset_dict, transform=self.train_transform
        # )

        # all_dataset = CustomizeDataset(train_dict, transform=self.train_transform)
        all_dict = train_dict

        return all_dict

    # def get_dataloaders(self):

    #     if self.config["DATASET"]["CV"]:
    #         from sklearn.model_selection import KFold

    #         trian_dataloader_list = []
    #         val_dataloader_list = []

    #         kfold = KFold(n_splits=self.num_folds)

    #         for fold, (train_idx, valid_idx) in enumerate(
    #             kfold.split(self.train_val_dataset_dict["data"])
    #         ):

    #             trian_data, train_labels = (
    #                 self.train_val_dataset_dict["data"][train_idx],
    #                 self.train_val_dataset_dict["labels"][train_idx],
    #             )

    #             val_data, val_labels = (
    #                 self.train_val_dataset_dict["data"][valid_idx],
    #                 self.train_val_dataset_dict["labels"][valid_idx],
    #             )

    #             train_dataset_dict = {}
    #             train_dataset_dict["data"] = trian_data
    #             train_dataset_dict["labels"] = train_labels

    #             val_dataset_dict = {}
    #             val_dataset_dict["data"] = val_data
    #             val_dataset_dict["labels"] = val_labels

    #             train_dataset = CustomizeDataset(
    #                 train_dataset_dict, transform=self.train_transform
    #             )
    #             val_dataset = CustomizeDataset(
    #                 val_dataset_dict, transform=self.train_transform
    #             )

    #             train_dataloader = DataLoader(
    #                 train_dataset,
    #                 batch_size=self.config["TRAIN"]["BATCH_SIZE"],
    #                 shuffle=True,
    #                 num_workers=self.config["DATASET"]["WORKERS"],
    #                 pin_memory=self.config["DATASET"]["PIN_MEMORY"],
    #             )

    #             val_dataloader = DataLoader(
    #                 val_dataset,
    #                 batch_size=self.config["TRAIN"]["BATCH_SIZE"],
    #                 shuffle=False,
    #                 num_workers=self.config["DATASET"]["WORKERS"],
    #                 pin_memory=self.config["DATASET"]["PIN_MEMORY"],
    #             )

    #             print("train_dataloader: {}".format(len(train_dataloader)))
    #             print("val_dataloader: {}".format(len(val_dataloader)))

    #             trian_dataloader_list.append(train_dataloader)
    #             val_dataloader_list.append(val_dataloader)

    #         return trian_dataloader_list, val_dataloader_list
    #     else:

    #         test_dataloader = DataLoader(
    #             self.test_dataset,
    #             batch_size=self.config["TRAIN"]["BATCH_SIZE"],
    #             shuffle=False,
    #             num_workers=self.config["DATASET"]["WORKERS"],
    #             pin_memory=self.config["DATASET"]["PIN_MEMORY"],
    #         )

    #         if self.config["DATASET"]["VAL"]:
    #             len_train_val_dataset = len(self.train_val_dataset)
    #             train_len = round(0.85 * len_train_val_dataset)
    #             val_len = len_train_val_dataset - train_len
    #             train_dataset, val_dataset = torch.utils.data.random_split(
    #                 self.train_val_dataset,
    #                 [train_len, val_len],
    #                 generator=torch.Generator().manual_seed(42),
    #             )

    #             train_dataloader = DataLoader(
    #                 train_dataset,
    #                 batch_size=self.config["TRAIN"]["BATCH_SIZE"],
    #                 shuffle=True,
    #                 num_workers=self.config["DATASET"]["WORKERS"],
    #                 pin_memory=self.config["DATASET"]["PIN_MEMORY"],
    #             )

    #             val_dataloader = DataLoader(
    #                 val_dataset,
    #                 batch_size=self.config["TRAIN"]["BATCH_SIZE"],
    #                 shuffle=False,
    #                 num_workers=self.config["DATASET"]["WORKERS"],
    #                 pin_memory=self.config["DATASET"]["PIN_MEMORY"],
    #             )

    #             return train_dataloader, val_dataloader, test_dataloader
    #         else:
    #             train_dataset = self.train_val_dataset
    #             train_dataloader = DataLoader(
    #                 train_dataset,
    #                 batch_size=self.config["TRAIN"]["BATCH_SIZE"],
    #                 shuffle=True,
    #                 num_workers=self.config["DATASET"]["WORKERS"],
    #                 pin_memory=self.config["DATASET"]["PIN_MEMORY"],
    #             )
    #             return train_dataloader, test_dataloader


class UniDataloader:
    def __init__(self, config):
        super(UniDataloader, self).__init__()

        self.config = config

        self.dataset_class = UniDataset(config)

        self.all_data_dict = self.dataset_class.get_datasets()

    def get_dataloader(self, split, batch_size=None):
        assert split in ("train", "val", "test", "cv"), "Unknown split '{}'".format(
            split
        )

        kwargs = {
            "num_workers": self.config["DATASET"]["WORKERS"],
            "pin_memory": self.config["DATASET"]["PIN_MEMORY"],
        }

        # if self.config["DATASET"]["FIX_TRAIN_DATA"]:
        #     shuffle, drop_last = [False, False]
        # else:
        #     shuffle, drop_last = [True, True] if split == "train" else [False, False]

        # if self.config is None:
        #     self.batch_size = batch_size
        # else:
        #     self.batch_size = self.config["TRAIN"]["BATCH_SIZE"]

        # return torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     drop_last=drop_last,
        #     shuffle=shuffle,
        #     **kwargs
        # )

        if split == "cv":

            from sklearn.model_selection import KFold

            trian_dataloader_list = []
            val_dataloader_list = []

            kfold = KFold(n_splits=self.config["DATASET"]["NUM_FOLD"])

            CN_data_train_list = []
            CN_data_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["data"]["CN"])
            ):
                each_fold_CN_train = self.all_data_dict["data"]["CN"][train_idx]
                each_fold_CN_val = self.all_data_dict["data"]["CN"][valid_idx]
                CN_data_train_list.append(each_fold_CN_train)
                CN_data_val_list.append(each_fold_CN_val)

            EMCI_data_train_list = []
            EMCI_data_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["data"]["EMCI"])
            ):
                each_fold_EMCI_train = self.all_data_dict["data"]["EMCI"][train_idx]
                each_fold_EMCI_val = self.all_data_dict["data"]["EMCI"][valid_idx]
                EMCI_data_train_list.append(each_fold_EMCI_train)
                EMCI_data_val_list.append(each_fold_EMCI_val)

            LMCI_data_train_list = []
            LMCI_data_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["data"]["LMCI"])
            ):
                each_fold_LMCI_train = self.all_data_dict["data"]["LMCI"][train_idx]
                each_fold_LMCI_val = self.all_data_dict["data"]["LMCI"][valid_idx]
                LMCI_data_train_list.append(each_fold_LMCI_train)
                LMCI_data_val_list.append(each_fold_LMCI_val)

            AD_data_train_list = []
            AD_data_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["data"]["AD"])
            ):
                each_fold_AD_train = self.all_data_dict["data"]["AD"][train_idx]
                each_fold_AD_val = self.all_data_dict["data"]["AD"][valid_idx]
                AD_data_train_list.append(each_fold_AD_train)
                AD_data_val_list.append(each_fold_AD_val)

            for fold_i in range(self.config["DATASET"]["NUM_FOLD"]):
                each_train_ONE_data = np.append(
                    np.array(AD_data_train_list[fold_i]),
                    np.array(LMCI_data_train_list[fold_i]),
                    axis=0,
                )
                each_train_ZERO_data = np.append(
                    np.array(CN_data_train_list[fold_i]),
                    np.array(EMCI_data_train_list[fold_i]),
                    axis=0,
                )
                each_train_ONE_label = np.ones(each_train_ONE_data.shape[0])
                each_train_ZERO_label = np.zeros(each_train_ZERO_data.shape[0])

                each_train_data = np.append(
                    each_train_ONE_data, each_train_ZERO_data, axis=0
                )
                each_train_label = np.append(
                    each_train_ONE_label, each_train_ZERO_label, axis=0
                )

                each_train_dataset_dict = {}
                each_train_dataset_dict["data"] = each_train_data
                each_train_dataset_dict["labels"] = each_train_label

                # validation
                each_val_ONE_data = np.append(
                    np.array(AD_data_val_list[fold_i]),
                    np.array(LMCI_data_val_list[fold_i]),
                    axis=0,
                )
                each_val_ZERO_data = np.append(
                    np.array(CN_data_val_list[fold_i]),
                    np.array(EMCI_data_val_list[fold_i]),
                    axis=0,
                )
                each_val_ONE_label = np.ones(each_val_ONE_data.shape[0])
                each_val_ZERO_label = np.zeros(each_val_ZERO_data.shape[0])

                each_val_data = np.append(each_val_ONE_data, each_val_ZERO_data, axis=0)
                each_val_label = np.append(
                    each_val_ONE_label, each_val_ZERO_label, axis=0
                )

                each_val_dataset_dict = {}
                each_val_dataset_dict["data"] = each_val_data
                each_val_dataset_dict["labels"] = each_val_label

                # create dataloader
                train_dataset = CustomizeDataset(
                    each_train_dataset_dict,
                    transform=self.dataset_class.train_transform,
                )
                val_dataset = CustomizeDataset(
                    each_val_dataset_dict, transform=self.dataset_class.train_transform
                )

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=self.config["TRAIN"]["BATCH_SIZE"],
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.config["DATASET"]["WORKERS"],
                    pin_memory=self.config["DATASET"]["PIN_MEMORY"],
                )

                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.config["TRAIN"]["BATCH_SIZE"],
                    shuffle=False,
                    num_workers=self.config["DATASET"]["WORKERS"],
                    pin_memory=self.config["DATASET"]["PIN_MEMORY"],
                )

                print("train_dataloader: {}".format(len(train_dataloader)))
                print("val_dataloader: {}".format(len(val_dataloader)))

                trian_dataloader_list.append(train_dataloader)
                val_dataloader_list.append(val_dataloader)

            return trian_dataloader_list, val_dataloader_list

    def get_train_dataloader(self):

        self.train_loader = self.get_dataloader("train")

        return self.train_loader

    def get_val_dataloader(self):

        self.val_loader = self.get_dataloader("val")

        return self.val_loader

    def get_test_dataloader(self):

        self.test_loader = self.get_dataloader("test")

        return self.test_loader

    def get_cv_dataloader(self):
        self.train_loader_list, self.val_loader_list = self.get_dataloader("cv")
        return self.train_loader_list, self.val_loader_list


if __name__ == "__main__":
    import warnings

    from rich import print

    warnings.filterwarnings("ignore")

    from expr_setting import ExprSetting

    if __name__ == "__main__":

        import yaml

        config_path = "configs/config_v3.yaml"

        # with open(config_path) as file:
        #     config = yaml.safe_load(file)
        # print("==>> config: ", config)

        expr_setting = ExprSetting(config_path)

        lr_logger, model_checkpoint, early_stop, model_class, dataloader_class, config = (
            expr_setting.lr_logger,
            expr_setting.model_checkpoint,
            expr_setting.early_stop,
            expr_setting.model_class,
            expr_setting.dataloader_class,
            expr_setting.config,
        )

        trian_dataloader_list, val_dataloader_list = (
            dataloader_class.get_cv_dataloader()
        )
        print("trian_dataloader_list: {}".format(len(trian_dataloader_list)))
        print("val_dataloader_list: {}".format(len(val_dataloader_list)))

        for fold_index in range(config["DATASET"]["NUM_FOLD"]):
            train_dataloader = trian_dataloader_list[fold_index]
            val_dataloader = val_dataloader_list[fold_index]

        # dataset_class = ADNIDataset(config)

        # dataset_class = prepare_dataset.get_dataset_class(config)
        # dataset_instance = ADNIDataset(config)

        # if config["DATASET"]["CV"]:

        #     (
        #         trian_dataloader_list,
        #         val_dataloader_list,
        #     ) = dataset_instance.get_dataloaders()
        #     print("trian_dataloader_list: {}".format(len(trian_dataloader_list)))
        #     print("val_dataloader_list: {}".format(len(val_dataloader_list)))

        #     for fold_index in range(config["DATASET"]["NUM_FOLD"]):
        #         train_dataloader = trian_dataloader_list[fold_index]
        #         val_dataloader = val_dataloader_list[fold_index]

        # else:
        #     if config["DATASET"]["VAL"]:
        #         print("creating training, val and testing data ... ")
        #         # dataset_instance = dataset_class(config, validation=True)
        #         (
        #             train_dataloader,
        #             val_dataloader,
        #             test_dataloader,
        #         ) = dataset_instance.get_dataloaders()
        #         print(
        #             "train_dataloader: {}, val_dataloader: {}, test_dataloader: {}".format(
        #                 len(train_dataloader), len(val_dataloader), len(test_dataloader)
        #             )
        #         )

        #     else:
        #         print("creating training and testing data ... ")
        #         # dataset_instance = dataset_class(config)
        #         train_dataloader, test_dataloader = dataset_instance.get_dataloaders()
        #         print(
        #             "train_dataloader: {}, test_dataloader: {}".format(
        #                 len(train_dataloader), len(test_dataloader)
        #             )
        #         )

        #         for batch_idx, (data, labels) in enumerate(train_dataloader):
        #             print(
        #                 "batch_idx: {}, data: {}, labels: {}".format(
        #                     batch_idx, data.shape, labels
        #                 )
        #             )

