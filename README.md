# graph_menet

Code for **Learning Multi-resolution Graph Edge Embedding for Discovering Brain Network Dysfunction in Neurological Disorders**.

## Code Structure
The codes include the following files:

- **configs/config_v0.yaml**: the setting of hyperparameters. 
- **dataloader_v0.py**: data loader generator.
- **model_v0.py**: the MENET model.
- **expr_setting.py**: the settings of the experiments, including the logger, checkpoint.
- **utils.py**: functions related to the project.
- **main_train.py**: the primary function to merge all the above functions to run experiments.
- **main_dist_train.sh**: The shell script to run the experiments.

In the project folder, a Dockerfile is used to set up the environment to run the experiments.
To generate the docker image, we can use the following command in the project folder:
```
docker build -t menet_torch:v1 .
```

## Data Prepare

### Use ADNI dataset
Each brain network in the experiment is the structural brain networks using Destrieux atlas with 148 ROIs. Each brain network is given as an adjacency matrix whose elements denote the number of neuron ﬁber tracts connecting two diﬀerent ROIs. In the experiments, we merged control (CN) and Early Mild Cognitive Impairment (EMCI) groups as Pre-clinical AD group and combined Late Mild Cognitive Impairment (LMCI) and AD groups as Prodromal AD group to ensure suﬃcient sample size and compare their subtle diﬀerences. 

When you create the ADNI dataset, please organize the samples with the following information. 

The number of samples in each class is as follows:
```
Preclinical AD
    |-- CN: 109
    |-- EMCI: 167
Prodromal AD:
    |-- LMCI: 94
    |-- AD: 77
```
If you use the ADNI dataset, please refer to the log information we provide. The epoch number is set to 200.

### Use your own dataset
Since our model directly works with graph edges, you only need to get the adjacency matrix of each graph to do graph classification. Please change the `NUM_ROI` to the number of graph nodes.



## Run the experiments
The configuration of the model is located in the `config_vXX.yaml` file. Please use the following command to run experiments.
```
bash main_dist_train.sh --config=./configs/config_v0.yaml
```
The dataset should be put in the `data/` folder, and the results will be saved in the `work_dirs/` folder.


## Update
In our original implementation, the features of MENET are concatenated together to feed into an MLP. While in our latest implementation, those features are added to be fed into an MLP. The number of parameters will be reduced, and the performance can increase further.

The number of scales is set to 5 in the original paper. This is a hyperparameter. In our experiments, the number does not need to be a large number. 