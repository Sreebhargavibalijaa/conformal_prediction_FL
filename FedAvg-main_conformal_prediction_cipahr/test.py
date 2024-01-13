
# def get_indices():
#     from pprint import PrettyPrinter
#     pp = PrettyPrinter(compact=True, indent=4, depth=3)
#     from functools import partial
#     import json
#     from collections import defaultdict
#     from operator import itemgetter
#     from pathlib import Path
#     import pandas as pd
#     import torch
#     import numpy as np
#     from tdigest import TDigest

#     import matplotlib.pyplot as plt; plt.style.use('bmh')
#     import matplotlib as mpl
#     from importlib import reload
#     import conformal  as cp
#     import temperature as ts
#     import helpers as helpers
#     reload(helpers)
#     reload(cp)
#     reload(ts)
#     from matplotlib import rcParams
#     rcParams['font.family'] = 'serif'
#     rcParams['font.sans-serif'] = ['Times']
#     def get_coverage(psets, targets, precision=None):
#         psets = psets.clone().detach()
#         targets = targets.clone().detach()
#         n = psets.shape[0]
#         coverage = psets[torch.arange(n), targets].float().mean().item()
#         if precision is not None:
#             coverage = round(coverage, precision)
#         return coverage
#     def inference_lac(scores, qhat, allow_empty_sets=False):
#         n = scores.size(0)

#         elements_mask = scores >= (1 - qhat)

#         if not allow_empty_sets:
#             elements_mask[torch.arange(n), scores.argmax(1)] = True

#         return elements_mask
#     def calibrate_lac(scores, targets, alpha=0.1, return_dist=False):
#         # assert scores.size(0) == targets.size(0)
#         # assert targets.size(0)
#         n = torch.tensor(targets.size(0))
#         assert n
#         # score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(-1), 1).flatten()
#         score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(-1).unsqueeze(-1), 1).flatten()

#         assert (
#             0 <= torch.ceil((n + 1) * (1 - alpha)) / n <= 1
#         ), f"{alpha=} {n=} {torch.ceil((n+1)*(1-alpha))/n=}"
#         # qhat = torch.quantile(
#         #     score_dist, torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
#         # )
#         subsample_size = 1000  # Adjust the size based on your needs
#         subsample = score_dist[:subsample_size]
#         qhat = torch.quantile(subsample.float(), torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")

#     #     qhat = torch.quantile(
#     #         score_dist.float(),  # or score_dist.double() if higher precision is desired
#     #         torch.ceil((n + 1) * (1 - alpha)) / n,
#     #         interpolation="higher"
#     # )
#         return (qhat, score_dist) if return_dist else qhat


#     import torch
#     from torchvision import datasets, transforms
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.calibration import CalibratedClassifierCV
#     from sklearn.model_selection import train_test_split

#     # Load MNIST data
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#     mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

#     # Split MNIST data into 10 clients
#     num_clients = 20
#     client_data = []
#     client_labels = []
#     lac_scores= []
#     for i in range(num_clients):
#         # Split the data into 10 equal parts
#         start_idx = i * len(mnist_train) // num_clients
#         end_idx = (i + 1) * len(mnist_train) // num_clients

#         # Extract data and labels for the current client
#         # client_train_data_i = mnist_train.data[start_idx:int(0.8*end_idx)].numpy()
#         # client_test_data_i = mnist_train.data[int(0.8*end_idx):end_idx].numpy()
#         # client_train_labels_i = mnist_train.targets[start_idx:int(0.8*end_idx)].numpy()
#         # client_test_labels_i = mnist_train.targets[int(0.8*end_idx):end_idx].numpy()
#         client_train_data_i = mnist_train.data[int(0.8*start_idx):int(0.9*end_idx)].numpy()
#         client_test_data_i = mnist_train.data[int(0.9*end_idx):end_idx].numpy()
#         client_train_labels_i = mnist_train.targets[int(0.8*start_idx):int(0.9*end_idx)].numpy()
#         client_test_labels_i = mnist_train.targets[int(0.9*end_idx):end_idx].numpy()

#         # client_data.append(client_train_data_i)
#         # client_labels.append(client_test_labels_i)
#     # for i in range(num_clients):
#         client_train_data_i = torch.tensor(client_train_data_i)
#         client_test_data_i = torch.tensor(client_test_data_i)
#         client_train_labels_i = torch.tensor(client_train_labels_i)
#         client_test_labels_i = torch.tensor(client_test_labels_i)
#         value = calibrate_lac(client_test_data_i, client_test_labels_i, alpha=0.1, return_dist=False)
#         psets = inference_lac(client_train_labels_i, value, allow_empty_sets=False)
#         coverage = get_coverage(psets, client_train_data_i, precision=None)
     
#         ##Coverage_value = get_coverage(psets,value)
#         coverage = coverage.tolist()
#         lac_scores.append(coverage)
#         print(value)

#     #     conformal_scores_per_client.append(conformal_scores_i)
#     #     targets_per_client.append(targets_i)

#     # # Print conformal scores and targets for each client
#     # for i in range(num_clients):
#     #     print(f"Client {i + 1} - Conformal Scores Shape: {conformal_scores_per_client[i].shape}, Targets Shape: {targets_per_client[i].shape}")
#     lac_scores_list = lac_scores.copy()
#     lac_scores_list_2 = lac_scores.copy()
#     lac_scores_list.sort()
#     lac_scores_list_indices = []
#     for i in range(0,14):
#         lac_scores_list_indices.append(lac_scores_list_2.index(lac_scores_list[i]))
#         lac_scores_list_2[lac_scores_list_2.index(lac_scores_list[i])] = -1000

#     print(lac_scores_list_indices)

#     # weights_avg = []
#     # for j in range(lac_scores_list_indices):
#     #     weights_avg = weights_avg+ weights[j]

#     # global_model.set_weights(weights_avg)
#     # global_model.acc
#     return lac_scores_list_indices
 