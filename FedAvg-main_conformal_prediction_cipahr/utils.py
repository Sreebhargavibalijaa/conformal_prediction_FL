from typing import Any, Dict, List
import argparse
import os
import copy
import torch
import wandb
import statistics

class Logger:
    def __init__(self, args):
        self.args = args
        if args.wandb:
            wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
            self.wandb = wandb

    def log(self, logs: Dict[str, Any]) -> None:
        if self.wandb:
            self.wandb.log(logs)


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg #average_weights_losses_30
def average_weights_quant(weights: List[Dict[str, torch.Tensor]],client_losses) -> Dict[str, torch.Tensor]:
        weights_avg = copy.deepcopy(weights[0])
        for i in range(1, len(weights)):
            weights += weights[i]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

        return weights_avg
def average_weights_time_70_prune(weights: List[Dict[str, torch.Tensor]],client_losses) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        client_losses_final = client_losses
        client_losses.sort()
        for j in range(0,6):
            t = client_losses_final.index(client_losses[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg
def average_weights_time_30_prune(weights: List[Dict[str, torch.Tensor]],client_losses) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        client_losses_final = client_losses
        client_losses.sort()
        for j in range(0,14):
            t = client_losses_final.index(client_losses[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg
def average_weights_time_50_prune(weights: List[Dict[str, torch.Tensor]],client_losses) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        client_losses_final = client_losses
        client_losses.sort()
        for j in range(0,10):
            t = client_losses_final.index(client_losses[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg
def average_weights_losses_30(weights: List[Dict[str, torch.Tensor]],times) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        times_final = times
        times.sort()
        for j in range(0,14):
            t = times_final.index(times[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg
def average_weights_losses_50(weights: List[Dict[str, torch.Tensor]],times) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        times_final = times
        times.sort()
        for j in range(0,14):
            t = times_final.index(times[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg
def average_weights_losses_70(weights: List[Dict[str, torch.Tensor]],times) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        times_final = times
        times.sort()
        for j in range(0,6):
            t = times_final.index(times[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg
def average_weights_time_50_prune(weights: List[Dict[str, torch.Tensor]],times) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        times_final = times
        times.sort()
        for j in range(0,10):
            t = times_final.index(times[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg
def average_weights_time_70_prune(weights: List[Dict[str, torch.Tensor]],times) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        times_final = times
        times.sort()
        for j in range(0,6):
            t = times_final.index(times[j])
            weights_avg[key] += weights[t][key]

        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


def fedmed(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        weights_med = []
        for i in range(1, len(weights)):
            weights_med.append(weights[i][key])
        weights_med.sort() 
        mid = len(weights_med) // 2
        res = (weights_med[mid] + weights_med[~mid]) / 2
        weights_avg[key] = res
        # weights_avg[key] = weights_avg[key]

    return weights_avg #

def fedmode(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        weights_med = []
        for i in range(1, len(weights)):
            weights_med.append(weights[i][key])
            weights_med.sort() 
            mid = len(weights_med) // 2
            res = (weights_med[mid] + weights_med[~mid]) / 2

            weights_avg[key] = res
        weights_avg[key] = weights_avg[key]

    return statistics.mode(weights_med)

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--model_name", type=str, default="cnn")

    parser.add_argument("--non_iid", type=int, default=1)  # 0: IID, 1: Non-IID
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prune_percent", type=int, default=30)

    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="FedAvg")
    parser.add_argument("--exp_name", type=str, default="exp")

    return parser.parse_args()
