
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from typing import Dict, Callable, Optional, Tuple

import numpy as np

import flwr as fl
from flwr.common import Metrics

import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from utils import Net, train, test
from dataset_utils import getCIFAR10, do_fl_partitioning, get_dataloader

# # Define metric aggregation function
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     print(metrics)
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}

# # Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "~/.flower/data/cifar-10"


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0", pretrained=True)
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
    

train_path, testset = getCIFAR10()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,
    eval_fn=get_eval_fn(testset),  # centralised testset evaluation of global model
)

# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config={"num_rounds": 6},
    strategy=strategy,
)
