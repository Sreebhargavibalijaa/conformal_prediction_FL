from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from data import MNISTDataset, FederatedSampler
from models import CNN, MLP
from utils import arg_parser, average_weights,fedmed, Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.quantization as quantization
from torchvision.datasets import CIFAR10
from torchvision import datasets

class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10Dataset, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """


    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )
        self.logger = Logger(args)

        self.train_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            n_shards=self.args.n_shards,
            non_iid=self.args.non_iid,
        )

        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                self.device
            )
            self.target_acc = 0.97
        elif self.args.model_name == "cnn":
            self.root_model = CNN(n_channels=1, n_classes=10).to(self.device)
            self.target_acc = 0.99
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        self.reached_target_at = None  # type: int

    def _get_data(
        self, root: str, n_clients: int, n_shards: int, non_iid: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            n_shards (int): number of shards.
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """
        train_set = CIFAR10Dataset(root=root, train=True)
        test_set = CIFAR10Dataset(root=root, train=False)

        sampler = FederatedSampler(
            train_set, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards
        )

        train_loader = DataLoader(train_set, batch_size=128, sampler=sampler)
        test_loader = DataLoader(test_set, batch_size=128)

        return train_loader, test_loader

    def calibrate_model(self,model, client_dataset):
            model.eval()
            data_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
            for data, target in data_loader:
                model(data)
    def quantize_modell(self,model, client_dataset):
    # Quantize the model
            quantized_model = quantization.quantize(model, run_fn=self.calibrate_model, run_args=[client_dataset])

            return quantized_model

# Define a calibration functi
    def quantize_models_final(self,model, idx):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, random_split
        from torch.quantization import quantize, prepare, convert
        print("sree" + str(idx))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load the MNIST dataset
        mnist_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # Define the number of clients
        num_clients = 20

        # Split the dataset into 10 clients
        client_datasets = random_split(mnist_dataset, [len(mnist_dataset)//num_clients]*num_clients)

        # Define a list to store quantized models for each client
        quantized_models = []

        # Loop over each client
        all_quant_models = []
        # for i, client_dataset in enumerate(client_datasets):
        print(f"Training and quantizing model for Client {idx}...")

        # Create a CNN model for each client
        # Train the model on the client's dataset (you may need to adjust training settings)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(client_datasets[idx], batch_size=64, shuffle=True)

        for epoch in range(5):  # Adjust the number of epochs as needed
            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        quantized_client_model = self.quantize_modell(model, client_datasets[idx])

    # Append the quantized model to the list
        quantized_models.append(quantized_client_model)
        return quantized_client_model.state_dict()

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        import torch
        model = copy.deepcopy(root_model)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )

        for epoch in range(self.args.n_client_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                logits = model(data)
                loss = F.nll_loss(logits, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)
            
            # Calculate average accuracy and loss
            epoch_loss /= idx
            epoch_acc = epoch_correct / epoch_samples

            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
                end="\r",
            )

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, random_split
        from torch.quantization import quantize, prepare, convert

        # print("sree" + str(client_idx))
        q_model = self.quantize_models_final(model,client_idx)
        # quantized_model = self.quantized_model_accuracies(model, logits, target)
        return model, epoch_loss / self.args.n_client_epochs, q_model

    def train(self) -> None:
        """Train a server model."""
        train_losses = []
        weights = []
        for epoch in range(self.args.n_epochs):
            clients_models = []
            clients_losses = []
            quantized_models= []
            quantized_models_weights = []
            # Randomly select clients
            # m = max(int(self.args.frac * self.args.n_clients),1)
            idx_clients = np.random.choice(range(self.args.n_clients), 20 , replace=False)
            # Train clients
            self.root_model.train()
            client_accuracies = []
            for client_idx in idx_clients:
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)
                # Train client
                client_model, client_loss, quantized_model = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                quantized_models_weights.append(quantized_model)
                finalls = [a for a in list(quantized_model.values())]
                weights = weights + finalls

                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)
                correct = 0
                total = 0
                with torch.no_grad():

                    for data in self.train_loader:
                        inputs, labels = data
                        outputs = client_model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                client_accuracy = 100 * correct / total
                client_accuracies.append(client_accuracy)
                # quantized_models_weights.append(quantized_model.state_dict())
            # Update server model based on clients models

            from statistics import mean
            # avg_updated_weights = mean(quantized_models_weights)
            # updated_weights = fedmed(clients_models)

# Load weights into the model
            self.root_model.load_state_dict(dict(zip(self.root_model.state_dict().keys(),weights)))

            # self.root_model.load_state_dict(weights)

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss, total_acc = self.test()
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "test/loss": total_loss,
                    "test/acc": total_acc,
                    "round": epoch,
                }
                if total_acc >= self.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    logs["reached_target_at"] = self.reached_target_at
                    print(
                        f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                    )

                # self.logger.log(logs)

                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                print(
                    f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                )

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break

    def test(self) -> Tuple[float, float]:
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)

            logits = self.root_model(data)
            loss = F.nll_loss(logits, target)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

        # calculate average accuracy and loss
        total_loss /= idx
        total_acc = total_correct / total_samples

        return total_loss, total_acc


if __name__ == "__main__":
    args = arg_parser()
    fed_avg = FedAvg(args)
    fed_avg.train()
