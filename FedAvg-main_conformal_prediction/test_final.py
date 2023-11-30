
def get_indices(client_datasets,num_clients,n_client_epochs,model):
    print("len(client_datasets")
    print(len(client_datasets))
    print(len(client_datasets[0]))
    print(type(client_datasets))
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    from torch.quantization import quantize, prepare, convert
    # from pprint import PrettyPrinter
    # pp = PrettyPrinter(compact=True, indent=4, depth=3)
    # from functools import partial
    # import json
    # from collections import defaultdict
    # from operator import itemgetter
    # from pathlib import Path
    # import pandas as pd
    # import torch
    # import numpy as np
    # from tdigest import TDigest

    # import matplotlib.pyplot as plt; plt.style.use('bmh')
    # import matplotlib as mpl
    # from importlib import reload
    # import conformal  as cp
    # import temperature as ts
    # import helpers as helpers
    # reload(helpers)
    # reload(cp)
    # reload(ts)
    # from matplotlib import rcParams
    # rcParams['font.family'] = 'serif'
    # rcParams['font.sans-serif'] = ['Times']
    # def get_coverage(psets, targets, precision=None):
    #     psets = psets.clone().detach()
    #     targets = targets.clone().detach()
    #     n = psets.shape[0]
    #     coverage = psets[torch.arange(n), targets].float().mean().item()
    #     if precision is not None:
    #         coverage = round(coverage, precision)
    #     return coverage
    # def inference_lac(scores, qhat, allow_empty_sets=False):
    #     n = scores.size(0)

    #     elements_mask = scores >= (1 - qhat)

    #     if not allow_empty_sets:
    #         elements_mask[torch.arange(n), scores.argmax(1)] = True

    #     return elements_mask
    # def calibrate_lac(scores, targets, alpha=0.1, return_dist=False):
    #     # assert scores.size(0) == targets.size(0)
    #     # assert targets.size(0)
    #     n = torch.tensor(targets.size(0))
    #     assert n
    #     # score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(-1), 1).flatten()
    #     score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(-1).unsqueeze(-1), 1).flatten()

    #     assert (
    #         0 <= torch.ceil((n + 1) * (1 - alpha)) / n <= 1
    #     ), f"{alpha=} {n=} {torch.ceil((n+1)*(1-alpha))/n=}"
    #     # qhat = torch.quantile(
    #     #     score_dist, torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    #     # )
    #     subsample_size = 1000  # Adjust the size based on your needs
    #     subsample = score_dist[:subsample_size]
    #     qhat = torch.quantile(subsample.float(), torch.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")

    # #     qhat = torch.quantile(
    # #         score_dist.float(),  # or score_dist.double() if higher precision is desired
    # #         torch.ceil((n + 1) * (1 - alpha)) / n,
    # #         interpolation="higher"
    # # )
    #     return (qhat, score_dist) if return_dist else qhat


    # import torch
    # from torchvision import datasets, transforms
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.calibration import CalibratedClassifierCV
    # from sklearn.model_selection import train_test_split

    # # Load MNIST data
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # # Split MNIST data into 10 clients
    # num_clients = 20
    # client_data = []
    # client_labels = []
    # lac_scores= []
    # for i in range(num_clients):
    #     # Split the data into 10 equal parts
    #     start_idx = i * len(mnist_train) // num_clients
    #     end_idx = (i + 1) * len(mnist_train) // num_clients

    #     # Extract data and labels for the current client
    #     # client_train_data_i = mnist_train.data[start_idx:int(0.8*end_idx)].numpy()
    #     # client_test_data_i = mnist_train.data[int(0.8*end_idx):end_idx].numpy()
    #     # client_train_labels_i = mnist_train.targets[start_idx:int(0.8*end_idx)].numpy()
    #     # client_test_labels_i = mnist_train.targets[int(0.8*end_idx):end_idx].numpy()
    #     client_train_data_i = mnist_train.data[int(0.8*start_idx):int(0.9*end_idx)].numpy()
    #     client_test_data_i = mnist_train.data[int(0.9*end_idx):end_idx].numpy()
    #     client_train_labels_i = mnist_train.targets[int(0.8*start_idx):int(0.9*end_idx)].numpy()
    #     client_test_labels_i = mnist_train.targets[int(0.9*end_idx):end_idx].numpy()

    #     # client_data.append(client_train_data_i)
    #     # client_labels.append(client_test_labels_i)
    # # for i in range(num_clients):
    #     client_train_data_i = torch.tensor(client_train_data_i)
    #     client_test_data_i = torch.tensor(client_test_data_i)
    #     client_train_labels_i = torch.tensor(client_train_labels_i)
    #     client_test_labels_i = torch.tensor(client_test_labels_i)
    #     value = calibrate_lac(client_test_data_i, client_test_labels_i, alpha=0.1, return_dist=False)
    #     psets = inference_lac(client_train_labels_i, value, allow_empty_sets=False)
    #     coverage = get_coverage(psets, client_train_data_i, precision=None)
     
    #     ##Coverage_value = get_coverage(psets,value)
    #     coverage = coverage.tolist()
    #     lac_scores.append(coverage)
    #     print(value)

    # #     conformal_scores_per_client.append(conformal_scores_i)
    # #     targets_per_client.append(targets_i)

    # # # Print conformal scores and targets for each client
    # # for i in range(num_clients):
    # #     print(f"Client {i + 1} - Conformal Scores Shape: {conformal_scores_per_client[i].shape}, Targets Shape: {targets_per_client[i].shape}")
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical

    # Load MNIST data

    #for loop
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
    # x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # num_clients = 20
    # x_clients = np.array_split(x_train, num_clients)
    # y_clients = np.array_split(y_train, num_clients)
    psets_scores = []
    for i in range(num_clients):
        # Use a pre-trained model or train a model on MNIST
        # Replace the following line with your own model loading or training code
        # model = load_model('your_mnist_model.h5')

        # For demonstration purposes, let's use a simple example model
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(10, 5)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(5, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

# Instantiate the model
        model = SimpleNN()

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
     
        train_size = int(0.8 * len(client_datasets[i]))
        test_size = len(client_datasets[i]) - train_size

        train_set, test_set = random_split(client_datasets[i], [train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)



        print("Training client "+ str(i))



        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.fit(x_clients[i], y_clients[i], epochs=5, batch_size=64, validation_data=(x_test, y_test))
        for epoch in range(5):  # Adjust the number of epochs as needed
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_correct += (output.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)
            epoch_loss /= idx
            epoch_acc = epoch_correct / epoch_samples
            print(
            f"Client #{idx} | Epoch: {epoch}/{n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
            end="\r",
        )
        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predictions.append(outputs)

        # Concatenate predictions into a single tensor
        smx = torch.cat(predictions, dim=0)


        # Get softmax scores from the model
        # smx = model.predict(data)

        # Problem setup
        n = 1000  # number of calibration points
        alpha = 0.1  # 1-alpha is the desired coverage

        # Split the softmax scores into calibration and validation sets (save the shuffling)
        idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx, :], smx[~idx, :]
        cal_labels, val_labels = test_loader.argmax(axis=1)[idx], test_loader.argmax(axis=1)[~idx]

        # Conformal prediction
        # (The rest of the code remains the same)

        # 1: get conformal scores. n = cal_labels.shape[0]
        cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
        # 2: get adjusted quantile
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(cal_scores, q_level, interpolation='higher')
        prediction_sets = val_smx >= (1 - qhat)  # 3: form prediction sets

        # Calculate empirical coverage
        empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
        print(f"The empirical coverage is: {empirical_coverage}")
        psets_scores.append(empirical_coverage)
        # Show some examples
        # for i in range(10):
        #     rand_index = np.random.choice(np.where(~idx)[0])
        #     img = x_test[rand_index].reshape((28, 28))
        #     prediction_set = smx[rand_index] > 1 - qhat
        #     plt.figure()
        #     plt.imshow(img, cmap='gray')
        #     plt.axis('off')
        #     plt.show()
        #     print(f"The prediction set is: {list(np.where(prediction_set)[0])}")







    lac_scores_list = psets_scores.copy()
    lac_scores_list_2 = psets_scores.copy()
    lac_scores_list.sort()
    lac_scores_list_indices = []
    print(len(psets_scores))
    for i in range(0,14):
        lac_scores_list_indices.append(lac_scores_list_2.index(lac_scores_list[i]))
        lac_scores_list_2[lac_scores_list_2.index(lac_scores_list[i])] = -1000

    print(lac_scores_list_indices)

        # weights_avg = []
        # for j in range(lac_scores_list_indices):
        #     weights_avg = weights_avg+ weights[j]

        # global_model.set_weights(weights_avg)
        # global_model.acc
    return lac_scores_list_indices
 