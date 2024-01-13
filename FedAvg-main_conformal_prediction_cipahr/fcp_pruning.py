
def get_indices(client_datasets,num_clients,n_client_epochs,model, Pruning_percentage,model_name):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    from torch.quantization import quantize, prepare, convert
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from models import CNN, MLP


    psets_scores = []
    for i in range(num_clients):
# Instantiate the model
        if model_name == "mlp":
            model = MLP(input_size=784, hidden_size=128, n_classes=10).to(
                "cpu"
            )
        elif model_name == "cnn":
            model = CNN(n_channels=3, n_classes=10).to("cpu")
        else:
            raise ValueError(f"Invalid model name ")
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
     
        train_size = int(0.8 * len(client_datasets[i]))
        test_size = len(client_datasets[i]) - train_size

        train_set, test_set = random_split(client_datasets[i], [train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        print("Training client: "+ str(i))
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
            epoch_loss /= 2
            epoch_acc = epoch_correct / epoch_samples
            print(
            f"Client #{i} | Epoch: {epoch}/{n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
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
        # Problem setup
        n = 1000  # number of calibration points
        alpha = 0.1  # 1-alpha is the desired coverage

        # Split the softmax scores into calibration and validation sets (save the shuffling)
        idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx, :], smx[~idx, :]
        #test_loader.argmax(axis=1)[idx], test_loader.argmax(axis=1)[~idx]
        all_labels = []
        idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx, :], smx[~idx, :]

        # Iterate over the test_loader
        for _, labels in test_loader:
            all_labels.extend(labels.cpu().numpy())

        all_labels = np.array(all_labels)

        # Apply the same indexing to get calibration and validation labels
        cal_labels, val_labels = all_labels[idx], all_labels[~idx]

        # Conformal prediction
        # 1: get conformal scores. n = cal_labels.shape[0]
        cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
        # 2: get adjusted quantile
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(cal_scores, q_level, interpolation='higher')
        prediction_sets = val_smx >= (1 - qhat)  # 3: form prediction sets

        # Calculate empirical coverage
        correct_predictions = prediction_sets[torch.arange(prediction_sets.size(0)), val_labels]

        # Calculate empirical coverage
        empirical_coverage = correct_predictions.float().mean().item()

        print(f"The empirical coverage is: {empirical_coverage}")
        psets_scores.append(empirical_coverage)


    lac_scores_list = psets_scores.copy()
    lac_scores_list_2 = psets_scores.copy()
    lac_scores_list.sort()
    lac_scores_list_indices = []
    print(len(psets_scores))
    for i in range(0,int(num_clients*(1-(Pruning_percentage/100)))):
        lac_scores_list_indices.append(lac_scores_list_2.index(lac_scores_list[i]))
        lac_scores_list_2[lac_scores_list_2.index(lac_scores_list[i])] = -1000

    print(lac_scores_list_indices)
    return lac_scores_list_indices
 