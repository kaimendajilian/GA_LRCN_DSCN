import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from sklearn.metrics import roc_auc_score
import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import preprocess
import pandas as pd
import matplotlib.pyplot as plt


from model import GA_LRCN_DSCN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

posi = r"C:\Users\data\ALKBH5_Baltz2012.train.positives.fa"
nega = r"C:\Users\data\ALKBH5_Baltz2012.train.negatives.fa"
train_bags, train_labels = preprocess.get_data(posi, nega, window_size=501)

train_bags, train_labels = np.array(train_bags), np.array(train_labels)
shuffle_index = torch.randperm(2170)
train_x = np.zeros_like(train_bags)
train_y = np.zeros_like(train_labels)
for i in range(2170):
    train_x[i] = train_bags[shuffle_index[i]]
    train_y[i] = train_labels[shuffle_index[i]]
train_bags, train_labels = train_x, train_y

val_p = r"C:\Users\data\ALKBH5_Baltz2012.val.positives.fa"
val_n = r"C:\Users\data\ALKBH5_Baltz2012.val.negatives.fa"

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_folder = os.path.join('result', current_time)
os.makedirs(result_folder, exist_ok=True)

val_bags, val_labels = preprocess.get_data(val_p, val_n, window_size=501)

# 数据打乱
val_bags, val_labels = np.array(val_bags), np.array(val_labels)
shuffle_index = torch.randperm(240)
val_x = np.zeros_like(val_bags)
val_y = np.zeros_like(val_labels)
for i in range(240):
    val_x[i] = val_bags[shuffle_index[i]]
    val_y[i] = val_labels[shuffle_index[i]]
val_bags, val_labels = val_x, val_y

train_bags = torch.tensor(np.array(train_bags), dtype=torch.float32)
train_labels = torch.tensor(np.array(train_labels), dtype=torch.long)
val_bags = torch.tensor(np.array(val_bags), dtype=torch.float32)
val_labels = torch.tensor(np.array(val_labels), dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_bags = train_bags.to(device)
train_labels = train_labels.to(device)
val_bags = val_bags.to(device)
val_labels = val_labels.to(device)

result_model = os.path.join(result_folder, "model.txt")

train_losses = []
train_accuracies = []
val_accuracies = []
train_auces = []
val_aucses = []

num_epochs = 100
parameters = {
    'batch_size': [80],
    'Dropout': [0.2],
    'out_channels': [2],
    'Linear': [[1010, 256, 64]],
    'ConvBlock_conv': [[3, 5, 7]]
}

param_grid = ParameterGrid(parameters)
result_file = os.path.join(result_folder, "results.txt")
for idx, params in enumerate(param_grid):
    set_seed(42)
    print(params)
    # 创建模型
    model = GA_LRCN_DSCN(Dropout=params['Dropout'], out_channels=params['out_channels'], Linear=params['Linear'],
                         ConvBlock_conv=params['ConvBlock_conv'])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    with open(result_model, 'w') as file:
        file.write(str(model))
    batch_size = params['batch_size']
    with open(result_file, 'w') as file:
        for epoch in range(num_epochs):
            running_loss = 0.0
            num_batches = len(train_bags) // batch_size
            correct_train = 0
            total_train = 0
            train_scores = []
            train_true_labels = []
            for i in range(num_batches):
                batch_bags = train_bags[i * batch_size: (i + 1) * batch_size]
                batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]
                outputs = model(batch_bags)
                loss = criterion(outputs, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted_train = torch.max(outputs, 1)
                total_train += batch_labels.size(0)
                correct_train += (predicted_train == batch_labels).sum().item()
                train_scores.extend(outputs[:, 1].tolist())
                train_true_labels.extend(batch_labels.tolist())
            train_auc = roc_auc_score(train_true_labels, train_scores)
            average_loss = running_loss / num_batches
            train_accuracy = correct_train / total_train
            model.eval()
            with torch.no_grad():
                outputs_val = model(val_bags)
                _, predicted_val = torch.max(outputs_val, 1)
                correct_val = (predicted_val == val_labels).sum().item()
                val_accuracy = correct_val / val_labels.size(0)
                val_auc = roc_auc_score(val_labels.cpu().numpy(), predicted_val.cpu().numpy())
            # print ACC、AUC
            file.write(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f},Train AUC: {train_auc:.4f}, Validation Accuracy: {val_accuracy:.4f},Validation AUC: {val_auc:.4f}\n")
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f},Train AUC: {train_auc:.4f}, Validation Accuracy: {val_accuracy:.4f},Validation AUC: {val_auc:.4f}")

            train_losses.append(average_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_auces.append(train_auc)
            val_aucses.append(val_auc)
        # find MAX ACC、AUC & MIN Loss
        min_train_loss = min(train_losses)
        min_train_loss_index = train_losses.index(min_train_loss)
        max_train_accuracy = max(train_accuracies)
        max_train_accuracy_index = train_accuracies.index(max_train_accuracy)
        max_val_accuracy = max(val_accuracies)
        max_val_accuracy_index = val_accuracies.index(max_val_accuracy)
        max_train_auc = max(train_auces)
        max_train_auc_index = train_auces.index(max_train_auc)
        max_val_auc = max(val_aucses)
        max_val_auc_index = val_aucses.index(max_val_auc)

        file.write("Train Loss MIN: {}，local: {}\n".format(min_train_loss, min_train_loss_index + 1))
        file.write("Train Accuracy MAX: {}，local: {}\n".format(max_train_accuracy, max_train_accuracy_index + 1))
        file.write("Validation Accuracy MAX: {}，local: {}\n".format(max_val_accuracy, max_val_accuracy_index + 1))
        file.write("Train AUC MAX: {}，local: {}\n".format(max_train_auc, max_train_auc_index + 1))
        file.write("Validation AUC MAX: {}，local: {}\n".format(max_val_auc, max_val_auc_index + 1))

epochs = range(1, num_epochs + 1)

data = {
    'Epochs': epochs,
    'Training Loss': train_losses,
    'Training Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies,
    'Training AUC': train_auces,
    'Validation AUC': val_aucses
}
result_loss_acc_auc = pd.DataFrame(data)

excel_file = os.path.join(result_folder, 'result_loss_acc_auc.csv')
result_loss_acc_auc.to_csv(excel_file, index=False)

plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(os.path.join(result_folder, 'training_loss_plot.svg'))
plt.show()

plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(os.path.join(result_folder, 'training_validation_accuracy_plot.svg'))
plt.show()

plt.plot(epochs, train_auces, label='Training AUC')
plt.plot(epochs, val_aucses, label='Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('Training and Validation AUC')
plt.legend()
plt.savefig(os.path.join(result_folder, 'training_validation_auc_plot.svg'))
plt.show()
