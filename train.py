import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data.loaders import get_cifar_loader
from models.model import ResNet18
from torch.optim.lr_scheduler import StepLR

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128

model_number = 13

module_path = os.getcwd()
home_path = module_path
loss_path = os.path.join(home_path, f'model_data/{model_number}/reports', 'loss')
models_path = os.path.join(home_path, f'model_data/{model_number}/reports')
data_path = os.path.join(home_path, 'data')
os.makedirs(loss_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

train_loader, valid_loader, test_loader = get_cifar_loader(root=data_path, batch_size=batch_size)

# n_class = 10

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f"model_data/{model_number}/runs/experiment_{timestamp}")

def get_accuracy(model, data_loader, device):
    ## --------------------
    # Add code as needed
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, pred = torch.max(output.data, 1)
            total += y.size(0)
            correct += (pred==y).sum().item()
    return correct / total

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [] 
    train_accuracy_curve = [] 
    val_accuracy_curve = [] 
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        # if scheduler is not None:
        #     scheduler.step()
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        # loss_list = []  # use this to record the loss value of each step
        # grad = []  # use this to record the loss gradient of each step
        # learning_curve[epoch] = 0  # maintain this to plot the training curve

        for batch_idx, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, prediction = output.max(1)
            total += y.size(0)
            correct += (prediction==y).sum().item()

            writer.add_scalar('Training batch loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        scheduler.step()

        # 梯度检查
        total_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                total_grad += grad_mean
        grad = total_grad/len(list(model.parameters()))
        print(f"\nEpoch {epoch} - 平均梯度: {total_grad/len(list(model.parameters())):.6f}")
        grads.append(grad)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        learning_curve.append(epoch_loss)
        train_accuracy_curve.append(epoch_acc)

        val_acc = get_accuracy(model, val_loader, device)
        val_accuracy_curve.append(val_acc)

        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

        writer.add_scalar('Training epoch loss', epoch_loss, epoch)
        writer.add_scalar('Training accuracy', epoch_acc, epoch)
        writer.add_scalar('Validation accuracy', val_acc, epoch)
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch+1}/{epochs_n}:')
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%')
        print(f'Val Acc: {val_acc*100:.2f}%')

        display.clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.plot(learning_curve, label="Training Loss")
        ax1.set_title("Training Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax2.plot(train_accuracy_curve, label='Training Accuracy')
        ax2.plot(val_accuracy_curve, label='Validation Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # plt.savefig(os.path.join(figures_path, f'training_curves_epoch_{epoch}.png'))
        plt.close()
    
    writer.close()
    return learning_curve, val_accuracy_curve, grads

n_epoch = 120
loss_save_path = loss_path
lr = 0.1

print("-"*70)
print(f"learning rate: {lr}")

set_random_seeds(seed_value=2020, device=device)
model = ResNet18()

model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, 10) # 将最后的全连接层改掉
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

best_model_path = os.path.join(models_path, f'best_model.pth')

loss, val_acc , grads = train(model, optimizer, criterion, train_loader, valid_loader, scheduler=scheduler, epochs_n=n_epoch, best_model_path=best_model_path)
np.savetxt(os.path.join(loss_save_path, f'loss_{lr}.txt'), loss, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(loss_save_path, f'val_acc_{lr}.txt'), val_acc, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(loss_save_path, f'grads_{lr}.txt'), grads, fmt='%s', delimiter=' ')
