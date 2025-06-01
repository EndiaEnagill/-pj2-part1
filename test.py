import torch
import torch.nn as nn
from data.loaders import get_cifar_loader
from models.model import ResNet18
import os

# set device
print("-"*70)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

module_path = os.getcwd()
home_path = module_path
data_path = os.path.join(home_path, 'data')

n_class = 10
batch_size = 128
train_loader,valid_loader,test_loader = get_cifar_loader(batch_size=batch_size,root=data_path)
model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, 10) # 将最后的全连接层改掉

# 载入权重
model.load_state_dict(torch.load('modal_data/2/reports/best_model.pth', map_location=torch.device('cpu'), weights_only=True))
model = model.to(device)

total_sample = 0
right_sample = 0
model.eval()  # 验证模型
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data).to(device)
    # convert output probabilities to predicted class(将输出概率转换为预测类)
    _, pred = torch.max(output, 1)    
    # compare predictions to true label(将预测与真实标签进行比较)
    correct_tensor = pred.eq(target.data.view_as(pred))
    # correct = np.squeeze(correct_tensor.to(device).numpy())
    total_sample += batch_size
    for i in correct_tensor:
        if i:
            right_sample += 1
print("Accuracy:",100*right_sample/total_sample,"%")