"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from .cutout import Cutout

class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root=r'D:\作业\神经网络与深度学习\PJ2\codes\Network_on_cifar10\data', batch_size=128, valid_size=0.2, train=True, shuffle=True, num_workers=0, n_items=-1):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，再把图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_test)
    test_data = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    # if n_items > 0:
    #     dataset = PartialDataset(dataset, n_items)

    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

# if __name__ == '__main__':
#     train_loader = get_cifar_loader()
#     for X, y in train_loader:
#         print(X[0])
#         print(y[0])
#         print(X[0].shape)
#         img = np.transpose(X[0], [1,2,0])
#         plt.imshow(img*0.5 + 0.5)
#         plt.savefig('sample.png')
#         print(X[0].max())
#         print(X[0].min())
#         break