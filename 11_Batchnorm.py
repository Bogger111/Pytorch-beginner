import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='./data',train=True,download=True,transform=transform_train)
test_data = datasets.CIFAR10(root='./data',train=False,download=True,transform=transform_test)
train_loader = DataLoader(train_data,batch_size=128,shuffle=True)
test_loader = DataLoader(test_data,batch_size=128,shuffle=False)

class Net(nn.Module):
    def __init__(self, use_dropout = False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Linear(64*8*8,32),
            nn.ReLU(),
        )
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = nn.linaer(32,10)
        return x
    
