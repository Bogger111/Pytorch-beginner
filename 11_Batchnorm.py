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

def model_train(model,data_loader,use_scheduler=False,lr=0.01,epoch=5):
    model.train()

    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.8)

    elosses = []
    blosses = []
    accuracies = []

    for times in range(epoch):
        correct = 0
        total = 0
        accuracy = 0

        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred,y)
            pred_class = pred.argmax(dim = 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct += (pred_class == y).sum().item()
            total += y.size(0)

            run_losses += loss.item()

            blosses.append(loss.item())
            accuracies.append((pred_class == y).sum().item()/y.size(0))
        acc = correct/total
        avr_loss = run_losses/len(train_loader)
        elosses.append(avr_loss)
        if (times+1)%10 == 0 :
            print(f"Times: {times+1}, Loss: {avr_loss}, Accuracy: {acc}.")

        if use_scheduler:
            scheduler.step()

    return elosses,blosses,accuracies

def model_test(model,test_loader):

    model.eval()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            pred_class = pred.argmax(dim=1)

            correct += (pred_class == y).sum().item()
            total += y.size(0)
            
    return correct/total

