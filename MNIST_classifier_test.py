import time

from networks import LeNet_plus_plus
import torch
import numpy as np
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("testing")

model_file_name = "models/MNIST_classifier.model"
net = LeNet_plus_plus()
batch_size = 128

test_mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=batch_size, shuffle=True)

net.load_state_dict(torch.load(model_file_name, map_location=device))
net = net.to(device)
net.eval()

for batch, (x, y) in enumerate(test_loader):
    x, y = x.to(device), y.to(device)

    pred = net(x)

    print(pred.argmax(1))
    print(y)
    print((pred.argmax(1) == y).type(torch.float).sum().item())  # how many correct

    time.sleep(100)
