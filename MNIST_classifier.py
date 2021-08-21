import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import pathlib
from networks import LeNet_plus_plus

batch_size = 128
epochs = 100
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=batch_size, shuffle=True)

net = LeNet_plus_plus().to(device)
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

net.train()

for epoch in range(epochs):
    total_loss = 0
    for batch, (x, y) in enumerate(train_loader):
        #
        # print(x.shape)
        # plt.imshow(x[0][0].to("cpu"), "gray")
        # plt.show()

        x, y = x.to(device), y.to(device)
        print(x)
        pred = net(x)
        loss = loss_fn(pred, y)

        total_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Epoch {}: Loss {}'.format(epoch, total_loss))

    results_dir = pathlib.Path("models")
    save_dir = results_dir / f"MNIST_classifier_training.model"
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_dir)
