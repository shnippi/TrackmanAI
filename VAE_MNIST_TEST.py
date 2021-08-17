from AE_networks import VAE_net, VAE_net_64, VanillaVAE
import torch
import numpy as np
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("testing")

model_file_name = "models/VAE_MNIST.model"
net = VanillaVAE()
batch_size = 16

test_mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()]
    ),
)
test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=batch_size, shuffle=True)

net.load_state_dict(torch.load(model_file_name, map_location=device))
net = net.to(device)
net.eval()

for batch, (x, y) in enumerate(test_loader):


    x, y = x.to(device), y.to(device)
    plt.imshow(x[0][0].to("cpu"), "gray")
    plt.show()

    output, original, mu, logVar = net(x)


    # output[0][0][0] since mu and std also get returend, otherwise output[0][0]
    plt.imshow(output[0][0].to("cpu").detach(), "gray")
    plt.show()