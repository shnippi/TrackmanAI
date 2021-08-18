from AE_networks import VAE_net, VAE_net_64, VanillaVAE
import torch
import numpy as np
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("testing")

model_file_name = "models/VAE_64_100eps_vanilla.model"
net = VanillaVAE()
batch_size = 16


def npy_loader(path):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    sample = torch.from_numpy(np.load(path))
    np.load = np_load_old
    return sample


test_dataset = datasets.DatasetFolder(
    root='data/data_test_64_game',
    loader=npy_loader,
    extensions='.npy',
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net.load_state_dict(torch.load(model_file_name, map_location=device))
net = net.to(device)
net.eval()

for idx, data in enumerate(test_loader, 0):
    imgs, _ = data
    imgs = imgs.to(device)
    imgs = imgs.permute(0, 1, 4, 2, 3)  # switch from NHWC to NCHW
    imgs = transforms.Grayscale().forward(imgs)  # convert to grayscale
    # print(imgs.shape)

    for file in imgs:
        file = file / 256

        # iterate over batch
        for batch in torch.split(file, batch_size):
            plt.imshow(batch[0][0].to("cpu"), "gray")
            plt.show()

            output, original, mu, logVar = net(batch)


            # output[0][0][0] since mu and std also get returend, otherwise output[0][0]
            plt.imshow(output[0][0].to("cpu").detach(), "gray")
            plt.show()