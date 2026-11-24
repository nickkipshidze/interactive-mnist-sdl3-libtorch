# Make sure to download the MNIST dataset first by running this script

import torchvision

if __name__ == "__main__":
    torchvision.datasets.MNIST(
        root="./",
        download=True
    )
