import numpy as np
import torchvision.transforms as T

from torch.utils.data import Subset, DataLoader
from torchvision.datasets import VOCSegmentation


def load_datasets(batch_size=16,
                 image_resize=256,
                 train_dataset_size=1000,
                 test_dataset_size=100,
                 download=False):
    """
    Get datasets and data loaders from torchvision
    """
    transform = T.Compose([
        T.Resize((image_resize, image_resize)), # to make life easier
        T.ToTensor(),
    ])

    train_dataset = VOCSegmentation(
                        './data/VOCSegmentation',
                        image_set='train',
                        download=download,
                        transform=transform,
                        target_transform=transform
                        )    
    
    test_dataset = VOCSegmentation(
                        './data/VOCSegmentation',
                        image_set='val',
                        download=download,
                        transform=transform,
                        target_transform=transform
                        )
    
    """
    Truncate the datasets
    """
    train_dataset = Subset(train_dataset, np.arange(train_dataset_size))
    
    test_dataset = Subset(test_dataset, np.arange(test_dataset_size))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                             )

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                            )

    return train_loader, test_loader
