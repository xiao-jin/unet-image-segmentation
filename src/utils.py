import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_target(target):
    """
    Preprocess the target to be in range [0,255]
    and remove the white border of objects
    """
    # convert target to index labels
    target = (target * 255).long() # convert [0., 1.] to [0, 255]
    target = target.squeeze(dim=1) #target.reshape(target.shape[0], *tuple(target.shape[-2:]))
    
    # And to make life easier, we don't care about the white borders
    target[target == 255] = 0

    return target

def get_weight(train_dataset):
    """
    Get the weights per class for the cross entropy loss
    """
    all_targets = []
    all_targets.append([target.numpy() * 255 for _, target in train_dataset])
    all_targets = np.array(all_targets)
    all_targets[all_targets == 255] = 0
    unique = np.unique(all_targets, return_counts = True)

    weights = unique[1].sum() / unique[1]
    return torch.tensor(weights).float().to(get_device())


def get_accuracy(pred, target):
    """
    Get the pixel level accuracy as the target
    """
    correct = (pred.argmax(dim=1) == target).sum().item()
    total = target.numpy().size
    
    return correct / total


def save_result(epoch, input, pred, target, name, only_show=False):
    """
    Visualize the input, prediction and target
    Data from the first images in the batch
    """
    input, pred, target = input[0], pred[0].argmax(0), target[0]

    # Credits to Pytorch tutortial https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    input = Image.fromarray((input*255).byte().cpu().transpose(0,2).transpose(0,1).numpy())
    pred = Image.fromarray(pred.byte().cpu().numpy())
    target = Image.fromarray(target.byte().cpu().numpy())

    pred.putpalette(colors)
    target.putpalette(colors)

    fig=plt.figure(figsize=(16,16))
    fig.add_subplot(1,3,1);plt.imshow(input)
    fig.add_subplot(1,3,2);plt.imshow(pred)
    fig.add_subplot(1,3,3);plt.imshow(target)

    if only_show:
        plt.show()
    else:
        plt.savefig(f'./results/{name}_{epoch}.png')
    plt.close(fig)


def save_model(epoch, model):
    torch.save(model.state_dict(), f'./saved_models/model_{epoch}.pt')
    