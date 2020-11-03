import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from PIL import Image

from model import UNet
import utils


def train(model,
          dataloader,
          epochs, 
          optimizer,
          criterion,
          save_output_every=10,
          save_model_every=50,
          only_show=False
          ):
    device = utils.get_device()

    for epoch in tqdm(range(1, epochs + 1)):
        losses_per_epoch = []
        accuracies_per_epoch = []
        # go over all batches
        for step, (input, target) in enumerate(dataloader):
            model.train()
            input, target = input.to(device), target.to(device)

            target = utils.preprocess_target(target)
            pred = model(input)
            
            optimizer.zero_grad()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            acc = utils.get_accuracy(pred, target)

            losses_per_epoch.append(loss.item())
            accuracies_per_epoch.append(acc)

        mean_loss = np.mean(losses_per_epoch)
        mean_acc = np.mean(accuracies_per_epoch)

        print(f'{"-"*30} Epoch {epoch} {"-"*30}')
        print('Loss: %.3f   Accuracy: %.3f' % (mean_loss, mean_acc))

        if epoch % save_output_every == 0:
            utils.save_result(epoch, input, pred, target, name='epoch', only_show=only_show)

        if epoch % save_model_every == 0:
            utils.save_model(epoch, model)
