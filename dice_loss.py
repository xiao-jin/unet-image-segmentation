import torch.nn.functional as F

def dice_loss(pred, target, smooth=1):
    pred = F.softmax(pred)

    intersection = (pred * target).sum(dim=1)
    unionset = pred.sum(dim=1) + target.sum(dim=1)
    loss = 2. * (intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / target.shape[0]