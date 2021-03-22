from .utils import *

def eval(model, dataloader, only_show=False):
    device = get_device()
    for step, (input, target) in enumerate(dataloader):
        model.eval()
        input, target = input.to(device), target.to(device)

        target = preprocess_target(target)
        pred = model(input)

        save_result(step+1, input, pred, target, name='eval', only_show=only_show)
        