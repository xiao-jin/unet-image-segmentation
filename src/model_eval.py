import utils

def eval(model, dataloader, only_show=False):
    device = utils.get_device()
    for step, (input, target) in enumerate(dataloader):
        model.eval()
        input, target = input.to(device), target.to(device)

        target = utils.preprocess_target(target)
        pred = model(input)

        utils.save_result(step+1, input, pred, target, name='eval', only_show=only_show)
        