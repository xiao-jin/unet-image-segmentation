import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

import dataloader
from model import UNet
from model_eval import eval
from model_trainer import train
from dice_loss import dice_loss

os.makedirs('./saved_models/', exist_ok=True)
os.makedirs('./results/', exist_ok=True)

def main(args):
    model = UNet(out_channels=21)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_dataloader, test_dataloader = dataloader.load_datasets(
                                         batch_size=args.batch_size,
                                         image_resize=args.image_resize,
                                         train_dataset_size=args.train_data_size,
                                         test_dataset_size=args.test_data_size)

    print(f'Start training for {args.epochs} epochs')
    train(model=model,
          dataloader=train_dataloader,
          epochs=args.epochs,
          optimizer=optimizer,
          criterion=criterion,
          save_output_every=1,
          )

    print(f'Training finished')
    print(f'Start evaluating with {len(test_dataloader.dataset)} images')

    eval(model, test_dataloader)

    print('All done')


if __name__ == "__main__":
    # MacOS issue
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Total epochs to be trained')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size, warning: U-Net used a lot of VRAM')
    parser.add_argument('--image_resize', type=int, default=256,
                        help='Resize VOC images to WxH')
    parser.add_argument('--train_data_size', type=int, default=1000,
                        help='Truncate the training data set')
    parser.add_argument('--test_data_size', type=int, default=100,
                        help='Truncate the test data set')

    args = parser.parse_args()
    main(args)
