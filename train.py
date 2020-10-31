import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


from model import UNet

def show_img(img):
    plt.imshow(img)
    plt.show()

def main(img_url):
    unet = UNet(1)

    img = cv2.imread(img_url)
    

    input = torch.tensor(img).float()


    # get the dimentions of the input image
    d1,d2,d3 = input.shape
    # reshape the input as (batch, w, h, c)
    input = input.reshape(1, d3, d1, d2)

    print(input.shape)

    output = unet(input)

    img_output = output.detach().numpy()
    img_output = img_output.reshape(d1, d2, d3)
    # img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)

    plt.imshow(img_output, cmap='gray')
    plt.show()






if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    img_url = ''
    main(img_url)