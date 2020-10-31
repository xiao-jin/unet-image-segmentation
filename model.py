import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def conv_module(self, in_channels, out_channels):
        """
        Constructs a torch module Sequential
        :param in_channel: The number of input channels into the conv layers
        :param out_channel: The number of middle and out channels of the conv layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.K, padding=self.P),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=self.K, padding=self.P),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def upsample_module(self, in_channels):        
        return nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=self.S), #Nearest2D is used to get sharp edges around segmenations
            nn.ConvTranspose2d(in_channels=in_channels, 
                               out_channels=in_channels // 2,
                               kernel_size=self.K,
                               padding=1),
        )

    def __init__(self, n_input, in_channel=3, out_channels=3, kernel_size=3, striding=2, padding=1):
        """
        Initiate Unet
        :param in_channel: The number of input channels, default 3 for RGB images
        :param out_channel: The number of out channels, default 3 for RGB images
        :param kernel_size: The kernal size of the convolution, default 3x3
        :param striding: The striding length, default 1
        :param padding: The padding of convolution, default 1
        :param n_input: The dimensionality of the input
        """
        super(UNet, self).__init__()

        # Assign and abbreviate the variables as CV conventions
        self.C = in_channel
        self.K = kernel_size
        self.S = striding
        self.P = padding

        # Convolution components int he U-Net
        self.conv1 = self.conv_module(in_channel, 64)
        self.conv2 = self.conv_module(64, 128)
        self.conv3 = self.conv_module(128, 256)
        self.conv4 = self.conv_module(256, 512)
        self.conv5 = self.conv_module(512, 1024)
        self.conv6 = self.conv_module(1024, 512)
        self.conv7 = self.conv_module(512, 256)
        self.conv8 = self.conv_module(256, 128)
        self.conv9 = self.conv_module(128, 64)

        # Upsampling modules
        self.upsample1 = self.upsample_module(1024)
        self.upsample2 = self.upsample_module(512)
        self.upsample3 = self.upsample_module(256)
        self.upsample4 = self.upsample_module(128)
        
        # Functional modules
        self.maxpool2D = nn.MaxPool2d(kernel_size=2, stride=self.S, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=self.S)


    def forward(self, input):
        # Encoder - downsampling
        conv1_out = self.conv1(input)
        print('conv1', conv1_out.shape)
        conv2_out = self.conv2(self.maxpool2D(conv1_out))
        print('conv2', conv2_out.shape)
        conv3_out = self.conv3(self.maxpool2D(conv2_out))
        print('conv3', conv3_out.shape)
        conv4_out = self.conv4(self.maxpool2D(conv3_out))
        print('conv4', conv4_out.shape)

        # Middle layer
        conv5_out = self.conv5(self.maxpool2D(conv4_out))
        print('conv5', conv5_out.shape)

        # Decoder - upsampling
        # using the same object to optimize memory usage
        next = self.upsample1(conv5_out)
        next = torch.cat([conv4_out, next], dim=1)
        print('conv6_in', next.shape)
        next = self.conv6(next)
        print('conv6_out', next.shape)


        next = self.upsample2(next)
        next = torch.cat([conv3_out, next], dim=1)
        print('conv7_in', next.shape)
        next = self.conv7(next)
        print('conv7_out', next.shape)

        next = self.upsample3(next)
        next = torch.cat([conv2_out, next], dim=1)
        print('conv8_in', next.shape)
        next = self.conv8(next)
        print('conv8_out', next.shape)

        next = self.upsample4(next)
        next = torch.cat([conv1_out, next], dim=1)
        print('conv9_in', next.shape)
        next = self.conv9(next)
        print('conv9_out', next.shape)

        out = self.conv1x1(next)
        return out
        