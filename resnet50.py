import torch.nn as nn
import torch.nn.functional as F


class IdentityBlock(nn.Module):

    def __init__(self, input_size, filters, filters_out, strides=(1, 1)):
        super(IdentityBlock, self).__init__()
        padding = 1
        self.down_sample = False
        self.input_size = input_size

        self.conv1 = nn.Conv2d(self.input_size, filters,
                               (1, 1),
                               stride=strides)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=filters)

        self.conv2 = nn.Conv2d(filters, filters,
                               (3, 3),
                               padding=(padding, padding))
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=filters)

        self.conv3 = nn.Conv2d(filters, filters_out,
                               (1, 1))
        self.gn3 = nn.GroupNorm(num_groups=8, num_channels=filters_out)

        if strides == (2, 2):
            self.down_sample = True
            self.conv4 = nn.Conv2d(self.input_size, filters_out, (1, 1),
                                   stride=strides)
            self.gn4 = nn.GroupNorm(num_groups=8, num_channels=filters_out)

    def forward(self, input):
        x = F.relu(self.gn1(self.conv1.forward(input)))
        x = self.gn2(self.conv2.forward(x))
        x = self.gn3(self.conv3.forward(x))
        if self.down_sample:
            input = self.gn4(self.conv4.forward(input))
            output = F.relu(x + input)
        else:
            output = x
        return output


class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50, self).__init__()
        self.conv = nn.Conv2d(3, 64, (3, 3), stride=(2, 2), padding=(1, 1))
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1))

        self.block1 = IdentityBlock(64, 64, 256)
        self.block2 = IdentityBlock(256, 64, 256)
        self.block3 = IdentityBlock(256, 64, 256, strides=(2, 2))

        self.block4 = IdentityBlock(256, 128, 512)
        self.block5 = IdentityBlock(512, 128, 512)
        self.block6 = IdentityBlock(512, 128, 512)
        self.block7 = IdentityBlock(512, 128, 512, strides=(2, 2))

        self.block8 = IdentityBlock(512, 256, 1024)
        self.block9 = IdentityBlock(1024, 256, 1024)
        self.block10 = IdentityBlock(1024, 256, 1024)
        self.block11 = IdentityBlock(1024, 256, 1024)
        self.block12 = IdentityBlock(1024, 256, 1024)
        self.block13 = IdentityBlock(1024, 256, 1024, strides=(2, 2))

        self.block14 = IdentityBlock(1024, 512, 2048)
        self.block15 = IdentityBlock(2048, 512, 2048)
        self.block16 = IdentityBlock(2048, 512, 256, strides=(2, 2))

        self.avg_pool = nn.AvgPool2d((3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, images):
        x = self.bn(self.conv(images))
        # conv2
        x = self.block3(self.block2(self.block1(x)))
        # conv3
        x = self.block7(self.block6(self.block5(self.block4(x))))
        # conv4
        x = self.block13(self.block12(self.block11(
            self.block10(self.block9(self.block8(x))))))
        # conv5
        x = self.block16(self.block15(self.block14(x)))
        # avg pool
        output = self.avg_pool(x)
        return output
