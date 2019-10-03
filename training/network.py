import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder3(nn.Module):

    def __init__(self):
        super(AutoEncoder3, self).__init__()
        self.conv_1 = nn.Conv1d(1, 32, kernel_size=3, padding=0, stride=1)
        self.conv_2 = nn.Conv1d(32, 64, kernel_size=3, padding=0, stride=1)
        self.conv_3 = nn.Conv1d(64, 128, kernel_size=3, padding=0, stride=1)
        self.deconv_1 = nn.ConvTranspose1d(128, 64, kernel_size=3)
        self.deconv_2 = nn.ConvTranspose1d(64, 32, kernel_size=3)
        self.deconv_3 = nn.ConvTranspose1d(32, 1, kernel_size=3)

    def forward(self, x):
        # encoder
        # L1
        x = self.conv_1(x)
        size_1 = x.size()
        x, indices_1 = F.max_pool1d_with_indices(x, kernel_size=2)
        # L2
        x = self.conv_2(x)
        size_2 = x.size()
        x, indices_2 = F.max_pool1d_with_indices(x, kernel_size=2)
        # L3
        x = self.conv_3(x)
        size_3 = x.size()
        x, indices_3 = F.max_pool1d_with_indices(x, kernel_size=2)

        # decoder
        # L1
        x = F.max_unpool1d(x, kernel_size=2, indices=indices_3, output_size=list(size_3))
        x = self.deconv_1(x)
        # L2
        x = F.max_unpool1d(x, kernel_size=2, indices=indices_2, output_size=list(size_2))
        x = self.deconv_2(x)
        # L3
        x = F.max_unpool1d(x, kernel_size=2, indices=indices_1, output_size=list(size_1))
        x = self.deconv_3(x)
        return x


class AutoEncoder4(nn.Module):
    def __init__(self):
        super(AutoEncoder4, self).__init__()
        self.conv_1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.deconv_1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.deconv_2 = nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1)
        self.deconv_3 = nn.ConvTranspose1d(16, 8, kernel_size=3, padding=1)
        self.deconv_4 = nn.ConvTranspose1d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # encoder
        # L1
        x = self.conv_1(x)
        size_1 = x.size()
        x, indices_1 = F.max_pool1d_with_indices(x, kernel_size=2)
        # L2
        x = self.conv_2(x)
        size_2 = x.size()
        x, indices_2 = F.max_pool1d_with_indices(x, kernel_size=2)
        # L3
        x = self.conv_3(x)
        size_3 = x.size()
        x, indices_3 = F.max_pool1d_with_indices(x, kernel_size=2)
        # L4
        x = self.conv_4(x)
        size_4 = x.size()
        x, indices_4 = F.max_pool1d_with_indices(x, kernel_size=2)

        # decoder
        # L1
        x = F.max_unpool1d(x, kernel_size=2, indices=indices_4, output_size=list(size_4))
        x = self.deconv_1(x)
        # L2
        x = F.max_unpool1d(x, kernel_size=2, indices=indices_3, output_size=list(size_3))
        x = self.deconv_2(x)
        # L3
        x = F.max_unpool1d(x, kernel_size=2, indices=indices_2, output_size=list(size_2))
        x = self.deconv_3(x)
        # L4
        x = F.max_unpool1d(x, kernel_size=2, indices=indices_1, output_size=list(size_1))
        x = self.deconv_4(x)
        return x