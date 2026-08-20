from torch.nn import Module, Conv2d, SiLU, MaxPool2d, Flatten, Dropout, Linear, BatchNorm2d
from torch.nn.init import kaiming_normal_, zeros_

class HMNISTModel(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.nonlin1 = SiLU()
        self.norm1 = BatchNorm2d(64 * 64 * 32)
        self.conv2 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.nonlin2 = SiLU()
        self.norm2 = BatchNorm2d(64 * 64 * 32)
        self.conv3 = Conv2d(32, 64, kernel_size=3, padding=1),
        self.nonlin3 = SiLU()
        self.norm3 = BatchNorm2d(64 * 64 * 64)
        self.pool1 = MaxPool2d(kernel_size=4)
        self.conv4 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.nonlin4 = SiLU()
        self.norm4 = BatchNorm2d(16 * 16 * 128)
        self.pool2 = MaxPool2d(kernel_size=2)
        self.conv5 = Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm5 = BatchNorm2d(8 * 8 * 256)
        self.nonlin5 = SiLU()
        self.flatten = Flatten()
        self.linear1 = Linear(8 * 8 * 256, 128)
        self.dropout1 = Dropout(0.25)
        self.nonlin6 = SiLU()
        self.dropout2 = Dropout(0.5)
        self.linear2 = Linear(128, n_classes)

        for mod in self.modules():
            if isinstance(mod, Conv2d) or isinstance(mod, Linear):
                #kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    zeros_(mod.bias)

    def forward(self, X):
        for mod in self.modules():
            X = mod(X)
        return X