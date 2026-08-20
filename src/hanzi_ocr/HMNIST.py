from torch.nn import Module, Conv2d, SiLU, AdaptiveAvgPool2d, MaxPool2d, Flatten, Dropout, Linear
from torch.nn.init import kaiming_normal_, zeros_

class HMNISTModel(Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=4, padding=1)
        self.elu1 = SiLU()
        self.conv2 = Conv2d(32, 64, kernel_size=4, padding=1)
        self.elu2 = SiLU()
        self.conv3 = Conv2d(32, 64, kernel_size=2, padding=1)
        self.adaptive = AdaptiveAvgPool2d(output_size=8)
        self.flatten = Flatten()
        self.linear1 = Linear(1, 128)
        self.dropout1 = Dropout(0.25)
        self.elu3 = SiLU()
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