# from .densenet import *
# from .dpn import *
# from .efficientnet import *
# from .googlenet import *
# from .lenet import *
# from .mobilenet import *
# from .mobilenetv2 import *
# from .pnasnet import *
# from .preact_resnet import *
from .resnet import *
from .resnext import *
# from .senet import *
# from .shufflenet import *
# from .shufflenetv2 import *
from .vgg import *

import torch.nn as nn


class pure_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(x)
