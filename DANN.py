import torch.nn as nn


class DANN(nn.Module):
    def __init__(self, input_size=128):
        super(DANN, self).__init__()
        self.input_size = input_size
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.input_size, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_feature):
        # input_feature = input_feature.view(-1, self.input_size)
        # input_feature = input_feature.view(-1)
        domain_output = self.domain_classifier(input_feature)

        return domain_output
