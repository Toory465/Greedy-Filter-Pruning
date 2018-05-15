import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 6, 5),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(6, 16, 5),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                      )


        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(120, 84),
                                        nn.Linear(84, 10),
                                        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x