import torch
import torch.nn as nn
#import torchvision

class Alexnet(nn.Module):
    def __init__(self, class_num = 3755):
        super(Alexnet, self).__init__()
        self.featureExtraction = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 96, kernel_size= 11, stride= 4, bias= 0),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 0),
            nn.Conv2d(in_channels= 96, out_channels= 256, kernel_size= 5, stride= 1, padding = 2, bias= 0),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size= 3, stride=1, padding= 1, bias=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size= 3, stride= 1, padding= 1, bias=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 0)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features= 256*6*6, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Dropout(0.5),
            nn.Linear(in_features= 4096, out_features= 4096),
            nn.ReLU(inplace= True),
            nn.Linear(in_features= 4096, out_features= class_num)
        )

    def forward(self, x):
        x = self.featureExtraction(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = Alexnet()
    print(model)
    input = torch.randn(8, 3, 227, 227)
    out = model(input)
    print(out.shape)
    print(out)
