import torch.nn as nn
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:2")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class PrintLayer(nn.Module):
        def __init__(self):
            super(PrintLayer, self).__init__()
    
        def forward(self, x):
            # Do your print / debug stuff here
            print('asd')
            print(x.size())
            return x

seq_model = nn.Sequential( #NEM elvetendő
           nn.Conv2d(3, 64, kernel_size=3),
           #PrintLayer(),
           nn.MaxPool2d(2),
           #PrintLayer(),
           nn.LeakyReLU(0.25),
           #PrintLayer(),
                        ##nn.Dropout2d(),
           #PrintLayer(),
           nn.Conv2d(64, 128, kernel_size=3),
           #PrintLayer(),
           nn.MaxPool2d(2),
           #PrintLayer(),
           nn.LeakyReLU(0.25),
           nn.Conv2d(128,256, kernel_size=3),
           #PrintLayer(),
           nn.MaxPool2d(2),
           #PrintLayer(),
           nn.LeakyReLU(),
           #PrintLayer(),
           Flatten(),
           #PrintLayer(),
           nn.Linear(475136, 200),
           #PrintLayer(),
           nn.LeakyReLU(),
           #PrintLayer(),
           nn.Linear(200, 100),
           nn.LeakyReLU(),
           #PrintLayer(),
           nn.Linear(100, 4),
           #PrintLayer(),
           nn.LeakyReLU(),
           
         )

