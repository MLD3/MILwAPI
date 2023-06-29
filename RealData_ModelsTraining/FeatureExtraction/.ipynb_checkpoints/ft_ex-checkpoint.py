import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class featureEX_sup(nn.Module):
    def __init__(self):
        super(featureEX_sup, self).__init__()
        self.ftex = models.densenet121(weights = models.DenseNet121_Weights)
        self.ftex.fc = torch.nn.Identity()
        self.patchclass = nn.Linear(1000, 2)
        
    
    def forward(self, x):
        size = 512 # patch size
        stride = 512 # patch stride
        patches = x.unfold(2, size, stride).unfold(3, size, stride)
        outs = []
        for i in range(patches.shape[2]):
            for j in range(patches.shape[3]):
                inp = patches[:, :, i, j, :, :]
                out = self.ftex(inp)
                outs.append(out.squeeze(0))
                
        H2 = torch.stack(outs)
        Y_prob = self.patchclass(H2)
        return H2, Y_prob
    
    