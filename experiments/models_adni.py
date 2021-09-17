import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,"/analysis/fabiane/phd/patch_individual_filters/")
from pif import PatchIndividualFilters3D


class ModelA(nn.Module):
    def __init__(self, drp_rate=0.3):
        super(ModelA, self).__init__()
        self.drp_rate = drp_rate
        self.dropout = nn.Dropout3d(p=self.drp_rate)
        self.Conv_1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
        self.Conv_2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.Conv_3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0)
        self.Conv_4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0)
        self.Conv_5 = nn.Conv3d(64, 36, kernel_size=3, stride=1, padding=0)
        self.pool_4 = nn.MaxPool3d(kernel_size=4, stride=2, padding=0)
        
        self.classifier_scratch = nn.Sequential(
            nn.Linear(1296, 80),
            nn.ELU(inplace=True),
            nn.Linear(80, 1)
        )

    def encode(self, x, print_size=False):
        if print_size:
            print("Conv 1 " + str(x.shape))
        x = F.elu(self.Conv_1(x))
        if print_size:
            print("Pool 1 " + str(x.shape))
        h = self.dropout(self.pool_1(x))
        if print_size:
            print("Conv 2 " + str(h.shape))
        x = F.elu(self.Conv_2(h))
        if print_size:
            print("Pool 2 " + str(x.shape))
        h = self.dropout(self.pool_2(x))
        if print_size:
            print("Conv 3 " + str(h.shape))
        x = F.elu(self.Conv_3(h))
        if print_size:
            print("Conv 4 " + str(x.shape))
        x = F.elu(self.Conv_4(x))
        if print_size:
            print("Conv 5 " + str(x.shape))
        x = F.elu(self.Conv_5(x))
        if print_size:
            print("Pool 4 " + str(x.shape))
        h = self.dropout(self.pool_4(x))
        
        return h

    def forward(self, x):
        print_size = False
        x = self.encode(x, print_size=print_size)
        x = self.flatten(x)
        x = self.classifier_scratch(x)
        return x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)
    
class ModelAPIF(nn.Module):
    def __init__(self, drp_rate=0.3):
        super(ModelAPIF, self).__init__()
        self.drp_rate = drp_rate
        self.drop = nn.Dropout3d(p=self.drp_rate)
        self.Conv_1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
        self.Conv_2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.Conv_3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0)
        self.Conv_4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool_4 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)

        self.pif = PatchIndividualFilters3D([10,13,10],
                                            filter_shape=(3,3,3),
                                            patch_shape=(5,5,5),
                                            num_local_filter_in=64,
                                            num_local_filter_out=3,
                                            overlap=1,
                                            reassemble=False,
                                            debug=False)

        self.classifier_scratch = nn.Sequential(
            nn.Linear(1134, 100),
            nn.ELU(inplace=True),
            nn.Linear(100, 1)
        )

    def encode(self, x, print_size=False):
        if print_size:
            print("Conv 1 " + str(x.shape))
        x = F.elu(self.Conv_1(x))
        if print_size:
            print("Pool 1 " + str(x.shape))
        h = self.drop(self.pool_1(x))
        if print_size:
            print("Conv 2 " + str(h.shape))
        x = F.elu(self.Conv_2(h))
        if print_size:
            print("Pool 2 " + str(x.shape))
        h = self.drop(self.pool_2(x))
        if print_size:
            print("Conv 3 " + str(h.shape))
        x = F.elu(self.Conv_3(h))
        if print_size:
            print("Conv 4 " + str(x.shape))
        x = F.elu(self.Conv_4(x))
        if print_size:
            print("PIF " + str(x.shape))
        h = F.elu(self.pif(x))
    
        return h

    def forward(self, x):
        print_size = False
        x = self.encode(x, print_size=print_size)
        x = self.flatten(x)
        x = self.classifier_scratch(x)
        return x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)
    

class ModelB(nn.Module):
    def __init__(self, drp_rate=0.3):
        super(ModelB, self).__init__()
        self.drp_rate = drp_rate
        self.dropout = nn.Dropout3d(p=self.drp_rate)
        self.Conv_1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
        self.Conv_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.Conv_3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.Conv_4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.Conv_5 = nn.Conv3d(64, 36, kernel_size=3, stride=1, padding=0)
        self.pool_4 = nn.MaxPool3d(kernel_size=4, stride=2, padding=0)
        
        self.classifier_scratch = nn.Sequential(
            nn.Linear(1296, 80),
            nn.ELU(inplace=True),
            nn.Linear(80, 1)
        )

    def encode(self, x, print_size=False):
        if print_size:
            print("Conv 1 " + str(x.shape))
        x = F.elu(self.Conv_1(x))
        if print_size:
            print("Pool 1 " + str(x.shape))
        h = self.dropout(self.pool_1(x))
        if print_size:
            print("Conv 2 " + str(h.shape))
        x = F.elu(self.Conv_2(h))
        if print_size:
            print("Pool 2 " + str(x.shape))
        h = self.dropout(self.pool_2(x))
        if print_size:
            print("Conv 3 " + str(h.shape))
        x = F.elu(self.Conv_3(h))
        if print_size:
            print("Conv 4 " + str(x.shape))
        x = F.elu(self.Conv_4(x))
        if print_size:
            print("Conv 5 " + str(x.shape))
        x = F.elu(self.Conv_5(x))
        if print_size:
            print("Pool 4 " + str(x.shape))
        h = self.dropout(self.pool_4(x))
        
        return h

    def forward(self, x):
        print_size = False
        x = self.encode(x, print_size=print_size)
        x = self.flatten(x)
        x = self.classifier_scratch(x)
        return x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)

class ModelBPIF(nn.Module):
    def __init__(self, drp_rate=0.3):
        super(ModelBPIF, self).__init__()
        self.drp_rate = drp_rate
        self.drop = nn.Dropout3d(p=self.drp_rate)
        self.Conv_1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)
        self.Conv_2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.Conv_3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.Conv_4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool_4 = nn.MaxPool3d(kernel_size=3, stride=3, padding=0)

        self.pif = PatchIndividualFilters3D([10,13,10],
                                            filter_shape=(3,3,3),
                                            patch_shape=(5,5,5),
                                            num_local_filter_in=64,
                                            num_local_filter_out=3,
                                            overlap=1,
                                            reassemble=False,
                                            debug=False)

        self.classifier_scratch = nn.Sequential(
            nn.Linear(1134, 100),
            nn.ELU(inplace=True),
            nn.Linear(100, 1)
        )

    def encode(self, x, print_size=False):
        if print_size:
            print("Conv 1 " + str(x.shape))
        x = F.elu(self.Conv_1(x))
        if print_size:
            print("Pool 1 " + str(x.shape))
        h = self.drop(self.pool_1(x))
        if print_size:
            print("Conv 2 " + str(h.shape))
        x = F.elu(self.Conv_2(h))
        if print_size:
            print("Pool 2 " + str(x.shape))
        h = self.drop(self.pool_2(x))
        if print_size:
            print("Conv 3 " + str(x.shape))
        x = F.elu(self.Conv_3(h))
        if print_size:
            print("Conv 4 " + str(x.shape))
        x = F.elu(self.Conv_4(x))
        if print_size:
            print("PIF " + str(x.shape))
        h = F.elu(self.pif(x))
    
        return h

    def forward(self, x):
        print_size = False
        x = self.encode(x, print_size=print_size)
        x = self.flatten(x)
        x = self.classifier_scratch(x)
        return x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)
