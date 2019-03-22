import torchvision.models as models
import torch.nn as nn
import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

class TSRNet(nn.Module):

    def __init__(self, num_traffic_signs):
        super(TSRNet, self).__init__()
        self.__net = models.resnet50(pretrained=True)
        self.__net.fc = nn.Linear(self.__net.fc.in_features, num_traffic_signs, bias=True)
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        return torch.min(0.99999997 * torch.ones((x.shape[0], self.__net.fc.out_features)), self.__sigmoid(self.__net(x)))