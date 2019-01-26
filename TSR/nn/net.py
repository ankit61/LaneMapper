import torchvision.models as models
import torch.nn as nn

class TSRNet(torch.nn.Module):

    def __init__(self, num_signs):
        self.__vgg_features = models.vgg16(pretrained=True).features
        
        self.__classifier = models.vgg16(pretrained=True).classifier
        self.__classifier[-1] = nn.Linear(self.__vgg_classifier[-1].in_features, num_signs + 1) #adding 1 for background class

        self.__classifier = models.vgg16(pretrained=True).classifier
        self.__classifier[-1] = nn.Linear(self.__vgg_classifier[-1].in_features, num_signs + 1) #adding 1 for background class        
        
        self.__bbx_regressor = models.vgg16(pretrained=True).classifier
        self.__classifier[-1] = nn.Linear(self.__vgg_classifier[-1].in_features, 4 * num_signs)

    def forward(self, x):
        features = self.__vgg_features(x)
        class_probabilities = self.__classifier(features)
        pred_bbxs = self.__bbx_regressor(features)

        return class_probabilities, pred_bbxs