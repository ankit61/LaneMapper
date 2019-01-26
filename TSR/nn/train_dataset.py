import torch
import pandas as pd

class TrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, root_dir, transforms):
        self.__df = pd.read_csv(csv_file, header=None)
        self.__transforms = transforms

    def __len__(self):
        return 

    def __getitem__(self, idx):