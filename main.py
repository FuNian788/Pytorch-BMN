import torch

from opt import MyConfig
from dataset import MyDataset

if __name__ == "__main__":

    arg = MyConfig()
    arg.parse()

    train_dataset = MyDataset(opt=arg)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batch_size)    