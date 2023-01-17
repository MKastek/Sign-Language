import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from numpy import asarray
from PIL import Image


class TrainDataset(Dataset):

    def __init__(self, data_path):

        array = np.zeros(shape=(2059,100,100,3))
        label = []
        i = 0
        for dir in data_path.iterdir():
            for file in dir.iterdir():
                image = Image.open(file)
                # convert image to numpy array
                data = asarray(image)
                array[i,:, :, :] = data
                label.append(int(str(dir)[-1]))
                i += 1
        self.images = array.reshape(2059, 3, 100, 100)
        self.img_label = label

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        self.image = torch.tensor(self.images[idx], dtype=torch.float32)
        self.label = torch.tensor(self.img_label[idx])
        return self.image, self.label


class TestDataset(Dataset):

    def __init__(self, data_path):
        array = np.zeros(shape=(10, 100, 100, 3))
        label = []
        i = 0
        for file in data_path.iterdir():
            image = Image.open(file)
            # convert image to numpy array
            data = asarray(image)
            array[i,:, :, :] = data
            label.append(int(str(file)[-5]))
            i += 1
        self.images = array.reshape(10, 3, 100, 100)
        self.img_label = label

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        self.image = torch.tensor(self.images[idx], dtype=torch.float32)
        self.label = torch.tensor(self.img_label[idx])
        return self.image, self.label


if __name__ == '__main__':
    data_path = Path().cwd() / 'data' / 'examples'
    test_dataset = TestDataset(data_path)
