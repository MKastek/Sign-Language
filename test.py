import numpy as np
import torch
from read_data import TestDataset
from pathlib import Path
from torch.utils.data import DataLoader


def check_accuracy_test(test_loader, model):
    predicted_values = np.array([])
    true_values = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, targets) in enumerate(test_loader):
            scores = model(image)
            _, predicted = torch.max(scores, 1)
            print(predicted)
            print(targets)
            predicted_values = np.append(predicted_values, predicted)
            true_values = np.append(true_values, targets)


if __name__ == '__main__':
    data_path = Path().cwd() / 'data' / 'examples'
    test_dataset = TestDataset(data_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    model = torch.load(Path().cwd() / 'model' / 'CNN-model.pt')
    check_accuracy_test(test_loader, model)