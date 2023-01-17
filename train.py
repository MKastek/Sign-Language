from pathlib import Path
from read_data import TrainDataset
from torch.utils.data import DataLoader
import torch
from model import CNN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(learning_rate=0.001, num_epochs=10):

    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        save_path = Path('checkpoint') / filename
        torch.save(state, save_path)

    def load_checkpoint(checkpoint):
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    data_path = Path().cwd() / 'data' / 'dataset'

    # Hyperparameters
    in_channel = 1
    num_classes = 10
    learning_rate = learning_rate
    batch_size = 64
    num_epochs = num_epochs
    load_model = False


    # Load Dataset
    train_dataset = TrainDataset(data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize network
    model = CNN(in_channel=in_channel, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load( Path('checkpoint') / "my_checkpoint.pth.tar"))

    loss_vals = []
    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
        epoch_loss = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # CUDA
            #print(f"{targets.size(dim=0)}")
            data = data.to(device=device)
            targets = targets.to(device=device)

            optimizer.zero_grad()
            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        loss_vals.append(sum(epoch_loss) / len(epoch_loss))
    return model, device, train_loader, loss_vals


def check_accuracy_train(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f" {num_correct} / {num_samples} {(float(num_correct/num_samples))*100:.2f}")
    model.train()


def plot_loss_curve(losses, learning_rate, num_epochs):
    plt.plot(losses, '.-')
    plt.title(f"Loss curve\n learning rate: {learning_rate}, epochs: {num_epochs}")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.7)
    plt.savefig(Path() / "plots" / f"Loss_curve_epochs_{num_epochs}_lr_{learning_rate}.png")
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    for learn in [0.0001]:
        for epoch in [10]:
            learning_rate = learn
            num_epochs = epoch
            model, device, train_loader, losses = train(num_epochs=epoch, learning_rate=learn)
            torch.save(model, Path().cwd() / 'model' / 'CNN-model.pt')
            print(f"learn: {learn}, epoch: {epoch}")
            check_accuracy_train(loader=train_loader, model=model)
            plt.plot(losses,'.-')
            plt.title(f"Loss curve\n learning rate: {learning_rate}, epochs: {num_epochs}")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.grid(which="minor", alpha=0.3)
            plt.grid(which="major", alpha=0.7)
            plt.savefig(Path() / "plots" / f"Loss_curve_epochs_{num_epochs}_lr_{learning_rate}.png")
            plt.cla()
            plt.clf()
