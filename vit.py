from transformer_package.models import ViT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset

from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchsummary import summary

import argparse
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--img_size',type=str,default = 28, help='image size')
parser.add_argument('--channel_size',type=str,default = 1, help='channel size')
parser.add_argument('--patch_size',type=str,default = 7, help='patch size')
parser.add_argument('--classes',type=str,default = 10, help='classes size')
parser.add_argument('--lr', type=int, default=5e-5, help='learning rate')
parser.add_argument('--epochs', type=int, default=25, help='epoch')
parser.add_argument('--dataset',type=str,default = "oracle-minist", help='training dataset')

args = parser.parse_args()
LR = args.lr
NUM_EPOCHES = args.epochs
data = f'data/{args.dataset}'
batch_size = args.batch_size
image_size = args.img_size
channel_size = args.channel_size
patch_size = args.patch_size
classes = args.classes
embed_size = 512
num_heads = 8
num_layers = 3
hidden_size = 256
dropout = 0.2



def vit_model():
    model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(device)
    print(model)
    return model

def train(model):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    loss_hist = {}
    loss_hist["train accuracy"] = []
    loss_hist["train loss"] = []

    for epoch in range(1, NUM_EPOCHES+1):
        model.train()

        epoch_train_loss = 0

        y_true_train = []
        y_pred_train = []

        for batch_idx, (img, labels) in enumerate(trainloader):
            img = img.to(device)
            labels = labels.to(device)

            preds = model(img)

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())

            epoch_train_loss += loss.item()

        loss_hist["train loss"].append(epoch_train_loss)

        total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x==y])
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total

        loss_hist["train accuracy"].append(accuracy)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
        print("-------------------------------------------------")

    torch.save(model, "model/vit.pt")

    plt.plot(loss_hist["train accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(loss_hist["train loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def test():
    model = torch.load("model/vit.pt")
    with torch.no_grad():
        model.eval()

    y_true_test = []
    y_pred_test = []

    for batch_idx, (img, labels) in enumerate(testloader):
        img = img.to(device)
        labels = labels.to(device)

        preds = model(img)

        y_pred_test.extend(preds.detach().argmax(dim=-1).tolist())
        y_true_test.extend(labels.detach().tolist())

    total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x==y])
    total = len(y_pred_test)
    accuracy = total_correct * 100 / total

    print("Test Accuracy%: ", accuracy, "==", total_correct, "/", total)



if __name__ == "__main__":
    dataset = Dataset(data,image_size,batch_size)
    trainloader,testloader = dataset.preprocessing()
    model = vit_model()
    for img, label in trainloader:
        img = img.to(device)
        label = label.to(device)

        print("Input Image Dimensions: {}".format(img.size()))
        print("Label Dimensions: {}".format(label.size()))
        print("-"*100)

        out = model(img)

        print("Output Dimensions: {}".format(out.size()))
        break

    train(model)
    test()



