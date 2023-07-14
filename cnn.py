import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class VirusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.data = []

        for class_ in self.classes:
            class_dir = os.path.join(self.root_dir, class_)
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if os.path.isfile(file_path):
                    self.data.append((class_, file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, self.classes.index(label)


class ConvNet(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()  
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)))
            elif i == 1:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)))
                
        self.fc = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.fc is None:
            n_features = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(n_features, self.num_classes)
            
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(model, criterion, optimizer, dataloader, num_epochs=25):
    model.train()
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions.double() / len(dataloader.dataset)
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))
        # Guardar el modelo después de cada época
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        
    return losses, accuracies


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),  
    transforms.ToTensor()
])

train_dataset = VirusDataset(root_dir='virus/dataset/augmented_train', transform=transform)
validation_dataset = VirusDataset(root_dir='virus/dataset/validation', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

num_layers_list = [3]

for num_layers in num_layers_list:
    model = ConvNet(len(train_dataset.classes), num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses, accuracies = train_model(model, criterion, optimizer, train_loader, num_epochs=25)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'ConvNet {num_layers} layers - Loss vs. No. of epochs');

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(f'ConvNet {num_layers} layers - Accuracy vs. No. of epochs');
    plt.show()

def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions.extend(predicted)
            targets.extend(labels)

        accuracy = 100 * correct / total
        print('Test Accuracy of the model on the test images: {} %'.format(accuracy))

        precision = precision_score(targets, predictions, average='macro')
        recall = recall_score(targets, predictions, average='macro')
        f1 = f1_score(targets, predictions, average='macro')

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-score: {f1}')

        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
evaluate_model(model, validation_loader)

# Load the saved model
model = ConvNet(len(train_dataset.classes), 3).to(device)
model.load_state_dict(torch.load('./model_epoch_24.pth'))  # replace 'model_epoch_24.pth' with your checkpoint file
model.eval()
