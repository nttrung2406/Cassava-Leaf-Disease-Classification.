import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from PIL import Image
import os
import time
import cv2
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
image_ids = train_df['image_id'].values
labels = train_df['label'].unique()

def displayLabel(train_df):
    label_counts = train_df['label'].value_counts()
    plt.bar(label_counts.index, label_counts.values)
    plt.xlabel('Label ID')
    plt.ylabel('Count')
    plt.title('Distribution of Label IDs')
    plt.show()

def plot_selected_images(train_df, labels):
    selected_images = []
    for label in labels:
        image_id = train_df[train_df['label'] == label]['image_id'].iloc[0]
        selected_images.append(image_id)
    
    fig, axes = plt.subplots(nrows=len(selected_images), figsize=(8, 8))
    
    for i, image_id in enumerate(selected_images):
        img_path = f'train_tfimages\\{image_id}'  # Assuming the images are stored in a directory named 'train_images'
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {labels[i]}')
    
    plt.tight_layout()
    plt.show()

plot_selected_images(train_df, labels)

data_directory = 'train_tfimages'
img = cv2.imread(os.path.join('train_tfimages', '6103.jpg'))
img.shape

class CustomDataset(Dataset):
    def __init__(self, image_ids, labels, transform=None):
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join('train_images/'+image_id)
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
X_train, X_val, y_train, y_val = train_test_split(image_ids, labels, test_size=0.2, random_state=42)

def create_data_loaders(image_ids, labels, batch_size=32, test_size=0.2, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    
    train_dataset = CustomDataset(X_train, y_train, transform)
    val_dataset = CustomDataset(X_val, y_val, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(image_ids, labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_resnet_model(num_classes):
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model(model, train_loader, num_epochs=100):
    # Compute class weights
    y_train = [labels for _, labels in train_loader.dataset]
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f} seconds')

def evaluate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

num_classes = 5
model = create_resnet_model(num_classes)
train_model(model, train_loader, num_epochs=1)
evaluate_model(model, val_loader)


def generate_train_predictions_csv(model, train_loader, device, X_train):
    train_predictions = []

    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_predictions.extend(predicted.tolist())

    train_results_df = pd.DataFrame({'image_id': X_train, 'label': train_predictions})
    train_results_df.to_csv('train_predictions.csv', index=False)

generate_train_predictions_csv(model, train_loader, device, X_train)
