import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import AnomalyClassifier

def train_model(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# Dummy dataset
train_data = torch.rand(100, 768)
train_labels = torch.randint(0, 2, (100,))
train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=10, shuffle=True)

# Initialize and train
model = AnomalyClassifier(768, 256, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, optimizer, criterion)
