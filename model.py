import torch
import torch.nn as nn

class AnomalyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AnomalyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

if __name__ == "__main__":
    model = AnomalyClassifier(input_size=768, hidden_size=256, output_size=2)
    test_input = torch.rand(1, 768)
    output = model(test_input)
    print("Model Output:", output)
