import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.nn import init

data = pd.read_csv("top_10_features.csv")

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Make y a column vector
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(10,16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16,8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8,1)

        init.kaiming_uniform(self.fc1.weight)
        init.kaiming_uniform(self.fc2.weight)
        init.kaiming_uniform(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = nn.functional.sigmoid(self.fc3(x))
        return x

model = ANN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay= 1e-4)

epochs = 50
batch_size = 32
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred >= 0.5).float() 
    accuracy = accuracy_score(y_test, y_pred_class.numpy())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

