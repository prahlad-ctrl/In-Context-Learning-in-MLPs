import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

batch_size = 128
context_len = 5
epochs = 20000
lr = 1e-3

class InContextRegression:
    def __init__(self, batch_size, context_len):
        self.batch_size = batch_size
        self.context_len = context_len
        self.input_dim = (2* context_len)+ 1

    def get_batch(self):
        a = torch.randn(self.batch_size, 1)
        b = torch.randn(self.batch_size, 1)
        x_all = torch.randn(self.batch_size, self.context_len + 1)
        y_all = a* x_all+ b

        inputs = []
        for i in range(self.context_len):
            inputs.append(x_all[:, i:i+1])
            inputs.append(y_all[:, i:i+1])
        inputs.append(x_all[:, -1:]) 
        
        batch_x = torch.cat(inputs, dim=1)
        batch_y = y_all[:, -1:]
        return batch_x, batch_y

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

def train():
    task_gen = InContextRegression(batch_size, context_len)
    model = SimpleMLP(task_gen.input_dim, hidden_size= 64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    for i in range(epochs):
        x, y = task_gen.get_batch()
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 1000 == 0:
            print(f"epochs: {i}, loss = {loss.item():.4f}")
            
    return model, losses

def stress_test(model):    
    def run(mod= "base", len= 100):
        total_loss = 0
        for i in range(len):
            a = torch.randn(batch_size, 1)
            b = torch.randn(batch_size, 1)
            x_all = torch.randn(batch_size, context_len + 1)
            y_all = a* x_all+ b
            
            inputs = []
            if mod == "shuffle":
                indices = torch.randperm(context_len)
                for i in indices:
                    inputs.append(x_all[:, i:i+1])
                    inputs.append(y_all[:, i:i+1])
                inputs.append(x_all[:, -1:])
                
            elif mod == "query_shift":
                inputs.append(x_all[:, -1:])
                for i in range(context_len):
                    inputs.append(x_all[:, i:i+1])
                    inputs.append(y_all[:, i:i+1])
            
            else:
                for i in range(context_len):
                    inputs.append(x_all[:, i:i+1])
                    inputs.append(y_all[:, i:i+1])
                inputs.append(x_all[:, -1:])

            batch_x = torch.cat(inputs, dim=1)
            batch_y = y_all[:, -1:]
            
            with torch.no_grad():
                pred = model(batch_x)
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                total_loss += loss.item()
        return total_loss / len

    base = run("base")
    shuff = run("shuffle")
    shift = run("query_shift")
    print(f"base MLP loss: {base:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.bar(['base', 'shuffle', 'query shift'], [base, shuff, shift])
    plt.title("MLP loss type")
    plt.ylabel("MSE Loss")
    plt.show()

trained_model, _ = train()
stress_test(trained_model)