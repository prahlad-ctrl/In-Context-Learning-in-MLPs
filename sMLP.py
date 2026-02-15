import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 128
context_len = 5
epochs = 20000
lr = 1e-3

class InContextRegression:
    def __init__(self, batch_size, context_len):
        self.batch_size = batch_size
        self.context_len = context_len
        self.input_dim = (2* context_len)+ 1 # [x1,y1, x2,y2, ..., xk,yk, query_x]

    def get_batch(self):
        a = torch.randn(self.batch_size, 1)
        b = torch.randn(self.batch_size, 1)
        x_all = torch.randn(self.batch_size, self.context_len + 1)
        y_all = a* x_all+ b # same linear rel

        inputs = []
        for i in range(self.context_len):
            inputs.append(x_all[:, i:i+1])
            inputs.append(y_all[:, i:i+1])
        inputs.append(x_all[:, -1:]) 
        
        batch_x = torch.cat(inputs, dim=1) # same dim(batch_size, 2*context_len + 1)
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
    for i in range(epochs): # forced to learn a general strategy, will see, should work ig
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
    def run(mod= "base", len= 100): # normal order
        total_loss = 0
        for i in range(len):
            a = torch.randn(batch_size, 1)
            b = torch.randn(batch_size, 1)
            x_all = torch.randn(batch_size, context_len + 1)
            y_all = a* x_all+ b
            
            inputs = []
            if mod == "shuffle": # random order
                indices = torch.randperm(context_len)
                for i in indices:
                    inputs.append(x_all[:, i:i+1])
                    inputs.append(y_all[:, i:i+1])
                inputs.append(x_all[:, -1:])
                
            elif mod == "query_shift": # query at start
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
    shift = run("query_shift") #pretty sure this proves the memorization behavior of MLPs, query test
    print(f"base MLP loss: {base:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.bar(['base', 'shuffle', 'query shift'], [base, shuff, shift])
    plt.title("MLP loss type - test 1")
    plt.ylabel("MSE Loss")
    plt.savefig("test1_results.png")
    plt.show() # can see how shifting query increases losss
    
def neuro_test(model): # just addition visuals 
    W1 = model.net[0].weight.data.cpu().numpy()    
    labels = []
    
    for i in range(context_len):
        labels.append(f"x{i+1}")
        labels.append(f"y{i+1}")
    labels.append("query")
    
    plt.figure(figsize= (10, 6))
    sns.heatmap(abs(W1))
    y_ticks = np.arange(0, 64, 3)
    plt.yticks(ticks= y_ticks + 0.5, labels= y_ticks)
    plt.title("neuron relation - test 2")
    plt.ylabel("neuron index")
    plt.tight_layout()
    plt.savefig("test2_results.png")
    plt.show() # the no dark column indicate that neurons are specialized to fixed positions

trained_model, _ = train()
stress_test(trained_model)
neuro_test(trained_model)


'''the MLPs do achieve low error on in-context regression but stress tests reveal that 
they rely on fixed positional correlations rather than dynamically inferring task structure,
unlike transformers which compute task-specific solutions via attention at inference time'''

'''MLPs can perform in-context regression in a limited, shallow sense (low level i would say)'''