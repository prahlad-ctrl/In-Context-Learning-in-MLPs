import torch
import torch.nn as nn
import torch.optim as optim

lr = 3e-4 # magic value by andrej karpathy (it actyually did do the work tho)
epochs = 20000

class InContextRegression:
    def __init__(self, batch_size, context_len):
        self.batch_size = batch_size
        self.context_len = context_len

    ' generating random values '
    def get_batch(self):
        a = torch.randn(self.batch_size, 1)
        b = torch.randn(self.batch_size, 1)
        x_all = torch.randn(self.batch_size, self.context_len + 1) # all examples + query point
        y_all = a* x_all+ b # plain linear relation for this comp

        inputs = []
        for i in range(self.context_len):
            inputs.append(x_all[:, i: i+1]) # dim(batch_size, 1) for both
            inputs.append(y_all[:, i: i+1])
        inputs.append(x_all[:, -1:])
        
        batch_x = torch.cat(inputs, dim=1) # dim(batch_size, 2*context_len + 1)
        batch_y = y_all[:, -1:]
        
        return batch_x, batch_y

class InTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=4, max_len=50):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.0,
            activation="gelu", # works better than relu for transformers, though in actual paper relu is used ig
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, 1) # scaled ^y

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        last_token = x[:, -1, :]
        
        return self.output_head(last_token)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_gen = InContextRegression(batch_size= 512, context_len= 5)
    model = InTransformer(d_model= 128, nhead= 4, num_layers= 4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.MSELoss()    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5) # experimenting with this one, will decide 

    for i in range(epochs):
        x_batch, y_batch = task_gen.get_batch()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        scale_factor = 10.0
        y_scaled = y_batch / scale_factor
        preds_scaled = model(x_batch)
        loss = criterion(preds_scaled, y_scaled)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # exploding gradients made me train multiple times
        optimizer.step()
        
        real_loss = loss.item()*(scale_factor** 2)
        scheduler.step(real_loss)

        if i % 1000 == 0:
            print(f"epochs {i}, MSE loss= {real_loss:.4f}")

    return model, device

def evaluate(model, device):
    task_gen = InContextRegression(batch_size= 512, context_len= 5)
    total_loss = 0
    model.eval()
    for i in range(100):
        x, y = task_gen.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds_scaled = model(x)
            preds_real = preds_scaled* 10.0
            loss = torch.nn.functional.mse_loss(preds_real, y)
            total_loss += loss.item()
    return total_loss/ 100

model, device = train()
final_loss = evaluate(model, device)
print(f"\nfinal loss: {final_loss:.4f}")

'''transformer implement in-context learning via attention-based, token-wise computation,
whereas MLPs rely on parameterized memorization structure that simulate the task implicitly'''

'''So is Attention necessary for in-context learning?
NAHHH not really, but it provides a more flexible and generalizable mechanism compared to the MLPs'''