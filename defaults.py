import torch

from config import num_cpus

cpus = num_cpus()
cmap = "viridis"
return_fig = False
silent = False
batch_size = 256
learning_rate = 1e-5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Training on device: {device}")
