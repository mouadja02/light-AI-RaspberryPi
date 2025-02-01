import torch
import numpy as np


model_path = "mlp_mnist_model.pth"
model_weights = torch.load(model_path, weights_only=True)

for layer_name, weights in model_weights.items():
    np.savetxt(f"{layer_name}.txt", weights.numpy().flatten(), delimiter=",")
