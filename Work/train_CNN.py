import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# -------------------------------------------------------------------------
# CNN Architecture Definition
# -------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input channels = 1 for grayscale
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Corrected to 64 * 7 * 7
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (0-9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# -------------------------------------------------------------------------
# Transforms + Dataset
# -------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

full_dataset = datasets.ImageFolder(root='bmpProcessed', transform=transform)
num_classes = 10
targets = [sample[1] for sample in full_dataset.samples]
train_indices = []
test_indices = []

random.seed(42)
for class_idx in range(num_classes):
    idxs_class = [i for i, t in enumerate(targets) if t == class_idx]
    random.shuffle(idxs_class)
    train_indices.extend(idxs_class[:26])
    test_indices.extend(idxs_class[26:30])

train_dataset = Subset(full_dataset, train_indices)
test_dataset  = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------------------------------------------------------
# Instantiate CNN
# -------------------------------------------------------------------------
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------
num_epochs = 20
print("Training CNN...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Average Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "cnn_digits_model.pth")

# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"\nCNN Test Accuracy: {accuracy:.2f}%")

# -------------------------------------------------------------------------
# Export Weights => cnn_weights.h
# -------------------------------------------------------------------------
model.load_state_dict(torch.load("cnn_digits_model.pth", weights_only=True))
model.eval()

conv_layers = [model.conv1, model.conv2]
fc_layers = [model.fc1, model.fc2]

with open("cnn_weights.h", "w") as f:
    f.write("// cnn_weights.h : Exported weights for CNN\n")
    f.write("// Generated by train_CNN.py\n\n")

    f.write(f"#define CNN_NUM_CONV_LAYERS {len(conv_layers)}\n")
    f.write(f"#define CNN_NUM_FC_LAYERS {len(fc_layers)}\n\n")

    f.write("typedef struct {\n")
    f.write("    int in_channels;\n")
    f.write("    int out_channels;\n")
    f.write("    int kernel_h;\n")
    f.write("    int kernel_w;\n")
    f.write("    int padding;\n")
    f.write("    const float *weight;\n")
    f.write("    const float *bias;\n")
    f.write("} ConvLayerDef;\n\n")
    
    f.write("#define ACT_NONE 0\n")
    f.write("#define ACT_RELU 1\n\n")

    f.write("typedef struct {\n")
    f.write("    int in_features;\n")
    f.write("    int out_features;\n")
    f.write("    int activation_type;\n")
    f.write("    const float *weight;\n")
    f.write("    const float *bias;\n")
    f.write("} FCLayerDef;\n\n")

    # Export Conv Layers
    for idx, conv in enumerate(conv_layers):
        W = conv.weight.detach().cpu().numpy()
        b = conv.bias.detach().cpu().numpy()

        weight_name = f"CONV{idx}_WEIGHT"
        bias_name = f"CONV{idx}_BIAS"

        f.write(f"static const float {weight_name}[] = {{\n")
        out_c, in_c, kH, kW = W.shape
        for oc in range(out_c):
            for ic in range(in_c):
                for kh in range(kH):
                    row_str = ", ".join(str(x) for x in W[oc, ic, kh, :])
                    f.write(f"    {row_str},\n")
        f.write("};\n\n")

        f.write(f"static const float {bias_name}[] = {{ ")
        f.write(", ".join(str(x) for x in b))
        f.write("};\n\n")

    f.write("static ConvLayerDef CNN_CONV_LAYERS[CNN_NUM_CONV_LAYERS] = {\n")
    for idx, conv in enumerate(conv_layers):
        W = conv.weight.detach().cpu().numpy()
        out_c, in_c, kH, kW = W.shape
        padding = conv.padding[0]
        weight_name = f"CONV{idx}_WEIGHT"
        bias_name = f"CONV{idx}_BIAS"
        f.write(f"    {{ {in_c}, {out_c}, {kH}, {kW}, {padding}, {weight_name}, {bias_name} }},\n")
    f.write("};\n\n")

    # Export FC Layers
    for idx, fc in enumerate(fc_layers):
        W = fc.weight.detach().cpu().numpy()
        b = fc.bias.detach().cpu().numpy()

        weight_name = f"FC{idx}_WEIGHT"
        bias_name = f"FC{idx}_BIAS"

        out_f, in_f = W.shape
        f.write(f"static const float {weight_name}[] = {{\n")
        for i in range(out_f):
            row_str = ", ".join(str(x) for x in W[i])
            f.write(f"    {row_str},\n")
        f.write("};\n\n")

        f.write(f"static const float {bias_name}[] = {{ ")
        f.write(", ".join(str(x) for x in b))
        f.write("};\n\n")

    f.write("static FCLayerDef CNN_FC_LAYERS[CNN_NUM_FC_LAYERS] = {\n")
    for idx, fc in enumerate(fc_layers):
        W = fc.weight.detach().cpu().numpy()
        out_f, in_f = W.shape
        act_type = "ACT_RELU" if idx < len(fc_layers) - 1 else "ACT_NONE"
        weight_name = f"FC{idx}_WEIGHT"
        bias_name = f"FC{idx}_BIAS"
        f.write(f"    {{ {in_f}, {out_f}, {act_type}, {weight_name}, {bias_name} }},\n")
    f.write("};\n\n")

print("CNN export completed. File generated: cnn_weights.h")
