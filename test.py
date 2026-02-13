import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("./models/VGG11_80.pt", map_location=device)
model.eval()

for name, param in model.named_parameters():
    # and param.dim() == 4
    if 'weight' in name and 'norm' not in name:
        print(name)
        print(param)
