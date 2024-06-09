import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.resnet import ResNet18_Weights
from torch import nn
working_dir = 'object_localization'

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)

#currently replaces final layer
model.fc = nn.Sequential(nn.Linear(512, 4))
def test():
    input_image = Image.open('test_image.jpg')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(input_image)
    #input_tensor = transforms.Normalize(transforms.PILToTensor(input_image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    print(output[0])

test()
