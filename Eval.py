import sys
import torch
from models.ResNet import ResNet
from models.VGG import VGG16
import torch.nn as nn
from torchvision import datasets, transforms
import os
from sklearn.metrics import classification_report
import torchvision


def print_report(model, dataloaders, class_names):
    y_pred = torch.tensor([], dtype=int).to(device)
    y_true = torch.tensor([], dtype=int).to(device)
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, outputs = torch.max(outputs, dim=1)
            y_pred = torch.cat((y_pred, outputs), 0)
            y_true = torch.cat((y_true, labels), 0)
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    print(classification_report(y_true, y_pred, target_names=class_names))
    return y_pred, y_true


if __name__ == "__main__":
    model_type = 'Custom_Resnet'
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'Pretrained_Resnet'

    if len(sys.argv) > 3:
        weight_dir = sys.argv[3]
    else:
        weight_dir = 'model_weights'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type == 'Pretrained_Resnet':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load(os.path.join(weight_dir, 'Pretrained_Resnet'), map_location=torch.device(device)))
    elif model_type == 'Custom_Resnet':
        model = ResNet()
        model.load_state_dict(torch.load(os.path.join(weight_dir, 'Custom_Resnet'), map_location=torch.device(device)))
    elif model_type == 'Custom_VGG':
        model = VGG16()
        model.load_state_dict(torch.load(os.path.join(weight_dir, 'Custom_VGG'), map_location=torch.device(device)))
    elif model_type == 'Pretrained_VGG':
        model = torchvision.models.vgg16(pretrained=True)
        model = nn.Sequential(model, nn.Linear(1000, 2))
        model.load_state_dict(torch.load(os.path.join(weight_dir, 'Pretrained_VGG'), map_location=torch.device(device)))

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    else:
        data_dir = 'data/hymenoptera_data'
    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4, shuffle=True, drop_last=True)
    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print_report(model, dataloaders, class_names)
