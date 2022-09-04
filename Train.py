from torchvision import transforms, datasets
import torch
import os
from models.ResNet import ResNet
from models.VGG import VGG16
import copy
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
import torchvision


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for features, labels in dataloaders[phase]:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                labels = torch.squeeze(labels)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(features)

                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * features.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'Pretrained_Resnet'

    if model_type == 'Pretrained_Resnet':
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, 2)
    elif model_type == 'Custom_Resnet':
        model = ResNet()
    elif model_type == 'Custom_VGG':
        model = VGG16()
    elif model_type == 'Pretrained_VGG':
        model = torchvision.models.vgg16(pretrained=True)
        model = nn.Sequential(model, nn.Linear(1000, 2))

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    else:
        data_dir = 'data/hymenoptera_data'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, drop_last=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary(model, (3, 224, 224))

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.Adam(model.parameters(), lr=0.005)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7)
    model = model.to(device)
    model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=20)

    if len(sys.argv) > 3:
        weight_dir = sys.argv[3]
    else:
        weight_dir = 'model_weights'

    torch.save(model.state_dict(), os.path.join(weight_dir, model_type))
