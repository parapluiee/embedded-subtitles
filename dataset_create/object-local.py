import torch
from torchvision import transforms
from torchvision.models.resnet import ResNet18_Weights
import torchvision
from torch import nn
from loaddataset import data_dict
from dataset_obj import CustomImageDataset
import numpy as np
import cv2
from torch.utils.data import DataLoader
import pandas as pd
def test(input_batch):
    with torch.no_grad():
        output = model(input_batch)

    print(output[0])

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the train loader
        i = 0
        for inputs, labels in train_loader:
            if (i % 4 == 0):
                print('Img in batch: ', i)
            # Move the inputs and labels to the device
            #inputs = inputs.to(device)
            #labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            #running_corrects += torch.sum(preds == labels.data)
            i+=1

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_dataset)
        #train_acc = running_corrects.double() / len(train_dataset)
        train_acc = 0
        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        #running_corrects = 0

        # Iterate over the batches of the validation loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Calculate the validation loss and accuracy
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)
        val_acc = 0
        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

def main():
    bw = True
    print('Loading Model')
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)
    model = torchvision.models.resnet18()
    for param in model.parameters():
        param.requires_grad = False
    #currently replaces final layer
    if (bw):
         model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)          
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 4))
    print('Model Loaded')

    print('Loading Data')
    train_dict = data_dict('data/train_data.obj')
    valid_dict = data_dict('data/valid_data.obj')

    df_train = pd.DataFrame.from_dict(train_dict, orient='index')
    df_valid = pd.DataFrame.from_dict(valid_dict, orient='index')

    ds_train = CustomImageDataset(df_train, 'data/w_text_train', transform = transforms.ToTensor())
    ds_valid = CustomImageDataset(df_valid, 'data/w_text_valid', transform = transforms.ToTensor())

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=32, shuffle=True)
    print ('Data Loaded')

    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=.9)
    criterion = torch.nn.CrossEntropyLoss()
    print('\nBegin Training')
    train(model, dl_train, dl_valid, criterion, optimizer, num_epochs=10)
main()
