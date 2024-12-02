from dataset_obj import CustomImageDataset
from torch.utils.data import DataLoader
import GIoU
from torchvision import transforms
import torch
import torchvision
from torch.nn import BCEWithLogitsLoss
from unet import UNet
import numpy as np
from torch.optim import Adam
from PIL import Image
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_acc = 0.0
        t_batch_size = train_loader.batch_size
        # Iterate over the batches of the train loader
        i = 0
        for inputs, labels in train_loader:
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
            if (epoch % 5 == 0):
                mask = (torch.sigmoid(outputs.detach()) > .5)
                overlap = mask == (torch.sigmoid(labels.detach()) > .5)
                running_acc += np.mean(overlap.numpy())
            # Update the running loss and accuracy
            running_loss += loss.item()
        
        if (epoch % 10 == 0):
            Image.fromarray(labels[0].detach().numpy()[0] * 255).show()
            mask = torch.sigmoid(outputs[0].detach().squeeze()).numpy()
            mask = mask > .5
            mask = mask * 255
            Image.fromarray(mask.astype(np.uint8)).show()


        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)
        #train_acc = running_acc / len(train_loader)
        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_acc = 0.0
        #running_pixels_off = 0.0
        # Iterate over the batches of the validation loader
        
        with torch.no_grad():
            i = 0
            for inputs, labels in val_loader:
                   # Move the inputs and labels to the device
                #inputs = inputs.to(device)
                #labels = labels.to(device)


                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Update the running loss and accuracy
                running_loss += loss.item()
                if (epoch % 5 == 0):
                    mask = (torch.sigmoid(outputs.detach()) > .5)
                    overlap = mask == (torch.sigmoid(labels.detach()) > .5)
                    running_acc += np.mean(overlap.numpy())
                #running_pixels_off += torch.abs(outputs - labels).sum(dim=1).mean()
        #pixels_off = running_pixels_off / len(val_loader)
        # Calculate the validation loss and accuracy

        val_loss = running_loss / len(val_loader)
        val_acc = running_acc / len(val_loader)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f} val loss: {:.4f}, val acc: {:.4f}' 
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

def main():
    bw = True
    batch_size = 16
    print('Loading Model')
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)
    #currently replaces final layer

    print('Model Loaded')

    print('Loading Data')

    ds = CustomImageDataset("data/input.npy", "data/input_shape.npy", "data/labels.npy", "data/label_shape.npy")
    size = np.load("data/input_shape.npy")
    print(size)
    model = UNet(size[1], size[2])
    train_set, val_set = torch.utils.data.random_split(ds, [.8, .2])
    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    print ('Data Loaded')

    optimizer = Adam(model.parameters(), lr=.001)
    #criterion = G_IoU()
    criterion = BCEWithLogitsLoss()
    #criterion = torchvision.ops.distance_box_iou_loss
    gpu = torch.cuda.is_available()
    print("GPU Available: ", gpu)  
    print('\nBegin Training')
    train(model, dl_train, dl_valid, criterion, optimizer, num_epochs=25)
    return model
#main()
#ds = CustomImageDataset("data/input.npy", "data/input_shape.npy", "data/labels.npy")