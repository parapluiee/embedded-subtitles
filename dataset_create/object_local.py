from dataset_obj import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from torch.nn import BCEWithLogitsLoss
from unet import UNet
import numpy as np
from torch.optim import Adam
from PIL import Image
from scipy import ndimage
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, margin=.5):
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
                mask = (torch.sigmoid(outputs.detach()) > margin)
                overlap = mask == (torch.sigmoid(labels.detach()) > margin)
                running_acc += np.mean(overlap.numpy())
            # Update the running loss and accuracy
            running_loss += loss.item()
        
        if (epoch % 10 == 0):
            compare_label_mask(labels[0].detach(), outputs[0].detach(), margin) 
 

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
                    running_acc += det_acc(outputs.detach(), labels.detach(), margin)
        if (epoch % 10 == 0):
            compare_label_mask(labels[0].detach(), outputs[0].detach(), margin, epoch + 1)

        val_loss = running_loss / len(val_loader)
        val_acc = running_acc / len(val_loader)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f} val loss: {:.4f}, val acc: {:.4f}' 
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

def compare_label_mask(label, mask, margin): 
    label_image = Image.fromarray((label.numpy()[0] * 255).astype(np.uint8))
    label_image.show()

    mask = torch.sigmoid(mask.squeeze()).numpy()
    mask = mask > margin
    #print(mask.shape)
    raw_mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    raw_mask_image.show()
    mask = flood_fill_square(mask)
    mask = mask * 255
    flood_fill_mask = Image.fromarray(mask.astype(np.uint8))
    flood_fill_mask.show()

def det_acc(outputs, labels, margin):
    mask = (torch.sigmoid(outputs.detach()) > .5)
    overlap = mask == (torch.sigmoid(labels.detach()) > .5)
    return np.mean(overlap.numpy())

def flood_fill_square(mask, buffer=0):
    label, num_label = ndimage.label(mask == 1)
    size = np.bincount(label.ravel())
    if (num_label == 0):
        return mask
    biggest_label = size[1:].argmax() + 1
    mask = label == biggest_label

    b = 0
    r = 0
    l = mask.shape[0]
    t = mask.shape[1]
    for i in range(0, mask.shape[0] - 1):
        for j in range(0, mask.shape[1]):
            if mask[i, j] == True:
                if i < l:
                    l = i
                if j < t:
                    t = j
                if i > r:
                    r = i
                if j > b:
                    b = j
    new_l = max(0, l-buffer)
    new_t = max(0, t-buffer)
    new_r = min(mask.shape[0]-1, r + buffer)
    new_b = min(mask.shape[1]-1, b + buffer)
    #print(f"Shape: {mask.shape}")
    half_mask = int(mask.shape[1] / 2)
    #l = min(new_l, min(0, new_r - int(mask.shape[1] / 2)))
    #r = max(new_r, max(mask.shape[1] -1, int(mask.shape[1] / 2) + new_l))
    if new_b >= half_mask:
        t = min(new_t, mask.shape[1] - new_b)
    if new_t <= half_mask:
        b = max(new_b, mask.shape[1] - new_t)
    #print(f"{l} {r} {t} {b}")
    for i in range(l, r):
        for j in range(t, b):
            mask[i, j] = True
    
    return mask
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
    train(model, dl_train, dl_valid, criterion, optimizer, num_epochs=40, margin=0.5)
    return model
#main()
#ds = CustomImageDataset("data/input.npy", "data/input_shape.npy", "data/labels.npy")