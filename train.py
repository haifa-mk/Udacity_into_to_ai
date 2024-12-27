

# Imports 
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import  models
from tqdm import tqdm  # For progress bars
from utils import load_data , save_checkpoint
from collections import OrderedDict






#command line
parser = argparse.ArgumentParser(description="A script for training an image classification model")
parser.add_argument('data_dir', type = str,
                    help = 'provide the directory of your data')

parser.add_argument('--save_dir', type = str, default = './',
                    help = 'Provide the save directory')
parser.add_argument('--arch', type = str, default = 'densenet121',
                help = "Choose the model architecture: 'vgg13' or 'densenet121' (default: 'vgg13')")

parser.add_argument('--learning_rate', type = float, default = 0.003,
                    help = "Learning rate for training the model (default: 0.001)")
parser.add_argument('--hidden_units', type = int, default = 512,
                    help = "Set the number of hidden units in the model's hidden layer (default: 512)")
parser.add_argument('--epochs', type = int, default = 20,
                    help = "Specify the number of training epochs (default: 20)")
#  added the  mps option  to run it on my mac
parser.add_argument('--gpu', action='store_true',
                    help = "Enable GPU acceleration if available (supports MPS or CUDA)")
args = parser.parse_args()

if args.arch not in ['vgg13', 'densenet121']:
    raise ValueError("Please choose either 'vgg13' or 'densenet121'.")




# Model setup based on architecture choice
def initialize_model(arch, hidden_units, learning_rate):
# TODO: Build and train your network
    if args.arch == 'vgg13':
        model= models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        Classifer= nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(25088,args.hidden_units,bias=True)),
                ('relu' ,nn.ReLU()),
                ('dropout',nn.Dropout(0.2)),
                ('f2', nn.Linear(args.hidden_units,80)),
                ('dropout',nn.Dropout(0.2)),
                ('f3', nn.Linear(80,102)),


                ('output',nn.LogSoftmax(dim=1)) ]))
        model.classifier = Classifer



    else :
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            Classifer= nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(1024,args.hidden_units,bias=True)),
                ('relu' ,nn.ReLU()),
                ('dropout',nn.Dropout(0.2)),
                ('f2', nn.Linear(args.hidden_units,102)),
                ('output',nn.LogSoftmax(dim=1)) ]))
            model.classifier = Classifer

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, optimizer



# Training the model

def train_model(model, trainloader, validloader, criterion, optimizer, num_epochs=20, print_every=2, device='cpu'):

    print("----------START TRAINING ----------------")
    running_loss = 0
    for e in range(num_epochs):

        # Training Phase
        train_accuracy = 0
        for images, labels in tqdm(trainloader):
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  
            log_ps = model.forward(images)  # Forward pass
            loss = criterion(log_ps, labels) 
            loss.backward()  # Backpropagation
            optimizer.step() 
            
            running_loss += loss.item() 

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equal = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

        
     
        # Print average losses and validation accuracy at specified intervals
        if (e + 1) % print_every == 0:
            test_loss = 0
            valid_accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in tqdm(validloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {e+1}/{num_epochs}.."
            f"Train loss: {running_loss/print_every:.3f}.. "
            f"Trainng accuracy: {train_accuracy/len(trainloader):.3f}.."
            f"Validation loss: {test_loss/len(validloader):.3f}.. "
            f"Validation accuracy: {valid_accuracy/len(validloader):.3f}" )

        
            model.train()
            running_loss = 0
    print("------------END-------------------")
    return model






if __name__ == "__main__":
    image_datasets , dataloaders =  load_data(args.data_dir)
    if args.gpu == 'cpu':
        device = 'cpu'
    else: 
        device = torch.device("mps" if args.gpu and torch.backends.mps.is_available() else "cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # Initialize the model and move it to the chosen device
    model, optimizer = initialize_model(args.arch, args.hidden_units, args.learning_rate)
    model.to(device)
    criterion = nn.NLLLoss()
    trained_model = train_model(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, num_epochs= args.epochs, device=device)
    save_checkpoint(args.arch,args.epochs,args.hidden_units, trained_model,optimizer, image_datasets['train'],args.save_dir,'model_checkpoint.pth')



    