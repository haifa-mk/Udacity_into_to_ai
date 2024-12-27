import torch
from torch import optim
from torchvision import datasets, transforms, models 
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
from torch import nn
from collections import OrderedDict
import os
import json

def load_data(data_dir='flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    
    data_transforms = {
    'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ]),
    'valid': transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                              ]),
    'test': transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                              ])  
    
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir,transform = data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir,transform = data_transforms['valid']),
        'test':datasets.ImageFolder(test_dir,transform= data_transforms['test'])
        
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders ={'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }

    return   image_datasets , dataloaders



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    transform= transforms.Compose([
        transforms.Resize(256),                
        transforms.CenterCrop(224), 
        transforms.ToTensor(),                  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
     # Open the image and apply transformations
    img = Image.open(image)
    img = transform(img).numpy()          
    
    return img



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))   
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax




def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get the top K probabilities and indices
    probs, indices = output.topk(topk)
    probs = probs.exp()  
    
    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices[0].tolist()]
    
    return probs[0].tolist(), classes   



def save_checkpoint(arch,epochs , hidden , model, optimizer,train_data , dir ,save_path='model_checkpoint.pth'):

    # Store necessary information in a dictionary
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': arch,
        'epoch': epochs,
        'hidden': hidden,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    # Save the checkpoint

    torch.save(checkpoint, os.path.join(dir ,save_path))

    print(f"Checkpoint saved to {save_path}")




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    hidden = checkpoint['hidden']

    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            Classifer= nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(1024,hidden,bias=True)),
                ('relu' ,nn.ReLU()),
                ('dropout',nn.Dropout(0.2)),
                ('f2', nn.Linear(hidden,102)),
                ('output',nn.LogSoftmax(dim=1)) ]))
            model.classifier = Classifer
    elif checkpoint['arch'] == 'vgg13':
        model= models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False 
        Classifer= nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(25088,hidden,bias=True)),
                ('relu' ,nn.ReLU()),
                ('dropout',nn.Dropout(0.2)),
                ('f2', nn.Linear(hidden,80)),
                ('dropout',nn.Dropout(0.2)),
                ('f3', nn.Linear(80,102)),
                ('output',nn.LogSoftmax(dim=1)) ]))
        model.classifier = Classifer
    



    for param in model.parameters():
        param.requires_grad = False
        
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model


def plot_prediction(image_path, model,json_file,top_k):
    '''Display the image and plot the top 5 class probabilities.'''
    
    top_probs, top_flowers = predict(image_path, model,top_k)
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    # Display the image
    plt.figure(figsize=(6,10))
    ax = plt.subplot(2,1,1)
    image = process_image(image_path)
    imshow(image, ax=ax)
    ax.axis('off') 
    names = [cat_to_name[key] for key in top_flowers]

    ax.set_title(names[0])

    # Display the top 5 classes as a bar chart
    plt.subplot(2,1,2)
    y_pos = range(len(top_flowers))
    plt.barh(y_pos, top_probs, color='skyblue')
    plt.yticks(y_pos, names)
    plt.gca().invert_yaxis()
    plt.xlabel('Probability')
    plt.title('Top 5 Class Probabilities')
    
    plt.tight_layout()
    plt.show()


