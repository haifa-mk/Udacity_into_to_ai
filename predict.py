import argparse
import torch 
from torch import nn
from torch import optim
from torchvision import  models 
from  utils  import  load_checkpoint , predict , plot_prediction
import json 
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str,  help='Path to the input image')
parser.add_argument('checkpoint', type=str, help='Path to the trained model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='File path for category-to-name mapping in JSON format')


#  added the  mps option  to run it on my mac
parser.add_argument('--gpu', action='store_true',
                    help = "Enable GPU acceleration if available (supports MPS or CUDA)")

args = parser.parse_args()

image_path = args.image
number_of_outputs = args.top_k
category_names = args.category_names
device = torch.device("mps" if args.gpu and torch.backends.mps.is_available() else "cuda" if args.gpu and torch.cuda.is_available() else "cpu")
checkpoint_path = args.checkpoint

json_file = args.category_names

print("Loading pre-trained model.")
model = load_checkpoint(checkpoint_path)
with open( json_file,'r') as f:
    cat_to_name = json.load(f)
print("results ... ")
top_probs, top_classes = predict(image_path, model, args.top_k)
print(f"Top Probabilities:{top_probs}")
top_classes= [cat_to_name[key] for key in top_classes]
print(f"Top Classes:{top_classes}")
plot_prediction(image_path,model,json_file,args.top_k)





