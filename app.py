import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import streamlit as st
from PIL import Image

# GPU or CPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Move Data on the device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
# ResNet9 model
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out)
        out = self.classifier(out)

        return out
    
# Image Prediction
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    model.eval()
    # Get predictions from model
    out = model(xb)
    # Pick index with highest probability
    prob = F.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

def convert_imageTo_tensor(image):
    image = image.convert('RGB')
    return transform(image)
    
# Classes (Rock, Paper, Scissors)
classes = ['Paper', 'Rock', 'Scissors']
device = get_default_device()

# Loading the model
model = torch.load('./model/3Fold-25Epochs-FirstModel/fold2.pth')
model.to(device)

#Image normalization
stats=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
transform = tt.Compose([tt.Resize((32,32)),tt.ToTensor(), tt.Normalize(*stats)])

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Rock Paper Scissors")

st.write("""Classifies between the following hand gestures:\n
            1. paper
            2. rock
            3. scissors""")
st.write("")

file_up = st.file_uploader("Upload a Gesture", type=["png","jpg","jpeg"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Please wait for a few seconds...')
    st.write("")
    
    image_tensor = convert_imageTo_tensor(image)
    labels = predict_image(image_tensor,model)
    st.write("The Hand Gesture is: ")
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])