import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import streamlit as st
from PIL import Image

print("is cuda avilable")
print(torch.cuda.is_available())

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
# SmallResNet9 model
class SmallResNet9(nn.Module):
    def __init__(self):
        super(SmallResNet9, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Sequential(nn.MaxPool2d(4), 
                      nn.Flatten(), 
                      nn.Dropout(0.2),
                      nn.Linear(256, num_of_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
    
def predict_image(img, model):
    # Convert to a batch of 1
    img = img.unsqueeze(0)
    img = img.to(device)
    # Get predictions from model
    print("##Printing shape of image",img.shape)
    yb = model(img)
    print(yb)
#     # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
#     # Retrieve the class label
    return valid_ds.classes[preds[0].item()]

def convert_imageTo_tensor(image):
    image = image.convert('RGB')
    return transform(image)
    
device = get_default_device()
model = torch.load('fold0.pth')
model.to(device)

stats=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
transform = tt.Compose([tt.Resize((32,32)), tt.ToTensor(), tt.Normalize(*stats)])
valid_ds = ImageFolder('./dataset', transform)

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
    st.image(image, caption='Please wait for a few seconds...', use_column_width=True)
    st.write("")
    
#     image = Image.open("./datasets/real-images/test/paper/nasmi_179.png")
    image_tensor = convert_imageTo_tensor(image)
    print("Transforming image")
    print(image_tensor.shape)
    labels = predict_image(image_tensor,model)
    st.write("The Hand Gesture is: ",labels)