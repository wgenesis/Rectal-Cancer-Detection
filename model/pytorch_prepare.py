import torch
from PIL import Image
from torchvision import transforms
from Model.pytorch import *
import cv2

from PIL import Image
device = torch.device('cuda:0')
image_path='E:/Python/TensorFlow/tensorflow/cancer2995/lung_colon_image_set/test/n/colonn38.jpeg'
transform=transforms.Compose([transforms.Resize((125, 125)),transforms.ToTensor()])
def prediect(img_path,model):
    image=cv2.imread(image_path)

    #image_tensor = torch.from_numpy(image)
    image_tensor=Image.fromarray(image)
    transform = transforms.Compose([transforms.Resize((125, 125)), transforms.ToTensor()])
    torch.no_grad()
    # img = Image.open(img_path)
    img = transform(image_tensor).unsqueeze(0)
    # img_ = img.to(device)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print(predicted[0].item())

    # net=torch.load('net.pkl')
    # #net=net.to(device)
    # torch.no_grad()
    # img=Image.open(img_path)
    # img=transform(img).unsqueeze(0)
    # img_ = img.to(device)
    # outputs = net(img_)
    # _, predicted = torch.max(outputs, 1)
    # print(predicted[0])
    # #print('this picture maybe :',classes[predicted[0]])
if __name__ == '__main__':
    net=Net()
    net.load_state_dict(torch.load('./net_params.pkl'))
    #net=torch.load('./net_params.pkl')
    prediect(image_path,net)

