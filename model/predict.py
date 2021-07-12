import os
import cv2
try:
    from tensorflow.keras import models
except:
    pass
try:
    import torch
    from PIL import Image
    from torchvision import transforms
except:
    pass
import numpy as np

def prepare(img_array):
    img_size = 50
    new_array = cv2.resize(img_array, (img_size, img_size))
    # img_array = cv2.imread(path)
    # new_array2 = cv2.resize(img_array, (img_size, img_size))
    # new_array=np.vstack([new_array,new_array2])
    return new_array.reshape(-1, img_size, img_size, 3)


def predict(image,model,img_size,frame):
    if frame=='tensorflow':
        new_array = cv2.resize(image, (img_size, img_size))
        new_image= new_array.reshape(-1, img_size, img_size, 3)
        return model.predict([new_image])[0][0]
    elif frame=='torch':
        image_tensor = Image.fromarray(image)
        transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        torch.no_grad()
        # img = Image.open(img_path)
        img = transform(image_tensor).unsqueeze(0)
        # img_ = img.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        #print(predicted[0].item())
        print(outputs)
        print(outputs[predicted[0].item()])
        return predicted[0].item()

def getImageName(path):
    Path = eval(repr(path).replace('\\', '/'))
    pathList=Path.split('/')
    return pathList[-1].split('.')[0]

def batchPredict(batchImage,model,img_size,frame,rescale=1.):
    result=None
    output=None
    result=[]
    if frame=='tensorflow':
        for i in range(int(len(batchImage)/10)):
            inputImageName = []
            inputImage = None
            for j in range(10):
                if inputImage is None:
                    inputImage=cv2.imread(batchImage[i*10+j])*rescale
                    inputImage=cv2.resize(inputImage,(img_size,img_size))
                else:
                    image=cv2.imread(batchImage[i*10+j])*rescale
                    image=cv2.resize(image,(img_size,img_size))
                    inputImage=np.vstack([inputImage,image])
                inputImageName.append(getImageName(batchImage[i * 10 + j]))
            output=model.predict([inputImage.reshape((-1,img_size,img_size,3))])
            print(output)
            for name,i in zip(inputImageName,range(output.shape[0])):

                result.append([name,str(output[i][0])])
        inputImage = None
        inputImageName = []
        for i in range(len(batchImage)%10):
            if inputImage is None:
                inputImage = cv2.imread(batchImage[int(len(batchImage)/10) * 10 + i])*rescale
                inputImage = cv2.resize(inputImage, (img_size, img_size))
            else:
                image = cv2.imread(batchImage[int(len(batchImage)/10) * 10 + i])*rescale
                image = cv2.resize(image, (img_size, img_size))
                inputImage = np.vstack([inputImage, image])
            inputImageName.append(getImageName(batchImage[int(len(batchImage)/10) * 10 + i]))
        output = model.predict([inputImage.reshape((-1, img_size, img_size, 3))])
        for name, i in zip(inputImageName, range(output.shape[0])):

            result.append([name, str(output[i][0])])
        return result
    elif frame=='torch':
        print(len(batchImage))
        for path in batchImage:
            print(path)
            transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
            torch.no_grad()
            img = Image.open(path)
            img = transform(img).unsqueeze(0)
            # img_ = img.to(device)
            outputs = model(img)
            print(outputs)
            _, predicted = torch.max(outputs, 1)
            result.append([getImageName(path),str(predicted[0].item())])
        return result


