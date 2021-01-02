import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,9,padding=4,bias=False)
        self.conv2=nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv3=nn.Conv2d(64,256,3,padding=1,bias=False)
        self.conv4=nn.Conv2d(64,256,3,padding=1,bias=False)
        self.conv5=nn.Conv2d(64,3,9,padding=4,bias=False)
        self.bn=nn.BatchNorm2d(64)
        self.ps=nn.PixelShuffle(2)
        self.prelu=nn.PReLU()
        
    def forward(self,x):
        temp1=self.conv1(x)
        out1=self.prelu(temp1)
        
        temp1=self.conv2(out1)
        temp2=self.bn(temp1)
        temp3=self.prelu(temp2)
        temp4=self.conv2(temp3)
        temp5=self.bn(temp4)
        out2=torch.add(temp5,out1)
        
        temp1=self.conv2(out2)
        temp2=self.bn(temp1)
        temp3=self.prelu(temp2)
        temp4=self.conv2(temp3)
        temp5=self.bn(temp4)
        out3=torch.add(temp5,out2)
        
        temp1=self.conv2(out3)
        temp2=self.bn(temp1)
        temp3=self.prelu(temp2)
        temp4=self.conv2(temp3)
        temp5=self.bn(temp4)
        out4=torch.add(temp5,out3)
               
        temp1=self.conv2(out4)
        temp2=self.bn(temp1)
        temp3=self.prelu(temp2)
        temp4=self.conv2(temp3)
        temp5=self.bn(temp4)
        out5=torch.add(temp5,out4)
        
        temp1=self.conv2(out5)
        temp2=self.bn(temp1)
        temp3=self.prelu(temp2)
        temp4=self.conv2(temp3)
        temp5=self.bn(temp4)
        out6=torch.add(temp5,out5)
        
        temp1=self.conv2(out6)
        temp2=self.bn(temp1)
        out7=torch.add(temp2,out1)
        
        temp1=self.conv3(out7)
        temp2=self.ps(temp1)
        out8=self.prelu(temp2)
        
        temp1=self.conv4(out8)
        temp2=self.ps(temp1)
        out9=self.prelu(temp2)
        
        out10=self.conv5(out9)
        return out10
    
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
gen = Generator().to(cuda)
gen.load_state_dict(torch.load("state_dict.pth")['state_dict'])

#change value of k if you want to test more images
k=5
Data_folder="dataset/img_align_celeba/img_align_celeba/"
image_list=os.listdir(Data_folder)[-k:]
images1=[]
images2=[]
for img in image_list:
    image1=cv2.imread(os.path.join(Data_folder,img))
    image1=cv2.resize(image1,(256,256))
    image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2=cv2.resize(cv2.GaussianBlur(image1,(5,5),cv2.BORDER_DEFAULT),(64,64)) 
    images1.append(image1)
    images2.append(image2)
images1=np.array(images1)
imagesPT=np.array(images2)/255
imagesPT=np.moveaxis(imagesPT,3,1)
imagesPT=torch.from_numpy(imagesPT).to(cuda).float()

out_images=gen(imagesPT)
out_images=out_images.to(torch.device("cpu"))
out_images=out_images.detach().numpy()
out_images=np.moveaxis(out_images,1,3)
out_images=np.clip(out_images,0,1)


fig,axes=plt.subplots(k,3,figsize=(15,15))
for i, axis in enumerate(axes):
    axis[0].axis('off')
    image=images2[i]
    axis[0].imshow(image)
    axis[0].title.set_text('Low Resolution')
    axis[1].axis('off')
    image=out_images[i]
    axis[1].imshow(image)
    axis[1].title.set_text('Super Resolution')
    axis[2].axis('off')
    image=images1[i]
    axis[2].imshow(image)
    axis[2].title.set_text('High Resolution')
plt.savefig('Result.png',pad_inches=0)
plt.show()
