# Importing Libraries

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchsummary import summary
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
import cv2
from matplotlib import pyplot as plt
from PIL import Image

#Data folder and images for training
Data_folder="dataset/img_align_celeba/img_align_celeba/"
images=os.listdir(Data_folder)

#Training on 5000 images
#to change the number of file for training change 1500 to another value
imageList=images[:5000]


#Generator
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


#Discrimiator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,3,padding=1,bias=False)
        self.conv2=nn.Conv2d(64,64,3,stride=2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,128,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128,128,3,stride=2,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(128)
        self.conv5=nn.Conv2d(128,256,3,padding=1,bias=False)
        self.bn4=nn.BatchNorm2d(256)
        self.conv6=nn.Conv2d(256,256,3,stride=2,padding=1,bias=False)
        self.bn5=nn.BatchNorm2d(256)
        self.conv7=nn.Conv2d(256,512,3,padding=1,bias=False)
        self.bn6=nn.BatchNorm2d(512)
        self.conv8=nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
        self.bn7=nn.BatchNorm2d(512)
        self.fc1=nn.Linear(512*16*16,1024)
        self.fc2=nn.Linear(1024,1)
        self.drop=nn.Dropout2d(0.3)
        
    def forward(self,x):
        temp1=self.conv1(x)
        out1=F.leaky_relu(temp1)
        
        temp1=self.conv2(out1)
        temp2=self.bn1(temp1)
        out2=F.leaky_relu(temp2)
        
        temp1=self.conv3(out2)
        temp2=self.bn2(temp1)
        out3=F.leaky_relu(temp2)
        
        temp1=self.conv4(out3)
        temp2=self.bn3(temp1)
        out4=F.leaky_relu(temp2)
        
        temp1=self.conv5(out4)
        temp2=self.bn4(temp1)
        out5 = F.leaky_relu(temp2)
        
        temp1=self.conv6(out5)
        temp2=self.bn5(temp1)
        out6 = F.leaky_relu(temp2)
        
        temp1=self.conv7(out6)
        temp2=self.bn6(temp1)
        out7 = F.leaky_relu(temp2)
        
        temp1=self.conv8(out7)
        temp2=self.bn7(temp1)
        out8 = F.leaky_relu(temp2)
        out8 = out8.view(-1,out8.size(1)*out8.size(2)*out8.size(3))
        
        temp1=self.fc1(out8)
        out9 = F.leaky_relu(temp1)
        
        temp1=self.fc2(out9)
        temp2=self.drop(temp1)
        out10 = torch.sigmoid(temp2)
        return out10



#Checking GPU
cuda=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

gen=Generator().to(cuda).float()
#summary(gen,(3,64,64))

disc=Discriminator().to(cuda).float()
#summary(disc,(3,256,256))

# Pretrained VGG19 model from torchvision library
vgg=models.vgg19(pretrained=True).to(cuda)

#Losses
gen_loss=nn.BCELoss()
vgg_loss=nn.MSELoss()
mse_loss=nn.MSELoss()
disc_loss=nn.BCELoss()

#Optimizer
gen_optimizer=optim.Adam(gen.parameters(),lr=0.0005)
disc_optimizer=optim.Adam(disc.parameters(),lr=0.0005)


#Conversion of original High Resolution images from 178*218 to 256*256
def load_full_size(image_list,img_folder):
    images=[]
    for file in (image_list):
        img=cv2.imread(os.path.join(img_folder,file))
        img=cv2.resize(img,(256,256)) 
        img=np.moveaxis(img, 2, 0)
        images.append(img)
    return np.array(images)


#Conversion from 256*256 images to 64*64 Low resolution images
def load_small_size(image_list,img_folder):
    images=[]
    for image in (image_list):
        img=cv2.imread(os.path.join(img_folder,image))
        img=cv2.resize(cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT),(64,64)) 
        img=np.moveaxis(img, 2, 0)
        images.append(img)
    return np.array(images)

#Loading generator model from Checkpoint
def load_checkpoint(model_path):
    checkpoint = torch.load(model_path)
    model=checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def test_model(image_list,model_path):
    images=[]
    for img in image_list:
        image=cv2.imread(os.path.join(Data_folder,img))
        image=cv2.resize(cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT),(64,64)) 
        images.append(image)
    images=np.array(images)/255
    imagearray=np.moveaxis(images,3,1)
    imagearray=torch.from_numpy(imagearray).float()
    model=load_checkpoint(model_path)
    out_image=model(imagearray)
    out_image=out_image.numpy()
    out_image=np.moveaxis(out_image,1,3)
    out_image=np.clip(out_image,0,1)
    return out_image

def show_images(images):
    fig,axes=plt.subplots(1,images.shape[0],figsize=(10,10))
    for i,axis in enumerate(axes):
        axis.axis('off')
        image=images[i]
        #image=Image.fromarray((image * 255).astype('uint8'))
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axis.imshow(image)
    plt.savefig(os.path.join(cwd,"out/SR")+"_"+str(epoch)+".png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    
epochs=1000
batch_size=16

#Weight file and output file
cwd=os.getcwd()
weight_folder=os.path.join(cwd,"SRPT_weights")
image_output=os.path.join(cwd,"out")

if not os.path.exists(weight_folder):
    os.makedirs(weight_folder)

if not os.path.exists(image_output):
    os.makedirs(image_output)


#Training Models
batch_count = len(imageList)//batch_size
d1loss_list=[]
d2loss_list=[]
gloss_list=[]
vloss_list=[]
mloss_list=[]
for epoch in range(epochs):
    temp1=[]
    temp2=[]
    temp3=[]
    temp4=[]
    temp5=[]
    
    for batch in tqdm(range(batch_count)):
        images_list=[img for img in imageList[batch*batch_size:(batch+1)*batch_size]]
        lr_images=load_small_size(images_list,Data_folder)/255
        hr_images=load_full_size(images_list,Data_folder)/255
        
        #Discrimiator
        disc_optimizer.zero_grad()
        gen_out=gen(torch.from_numpy(lr_images).to(cuda).float())
        fake_labels=disc(gen_out)
        real_labels=disc(torch.from_numpy(hr_images).to(cuda).float())
        d1_loss=(disc_loss(fake_labels,torch.zeros_like(fake_labels,dtype=torch.float)))
        d2_loss=(disc_loss(real_labels,torch.ones_like(real_labels,dtype=torch.float)))
        d2_loss.backward()
        d1_loss.backward(retain_graph=True)
        disc_optimizer.step()
        
        #Generator
        gen_optimizer.zero_grad()
        g_loss=gen_loss(fake_labels.data,torch.ones_like(fake_labels,dtype=torch.float))
        v_loss=vgg_loss(vgg.features[:7](gen_out),vgg.features[:7](torch.from_numpy(hr_images).to(cuda).float()))
        m_loss=mse_loss(gen_out,torch.from_numpy(hr_images).to(cuda).float())
        generator_loss=g_loss+v_loss+m_loss
        generator_loss.backward()
        gen_optimizer.step()
        
        #storing loss
        temp1.append(d1_loss.item())
        temp2.append(d2_loss.item())
        temp3.append(g_loss.item())
        temp4.append(v_loss.item())
        temp5.append(m_loss.item())
    
    d1loss_list.append(sum(temp1)/len(temp1))
    d2loss_list.append(sum(temp2)/len(temp2))
    gloss_list.append(sum(temp3)/len(temp3))
    vloss_list.append(sum(temp4)/len(temp4))
    mloss_list.append(sum(temp5)/len(temp5))
    print()
    print(f'Epoch {epoch+1}     d1_loss {d1loss_list[-1]}     d2_loss {d2loss_list[-1]}')
    print(f'Generator_Loss {gloss_list[-1]}     VGG_Loss {vloss_list[-1]}     Mean_Loss {mloss_list[-1]}')
    
    if(epoch%10==0):
        checkpoint = {'model': Generator(),'state_dict': gen.state_dict()}
        torch.save(checkpoint,os.path.join(weight_folder,"SR"+str(epoch+1)+".pth"))
        torch.cuda.empty_cache()
        out_images=test_model(images[-3:],os.path.join(weight_folder,"SR"+str(epoch+1)+".pth"))
        show_images(out_images)

#Plottig result
#Discriminator loss
d_loss_list=np.array(d1loss_list)+np.array(d2loss_list)
plt.plot(np.arange(1,len(d_loss_list)+1),d_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator loss')
plt.savefig('DISC.png')
plt.show()

#Generator loss
plt.plot(np.arange(1,len(gloss_list)+1),gloss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator loss')
plt.savefig('GEN.png')

#VGG loss
plt.plot(np.arange(1,len(vloss_list)+1),vloss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VGG loss')
plt.savefig('VGG.png')
plt.show()

#MSE loss
plt.plot(np.arange(1,len(mloss_list)+1),mloss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MSE loss')
plt.savefig('MSE.png')
plt.show()
plt.show()



