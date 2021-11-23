# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:16:20 2021

@author: DEEPIKA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.decomposition import NMF
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import PCA



Img=np.zeros((4000,4,4))
Img[0:199]=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]])   ##A
Img[200:399]=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,1]]) ##C
Img[400:599]=np.array([[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]]) ##D
Img[600:799]=np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,0]]) ##F
Img[800:999]=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]]) ##G
Img[1000:1199]=np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]]) ##H
Img[1200:1399]=np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]) ##I
Img[1400:1599]=np.array([[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,1,1,1]]) ##J
Img[1600:1799]=np.array([[1,0,0,1],[1,0,1,0],[1,1,1,0],[1,0,0,1]]) ##K
Img[1800:1999]=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]) ##L
Img[2000:2199]=np.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]]) ##N
Img[2200:2399]=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]) ##O
Img[2400:2599]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0]]) ##P
Img[2600:2799]=np.array([[1,1,1,0],[1,0,1,0],[1,1,1,0],[0,0,1,1]]) ##Q
Img[2800:2999]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,0],[1,0,0,1]]) ##R
Img[3000:3199]=np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) ##T
Img[3200:3399]=np.array([[0,1,1,0],[0,1,0,0],[0,0,1,0],[0,1,1,0]]) ##S
Img[3400:3599]=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]) ##U
Img[3600:3799]=np.array([[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,0]]) ##V
Img[3800:3999]=np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]]) ##X
Img[4000:4199]=np.array([[0,1,0,1],[0,1,0,1],[0,0,1,0],[0,0,1,0]]) ##Y
Img[4200:4399]=np.array([[1,1,1,0],[0,0,1,0],[0,1,0,0],[0,1,1,1]]) ##Z



Img= shuffle(Img[:4000])
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('Original_Images')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(Img[i],cmap="Greys")


Siz=Img[0:199] #size track

img_gn=np.zeros((4000,4,4))#For storing  gaussian_noissy image

#Gaussian_Noise_1
mn=0;
vr=0.1
gn1=np.random.normal(mn,vr,Siz.shape)
Img[400:599]=Img[400:599]+gn1
Img[1200:1399]=Img[1200:1399]+gn1
Img[1400:1599]=Img[1400:1599]+gn1
Img[1600:1799]=Img[1600:1799]+gn1
Img[1800:1999]=Img[1800:1999]+gn1

#Gaussian_Noise_2
mn=0;
vr=0.2
gn2=np.random.normal(mn,vr,Siz.shape)
Img[600:799]=Img[600:799]+gn2
Img[2000:2199]=Img[2000:2199]+gn2
Img[2200:2399]=Img[2200:2399]+gn2
Img[2400:2599]=Img[2400:2599]+gn2
Img[2600:2799]=Img[2600:2799]+gn2

#Gaussian_Noise_3
mn=0;
vr=0.3
gn3=np.random.normal(mn,vr,Siz.shape)
Img[400:599]=Img[400:599]+gn3
Img[3000:3199]=Img[3000:3199]+gn3
Img[3200:3399]=Img[3200:3399]+gn3
Img[3400:3599]=Img[3400:3599]+gn3
Img[3600:3799]=Img[3600:3799]+gn3

#Gaussian_Noise_4
mn=0;
vr=0.06
gn4=np.random.normal(mn,vr,Siz.shape)
Img[200:399]=Img[200:399]+gn4
Img[2200:2399]=Img[2200:2399]+gn4
Img[1600:1799]=Img[1600:1799]+gn4
Img[3600:3799]=Img[3600:3799]+gn4
Img[4000:4199]=Img[4000:4199]+gn4

#Gaussian_Noise_5
mn=0;
vr=0.08
gn5=np.random.normal(mn,vr,Siz.shape)
Img[800:999]=Img[800:999]+gn5
Img[2800:2900]=Img[4200:4399]+gn5
Img[4000:4199]=Img[4000:4199]+gn5
Img[3600:3799]=Img[3600:3799]+gn5
Img[1000:1199]=Img[1000:1199]+gn5
Img[2400:2799]=Img[2400:2799]+gn5


img_gn=Img

fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('Noisy_Image')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(img_gn[i],cmap="Greys")

A=img_gn.reshape(4000,16)

#Applying PCA

pca=PCA(16)
Pc=pca.fit(A)
p=pca.components_
Fn=p.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title(' PCA_OutPut ')
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(Fn[i],cmap="Greys")
    
    
#Applying NMF    

nmf = NMF(16)
K = nmf.fit(np.abs(A))
L=nmf.components_
C=L.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('NMF_OutPut')
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(C[i],cmap="Greys")
    

#Applying Divtionary Learning

dl = DictionaryLearning(16)
F= dl.fit(np.abs(A))
Q=dl.components_
DL=Q.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('Dictionary_OutPut')
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(DL[i],cmap="Greys")
      
