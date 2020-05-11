#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle


# # Loading the Dataset

# In[3]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        x = pickle.load(fo, encoding='bytes')
    return x[b'data'],x[b'labels']


# In[4]:


path="C:/Users/Devashi Jain/Desktop/IIIT-D/SML/Assignment3/train"
TrainingSet= pd.DataFrame()
Labels=[]

for folder in os.listdir(path):
    file=os.path.join(path,folder)
    Data,Label=unpickle(file)
    Data=pd.DataFrame(Data)
    Labels.extend(Label)
    TrainingSet=TrainingSet.append(Data)
TrainingSet.index=range(len(TrainingSet))   


# In[5]:


import numpy as np
def RGBtoGray(dataframe):
    dataframe=dataframe.values
    grayscale=np.zeros((len(dataframe),1024))
    for i in range(len(dataframe)):
        for j in range(1024):
            grayscale[i][j]=0.299*dataframe[i][j]+0.587*dataframe[i][j+1024]+0.114*dataframe[i][j+2048]
    return grayscale  


# In[6]:


gray=RGBtoGray(TrainingSet)


# In[7]:


import cv2
import matplotlib.pyplot as plt
a=np.array(gray[8]).reshape(32,32)
plt.imshow(a,cmap='gray')
plt.show()


# In[8]:


Train=pd.DataFrame(gray)


# In[9]:


Train


# In[10]:


def Patches(Image):
    row,col=Image.shape
    y=[]
    x=np.array_split(Image, 2,axis=0)
    for sp in x:
        y.extend(np.array_split(sp, 2,axis=1))
    return y


# In[11]:


def LBP(gray_image):
    gray_scale_image=np.zeros((len(gray_image)+(2),len(gray_image[0])+(2)))
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gray_scale_image[i+1][j+1]=gray_image[i][j]
#     print(gray_scale_image.shape)
    img1=gray_scale_image.astype(np.uint8)
#     cv2.imshow('BGR color space',img1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    gray=copy.deepcopy(gray_scale_image)
    row,col=gray_scale_image.shape
#     print(row,col)
    for i in range(1,row-1):
        for j in range(1,col-1):
            Bit=np.zeros(8)
            if gray_scale_image[i][j+1]>gray_scale_image[i][j]:
                Bit[0]=1
            if gray_scale_image[i-1][j+1]>gray_scale_image[i][j]:
                Bit[1]=1
            if gray_scale_image[i-1][j]>gray_scale_image[i][j]:
                Bit[2]=1
            if gray_scale_image[i-1][j-1]>gray_scale_image[i][j]:
                Bit[3]=1
            if gray_scale_image[i][j-1]>gray_scale_image[i][j]:
                Bit[4]=1
            if gray_scale_image[i+1][j-1]>gray_scale_image[i][j]:
                Bit[5]=1
            if gray_scale_image[i+1][j]>gray_scale_image[i][j]:
                Bit[6]=1
            if gray_scale_image[i+1][j+1]>gray_scale_image[i][j]:
                Bit[7]=1
            y=int(''.join(map(lambda Bit: str(int(Bit)), Bit)), 2)
            gray[i][j]=y
    gray=gray[1:-1, 1:-1]
    return gray
# LBP_Image=LBP(img)
# LBP_image1=LBP_Image.astype(np.uint8)
# print(LBP_image1.shape)
# cv2.imshow('BGR color space',LBP_image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[12]:


import copy
Bag=[]
Image_Vec=[]
for i in range(len(Train)):
    Image=[]
    a=np.array(Train.iloc[i]).reshape(32,32)
    patches=Patches(a)
    for j in range(len(patches)):
        LBP_Img=LBP(patches[j]) 
        x=LBP_Img.ravel()
        Bag.append(x)
        Image.append(x)
    Image_Vec.append(Image)


# In[13]:


len(Bag)


# In[14]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=40, random_state=0).fit(Bag)
centers=kmeans.cluster_centers_


# In[15]:


labels=kmeans.labels_
T=[]
for i in range(len(labels)):
    if(labels[i]==2):
        T.append(i)


# In[24]:


T


# In[25]:


import cv2
import matplotlib.pyplot as plt
a=np.array(centers[2]).reshape(16,16)
plt.imshow(a,cmap='gray')
plt.show()
a=np.array(Bag[250]).reshape(16,16)
plt.imshow(a,cmap='gray')
plt.show()
a=np.array(Bag[73]).reshape(16,16)
plt.imshow(a,cmap='gray')
plt.show()
a=np.array(Bag[96]).reshape(16,16)
plt.imshow(a,cmap='gray')
plt.show()
a=np.array(Bag[138]).reshape(16,16)
plt.imshow(a,cmap='gray')
plt.show()
a=np.array(Bag[161]).reshape(16,16)
plt.imshow(a,cmap='gray')
plt.show()


# In[16]:


Representatives=pd.DataFrame(centers)


# In[17]:


Representatives


# In[ ]:





# In[16]:


train_frame=pd.concat([Train,Labels],axis=1)


# In[ ]:


train_frame


# In[26]:


import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity
def Similarity(image,center):
    image_vec=np.zeros(40)
    for i in range(len(image)):
        w={}
        for j in range(len(center)):
            x=[]
            image[i] = image[i].reshape(1,-1)
            x.append(center[j])
            score=cosine_similarity(image[i],x)
            w[j]=score
        index=max(w.items(), key=operator.itemgetter(1))[0]
        image_vec[index]+=1
    return image_vec


# In[27]:


print(len(Image_Vec[0][0]))


# In[28]:


Data=[]
for i in range(len(Image_Vec)):
    Vector=Similarity(Image_Vec[i],centers)
    Data.append(Vector)


# In[29]:


print(len(Data))


# In[30]:


path="C:/Users/Devashi Jain/Desktop/IIIT-D/SML/Assignment3/test"
TestingSet= pd.DataFrame()
TestLabels=pd.DataFrame()
for folder in os.listdir(path):
    file=os.path.join(path,folder)
    print(file)
    TestData,Labels=unpickle(file)
    TestData=pd.DataFrame(TestData)
    TestLabels=TestLabels.append(Labels)
    TestingSet=TestingSet.append(TestData)
TestingSet.index=range(len(TestingSet)) 


# In[31]:


print(len(TestLabels))


# In[32]:


grayTest=RGBtoGray(TestingSet)


# In[33]:


Test=pd.DataFrame(grayTest)


# In[34]:


import copy
TestBag=[]
Test_Image_Vec=[]
for i in range(len(Test)):
    Image=[]
    a=np.array(Test.iloc[i]).reshape(32,32)
    patches=Patches(a)
    for j in range(len(patches)):
        LBP_Img=LBP(patches[j]) 
        x=LBP_Img.ravel()
        TestBag.append(x)
        Image.append(x)
    Test_Image_Vec.append(Image)


# In[68]:


TestData=[]
for i in range(len(Test_Image_Vec)):
    Vector=Similarity(Test_Image_Vec[i],centers)
    TestData.append(Vector)


# In[69]:


Train_Data=pd.DataFrame(Data)
Test_Data=pd.DataFrame(TestData)


# In[89]:


Train_Data


# In[90]:


Test_Data


# In[35]:


u=Test_Image_Vec[10]
for i in range(len(u)):
    a=np.array(u[i]).reshape(16,16)
    plt.imshow(a,cmap='gray')
    plt.show()


# In[88]:


print((Labels))


# In[78]:


from sklearn.naive_bayes import GaussianNB
def Classifier(train,trainlabel,test):
        clf = GaussianNB()
        clf.fit(train,trainlabel)
        predicted=clf.predict(test)
        return predicted
Predicted_Labels=Classifier(Train_Data,Labels,Test_Data)


# In[120]:


y=list((TestLabels).flatten())


# In[122]:


def accuracy(predicted,actual):
    c=0
    for i in range(len(predicted)):
        if(predicted[i]==actual[i]):
            c=c+1
    s=c/len(predicted)
    return s
Accuracy=accuracy(Predicted_Labels,y)
print(Accuracy)


# In[ ]:




