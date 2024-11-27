import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
import cv2
import random,os

def getName(filepath):
    return filepath.split('\\')[-1]
    
def importDataInfo(path):
    columns = ['Center','Left','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path, r'driving_log.csv'),names=columns)
    data['Center'] = data['Center'].apply(getName)
    return data

def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data['Steering'], nBins)
    print(hist, bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.05)
        plt.plot( (-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()
        
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
                
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
        
    print("Kaldırılan Resimler: ", len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)
    print("Kalan resimler: ", len(data))
    
    return data
        
def loadData(data, path):
    steering = []
    images = []
    for i in range( len(data) ):
        indexedData = data.iloc[i]
        images.append(os.path.join(path, 'IMG',indexedData[0] ))
        steering.append( float(indexedData[3] ))
        
    images = np.asarray( images )
    steering = np.asarray( steering )
    
    return images, steering
    
def augmentImage(imagePath, steering):
    img = mpimg.imread(imagePath)
    #pan
    
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1, 0.1), 'y': (-0.1,0.1) })
        img = pan.augment_image(img)
        
    # zoom
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    
    # brigtness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply( (0.4, 1.2) )
        img = brightness.augment_image(img)
        
    # flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering
    
    return img, steering

#img, st = augmentImage('test.jpg',0)
#plt.imshow(img)
#plt.show()
#%matplotlib inline


def preProcessing(img):
    img = img[60:135, :,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img , (3,3), 0)

    img = cv2.resize(img,(200,66))
    img = img / 255 # 255'e bölmek normalizasyon oluyor.
    return img

#plt.imshow(preProcessing(mpimg.imread('test.jpg')))
#plt.show()
#%matplotlib inline

def batchGen(imgList, steeringList, batchSize, trainFlag):
    #list olarak görselleri alır , direksyon verilerini alır
    # trainflag ? test data mı train data mı ?
    while True:
        imgBatchList = []
        steeringBatchList = []
        for i in range(batchSize):
            index = random.randint(0, len(imgList)-1)
            if trainFlag:
                # datayı zenginleştir.
                img, steering = augmentImage(imgList[index],steeringList[index])
                
            else:
                img = mpimg.imread( imgList[index] )
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatchList.append(img)
            steeringBatchList.append(steering)
        yield(np.asarray(imgBatchList), np.asarray(steeringBatchList))
    
# Sıralı model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, Input
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def getModel():
    lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.9)
    model = Sequential()
    model.add(Input(shape=(66, 200, 3)))
    model.add(Convolution2D(24, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(Adam(learning_rate=lr_schedule), loss='mse')
    return model
