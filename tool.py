import os
import subprocess
from connectFunction import *
global lastlayer


def residualBlock(file,filters):
    global lastLayer
    lasto=getLastlayer()
    
    addconv2d(file,0,filters,3,3,1,1,1)
    addBatchNorm(file,0,0.999,0.000011)
    addrelu(file,0)
    addconv2d(file,0,filters,3,3,1,1,1)
    addBatchNorm(file,0,0.999,0.000011)
    addAdd(file,lasto-getLastlayer())
    addrelu(file,0)
    
def residualBlockFirst(file,filters,strides):
    global lastLayer
    addconv2d(file,0,filters,3,3,strides,1,1)
    lasto=lastlayer
    addBatchNorm(file,0,0.999,0.000011)
    addrelu(file,0)
    addconv2d(file,0,filters,3,3,1,1,1)
    addBatchNorm(file,0,0.999,0.000011)
    addAdd(file,lasto-lastlayer)
    addrelu(file,0)
def createNN(file,out):
    filters = [64, 64, 128, 256, 512]
    kernels = [7, 3, 3, 3, 3]
    strides = [2, 1, 2, 2, 2]
    
    addconv2d(file,0,filters[0],kernels[0],kernels[0],strides[0],3,1)
    addBatchNorm(file,0,0.999,0.000011)
    addrelu(file,0)
    addpoolmax(file,0,3,2,1)
    
    residualBlock(file,filters[0])
    residualBlock(file,filters[0])
    residualBlockFirst(file,filters[2],strides[2])
    residualBlock(file,filters[2])
    residualBlockFirst(file,filters[3],strides[3])
    residualBlock(file,filters[3])
    residualBlockFirst(file,filters[4],strides[4])
    residualBlock(file,filters[4])
    adddropout(file,0,0.5)
    addFully(file,0,500)
    addrelu(file,0)
    addEnd(file,out)
    
def create_file(filename,batch,cha,hei,wei,traSam,testSam,out,leaRat,mom,leaDec):
    file=open(filename,"w+")
    global lastlayer
    lastlayer=0
    firstVal(file,batch,cha,hei,wei,traSam,testSam,out,leaRat,mom,leaDec)
    createNN(file,out)
    file.close()
    
filename = "resnet18.ini"
##  filename,batchSize,inputChanal,inputHeigth,inputWeigth,trainingSample,
#testSample,outChanal,learningRate,momentum,learningDecay
create_file(filename,1000,3,32,32,50000,10000,10,0.01,0.95,0.0)
os.system("AkaCudnNet.exe "+filename)
