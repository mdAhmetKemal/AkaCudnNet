import os
import subprocess

global lastlayer
lastlayer=0

def firstVal(file,batch,cha,hei,wei,traSam,testSam,out,leaRat,mom,leaDec):
    file.write("#total layer/optimizer\n")
    file.write("100 sgd\n")
    file.write("#batch size\n")
    file.write("%d\n" %(batch))
    file.write("#input chanal x heigth x weigth x traimSample x testSample\n")
    file.write("%d %d %d %d %d\n" % (cha,hei,wei,traSam,testSam) )
    file.write("#output chanal x heigth x weigth \n")
    file.write("%d\n" % (out) )
    file.write("#input Source\n")
    file.write("cifar10train.bin\n")
    file.write("#output Sourca\n")
    file.write("cifar10test.bin\n")
    file.write("#alpha beta gamma teta parameters\n")
    file.write("%f %f 1 %f" %(leaRat,mom,leaDec))
def getLastlayer():
    a =5
    return lastlayer
def addconv2d(file,origin,fea,h,w,stride,pad,dil):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d conv2d %d %d %d %d %d %d"%(inLay,outLay,fea,h,w,stride,pad,dil))

def addBatchNorm(file,origin,mom,epsi):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d batchNormCha %f %f"%(inLay,outLay,mom,epsi))
def addrelu(file,origin):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d relu"%(inLay,outLay))

def addpoolmax(file,origin,h,stride,pad):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d poolmax %d %d %d"%(inLay,outLay,h,stride,pad))
def addfully(file,origin,fully):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d fully %d"%(inLay,outLay,fully))
def adddropout(file,origin,dropout):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d dropout %f"%(inLay,outLay,dropout))
def addsoftmax(file,origin):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d softmax"%(inLay,outLay))
def addAdd(file,origin):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer
    #lastlayer+=1
    file.write("\nc %d %d add"%(inLay,outLay))
def addFully(file,origin,fully):
    global lastlayer
    inLay=lastlayer+origin
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d fully %d"%(inLay,outLay,fully))

def addconcat2(file,origin1,origin2):
    global lastlayer
    inLay1=lastlayer+origin1
    inLay2=lastlayer+origin2
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d concat"%(inLay1,outLay))
    file.write("\nc %d %d concat"%(inLay2,outLay))

def addEnd(file,out):
    global lastlayer
    inLay1=lastlayer
    outLay=lastlayer+1
    lastlayer+=1
    file.write("\nc %d %d fully %d"%(inLay1,outLay,out))
    file.write("\nc %d %d softmax"%(inLay1+1,outLay+1))
