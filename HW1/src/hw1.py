import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from hw1UI import Ui_Dialog
import cv2 as cv
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def on_trackbar(val):
        img1 = cv.imread("../image/dog.bmp",cv.IMREAD_COLOR)
        img2 = cv.imread("../image/dog.bmp",cv.IMREAD_COLOR)
        img2 = cv.flip(img1, 1)
        alpha = val / 100
        beta = ( 1.0 - alpha )
        dst = cv.addWeighted(img1, alpha, img2, beta, 0.0)
        cv.imshow('Blending Image', dst)

clickCounter = 0
src = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
def mouse_click(event, x, y, flags, param):
    global clickCounter
    img = cv.imread("../image/OriginalPerspective.png",cv.IMREAD_COLOR)

    if event == cv.EVENT_LBUTTONDOWN:
        src[clickCounter][0] = x
        src[clickCounter][1] = y
        clickCounter=clickCounter+1
        if clickCounter > 3:
            clickCounter = 0
            dst = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
            map = cv.getPerspectiveTransform(src, dst)
            dst = cv.warpPerspective(img, map, (450, 450))
            cv.imshow('Perspective Image', dst)

# CNN
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = "SGD"
def downloadMNIST():
    data_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root="./", train=True,download=True, transform=data_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(50000)))
    val_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(50000, 60000)))
    test_data = datasets.MNIST(root="./", train=False, download=True, transform=data_transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    return train_data, train_loader, test_data, test_loader, val_loader

def flatten(x):
    x = torch.flatten(x, start_dim=1)
    return x
LOSS=[]
LOSS_TRAIN=[]
ACC_TRAIN=[]
ACC_TEST=[]
FIG_X=[]
flag=0

class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.device = device
        
    def train_loop(self, model, train_loader, val_loader):
        for epoch in range(EPOCHS):
#             print("---------------- Epoch {} ----------------".format(epoch))
            self._training_step(model, train_loader, epoch)
            
            self._validate(model, val_loader, epoch)
    
    def test(self, model, test_loader):
            print("---------------- Testing ----------------")
            self._validate(model, test_loader, 0, state="Testing")
            
    def _training_step(self, model, loader, epoch):
        model.train()
        global flag
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            N = X.shape[0]
            
            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            
            ###################################
            LOSS.append(loss.data.item())
            FIG_X.append(flag)
            flag=flag+1
            ###################################
            
            if step >= 0 and (step % 100 == 0):
                self._state_logging(outs, y, loss, step, epoch, "Training")
            
            loss.backward()
            self.optimizer.step()
            
    def _validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]
                
                outs = model(X)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)
            ####################################
            if(state == "Validate"):
                LOSS_TRAIN.append(loss)
                ACC_TRAIN.append(self._accuracy(outs, y))
            elif(state == "Testing"):
                ACC_TEST.append(self._accuracy(outs, y))
            ####################################
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCHS, state, step, loss, acc))
            
    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv3 = None, None
        self.sub2, self.sub4 = None, None
        self.fc1, self.fc2, self.fc3 = None, None, None
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=False)
        )
        self.sub2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=False)
        )
        self.sub4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=False)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.sub2(out)
        out = self.conv2(out)
        out = self.sub4(out)
        out = flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
# end CNN

class MyDlg(QDialog):
    def __init__(self):
        super(MyDlg, self).__init__()

        # Set up the user interface from Designer.
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Part 1
        self.ui.button1_1.clicked.connect(self.clickButton1_1)
        self.ui.button1_2.clicked.connect(self.clickButton1_2)
        self.ui.button1_3.clicked.connect(self.clickButton1_3)
        self.ui.button1_4.clicked.connect(self.clickButton1_4)

        # Part 2
        self.ui.button2_1.clicked.connect(self.clickButton2_1)
        self.ui.button2_2.clicked.connect(self.clickButton2_2)

        # Part 3
        self.ui.button3_1.clicked.connect(self.clickButton3_1)
        self.ui.button3_2.clicked.connect(self.clickButton3_2)

        # Part 4
        self.ui.button4_1.clicked.connect(self.clickButton4_1)
        self.ui.button4_2.clicked.connect(self.clickButton4_2)
        self.ui.button4_3.clicked.connect(self.clickButton4_3)
        self.ui.button4_4.clicked.connect(self.clickButton4_4)

        # Part 5
        self.ui.button5_1.clicked.connect(self.clickButton5_1)
        self.ui.button5_2.clicked.connect(self.clickButton5_2)
        self.ui.button5_3.clicked.connect(self.clickButton5_3)
        self.ui.button5_4.clicked.connect(self.clickButton5_4)
        self.ui.button5_5.clicked.connect(self.clickButton5_5)


    # Part 1
    def clickButton1_1(self):
        img = cv.imread("../image/dog.bmp",cv.IMREAD_COLOR)
        cv.imshow('1.1', img)
    def clickButton1_2(self):
        img1 = cv.imread("../image/color.png",cv.IMREAD_COLOR)
        img2 = cv.imread("../image/color.png",cv.IMREAD_COLOR)
        for x in range (0,img1.shape[0],1):
            for y in range(0,img1.shape[1],1):
                img2[x, y][0] = img1[x, y][1]
                img2[x, y][1] = img1[x, y][2]
                img2[x, y][2] = img1[x, y][0]
        cv.imshow('Original Image', img1)
        cv.imshow('Conversion Image', img2)
    def clickButton1_3(self):
        img1 = cv.imread("../image/dog.bmp",cv.IMREAD_COLOR)
        img2 = cv.imread("../image/dog.bmp",cv.IMREAD_COLOR)
        img2 = cv.flip(img1, 1)
        cv.imshow('Original Image', img1)
        cv.imshow('Flipping Image', img2)
    def clickButton1_4(self):
        cv.namedWindow('Blending Image')
        trackbar_name = 'Blend: '
        cv.createTrackbar(trackbar_name, 'Blending Image' , 0, 100, on_trackbar)
        on_trackbar(0)

    # part 2
    def clickButton2_1(self):
        img1 = cv.imread("../image/QR.png",cv.IMREAD_GRAYSCALE)
        img2 = cv.imread("../image/QR.png",cv.IMREAD_GRAYSCALE)
        ret, img2 = cv.threshold(img1,80,255,cv.THRESH_BINARY)
        cv.imshow('Original Image', img1)
        cv.imshow('Thresholded Image', img2)
    def clickButton2_2(self):
        img1 = cv.imread("../image/QR.png",cv.IMREAD_GRAYSCALE)
        img2 = cv.imread("../image/QR.png",cv.IMREAD_GRAYSCALE)
        
        img2 = cv.adaptiveThreshold(img1,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,19,-1)
        cv.imshow('Original Image', img1)
        cv.imshow('Thresholded Image', img2)

    # part 3
    def clickButton3_1(self):
        img1 = cv.imread("../image/OriginalTransform.png",cv.IMREAD_COLOR)
        img2 = cv.imread("../image/OriginalTransform.png",cv.IMREAD_COLOR)

        Angle = self.ui.textEdit3_1.toPlainText()
        Scale = self.ui.textEdit3_2.toPlainText()
        Tx = self.ui.textEdit3_3.toPlainText()
        Ty = self.ui.textEdit3_4.toPlainText()
        if Angle == "":
            Angle = 0
        else:
            Angle = (int)(Angle)
        if Scale == "":
            Scale = 0
        else:
            Scale = (float)(Scale)
        if Tx == "":
            Tx = 0
        else:
            Tx = (int)(Tx)
        if Ty == "":
            Ty = 0
        else:
            Ty = (int)(Ty)
        Rot = cv.getRotationMatrix2D((130, 125), Angle, Scale)
        img2 = cv.warpAffine(img1, Rot, (img1.shape[1], img1.shape[0]))
        Mov = np.float32([[1, 0, Tx], [0, 1, Ty]])
        img2 = cv.warpAffine(img2, Mov, (img1.shape[1], img1.shape[0]))
        cv.imshow('Original Image', img1)
        cv.imshow('Rotation + Scale + Translation Image', img2)
    def clickButton3_2(self):
        img1 = cv.imread("../image/OriginalPerspective.png",cv.IMREAD_COLOR)
        clickCounter = 0
        cv.imshow('Original Image', img1)
        cv.setMouseCallback('Original Image',mouse_click)

    # Part 4
    def clickButton4_1(self):
        img = cv.imread("../image/School.jpg",cv.IMREAD_COLOR)
        # convert to Gray Scale
        imgGray = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (0,img.shape[0],1):
            for y in range(0,img.shape[1],1):
                imgGray[x, y][0] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][1] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][2] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
        # Gaussian Filter
        imgGaussian = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (1,(imgGray.shape[0])-1,1):
            for y in range(1,(imgGray.shape[1])-1,1):
                smooth = 0
                smooth = smooth + 0.0947416*imgGray[x-1, y-1][0] + 0.118318*imgGray[x-1, y][0] + 0.0947416*imgGray[x-1, y+1][0]
                smooth = smooth + 0.118318*imgGray[x, y-1][0] + 0.147761*imgGray[x, y][0] + 0.118318*imgGray[x, y+1][0]
                smooth = smooth + 0.0947416*imgGray[x+1, y-1][0] + 0.118318*imgGray[x+1, y][0] + 0.0947416*imgGray[x+1, y+1][0]
                imgGaussian[x, y][0] = smooth
                imgGaussian[x, y][1] = smooth
                imgGaussian[x, y][2] = smooth
        cv.imshow('Original Image', img)
        cv.imshow('Gaussian Smooth Image', imgGaussian)
    def clickButton4_2(self):
        img = cv.imread("../image/School.jpg",cv.IMREAD_COLOR)
        # convert to Gray Scale
        imgGray = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (0,img.shape[0],1):
            for y in range(0,img.shape[1],1):
                imgGray[x, y][0] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][1] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][2] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
        # Gaussian Filter
        imgGaussian = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (1,(imgGray.shape[0])-1,1):
            for y in range(1,(imgGray.shape[1])-1,1):
                smooth = 0
                smooth = smooth + 0.0947416*imgGray[x-1, y-1][0] + 0.118318*imgGray[x-1, y][0] + 0.0947416*imgGray[x-1, y+1][0]
                smooth = smooth + 0.118318*imgGray[x, y-1][0] + 0.147761*imgGray[x, y][0] + 0.118318*imgGray[x, y+1][0]
                smooth = smooth + 0.0947416*imgGray[x+1, y-1][0] + 0.118318*imgGray[x+1, y][0] + 0.0947416*imgGray[x+1, y+1][0]
                imgGaussian[x, y][0] = smooth
                imgGaussian[x, y][1] = smooth
                imgGaussian[x, y][2] = smooth
        # SobelX
        imgSobelX = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (1,(imgGaussian.shape[0])-1,1):
            for y in range(1,(imgGaussian.shape[1])-1,1):
                Gx = 0
                Gx = Gx - 1*imgGaussian[x-1, y-1][0] + 0*imgGaussian[x-1, y][0] + 1*imgGaussian[x-1, y+1][0]
                Gx = Gx - 2*imgGaussian[x, y-1][0] + 0*imgGaussian[x, y][0] + 2*imgGaussian[x, y+1][0]
                Gx = Gx - 1*imgGaussian[x+1, y-1][0] + 0*imgGaussian[x+1, y][0] + 1*imgGaussian[x+1, y+1][0]
                Gx = math.sqrt(Gx*Gx)
                if Gx > 250:
                    Gx = 0
                imgSobelX[x, y][0] = Gx
                imgSobelX[x, y][1] = Gx
                imgSobelX[x, y][2] = Gx
        cv.imshow('Original Image', img)
        cv.imshow('SobelX Image', imgSobelX)
    def clickButton4_3(self):
        img = cv.imread("../image/School.jpg",cv.IMREAD_COLOR)
        # convert to Gray Scale
        imgGray = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (0,img.shape[0],1):
            for y in range(0,img.shape[1],1):
                imgGray[x, y][0] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][1] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][2] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
        # Gaussian Filter
        imgGaussian = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (1,(imgGray.shape[0])-1,1):
            for y in range(1,(imgGray.shape[1])-1,1):
                smooth = 0
                smooth = smooth + 0.0947416*imgGray[x-1, y-1][0] + 0.118318*imgGray[x-1, y][0] + 0.0947416*imgGray[x-1, y+1][0]
                smooth = smooth + 0.118318*imgGray[x, y-1][0] + 0.147761*imgGray[x, y][0] + 0.118318*imgGray[x, y+1][0]
                smooth = smooth + 0.0947416*imgGray[x+1, y-1][0] + 0.118318*imgGray[x+1, y][0] + 0.0947416*imgGray[x+1, y+1][0]
                imgGaussian[x, y][0] = smooth
                imgGaussian[x, y][1] = smooth
                imgGaussian[x, y][2] = smooth
        # SobelY
        imgSobelY = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (1,(imgGaussian.shape[0])-1,1):
            for y in range(1,(imgGaussian.shape[1])-1,1):
                Gy = 0
                Gy = Gy + 1*imgGaussian[x-1, y-1][0] + 2*imgGaussian[x-1, y][0] + 1*imgGaussian[x-1, y+1][0]
                Gy = Gy + 0*imgGaussian[x, y-1][0] + 0*imgGaussian[x, y][0] + 0*imgGaussian[x, y+1][0]
                Gy = Gy - 1*imgGaussian[x+1, y-1][0] - 2*imgGaussian[x+1, y][0] - 1*imgGaussian[x+1, y+1][0]
                Gy = math.sqrt(Gy*Gy)
                if Gy > 250:
                    Gy = 0
                imgSobelY[x, y][0] = Gy
                imgSobelY[x, y][1] = Gy
                imgSobelY[x, y][2] = Gy
        cv.imshow('Original Image', img)
        cv.imshow('SobelY Image', imgSobelY)
    def clickButton4_4(self):
        img = cv.imread("../image/School.jpg",cv.IMREAD_COLOR)
        # convert to Gray Scale
        imgGray = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (0,img.shape[0],1):
            for y in range(0,img.shape[1],1):
                imgGray[x, y][0] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][1] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
                imgGray[x, y][2] = 0.299*img[x, y][0] + 0.587*img[x, y][1] + 0.114*img[x, y][2]
        # Gaussian Filter
        imgGaussian = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (1,(imgGray.shape[0])-1,1):
            for y in range(1,(imgGray.shape[1])-1,1):
                smooth = 0
                smooth = smooth + 0.0947416*imgGray[x-1, y-1][0] + 0.118318*imgGray[x-1, y][0] + 0.0947416*imgGray[x-1, y+1][0]
                smooth = smooth + 0.118318*imgGray[x, y-1][0] + 0.147761*imgGray[x, y][0] + 0.118318*imgGray[x, y+1][0]
                smooth = smooth + 0.0947416*imgGray[x+1, y-1][0] + 0.118318*imgGray[x+1, y][0] + 0.0947416*imgGray[x+1, y+1][0]
                imgGaussian[x, y][0] = smooth
                imgGaussian[x, y][1] = smooth
                imgGaussian[x, y][2] = smooth
        # Sobel
        imgSobel = np.zeros(shape=[img.shape[0], img.shape[1], 3], dtype=np.uint8)
        for x in range (1,(imgGaussian.shape[0])-1,1):
            for y in range(1,(imgGaussian.shape[1])-1,1):
                Gx = 0
                Gx = Gx - 1*imgGaussian[x-1, y-1][0] + 0*imgGaussian[x-1, y][0] + 1*imgGaussian[x-1, y+1][0]
                Gx = Gx - 2*imgGaussian[x, y-1][0] + 0*imgGaussian[x, y][0] + 2*imgGaussian[x, y+1][0]
                Gx = Gx - 1*imgGaussian[x+1, y-1][0] + 0*imgGaussian[x+1, y][0] + 1*imgGaussian[x+1, y+1][0]
                Gy = 0
                Gy = Gy + 1*imgGaussian[x-1, y-1][0] + 2*imgGaussian[x-1, y][0] + 1*imgGaussian[x-1, y+1][0]
                Gy = Gy + 0*imgGaussian[x, y-1][0] + 0*imgGaussian[x, y][0] + 0*imgGaussian[x, y+1][0]
                Gy = Gy - 1*imgGaussian[x+1, y-1][0] - 2*imgGaussian[x+1, y][0] - 1*imgGaussian[x+1, y+1][0]
                G = math.sqrt(abs(Gx)*abs(Gy))
                if G > 250:
                    G = 0
                imgSobel[x, y][0] = G
                imgSobel[x, y][1] = G
                imgSobel[x, y][2] = G
        cv.imshow('Original Image', img)
        cv.imshow('Sobel Image', imgSobel)

    # Part 5
    def clickButton5_1(self):
        train_data, train_loader, test_data, test_loader, val_loader = downloadMNIST()
        
        fig = plt.figure(figsize=(14, 14), num='5.1 10 Images and Labels of MNIST')
        columns = 5
        rows = 2
        for i in range(1, columns*rows +1):
            randomNum = random.randint(0,60000)
            train_image, train_image_label = train_data[randomNum]
            train_image = np.array(train_image, dtype='float')
            pixels = train_image.reshape((32, 32))
            fig.add_subplot(rows, columns, i)
            plt.xlabel(train_image_label)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(pixels, cmap='gray')
        fig.show()
        
    def clickButton5_2(self):
        print("hyperparameters:")
        print("batch size:", BATCH_SIZE)
        print("learning rate:", LEARNING_RATE)
        print("optimizer:", OPTIMIZER)
    def clickButton5_3(self):
        train_data, train_loader, test_data, test_loader, val_loader = downloadMNIST()
        global LOSS
        global FIG_X
        global flag
        flag = 0
        model_CNN_1 = CNN()
        device = "cpu"
        optimizer = torch.optim.SGD(params=model_CNN_1.parameters(),lr=LEARNING_RATE, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()
        trainer = Trainer(loss_function, optimizer, device)
        trainer.train_loop(model_CNN_1, train_loader, val_loader)
        trainer.test(model_CNN_1, test_loader)
        #plot
        plt.figure(num='5.3 Train 1 Epoch Loss')
        plt.plot(FIG_X[0:len(LOSS)], LOSS)
        plt.show()

    def clickButton5_4(self):
        img = cv.imread("./5_4.png",cv.IMREAD_COLOR)
        cv.imshow('5_4 Training Result', img)
    def clickButton5_5(self):
        Index = self.ui.textEdit5_5.toPlainText()
        if Index == "":
            Index = 0
        else:
            Index = (int)(Index)

        # Load Model
        model_CNN_50 = CNN()
        model_CNN_50.load_state_dict(torch.load("./model"))
        train_data, train_loader, test_data, test_loader, val_loader = downloadMNIST()
        # Plot Test Image
        fig = plt.figure(figsize=(14, 6), num='5.5')
        TEST_IMAGE, test_image_label = test_data[Index]
        test_image = np.array(TEST_IMAGE, dtype='float')
        pixels = test_image.reshape((32, 32))
        fig.add_subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(pixels, cmap='gray')
        fig.show()

        # Test Model
        model_CNN_50.eval()
        test = DataLoader(test_data[Index])
        test_x, test_y = test
        output = model_CNN_50(test_x)
        output = F.softmax(output, dim=1)
        output = output.tolist()[0]

        outputProbLabel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        fig.add_subplot(1, 2, 2)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.bar(outputProbLabel, output)

def main_start():
    app = QApplication(sys.argv)
    window = MyDlg()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main_start()