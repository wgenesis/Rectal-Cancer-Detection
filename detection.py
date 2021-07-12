from __future__ import division
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox, QFileDialog,QLabel
from gui.GUI import *

from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.Qt import QThread, QTextCursor
import traceback
import time
import pandas as pd
import cv2
from model.predict import *
try:
    from tensorflow.keras import models
except:
    pass
try:
    import torch
    from model.pytorch import *
    from torchvision import transforms
except:
    pass
import glob

class MyWindow(QMainWindow, Ui_MainWindow, QWidget):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)
        self.image=None
        self.colonModel=None
        self.batchImage=None
        self.batchResult=None
        self.frame=None
        self.imageFormat='*.jpg *.jpeg *.png *.jpe *.bmp *.pbm *.pgm *.ppm *.tiff *.tif'

    def initUI(self, ):
        self.setWindowTitle('Medical')
        self.setWindowIcon(QIcon('car2.ico'))
        self.colonSingleOpenImage.clicked.connect(self.openImage)
        self.colonLoadModel.clicked.connect(self.openModel)
        self.colonSingleBeginPredict.clicked.connect(self.singlePredict)
        self.colonBatchOpenDir.clicked.connect(self.openDir)
        self.colonBatchBeginPredict.clicked.connect(self.batchPredict)
        self.colonBatchClear.clicked.connect(self.colonBatchPredictClear)
        self.colonBatchSaveCSV.clicked.connect(self.outputCSV)
        self.show()

    def outputCSV(self):
        if self.batchResult is None:
            QMessageBox.warning(None, '错误', '请先生成识别结果！')
        else:
            try:
                saveFileName = QFileDialog.getSaveFileName(self, '导出CSV文件', 'ColonCancerPredict.csv')[0]
                MapDataFrame = pd.DataFrame(self.batchResult,columns=['SAMPLE','RESULT'])
                MapDataFrame.to_csv(saveFileName, sep=',')
                self.displayMessage('CSV文件导出成功')
            except:
                QMessageBox.critical(None, '错误', 'CSV导出失败！')
                self.displayMessage('CSV文件导出失败')

    def batchPredict(self):
        if self.colonModel is None:
            QMessageBox.warning(None, '错误', '请先载入预训练模型！')
        elif self.batchImage is None:
            QMessageBox.warning(None, '错误', '请先载入待识别图片！')
        else:
            self.displayMessage('开始识别')
            Time=time.time()
            result = batchPredict(self.batchImage, self.colonModel, 125, self.frame)
            self.displayMessage('识别完毕,耗时:'+str(time.time()-Time)+'s')
            self.batchResult=result
            for r in result:
                self.displayMessage(r[0]+'  ------->  '+r[1],type='batch')

    def colonBatchPredictClear(self):
        self.colonBatchPredict.clear()
        self.batchResult=None
        self.displayMessage('清除成功')

    def singlePredict(self):
        if self.colonModel is None:
            QMessageBox.warning(None, '错误', '请先载入预训练模型！')
        elif self.image is None:
            QMessageBox.warning(None, '错误', '请先载入待识别图片！')
        else:
            try:
                result=predict(self.image,self.colonModel,125,self.frame)
            except Exception as e:
                print('traceback.format_exc():\n%s' % traceback.format_exc())
            self.colonSinglePredict_text.setText(str(result))

    def displayImage(self):
        if self.image is not None:
            displayerQPixMap = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            mapQImage = QtGui.QImage(displayerQPixMap.data, displayerQPixMap.shape[1], displayerQPixMap.shape[0],displayerQPixMap.shape[1] * 3, QtGui.QImage.Format_RGB888)
            self.colonSingleImage.setPixmap(QtGui.QPixmap.fromImage(mapQImage))
            self.colonSingleImage.setScaledContents(True)

    def displayMessage(self,msg,type='main'):
        if type=='main':
            self.message.moveCursor(QTextCursor.End)
            self.message.append(msg)
        elif type=='batch':
            self.colonBatchPredict.moveCursor(QTextCursor.End)
            self.colonBatchPredict.append(msg)

    def openDir(self):
        dirFileName = QFileDialog.getExistingDirectory(self, '打开文件', '/')
        self.displayMessage('读取图片路径：' + dirFileName)
        imageFormat=self.imageFormat.split(' ')
        imagePath=[]
        try:
            for Format in imageFormat:
                try:
                    imagePath.extend(glob.glob(dirFileName+'/'+Format))
                except:
                    pass
            if len(imagePath)==0:
                QMessageBox.warning(None, '错误', '未找到图片！')
                self.displayMessage('图片载入失败')
            else:
                self.batchImage=imagePath
                self.displayMessage('成功载入'+str(len(imagePath))+'张图片')
        except:
            QMessageBox.critical(None, '错误', '打开失败！')
            self.displayMessage('图片载入失败')

    def openModel(self):
        modelFileName = QFileDialog.getOpenFileName(self, '打开文件', '/', '*.h5 *.pkl')
        self.displayMessage('打开预训练模型：' + modelFileName[0])
        try:
            if modelFileName[0][-2:]=='h5':
                self.colonModel=models.load_model(modelFileName[0])
                self.frame='tensorflow'
                self.displayMessage('成功载入模型')
            elif modelFileName[0][-3:]=='pkl':
                self.colonModel=Net()
                self.colonModel.load_state_dict(torch.load(modelFileName[0]))
                self.frame='torch'
                self.displayMessage('成功载入模型')

        except:
            QMessageBox.critical(None, '错误', '打开失败！')
            self.displayMessage('模型载入失败')

    def openImage(self):
        imageFileName = QFileDialog.getOpenFileName(self, '打开文件', '/', self.imageFormat)
        self.displayMessage('打开图片：' + imageFileName[0])
        try:
            image = cv2.imread(imageFileName[0])
            if image is not None:
                self.image = image
                self.displayMessage('成功载入图片')
                self.displayImage()
            else:
                QMessageBox.critical(None, '错误', '打开失败！')
                self.displayMessage('图片载入失败')
        except:
            QMessageBox.critical(None, '错误', '打开失败！')
            self.displayMessage('图片载入失败')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.initUI()
    app.exec_()
    sys.exit(0)