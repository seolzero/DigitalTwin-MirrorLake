import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

import subprocess
#import schedule
import time
import datetime
import os

import datapreprocessing_lidar as DataPreprocessing


##model define
class Net(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    #self.relu = nn.ReLU()
    self.sigmoid=nn.Sigmoid()
    #self.tanh=nn.Tanh()
    #self.leakyRelu=nn.LeakyReLU() 
    self.fc2 = nn.Linear(hidden_size, hidden_size*2)
    self.fc3 = nn.Linear(hidden_size*2, hidden_size)

    self.fc_f = nn.Linear(hidden_size, num_classes)
  
  def forward(self,x):
    out = self.fc1(x)
    #out = self.relu(out)
    out=self.sigmoid(out)
    #out=self.tanh(out)
    #out=self.leakyRelu(out)
    out = self.fc2(out)
    #out = self.relu(out)
    out=self.sigmoid(out)
    #out=self.tanh(out)
    #out=self.leakyRelu(out)
    out = self.fc3(out)
    #out = self.relu(out)
    out=self.sigmoid(out)
    #out=self.tanh(out)
    #out=self.leakyRelu(out)

    out = self.fc_f(out)
    #out = self.relu(out)
    out=self.sigmoid(out)
    #out=self.tanh(out)
    #out=self.leakyRelu(out)
    return out



def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("create folder...", directory)
            #log(filename, 'debug', ["create folder...", directory])
    except OSError:
        print('Error: Creating directory. ' + directory)
        #log(filename, 'error', 'Error: Creating directory. ' + directory)







class brain:
    def __init__(self,conf,model_name=""):
        self.model_name=model_name
        self.loss_function = nn.CrossEntropyLoss()
        self.conf=conf

    def data_loading(self,train_file_name,sheet_name):
        print("-------------data_loading------------")

        df = pd.read_excel(train_file_name, sheet_name=sheet_name)
        print(df)
        #print(df.dtypes)
        return df

        
    def set_train_data(self,data):
        print("-------------set_train_data------------")

        drop_col_data=DataPreprocessing.drop_col(data)
        astype_data=DataPreprocessing.astype_data(drop_col_data)
        fill_nan_data=DataPreprocessing.fill_nan(astype_data)
        
        #normalize (range/intensities)
        nor_df=DataPreprocessing.normailze_train_data(fill_nan_data)

        #rename dataframe columns
        rename_df=DataPreprocessing.rename_df(nor_df[0], nor_df[1])
        
        #merge (df_range + df_intensities)
        df_all=DataPreprocessing.merge_df(astype_data,nor_df[0],nor_df[1])

        print('data preprocessing finish')
        return df_all


    def set_lidar_data(self,data):
        #pd.set_option('display.max_columns',None)

        # print("-------------set_train_data------------")

        drop_col_data=DataPreprocessing.drop_col(data)
        astype_data=DataPreprocessing.astype_data(drop_col_data)
        fill_nan_data=DataPreprocessing.fill_nan(astype_data)
        
        nor_df=DataPreprocessing.normailze_lidar_data(fill_nan_data)
        
        rename_df=DataPreprocessing.rename_df(nor_df[0], nor_df[1])
        
        df_all=DataPreprocessing.lidar_merge_df(astype_data,nor_df[0],nor_df[1])

        print(df_all)
        # print('data preprocessing finish')
        return df_all



    
    def split_train_test(self,df_all):
        #split train, test data set by random sampling
        x=df_all.iloc[:,1:901]
        y=df_all.iloc[:,-1] 
        x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,shuffle=True,random_state=1004)

        return x_train, y_train, x_test, y_test


    def train(self,x_train,y_train, model=None, test_per = 100, save_per = 100):
        conf=self.conf

        cursor =0
        total_used_data=0

        x_train_size=len(x_train.columns)

        total_batch=len(x_train)
        loss_list=[]

        #class Net model load
        model=Net(conf['input_size'], conf['hidden_size'], conf['num_classes'])

        #optimizer reset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])


        print("training start")
        for epoch in range(conf['num_epochs']):
            #sets the gredients of all optimizered to zero
            self.optimizer.zero_grad()

            #data reshape
            x_train_values=x_train.values #
            x_train_array=np.array(x_train_values)
            train_x=torch.Tensor(x_train_array[cursor:cursor+conf['batch_size']]).view(conf['batch_size'],x_train_size)

            y_train_dum=pd.get_dummies(y_train)
            y_train_values=y_train_dum.values
            y_train_array=np.array(y_train_values)
            train_y=torch.Tensor(y_train_array[cursor:cursor+conf['batch_size']]).view(conf['batch_size'],len(y_train_dum.columns))


            cursor=cursor+conf['batch_size']
            total_used_data += conf['batch_size']
            if cursor > total_batch-conf['batch_size']:
                cursor = 0
                print("cursor_reset")


            #forward
            prediction=model(train_x)
            loss = self.loss_function(prediction, train_y)

            #backward
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.data)


            #train course print
            if epoch % 10 == 0:
                print('Epoch [%d/%d], Step[%d/%d],Total_used_data [%d], Loss: %.4f' %(epoch+1, conf['num_epochs'], cursor,total_batch, total_used_data, loss.data))
            
            if epoch % save_per == 0 and epoch > 0:
                log_data_list = [loss.data] 
                print('log_data_list',log_data_list)
                self.model_save(model, epoch, conf, log_data_list, model_name=self.model_name)
                print('model save')

        print("finish Training!")
    


    def test(self, model, epoch, test_data_x, test_data_y, conf, model_name=""):
        #@title Evaluating the accuracy of the model
        cost=0
        accuracy_list = []
        test_N = 20

        #with torch.no_grad():
        for test_i in range(test_N):
            correct = 0
            total = 0

            #data reshape
            x_test_values=test_data_x.values
            x_test_array=np.array(x_test_values)
            test_x=torch.Tensor(x_test_array)
            print(test_x)

            y_test_values=test_data_y.values
            y_test_array=np.array(y_test_values)
            test_y=torch.Tensor(y_test_array)
            print(test_y)


            output = model(test_x)

            loss = self.loss_function(output, test_y)

            correct += (predicted ==test_y).sum()
            total += test_y.size(0)


        accuracy=100*correct/total
        accuracy_list.append(accuracy)

        print("Accuracy : {:.2f}".format(100*correct/total))
        return loss / test_N


    def inference(self, model, lidar_data, conf):
        lidar_data_x=lidar_data.iloc[:,1:901]

        x_test_values=lidar_data_x.values
        x_test_array=np.array(x_test_values)
        test_x=torch.Tensor(x_test_array)


        preds=model(test_x)
        #print("inference=",preds)

        # lidar_data_y=lidar_y.iloc[0,-1]
        # y_train_values=lidar_data_y.values
        # y_train_array=np.array(y_train_values)
        
        # print("data_y",y_train_array)
        

        return preds
        
        
        
    def model_save(self, model, epoch, conf, log_data_list, model_name=""):
        path='./model_save/'

        save_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")

        model_logs = model_name

        for conf_item in conf:
            model_logs = model_logs + '\t' + str(conf[conf_item])
        for log in log_data_list:
            model_logs = model_logs + '\t' + str(log)

        model_logs = model_logs + '\t' + str(epoch)
        model_logs = model_logs + '\t' + save_time
        model_logs = model_logs + '\n'

        with open(path+'model_logs.txt', 'a+') as f:
            f.write(model_logs)

        createFolder(path+model_name)

        torch.save({
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'conf': conf
        }, path + model_name+'/' + str(epoch), _use_new_zipfile_serialization=False)


    def model_load(self, model_name, epoch):

        check_point = torch.load('./model_save/' + model_name +'/' + str(epoch))
        if check_point == None:
            print("ERROR MSG: there is no model data", model_name,str(epoch))
        print("Load model:", model_name, epoch)

        conf = check_point['conf']
        model = Net(conf['input_size'], conf['hidden_size'], conf['num_classes'])
        model.load_state_dict(check_point['model_state_dict'])
        self.conf = conf
        self.model_name = model_name
        return model, conf















    


    


