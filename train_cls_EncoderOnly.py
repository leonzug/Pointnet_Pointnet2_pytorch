"""
Author: Leon Zueger, ETH Zurich
Date: May 2020
"""
import os
import torch
import numpy as np
import json
from  data_utils.ShapeNetDataLoader_Splits import get_objectnames_from_split #serve importarlo anche qui?
from  data_utils.ShapeNetDataLoader_Splits import load_ShapeNet_pointclouds
#from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction
from models import pointnet2_cls_msg_encoderOnly
from tqdm import tqdm

def test(model,loss_function, dataLoader):
    loss=0
    for j_batch, data_batch in tqdm(enumerate(dataLoader), total=len(dataLoader), smoothing=0.9):
        pointclouds=data_batch[0] #TODO: add point droupout, e.g. 0.875 or perform other geometric operations, see train_cls.py
        codes=data_batch[1]
        #codes=codes[:,:,0] necessary only when using fake data
        codes=codes.contiguous()
        pointclouds, codes= pointclouds.cuda(), codes.cuda()
        pointclouds, codes= pointclouds.contiguous(), codes.contiguous()
        with torch.no_grad():
            encoder=model.eval()
            codes_predicted= encoder(pointclouds)
        codes_predicted=codes_predicted.contiguous()
        loss=loss+ loss_function(codes_predicted, codes)
    return loss
            

if __name__ == '__main__':

    #print(torch.cuda.is_available())
    #print(torch.cuda.current_device())
    #print(torch.cuda.get_device_name(0))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    device='cpu'
    print('Using device:', device)

    #if device.type == 'cuda':
    #    print(torch.cuda.get_device_name(0))
    #    print('Memory Usage:')
    #    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    #    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path=os.path.join(BASE_DIR, '..',"rawData/ShapeNetCore.v2/03636649" )
    codes_filename= os.path.join(BASE_DIR,"lampsOnly_latest.pth")
    split_filename=os.path.join(BASE_DIR,"sv2_lamps_train.json")
    #for now, take a part of training data as testing. TODO: import training and testing separately.

    inputs,labels= load_ShapeNet_pointclouds(data_path, split_filename, codes_filename,device)
    inputs=inputs.permute(0,2,1)
    #print(inputs)
    #########################this is for quicker code testing#################################
    #inputs=np.ones((50,3,1024))*0.999
    #inputs=torch.from_numpy(inputs).float()
    #print(inputs)
    #labels=2.000*np.ones((50,256,1))
    #labels=torch.from_numpy(labels).float()
    ##########################################################################################
    train_data = []
    for i in range(len(inputs)):
        train_data.append([inputs[i,:,:], labels[i,:]]) #this line for real data
        #train_data.append([inputs[i,:,:], labels[i,:,:]]) #this line for fake data for code testing

    divide_dataset=[int(len(train_data)*0.8),int(len(train_data)-int(len(train_data)*0.8))]


    training,testing=torch.utils.data.random_split(train_data,divide_dataset)

    trainingDataLoader=torch.utils.data.DataLoader(training,batch_size=2,shuffle=True,num_workers=4,drop_last=True)
    testingDataLoader=torch.utils.data.DataLoader(testing,batch_size=2,shuffle=True,num_workers=4,drop_last=True)

    encoder=pointnet2_cls_msg_encoderOnly.get_model(normal_channel=False).cuda() #TODO: incorporate normals for better result (possibly)
    loss_function=pointnet2_cls_msg_encoderOnly.get_loss().cuda()
    epochs= 100

    optimizer=torch.optim.Adam(encoder.parameters(),lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7) 
    print("Start training.")
    torch.backends.cudnn.enabled = False #slows down training!!! TODO: fix error
    best_testing_loss=1000000
    save_path=os.path.join(BASE_DIR,"best_encoder_parmeters.pth")
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        for i_batch, data_batch in tqdm(enumerate(trainingDataLoader),total=len(trainingDataLoader),smoothing=0.9):
            pointclouds=data_batch[0] #TODO: add point droupout, e.g. 0.875 or perform other geometric operations, see train_cls.py
            codes=data_batch[1]
            #print(pointclouds) TODO: sono giuste le dimensioni?
            #codes=codes[:,:,0] #remove this when using real data
            codes=codes.contiguous()
            pointclouds, codes= pointclouds.cuda(), codes.cuda()
            pointclouds, codes= pointclouds.contiguous(), codes.contiguous()
            optimizer.zero_grad() #set gradient to zero for new batch
            encoder=encoder.train()
            codes_predicted= encoder(pointclouds)
            codes_predicted=codes_predicted.contiguous()
            loss=loss_function(codes_predicted, codes)
            loss=loss.contiguous()
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()

        print("training finished")
        total_testing_loss=test(encoder, loss_function,testingDataLoader)
        if total_testing_loss < best_testing_loss:
            state = {
                'epoch': epoch, 
                'model_state_dict':encoder.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
            }
            torch.save(state,save_path)
            improvement_factor=best_testing_loss/total_testing_loss
            print("The model predicted better on the testing data by a factor of "+ str(improvement_factor)+ ".")
            print("Total valiation loss: " + str(total_testing_loss))
            print("Model saved to disk.")
            best_testing_loss=total_testing_loss
        else: 
            print("During this epoch the testing error increased.")




