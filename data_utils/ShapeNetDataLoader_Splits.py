import numpy as np
import torch
import json
import os
from pytorch3d.io import load_obj, save_obj
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import time

def get_objectnames_from_split(split): #return list of files in a given .json split
    files=[]
    for DataSet in split:
        for class_name in split[DataSet]:
            for single_object in split[DataSet][class_name]:
                files=files + [single_object]
    return files

'''
function takes paths of split file, raw ShapeNet data and path to to already generated codes
return pointclouds of the data and the codes
'''
def  load_ShapeNet_pointclouds(data_path, split_path, codes_path,this_device):
    codes=torch.load(codes_path)
    #print(codes['latent_codes']['weight'].shape)
    split=json.load(open(split_path))
    object_list=get_objectnames_from_split(split) 
    object_paths=[]
    i=0
    for object_name in object_list:
        object_paths.append(os.path.join(data_path,object_name,"models/model_normalized.obj"))
    start1=time.perf_counter()
    print(start1)
    print("Start loading of objects.")
    input_meshes=load_objs_as_meshes(object_paths,device=this_device,load_textures=False)
    loading_time=time.perf_counter()-start1
    print("Loading of objects finished. Time necessary to load "+str(len(object_paths))+" objects is: "+str(loading_time)+" seconds.")
    print("Start sampling of points.")
    start2=time.perf_counter()
    number_samples=4096
    input_pointclouds= sample_points_from_meshes(input_meshes,number_samples)
    sampling_time=time.perf_counter()-start2
    print("Sampling of points finished. Time necessary to sample "+str(number_samples)+" points from "+str(len(object_paths))+" objects each is: "+str(sampling_time)+" seconds.")

    return input_pointclouds, codes['latent_codes']['weight']



