import math
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Subcarrier selection for HALOC dataset
# 52 L-LTF subcarriers 
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_index += [i for i in range(6, 32)]
csi_vaid_subcarrier_index += [i for i in range(33, 59)]
# 56 HT-LTF: subcarriers 
#csi_vaid_subcarrier_index += [i for i in range(66, 94)]     
#csi_vaid_subcarrier_index += [i for i in range(95, 123)] 
CSI_SUBCARRIERS = len(csi_vaid_subcarrier_index) 

# HALOC dataset class
class HALOC(Dataset):
    def __init__(self, dataPath, windowSize):
        self.dataPath = dataPath
        self.windowSizeH = int(windowSize/2)

        # read data from .csv file
        data = pd.read_csv(dataPath)
        csi = data['data']
        self.x = data['x']
        self.y = data['y']
        self.z = data['z']

        # replace ".csv" with ".npy" in dataPath
        csiCachePath = dataPath.replace(".csv",".npy")

        # check if cached CSI data exists
        if os.path.exists(csiCachePath):
            csiComplex = np.load(csiCachePath)
        else:
            # extract CSI data from relevant columns
            csiComplex = np.zeros([len(csi), CSI_SUBCARRIERS], dtype=np.complex64)
            # select relevant columns and convert to complex numbers
            for s in tqdm(range(len(csi))):
                for i in range(CSI_SUBCARRIERS):
                    sample  = csi[s][1:-1].split(',')
                    sample = np.array([int(x) for x in sample])
                    csiComplex[s][i] = complex(sample[csi_vaid_subcarrier_index[i] * 2],sample[csi_vaid_subcarrier_index[i] * 2 - 1])
            np.save(csiCachePath,csiComplex)

        # extract feature from CSI
        self.features = np.abs(csiComplex) # amplitude
        #self.csiAmplitudes = np.angle(csiComplex) # phase

        # compute number of samples exleduing border regions
        self.dataSize = len(self.features)-windowSize

        # min-max scaling
        self.features = (self.features - np.min(self.features)) / (np.max(self.features) - np.min(self.features))

    def __len__(self):
        return self.dataSize

    def __getitem__(self, index):
        # add index offset to avoid index border regions
        index = index + self.windowSizeH+1 

        l = np.array([self.x[index], self.y[index], self.z[index]]) # get 3D location label
        featureWindow = self.features[index-self.windowSizeH:index+self.windowSizeH] # get feature window
        featureWindow = np.transpose(featureWindow, (1, 0)) # transpose to [self.windowSize,CSI_SUBCARRIERS]
        featureWindow = np.expand_dims(featureWindow, axis=0) # add channel dimension

        return featureWindow, l

