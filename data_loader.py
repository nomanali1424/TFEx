import torch 
from torch.utils.data import Dataset
import pandas as pd
from scipy.fft import fft, ifft
from scipy.signal import stft, istft
from utils.util import *
import random
import numpy as np 
import matplotlib.pyplot as plt 

random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_backgroundIdentification(f,t,original_spectrogram, original_signal, args):

    frequency_composition_abs = np.abs(original_spectrogram)
    measures = []
    for freq,freq_composition in zip(f,frequency_composition_abs):
        measures.append(np.mean(freq_composition)/np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    dummymatrix = np.zeros((len(f),len(t)))
    dummymatrix[selected_frequency,:] = 1  
    
    background_frequency = original_spectrogram * dummymatrix
    background_frequency = torch.tensor(background_frequency)
    _, xrec = istft(background_frequency,args.fs,nperseg=args.nperseg,noverlap=args.noverlap,boundary='zeros')
    xrec = xrec[:original_signal.shape[0]]
    xrec = xrec.reshape(original_signal.shape)
    return xrec, background_frequency

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, args):
        
        self.args = args
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].cpu().detach().numpy()

        #spectrogram
        f,t,spectrogram = stft(data, fs=self.args.fs, nperseg=self.args.nperseg, noverlap=self.args.noverlap, boundary='zeros')
        #obtain the rbp of spectrogram 
        xrec, background_spectrogram = sample_backgroundIdentification(f,t,spectrogram, self.data[idx], self.args) 
         
        return self.data[idx], self.labels[idx], spectrogram, background_spectrogram, xrec

def load_data(data_name, args):
    
    print(f"Dataset is {data_name}")

    if data_name=="cincecgtorso":
        # Read the file into a DataFrame using space as the separator
        file_path = 'data/CinCECGTorso/CinCECGTorso_TRAIN.txt'  
        file_path2 = 'data/CinCECGTorso/CinCECGTorso_TEST.txt'
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32)-1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)-1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor,data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0) 
    else:
        print("Unknown Data name")
    
    #return num_freq and num_slices
    f,t,spectrogram = stft(data, fs=args.fs, nperseg=args.nperseg, noverlap=args.noverlap, boundary='zeros')
    
    num_freq = f.shape[0]
    num_slices = t.shape[0]
    
    print(f"Data Shape: {data.shape}, labels shape: {labels.shape}")
    
    ds = TimeSeriesDataset(data, labels, args)
    
    return ds, num_freq, num_slices 