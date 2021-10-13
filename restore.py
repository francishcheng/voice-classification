
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import json
import torch
import librosa
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from utils import NeuralNetwork, get_dataloader, test, train, extract_features, model_path, map_path
import pickle
model = torch.load(model_path)
with open(map_path, 'rb') as f:
    lb_name_mapping = pickle.load(f)
print(lb_name_mapping)

files = os.listdir('./test/')
base_path = os.path.join(os.path.abspath('.'), 'test')
features_  = []
labels_ = []

for file in files:
    if file.endswith('.flac') or file.endswith('.ogg'):

        file_path = os.path.join(base_path, file)
        mfccs, chroma, mel, contrast, tonnetz = extract_features(file_path)
        features = np.concatenate([mfccs, chroma, mel, contrast, tonnetz, ])
        print(len(features))
        features_.append(features)
df = pd.DataFrame(np.concatenate([features_]), dtype=np.float32)
res = []
features = torch.tensor(df.values, dtype=torch.float32)
for feature in features:
    pred = model(feature)
    label = pred.argmax().item()
    res.append(lb_name_mapping[label])
dt = dict(zip(files, res))
for key in dt.keys():
    print(key, dt[key])