from flask import Flask, request
import time, os, glob
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import torch
import librosa
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from .utils import NeuralNetwork, get_dataloader, test, train, extract_features, model_path, map_path
import json
app = Flask(__name__, static_folder='static')
CORS(app)


@app.route('/save/', methods=['POST'])
def save_record():

    files = request.files
    files = files.getlist('file')
    form = request.form
    clip_names = form.getlist('clipnames')
    for clip_name, file in zip(clip_names, files):
        file.save('./clips/' + clip_name+'.ogg')
    return 'success'

@app.route('/train/', methods=['POST'])
def train_():
    form = request.form
    print(form)

    files = os.listdir('./clips/')
    base_path = os.path.join(os.path.abspath('.'), 'clips')
    features_  = []
    labels_ = []
    for file in files:
        if file.endswith('.flac') or file.endswith('.ogg'):
            person = file.split('-')[0]

            file_path = os.path.join(base_path, file)
            mfccs, chroma, mel, contrast, tonnetz = extract_features(file_path)
            features = np.concatenate([mfccs, chroma, mel, contrast, tonnetz])
            labels = str(person)
            features_.append(features)
            labels_.append(labels)
    df = pd.DataFrame(np.concatenate([features_]), dtype=np.float32)
    df['label'] = labels_

    col_names = df.columns
    feature_cols = col_names[:-1]
    print(feature_cols)
    label_cols =  'label' 
    label_set = df[label_cols].unique()
    label_num = len(df[label_cols].unique())
    feature_num = len(feature_cols)
    print(feature_num, 'feature')
    labels = df[label_cols]
    features = df[feature_cols]
    lb = LabelEncoder()
    lb.fit(labels)
    labels = lb.transform(labels)
    df[label_cols] = labels
    # lb_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
    lb_name_mapping = dict(zip(lb.transform(lb.classes_), lb.classes_ ))
    with open(map_path, 'wb') as f:
        pickle.dump(lb_name_mapping, f)
    model = NeuralNetwork(feature_num, len(label_set))


    train_loader, test_loader = get_dataloader(df, feature_cols, label_cols)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 300
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train(train_loader, model, loss_fn, optimizer)
        acc = test(test_loader, model, loss_fn)
        
    print("Done!")
    print('*'*10)
    print(acc)
    torch.save(model, model_path)
    print(lb_name_mapping)
    print('model saved!')
    return 'model saved'

@app.route('/calculate/', methods = ['POST'])
def calculate():

    # get files 
    files = request.files
    files = files.getlist('file')
    form = request.form
    clip_names = form.getlist('clipnames')
    file_names = []
    for clip_name, file in zip(clip_names, files):
        file_name = './test/' + clip_name+'.ogg'
        file_names.append(clip_name+'.ogg')
        file.save(file_name)
    print(file_names)

    model = torch.load(model_path)
    with open(map_path, 'rb') as f:
        lb_name_mapping = pickle.load(f)
    print(lb_name_mapping)

    files = file_names 
    # # base_path = os.path.join(os.path.abspath('.'), 'test')
    base_path = './test'
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
        print(pred)
        label = pred.argmax().item()
        print(label)
        res.append(lb_name_mapping[label])
    dt = dict(zip(files, res))
    for key in dt.keys():
        print(key, dt[key])

    return dt 

from flask import render_template
@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)


if __name__=="__main__":
    app.debug=True
    app.run()
    app.run(debug=True)