## recontech r&d project

import numpy as np
import pandas as pd
import time
from pyod.models.knn import KNN
from joblib import dump, load

#####
####This is my proof of concept for an lightweight, ML model for detecting anomalies
#### 
#### I prefer Golang over Python so this code is not perfect
#####


def ord_encoder(string):
    return [ord(ch) for ch in string]

### Training

## reading in the training data CSV 
trainingData = pd.read_csv('data.csv')
trainingArray = np.zeros((40,3,40)) #first 40 is number of events, last 40 is string length)

## encoding each column
for index, row in trainingData.iterrows():
    tempName = ord_encoder(str(row['name']))
    for name_index, tName in enumerate(tempName):
        trainingArray[index][0][name_index] = tName

    tempID = ord_encoder(str(row['id']))
    for id_index, tID in enumerate(tempID):
        trainingArray[index][1][id_index] = tID
    
    tempPath = ord_encoder(str(row['path']))
    for p_index, tPath in enumerate(tempPath):
        trainingArray[index][2][p_index] = tPath

## flatten to 2d array
flatTrainArray = trainingArray.reshape(40, 3*40)


### Validation
valData = pd.read_csv('validation.csv')
valArr = np.zeros((1,3,40))
for vIndex, vRow in valData.iterrows():
    tempVName = ord_encoder(str(vRow['name']))
    for vname_index, vName in enumerate(tempVName):
        valArr[vIndex][0][vname_index] = vName

    tempVID = ord_encoder(str(vRow['id']))
    for vid_index, vid in enumerate(tempVID):
        valArr[vIndex][1][vid_index] = vid
    
    vtempPath = ord_encoder(str(vRow['path']))
    for vp_index, vpath in enumerate(vtempPath):
        valArr[vIndex][2][vp_index] = vpath

#again flatten to 2d
flatVArr = valArr.reshape(1, 3*40)

model = KNN(n_neighbors=1)
model.fit(flatTrainArray)

#dump(model, 'anomaly.model') #uncomment for model size

start_ts = time.time()

predict = model.decision_function(flatVArr)

end_ts = time.time()

if predict > 0:
    print("Anomaly")
else:
    print("Normal")

print(f"Prediction Time [s]:{(end_ts-start_ts):.6f}")