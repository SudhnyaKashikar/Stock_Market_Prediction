#import clean_data
import create_arima_file
import lstm
import arima
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import datetime

with open('config.json') as json_file:
    data = json.load(json_file)
    for p in data['lstmConfig']:
        fileName = p['fileNameLSTM']
        layer1Neurons = p['layer1Neurons']
        layer2Neurons = p['layer2Neurons']
        activation = p['activation']
        innerActivation = p['innerActivation']
        loss = p['loss']
        optimizer = p['optimizer']
        epochs = p['epochs']
        batchSize = p['batchSize']
        toTrain = p['toTrain']
    for p in data['arimaConfig']:
        arimaFileName = p['fileNameArima']
        numberOfDaysToPredict = p['numberOfDaysToPredict']
    for q in data['dataCleaningConfig']:
        originalFileName = q['fileName']
days=1
while(days<21):
    lstmPredictions = lstm.runlstm(fileName,layer1Neurons,layer2Neurons,activation,innerActivation,loss,optimizer,epochs,batchSize,toTrain)
    days=days+1
#arimaPredictions = arima.runArima(arimaFileName,numberOfDaysToPredict)

#finalPredictions = (lstmPredictions+arimaPredictions)/2

data_csv=pd.read_csv(fileName)
total_data=len(data_csv)
train_end = round(0.8*total_data)
start=0
yt = data_csv.iloc [start:total_data ,4]
yt_ = yt.shift (-1)
data = pd.concat ([yt_],axis=1)
data. columns = ['yt_']
data = data.dropna()
y = data ['yt_']
y = np.array (y).reshape ((len( y), 1))
y_test = yt[ train_end +1:len(y)] 

abcd = y_test.values.reshape(len(y_test),1)

plt.plot( [row[0] for row in abcd], label="actual")
print("NEXT: ",lstmPredictions[-1])
#print("NEXT DAY's CLOSING PRICE: ", finalPredictions[-1])
plt.title('Prediction vs Actual | of file name '+originalFileName)
#plt.plot(finalPredictions, color = 'purple', label="prediction (merged)")
plt.plot(lstmPredictions, color= 'orange', label="prediction of LSTM")
#plt.plot(arimaPredictions, color= 'red', label="prediction of ARIMA")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)


ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
plt.show()
       
#plt.savefig('images/'+timestamp+'.png')
print('CREATED A GRAPH OF PREDICTION AND ACTUAL AT LOCATION: images/'+timestamp+'.png')


print("FINISHED RUNNING THE SCRIPT!")





#TO ADD H5 IN JSON
#TO ADD DIVISION OF DATASET IN JSON
