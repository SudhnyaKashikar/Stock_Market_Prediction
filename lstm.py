import numpy as np
import pandas as pd
import csv
import json
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import datetime   
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout

def runlstm(fileName,layer1Neurons,layer2Neurons,activation_,innerActivation_,loss_,optimizer_,epochs_,batchSize_,toTrain):


    #files = ["AXISBANK.NS.csv","BAJFINANCE.NS.csv","BPCL.NS.csv","HDFCBANK.NS.csv","HINDUNILVR.NS.csv","ICICIBANK.NS.csv"]
    files=(fileName)

    data_csv = pd.read_csv (files)

    total_data=len(data_csv)

    #how many data we will use 
    # (should not be more than dataset length )
    data_to_use = total_data
        
    # number of training data
    # should be less than data_to_use
    train_end = round(0.8*total_data)

        
    #most recent data is in the end 
    #so need offset
    start=total_data - data_to_use
        
        
    #currently doing prediction only for 1 step ahead
    steps_to_predict =1
        
        
    yt = data_csv.iloc [start:total_data ,4]    #Close price
    yt1 = data_csv.iloc [start:total_data ,1]   #Open
    yt2 = data_csv.iloc [start:total_data ,2]   #High
    yt3 = data_csv.iloc [start:total_data ,3]   #Low
    #vt = data_csv.iloc [start:total_data ,6]    # volume
    #ma2 = data_csv.iloc [start:total_data, 7]
    #ma5 = data_csv.iloc [start:total_data, 8]
    #ma10 = data_csv.iloc [start:total_data, 9]
    #bub = data_csv.iloc [start:total_data, 10]
    #bul = data_csv.iloc [start:total_data, 11]

    #yt = np.log(yt)
    #yt1 = np.log(yt1)
    #yt2 = np.log(yt2)
    #yt3 = np.log(yt3)
    #vt = np.log(vt)
    #ma2 = np.log(ma2)
    #ma5 = np.log(ma5)
    #ma10 = np.log(ma10)
    #bub = np.log(bub)
    #bul = np.log(bul)

        
    print ("yt head :")
    print (yt.head())

    #yt_ = yt.shift (-1)
    #yt_ = yt1.shift(-1)
    #yt_ = yt2.shift(-1)
    yt_ = yt3.shift(-1)
            
    data = pd.concat ([yt, yt_, yt1, yt2, yt3],axis=1)
    data. columns = ['yt', 'yt_', 'yt1', 'yt2', 'yt3']
        
    data = data.dropna()
            
    print (data)
            
    # target variable - closed price
    # after shifting
    y = data ['yt_']
        
            
    #       closed,   open,  high,   low    
    cols =['yt',  'yt1', 'yt2', 'yt3']
    x = data [cols]
        
        
        
    scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
    x = np. array (x).reshape ((len( x) ,len(cols)))
    x = scaler_x.fit_transform (x)


    scaler_y = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
    y = np.array (y).reshape ((len( y), 1))
    y = scaler_y.fit_transform (y)
        
            
    x_train = x [0: train_end,]
    x_test = x[ train_end +1:len(x),]
    y_train = y [0: train_end] 
    y_test = y[ train_end +1:len(y)]  
    x_train = x_train.reshape (x_train. shape + (1,)) 
    x_test = x_test.reshape (x_test. shape + (1,))

            

# ------ LSTM ALGORITHM STARTS HERE ------

    seed =2016
    np.random.seed (seed)
    if(toTrain == True):
        fit1 = Sequential ()
        fit1.add (LSTM (  10 , activation = 'tanh', inner_activation = 'hard_sigmoid', return_sequences = True, input_shape =(len(cols), 1) ))
        fit1.add (Dropout(0.2))
        #fit1.add (LSTM (  500 , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
        #fit1.add (Dropout(0.2))
        fit1.add (LSTM ( 30 , activation = 'tanh', inner_activation = 'hard_sigmoid'))
        fit1.add (Dropout(0.2))
        #fit1.add (LSTM ( 10 , activation = None, inner_activation = 'hard_sigmoid'))
        #fit1.add (Dropout(0.2))
        fit1.add (Dense (output_dim =1, activation = 'linear'))
            
        fit1.compile (loss = loss_ , optimizer = optimizer_)

        #fit1.save('original.h5')
        
        fit1.fit (x_train, y_train, batch_size = batchSize_, nb_epoch = epochs_, shuffle = False)
            
        print (fit1.summary())
                
        score_train = fit1.evaluate (x_train, y_train, batch_size =1)
        score_test = fit1.evaluate (x_test, y_test, batch_size =1)
        print (" in train MSE = ", round( score_train ,4)) 
        print (" in test MSE = ", score_test )
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        fit1.save('HOME_LOW.h5')
        print("SAVED A MODEL FILE WITH THE NAME lstm_model_"+timestamp+".h5")
    else:
        fit1=load_model('HOME_CLOSE.h5')
        print("MODEL LOADED FROM SAVED .h5 FILE.")
    count=0
    myCsvRow = [1]
    models=['HOME_OPEN.h5','HOME_HIGH.h5','HOME_LOW.h5','HOME_CLOSE.h5']
    while(count<4):
        fit1=load_model(models[count])
        
        pred1 = fit1.predict (x_test)

        print(count)
        pred1 = scaler_y.inverse_transform (np. array (pred1). reshape ((len( pred1), 1)))
            
            
        count=count+1

        prediction_data = pred1[-1][0]
        myCsvRow.append(prediction_data)

    print(myCsvRow)
    with open(r'dataset/CLEANED_DATASET.csv','a') as gf:
        writer=csv.writer(gf)
        writer.writerow(myCsvRow)

    print(pred1)




    '''


    fit2 = Sequential()
    fit2.add(LSTM (10, activation = 'tanh', inner_activation = 'hard_sigmoid', return_sequences = True, input_shape = (len(cols),1)))
    fit2.add(Dropout(0.2))
    fit2.add(LSTM (30, activation = 'tanh', inner_activation = 'hard_sigmoid'))
    fit2.add(Dropout(0.2))
    fit2.add(Dense (output_dim = 1, activation = 'linear'))

    fit2.compile (loss = "mean_squared_error", optimizer = "adam")
    fit2.fit(x_train, y_train, batch_size = 64, nb_epoch = 50, shuffle = False)

    print(fit2.summary())

    pred2 = fit2.predict(x_test)


    pred2 = scaler_y.inverse_transform (np. array (pred1). reshape ((len( pred1), 1)))


    '''



    fit1.summary()
    print ("Inputs: {}".format(fit1.input_shape))
    print ("Outputs: {}".format(fit1.output_shape))
    print ("Actual input: {}".format(x_test.shape))
    print ("Actual output: {}".format(y_test.shape))


    print ("prediction data:")
    print (prediction_data)


    print ("actual data")
    x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test), len(cols))))
    #print (x_test)      //only print to fix the errors


    #plt.plot(pred1, label="predictions")

    #plt.plot(pred2, label="predictions2")

        
    y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))
    #plt.plot( [row[0] for row in y_test], label="actual")

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)

    import matplotlib.ticker as mtick
    fmt = '₹%.0f'
    tick = mtick.FormatStrFormatter(fmt)

    ax = plt.axes()
    ax.yaxis.set_major_formatter(tick)


    #plt.show()
    '''
    toStorePrediction = prediction_data[0]
    toStoreActual = data_csv.iloc[total_data-1,4]
    differenceInPercentage = ((toStoreActual - toStorePrediction)/toStoreActual)*100
    toStoreDifference = 0
    if(differenceInPercentage > 0):
        toStoreDifference = differenceInPercentage
    row = [fname, toStorePrediction, toStoreActual, toStoreDifference]

        #reader = csv.reader(resultFile)
    if(flag == 0):
        #next(reader,None)
        flag = 1
    #writer.writerow(row)

    '''

    return (pred1)
        
    '''
    resultFile.close()

    resultFile2 = pd.read_csv("result.csv")
    totalLengthOfResultFile = len(resultFile2)
    startOfFile = 0;

    ab = resultFile2.iloc[startOfFile-1:totalLengthOfResultFile,3]
    cd = []
    totalCount = 0;

    for j in range (totalLengthOfResultFile):
        totalCount=totalCount+ab[j]

    for k in range (totalLengthOfResultFile):
        cd.append((ab[k]/totalCount)*100)

    print(cd)
    '''
