from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 
def parser(x):
    return datetime.strptime(x, '%d-%m-%Y')
 
def runArima(fileName,numberOfDays):
    series = read_csv(fileName, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    print(series)
    X = series.values
    size = round(len(X) * 0.8)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    print('Running ARIMA Script. Please wait...')
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        
    error = mean_squared_error(test, predictions)
    print('ARIMA TEST MSE: %.3f' % error)
    severalDays=model_fit.forecast(steps=numberOfDays)[0]
    day = 1
    for tHat in severalDays:
        print('Day %d: range - %f to %f' %(day, (tHat+(0.025*tHat)), (tHat-(0.025*tHat))))
        day += 1
    # plot
    #pyplot.plot(test)
    #pyplot.plot(predictions, color='red')
    #for i in range(0,len(x_test)-2):
    predictions.pop()
    predictions.pop()
    return (predictions)
