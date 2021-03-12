# ------ IMPORT ALL NECESSARY LIBRARIES ------
import csv
import time
import datetime
# ------ IMPORT SECTION ENDS ------



# ------ OPEN THE DATASET AND CREATE A CLEANED COPY ------
with open('dataset/CLEANED_DATASET.csv','r') as source:
  rdr = csv.reader(source)
  with open('dataset/ARIMA_CLEANED_DATASET.csv','w') as result:
      next(source)
      wtr = csv.writer(result)
      for r in rdr:
        wtr.writerow((r[0],r[4]))
# ------ CLEANING SECTION ENDS ------



# ------ TELL THE USER THAT THE FILE HAS BEEN CREATED ------
ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print('CREATED CLEANED DATASET FILE FOR ARIMA! at '+timestamp)
# ------ TELLING USER SECTION ENDS ------     