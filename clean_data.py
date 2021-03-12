#------ IMPORT ALL NECESSARY LIBRARIES ------
import csv
import json
import time
import datetime
#------ IMPORT SECTION ENDS ------



#------ READ FROM config.json ------
with open('config.json') as json_file:
    data = json.load(json_file)
    for p in data['dataCleaningConfig']:
        fileName = p['fileName']
        cleanedFileName = (p['cleanedFileName'])
#------ READING DONE ------



#------ WRITE THE CLEANED FILE ------
with open(fileName, 'r') as inp, open(cleanedFileName, 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[2]!="null" and row[6]!="0":
            writer.writerow(row)
#------ CLEANED FILE WRITING DONE ------



#------ TELL THE USER THAT THE FILE IS CLEANED ------
ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("CLEANED! at "+timestamp)
#------ TELLING SECTION ENDS ------