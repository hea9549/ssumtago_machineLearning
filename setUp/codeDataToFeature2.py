import numpy as np
import pandas as pd
import pymongo
from setUp import ssumPreprocessor

client = pymongo.MongoClient("mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago")

db_ssumtago = client['ssumtago']
surveys = db_ssumtago['surveys']
ssum_predect_survey = surveys.find_one({"id": 1})
questions = ssum_predect_survey['questions']
# pprint(questions)
client.close()

ssum_preprocessor = ssumPreprocessor.SsumPreprocessor(ssum_predect_survey)
# print(ssum_preprocessor.convert('01000112006', '02006037'))


dataset = pd.read_csv("./surveyDataCode.csv", dtype=object)

Label = '01000120001'

col_name_list = list(dataset)
all_processed_data = []
for v in col_name_list:
    cols = dataset[v]
    process_data = []

    if v == Label:
        continue

    for data in cols:
        process_data = np.append(process_data, np.array(ssum_preprocessor.convert(v, data)))

    process_data = process_data.reshape(len(cols), -1)

    if len(all_processed_data) == 0:
        all_processed_data = process_data
    else:
        all_processed_data = np.append(all_processed_data, process_data, axis=1)

cols=dataset[Label]

np_cols = np.zeros(len(cols))
for i in range(0, len(cols)):
    if cols[i] == '02001001':
        np_cols[i] = 1
    if cols[i] == '02001002':
        np_cols[i] = 0

all_processed_data=np.append(all_processed_data,np_cols.reshape(-1,1),axis=1)
print(all_processed_data.shape)