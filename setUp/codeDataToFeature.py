import pandas as pd
import pymongo
import numpy as np
from pprint import pprint
from mlxtend.preprocessing import one_hot
from sklearn.preprocessing import MinMaxScaler

client = pymongo.MongoClient("mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago")

db_ssumtago = client['ssumtago']
surveys = db_ssumtago['surveys']
ssum_predect_survey = surveys.find_one({"id": 1})
questions = ssum_predect_survey['questions']
# pprint(questions)
client.close()
#
# pprint(ssum_predect_survey)

dataset = pd.read_csv("./surveyDataCode.csv", dtype=object)

col_name_list = list(dataset)

#
# excludeList = [
#     "01000120001",
#     "zxc"
# ]

Label = '01000120001'
processed_data = []
num = 0
for v in col_name_list:
    cols = dataset[v]
    code = v[3:8]

    if v == Label:
        print("label")
        continue

    if code == '00120':
        # 객관식 카테코리 다중선택불가 그냥
        # hot encoding
        for question in questions:
            if question["code"] == v:
                answer_length = len(question["answers"])
                answers = question["answers"]
                dictMap = {}
                index = 0
                for answer in answers:
                    dictMap[answer["code"]] = index
                    index += 1

                data_array = list(map(str, cols))
                stringToInt = []
                for data in data_array:
                    stringToInt.append(dictMap[data])

                one_hot_result = one_hot(stringToInt, num_labels=answer_length)
                if len(processed_data) == 0:
                    processed_data = one_hot_result
                else:
                    processed_data = np.append(processed_data, one_hot_result, axis=1)

                num+=answer_length
                print("code:",v,"num:",answer_length)
        pass
    if code == '00112':
        # 객관식 카테고리 다중선택가능 sum알고리즘
        # 각각에 대하여 다른 알고리즘 적용
        for question in questions:
            if question["code"] == v:
                print("find", code, v)
                answer_length = len(question["answers"])
                stringToInt = []

                for answerCode in cols:
                    # print(int(int(answerCode[-2:])/20))

                    if int(int(answerCode[-2:])/1) == 1:
                        stringToInt.append(0)
                    elif int(int(answerCode[-2:])/2) == 1:
                        stringToInt.append(1)
                    elif int(int(answerCode[-2:])/4) == 1:
                        stringToInt.append(2)
                    elif int(int(answerCode[-2:])/10) == 1:
                        stringToInt.append(3)
                    elif int(int(answerCode[-2:])/20) == 1:
                        stringToInt.append(4)
                    else:
                        print("error")

                one_hot_result = one_hot(stringToInt, num_labels=answer_length)
                print(one_hot_result)
                if len(processed_data) == 0:
                    processed_data = one_hot_result
                else:
                    processed_data = np.append(processed_data, one_hot_result, axis=1)



        pass
    if code == '10000':
        # 주관식 값 그냥 그냥
        # normalization (min max scaler)
        for question in questions:
            if question["code"] == v:
                print("find", code, v)
                scaler = MinMaxScaler(feature_range=(0, 1))

                age_array = np.zeros(len(cols))
                print(age_array.shape)
                for i in range(len(cols)):
                    age_array[i] = cols[i][-2:]

                print(age_array)

                print(scaler.fit_transform(age_array.reshape(-1, 1)))
                processed_data = np.append(processed_data, scaler.fit_transform(age_array.reshape(-1, 1)), axis=1)
        pass

cols=dataset[Label]

np_cols = np.zeros(len(cols))
for i in range(0, len(cols)):
    if cols[i] == '02001001':
        np_cols[i] = 1
    if cols[i] == '02001002':
        np_cols[i] = 0

processed_data=np.append(processed_data,np_cols.reshape(-1,1),axis=1)

np.savetxt("./feature.csv",processed_data,delimiter=",")
