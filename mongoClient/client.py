import pymongo
import json
import csv
import numpy as np
import pandas as pd

surveyData = pd.read_csv("./surveyData/survey.csv", dtype=object)

col_name_list = list(surveyData)

questions = []
for v in col_name_list:
    cols = surveyData[v]
    answers = []
    for i in range(8):
        if pd.isnull(cols[i]):
            pass
        else:
            answer = {
                "name": "",
                "desc": "",
                "code": cols[i]
            }
            answers.append(answer)

    question = {
        "code": v,
        "desc": "",
        "name": "",
        "answers": answers,
    }

    questions.append(question)

client = pymongo.MongoClient("mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago")

db_ssumtago = client['ssumtago']

surveys = db_ssumtago['surveys']

survey = {
    "name": "ssumPredict",
    "desc": "썸의 성공 확률을 예측하는 설문지",
    "id": 1,
    "version": "1.0.0",
    "questions": questions
}

insertedItem = surveys.insert(survey)
print(insertedItem)
