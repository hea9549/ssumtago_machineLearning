import json

from bson import json_util
from flask import Flask
from flask import request
from flask_pymongo import PyMongo
import numpy as np

from app.decorator import json_schema
from setUp import ssumPreprocessor
from app import ssumPredictService

app = Flask(__name__)
app.config['MONGO_DBNAME'] = 'ssumtago'
app.config['MONGO_URI'] = "mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago"
mongo = PyMongo(app)

with open('./app/schema.json') as f:
    config = json.load(f)
    app.config["schema"] = config


@app.route('/')
def hello_world():
    ssum_predict_survey = mongo.db.surveys.find_one({"id": 1})
    return json.dumps(ssum_predict_survey, indent=4, default=json_util.default).encode('utf8')


@app.route('/json', methods=['POST'])
@json_schema('requestPredict', app)
def request_predict():
    request_body = request.json
    return "hello world"


@app.route('/predicts', methods=['POST'])
@json_schema('requestPredict', app)
def predict():
    request_predict_body = request.json
    ssum_predect_survey = mongo.db.surveys.find_one({"id": request_predict_body["surveyId"]})
    ssum_preprocessor = ssumPreprocessor.SsumPreprocessor(ssum_predect_survey)

    data = request_predict_body["data"]
    all_processed_data = []

    for data_element in data:
        if data_element["questionCode"] in ssum_predect_survey["excludeCodes"]:
            print(data_element["questionCode"])
            continue

        process_data = ssum_preprocessor.convert(data_element["questionCode"], data_element["answerCode"])
        if len(all_processed_data) == 0:
            all_processed_data = process_data
        else:
            all_processed_data = np.append(all_processed_data, process_data)

    result = ssumPredictService.predict_human([all_processed_data])

    return str(result)


if __name__ == '__main__':
    app.debug = True
    app.run()
