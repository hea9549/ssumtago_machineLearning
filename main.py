from flask import Flask, session, redirect, url_for, escape, request, jsonify
from jsonschema import validate, ValidationError
from bson import BSON
from bson import json_util
import tensorflow as tf
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
import pymongo
import json
from functools import wraps
from setUp import ssumPreprocessor
import numpy as np

app = Flask(__name__)

client = pymongo.MongoClient("mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago")

schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "definitions": {},
    "id": "http://example.com/example.json",
    "properties": {
        "data": {
            "items": {
                "properties": {
                    "answerCode": {
                        "type": "string"
                    },
                    "questionCode": {
                        "type": "string"
                    }
                },
                "required": [
                    "answerCode",
                    "questionCode"
                ],
                "type": "object"
            },
            "type": "array"
        },
        "surveyId": {
            "type": "integer"
        },
        "version": {
            "type": "string"
        }
    },
    "required": [
        "surveyId",
        "version",
        "data"
    ],
    "type": "object"
}


def json_schema(schema_name):
    """
    지정한 API 에 대해서 지정한 schema_name로 검사한다.
    :param schema_name: 검사대상 스키마 이름
    :return: 에러나면 40000 에러
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            try:
                print(request.json)
                # request.on_json_loading_failed = on_json_loading_failed_return_dict
                validate(request.json, schema)
                return func(*args, **kw)
            except ValidationError as e:
                print(e)
                return "error"
                # logger.exception(traceback.format_exc())
                # return ResponseData(code=HttpStatusCode.INVALID_PARAMETER).json

        return wrapper

    return decorator


@app.route('/')
def hello_world():
    db_ssumtago = client['ssumtago']
    surveys = db_ssumtago['surveys']
    ssum_predect_survey = surveys.find_one({"id": 1})

    # json_val = json.dumps(ssum_predect_survey)
    # print(json_val)
    return json.dumps(ssum_predect_survey, indent=4, default=json_util.default).encode('utf8')


@app.route('/json', methods=['POST'])
@json_schema('schema')
def json_schemaTest():
    return "hello world"


@app.route('/predicts', methods=['POST'])
def predict():
    request_body = request.json
    validate(request_body, schema)

    db_ssumtago = client['ssumtago']
    surveys = db_ssumtago['surveys']
    ssum_predect_survey = surveys.find_one({"id": request_body["surveyId"]})

    ssum_preprocessor = ssumPreprocessor.SsumPreprocessor(ssum_predect_survey)

    data = ssum_predect_survey["data"]
    all_processed_data = []

    for element in data:
        process_data = ssum_preprocessor.convert(element["questionCode"], element["answerCode"])
        if len(all_processed_data) == 0:
            all_processed_data = process_data
        else:
            all_processed_data = np.append(all_processed_data, process_data, axis=1)

    print(all_processed_data[0].shape)

    result = sess.run(predicted, feed_dict={X: all_processed_data, keep_prob: 1})

    return str(result)


if __name__ == '__main__':
    x_num_of_feature = 82
    num_of_unit = 128
    keep_prob = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, shape=[None, x_num_of_feature])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W1 = tf.get_variable("W1", shape=[x_num_of_feature, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([num_of_unit]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1 = tf.nn.dropout(L1, keep_prob)

    W2 = tf.get_variable("W2", shape=[num_of_unit, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([num_of_unit]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(L2, keep_prob)

    W3 = tf.get_variable("W3", shape=[num_of_unit, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([num_of_unit]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(L3, keep_prob)

    W4 = tf.get_variable("W4", shape=[num_of_unit, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([num_of_unit]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob)

    W5 = tf.get_variable("W5", shape=[num_of_unit, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([num_of_unit]))
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
    L5 = tf.nn.dropout(L5, keep_prob)

    W6 = tf.get_variable("W6", shape=[num_of_unit, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([num_of_unit]))
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
    L6 = tf.nn.dropout(L6, keep_prob)

    W7 = tf.get_variable("W7", shape=[num_of_unit, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([num_of_unit]))
    L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
    L7 = tf.nn.dropout(L7, keep_prob)

    W8 = tf.get_variable("W8", shape=[num_of_unit, num_of_unit],
                         initializer=tf.contrib.layers.xavier_initializer())
    b8 = tf.Variable(tf.random_normal([num_of_unit]))
    L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
    L8 = tf.nn.dropout(L8, keep_prob)

    W9 = tf.get_variable("W9", shape=[num_of_unit, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b9 = tf.Variable(tf.random_normal([1]))

    hypothesis = tf.sigmoid(tf.matmul(L8, W9) + b9)

    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))

    train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    #
    # sess.run(tf.global_variables_initializer())
    # # sess.run(tf.global_variables_initializer())
    # new_saver = tf.train.import_meta_graph('./model/ssum_predict.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
    #
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('./model/ssum_predict.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model'))

    print("tensorflow config finished")
    app.debug = True
    app.run()
