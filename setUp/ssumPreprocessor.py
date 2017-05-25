import pymongo
import pandas as pd
import numpy as np
from mlxtend.preprocessing import one_hot
from sklearn.preprocessing import MinMaxScaler


class SsumPreprocessor:
    def __init__(self, survey):
        self._survey = survey
        self._questions = survey["questions"]
        pass

    def convert(self, question_code, answer_code):
        result = []
        code = question_code[3:8]
        if code == '00120':
            result = self.one_hot_encoding(question_code, answer_code)
            pass

        if code == '00112':
            result = self.custom_alg(question_code, answer_code)
            pass

        if code == '10000':
            result = self.normalization(question_code, answer_code)
            pass

        return result

    def custom_alg(self, question_code, answer_code):

        question = self.find_question_from_question_list(question_code)

        value = 0
        if int(int(answer_code[-2:]) / 1) == 1:
            value = 0
        elif int(int(answer_code[-2:]) / 2) == 1:
            value = 1
        elif int(int(answer_code[-2:]) / 4) == 1:
            value = 2
        elif int(int(answer_code[-2:]) / 10) == 1:
            value = 3
        elif int(int(answer_code[-2:]) / 20) == 1:
            value = 4
        else:
            print("error")

        one_hot_result = one_hot([value], num_labels=len(question["answers"]))
        return one_hot_result[0]

    def find_question_from_question_list(self, question_code):
        for question in self._questions:
            if question["code"] == question_code:
                return question

    def normalization(self, question_code, answer_code):
        max = 30
        min = 15
        value = int(answer_code[-2:])
        return [(value - min) / (max - min)]

    def one_hot_encoding(self, question_code, answer_code):
        question = self.find_question_from_question_list(question_code)
        answers = question["answers"]
        dictMap = {}
        index = 0
        for answer in answers:
            dictMap[answer["code"]] = index
            index += 1

        one_hot_result = one_hot([dictMap[answer_code]], num_labels=len(question["answers"]))
        return one_hot_result[0]