import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mlxtend.preprocessing import one_hot


def preprocessing():
    dataset = pd.read_csv("../../data/dataset2.csv")

    # reply time
    reply_time_cols = dataset['replyTime']
    # print (preprocessing_replyTime(reply_time_cols))
    processedData = preprocessing_replyTime(reply_time_cols);

    # timeznoe for chatting
    timezone_for_chattings_cols = dataset['timeZoneForChattings']
    processedData = np.append(processedData, preprocessing_timezone_for_chattings(timezone_for_chattings_cols), axis=1)
    # print(processedData)

    # mainSender one hot encoding
    main_sender_cols = dataset['mainSender']
    # print(main_sender_cols)
    processedData = np.append(processedData, preprocessing_one_hot_encoding(main_sender_cols), axis=1)
    # print(processedData)

    # mainLeader one hot encoding
    main_leader_cols = dataset['mainLeader']
    processedData = np.append(processedData, preprocessing_one_hot_encoding(main_leader_cols), axis=1)
    print(processedData)

    # periodOfKnowing
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['periodOfKnowing']), axis=1)

    # lengthOfTimeTogether
    length_of_time_together_cols = dataset['lengthOfTimeTogether']
    processedData = np.append(processedData, preprocessing_length_of_time_together(length_of_time_together_cols),
                              axis=1)
    # print(processedData)

    # meetingCycle
    meeting_cycle = dataset['meetingCycle']
    processedData = np.append(processedData, preprocessing_meeting_cycle(meeting_cycle), axis=1)

    # costProcessing
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['costProcessing']), axis=1)
    # print(preprocessing_one_hot_encoding(dataset['costProcessing']))
    # processedData = np.append(processedData, )

    # skinShip
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['skinShip']), axis=1)

    # partnerAlcoholPreference
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['partnerAlcoholPreference']),
                              axis=1)

    # drunkExperience
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['drunkExperience']), axis=1)

    # dateType

    # commonHobby
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['commonHobby']), axis=1)

    # commonConcern
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['commonConcern']), axis=1)

    # seriousConversation
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['seriousConversation']), axis=1)
    # print(preprocessing_yes_no_dont_know(dataset['commonHobby']))

    # group
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['group']), axis=1)

    # ageDifference
    processedData = np.append(processedData, preprocessing_one_hot_encoding(dataset['ageDifference']), axis=1)

    # distance
    # age
    processedData = np.append(processedData, preprocessing_age(dataset['age']), axis=1)

    # sex
    processedData = np.append(processedData, preprocessing_sex(dataset['sex']), axis=1)
    print(preprocessing_sex(dataset['sex']))
    # howmanyLabel
    processedData = np.append(processedData,preprocessing_one_hot_encoding(dataset['howMany']),axis=1)

    print(preprocessing_one_hot_encoding(dataset['howMany']).shape)

    # print(preprocessing_age(dataset['Label']))
    return processedData

def preprocessing_sex(cols):
    np_cols = np.zeros(len(cols))

    for i in range(0, len(cols)):
        if cols[i] == '남':
            np_cols[i] = 1
        if cols[i] == '여':
            np_cols[i] = 0

    return np_cols.reshape(-1, 1)


def preprocessing_age(cols):
    np_cols = np.zeros(len(cols))

    for i in range(0, len(cols)):
        np_cols[i] = cols[i]

    scaler = MinMaxScaler(feature_range=(0, 1))

    return scaler.fit_transform(np_cols.reshape(-1, 1))


def preprocessing_yes_no_dont_know(cols):
    np_cols = np.zeros(len(cols))

    for i in range(0, len(cols)):
        if cols[i] == '네':
            np_cols[i] = 1
        if cols[i] == '아니오':
            np_cols[i] = 0
        if cols[i] == '모름':
            np_cols[i] = 0.5

    return np_cols.reshape(-1, 1)


# 1주에 한번
# 1주에 두번
# 1주에 세번
# 1주에 네번 이상
# 2주에 한번
# 1달에 한번 이하

def preprocessing_meeting_cycle(cols):
    np_cols = np.zeros(len(cols))

    for i in range(0, len(cols)):
        if cols[i] == '1주에 한번':
            np_cols[i] = 1
        if cols[i] == '1주에 두번':
            np_cols[i] = 2
        if cols[i] == '1주에 세번  ( ͡° ͜ʖ ͡°) ( ͡° ͜ʖ ͡°)':
            np_cols[i] = 3
        if cols[i] == '1주에 네번이상':
            np_cols[i] = 4
        if cols[i] == '2주에 한번':
            np_cols[i] = 0.5
        if cols[i] == '1달에 한번 이하':
            np_cols[i] = 0.25

    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(np_cols.reshape(-1, 1))


# 1시간 이내 -> 1
# 1시간 ~ 2시간 -> 2
# 2시간 ~ 4시간 -> 4
# 4~8시간 -> 8
# 8 ~ -> 16
def preprocessing_length_of_time_together(cols):
    np_cols = np.zeros(len(cols))

    for i in range(0, len(cols)):
        if cols[i] == '2시간~4시간':
            np_cols[i] = 4
        if cols[i] == '4시간~8시간':
            np_cols[i] = 8
        if cols[i] == '8시간이상  ( ͡° ͜ʖ ͡°) ( ͡° ͜ʖ ͡°)':
            np_cols[i] = 16
        if cols[i] == '1시간~2시간':
            np_cols[i] = 2
        if cols[i] == '1시간 이내':
            np_cols[i] = 1

    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(np_cols.reshape(-1, 1))


def preprocessing_one_hot_encoding(cols):
    array, size = category_array_to_int_array(cols)
    return one_hot(array).reshape(-1, size)


def preprocessing_main_leader(main_leader_cols):
    array, size = category_array_to_int_array(main_leader_cols)
    array = one_hot(array)
    return array.reshape(-1, size)


def preprocessing_main_sender(main_sender_cols):
    processed_main_sender_cols, size = category_array_to_int_array(main_sender_cols)

    processed_main_sender_cols_one_hot_encoding = one_hot(processed_main_sender_cols)

    return processed_main_sender_cols_one_hot_encoding.reshape(-1, size)


# 카톡답장 시간 처리
# 30분 이내 -> 5점
# 30~1시간 -> 4점
# 1시간~2시간 -> 3점
# 2시간~4시간 -> 2점
# 4시간~8시간 -> 1점
def preprocessing_replyTime(reply_time_cols):
    processed_reply_time_cols = np.zeros(len(reply_time_cols))
    print(processed_reply_time_cols.size)
    for i in range(0, len(reply_time_cols)):
        if reply_time_cols[i] == '30분이내':
            processed_reply_time_cols[i] = 5
        if reply_time_cols[i] == '30분~1시간':
            processed_reply_time_cols[i] = 4
        if reply_time_cols[i] == '1시간~2시간':
            processed_reply_time_cols[i] = 3
        if reply_time_cols[i] == '2시간~4시간':
            processed_reply_time_cols[i] = 2
        if reply_time_cols[i] == '4시간~8시간':
            processed_reply_time_cols[i] = 1

    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(processed_reply_time_cols.reshape(-1, 1))
    # return processed_reply_time_cols.reshape(-1, 1)


# 아침 1
# 점심 2
# 저녁 3
# 밤 4
# 새벽 5
def preprocessing_timezone_for_chattings(timezone_for_chattings_cols):
    processed_timezone_for_chattings_cols = np.zeros(len(timezone_for_chattings_cols))
    for i in range(0, len(timezone_for_chattings_cols)):
        value = 0
        if "아침" in timezone_for_chattings_cols[i]:
            value += 1
        if "점심" in timezone_for_chattings_cols[i]:
            value += 2
        if "저녁" in timezone_for_chattings_cols[i]:
            value += 3
        if "밤" in timezone_for_chattings_cols[i]:
            value += 4
        if "새벽" in timezone_for_chattings_cols[i]:
            value += 5

        processed_timezone_for_chattings_cols[i] = value

    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(processed_timezone_for_chattings_cols.reshape(-1, 1))
    # return processed_timezone_for_chattings_cols.reshape(-1, 1)


# categorical value to int
# return int_array, dicSize
def category_array_to_int_array(categorical_array):
    processed_int_array = np.zeros(len(categorical_array)).astype(int)
    dicMap = {}
    index = 0
    for i in range(0, len(categorical_array)):
        if dicMap.get(categorical_array[i]) == None:
            dicMap[categorical_array[i]] = index
            processed_int_array[i] = index
            index += 1
        else:
            processed_int_array[i] = dicMap.get(categorical_array[i])

    return processed_int_array, len(dicMap.keys())


processed_data = preprocessing()
np.savetxt("../data/when_to_confess_ternary_one_hot_encoding.csv",processed_data,delimiter=",")
