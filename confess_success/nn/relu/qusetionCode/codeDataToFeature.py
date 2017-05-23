import pandas as pd
import pymongo
from pprint import pprint

client = pymongo.MongoClient("mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago")
db_ssumtago = client['ssumtago']
surveys = db_ssumtago['surveys']
ssum_predect_survey = surveys.find_one({"id":1})
client.close()

pprint(ssum_predect_survey)


dataset = pd.read_csv("../../../data/dataset.csv")

col_name_list = list(dataset)

for v in col_name_list:
    cols = dataset[v]
    # 1. 객관식 category 다중 선택 불가 -> one hot encoding
    # 2. 객관식 number -> normalize
    # 3. 객관식 category 다중 선택 불가 ->
    # 4.
# def process_reply_time(param):
#     pass
#
#
# def process_reply_time_zone(param):
#     pass
#
#
# def process_message_first_sender(param):
#     pass
#
#
# def process_message_leader(param):
#     pass
#
#
# def process_period_of_knowing(param):
#     pass
#
#
# def process_time_spent_together(param):
#     pass
#
#
# def process_meeting_cycle(param):
#     pass
#
# def process_pay_process(param):
#     pass
#
#
# def process_skin_ship(param):
#     pass
#
#
# def process_partner_alcohol_preference(param):
#     pass
#
#
# def process_drunk_experience(param):
#     pass
#
#
# def process_common_hobby(param):
#     pass
#
#
# def process_common_concern(param):
#     pass
#
#
# def process_serious_conversation(param):
#     pass
#
#
# def process_age_difference(param):
#     pass
#
#
# def process_living_distance(param):
#     pass
#
#
# def process_age(param):
#     pass
#
#
# def process_sex(param):
#     pass
#
#
# def process_label(param):
#     pass
#
#
# def preprocessing():
#     dataset = pd.read_csv("../data/dataset.csv")
#
#     dataset
#     data = []
#     # 0100005		replyTime
#     data = data.append(process_reply_time(dataset['0100005']))
#
#     # 0101006		replyTimeZone
#     data = data.append(process_reply_time_zone(dataset['0101006']))
#
#     # 0100207		messageFirstSender
#     data = data.append(process_message_first_sender(dataset['0100207']))
#
#     # 0100008		messageLeader
#     data = data.append(process_message_leader(dataset['0100008']))
#
#     # 0100009		periodOfKnowing
#     data = data.append(process_period_of_knowing(dataset['0100009']))
#
#     # 0100110		timeSpentTogether
#     data = data.append(process_time_spent_together(dataset['0100110']))
#
#     # 0100111		meetingCycle
#     data = data.append(process_meeting_cycle(dataset['0100111']))
#
#     # 0100013		payProcess
#     data = data.append(process_pay_process(dataset['0100013']))
#
#     # 0100114		skinship
#     data = data.append(process_skin_ship(dataset['0100114']))
#
#     # 0100015		partnerAlcoholPreference
#     data = data.append(process_partner_alcohol_preference(dataset['0100015']))
#
#     # 0100116		drunkExperience
#     data = data.append(process_drunk_experience(dataset['0100116']))
#
#     # 0100019		commonHobby
#     data = data.append(process_common_hobby(dataset['0100019']))
#
#     # 0100020		commonConcern
#     data = data.append(process_common_concern(dataset['0100020']))
#
#     # 0100121		seriousConversation
#     data = data.append(process_serious_conversation(dataset['0100121']))
#
#     # 0100023		ageDifference
#     data = data.append(process_age_difference(dataset['0100023']))
#
#     # 0100024		livingDistance
#     data = data.append(process_living_distance(dataset['0100024']))
#
#     # 0102025		age
#     data = data.append(process_age(dataset['0102025']))
#
#     # 0100026		sex
#     data = data.append(process_sex(dataset['0100026']))
#
#
#     # 0100001		isSuccess Label
#     data = data.append(process_label(dataset['0100001']))
#     return data
#
# def preprocessing_sex(cols):
#     np_cols = np.zeros(len(cols))
#
#     for i in range(0, len(cols)):
#         if cols[i] == '남':
#             np_cols[i] = 1
#         if cols[i] == '여':
#             np_cols[i] = 0
#
#     return np_cols.reshape(-1, 1)
#
#
# def preprocessing_age(cols):
#     np_cols = np.zeros(len(cols))
#
#     for i in range(0, len(cols)):
#         np_cols[i] = cols[i]
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#
#     return scaler.fit_transform(np_cols.reshape(-1, 1))
#
#
# def preprocessing_yes_no_dont_know(cols):
#     np_cols = np.zeros(len(cols))
#
#     for i in range(0, len(cols)):
#         if cols[i] == '네':
#             np_cols[i] = 1
#         if cols[i] == '아니오':
#             np_cols[i] = 0
#         if cols[i] == '모름':
#             np_cols[i] = 0.5
#
#     return np_cols.reshape(-1, 1)
#
#
# # 1주에 한번
# # 1주에 두번
# # 1주에 세번
# # 1주에 네번 이상
# # 2주에 한번
# # 1달에 한번 이하
#
# def preprocessing_meeting_cycle(cols):
#     np_cols = np.zeros(len(cols))
#
#     for i in range(0, len(cols)):
#         if cols[i] == '1주에 한번':
#             np_cols[i] = 1
#         if cols[i] == '1주에 두번':
#             np_cols[i] = 2
#         if cols[i] == '1주에 세번  ( ͡° ͜ʖ ͡°) ( ͡° ͜ʖ ͡°)':
#             np_cols[i] = 3
#         if cols[i] == '1주에 네번이상':
#             np_cols[i] = 4
#         if cols[i] == '2주에 한번':
#             np_cols[i] = 0.5
#         if cols[i] == '1달에 한번 이하':
#             np_cols[i] = 0.25
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     return scaler.fit_transform(np_cols.reshape(-1, 1))
#
#
# # 1시간 이내 -> 1
# # 1시간 ~ 2시간 -> 2
# # 2시간 ~ 4시간 -> 4
# # 4~8시간 -> 8
# # 8 ~ -> 16
# def preprocessing_length_of_time_together(cols):
#     np_cols = np.zeros(len(cols))
#
#     for i in range(0, len(cols)):
#         if cols[i] == '2시간~4시간':
#             np_cols[i] = 4
#         if cols[i] == '4시간~8시간':
#             np_cols[i] = 8
#         if cols[i] == '8시간이상  ( ͡° ͜ʖ ͡°) ( ͡° ͜ʖ ͡°)':
#             np_cols[i] = 16
#         if cols[i] == '1시간~2시간':
#             np_cols[i] = 2
#         if cols[i] == '1시간 이내':
#             np_cols[i] = 1
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     return scaler.fit_transform(np_cols.reshape(-1, 1))
#
#
# def preprocessing_one_hot_encoding(cols):
#     array, size = category_array_to_int_array(cols)
#     return one_hot(array).reshape(-1, size)
#
#
# def preprocessing_main_leader(main_leader_cols):
#     array, size = category_array_to_int_array(main_leader_cols)
#     array = one_hot(array)
#     return array.reshape(-1, size)
#
#
# def preprocessing_main_sender(main_sender_cols):
#     processed_main_sender_cols, size = category_array_to_int_array(main_sender_cols)
#
#     processed_main_sender_cols_one_hot_encoding = one_hot(processed_main_sender_cols)
#
#     return processed_main_sender_cols_one_hot_encoding.reshape(-1, size)
#
#
# # 카톡답장 시간 처리
# # 30분 이내 -> 5점
# # 30~1시간 -> 4점
# # 1시간~2시간 -> 3점
# # 2시간~4시간 -> 2점
# # 4시간~8시간 -> 1점
# def preprocessing_replyTime(reply_time_cols):
#     processed_reply_time_cols = np.zeros(len(reply_time_cols))
#     print(processed_reply_time_cols.size)
#     for i in range(0, len(reply_time_cols)):
#         if reply_time_cols[i] == '30분이내':
#             processed_reply_time_cols[i] = 5
#         if reply_time_cols[i] == '30분~1시간':
#             processed_reply_time_cols[i] = 4
#         if reply_time_cols[i] == '1시간~2시간':
#             processed_reply_time_cols[i] = 3
#         if reply_time_cols[i] == '2시간~4시간':
#             processed_reply_time_cols[i] = 2
#         if reply_time_cols[i] == '4시간~8시간':
#             processed_reply_time_cols[i] = 1
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     return scaler.fit_transform(processed_reply_time_cols.reshape(-1, 1))
#     # return processed_reply_time_cols.reshape(-1, 1)
#
#
# # 아침 1
# # 점심 2
# # 저녁 3
# # 밤 4
# # 새벽 5
# def preprocessing_timezone_for_chattings(timezone_for_chattings_cols):
#     processed_timezone_for_chattings_cols = np.zeros(len(timezone_for_chattings_cols))
#     for i in range(0, len(timezone_for_chattings_cols)):
#         value = 0
#         if "아침" in timezone_for_chattings_cols[i]:
#             value += 1
#         if "점심" in timezone_for_chattings_cols[i]:
#             value += 2
#         if "저녁" in timezone_for_chattings_cols[i]:
#             value += 3
#         if "밤" in timezone_for_chattings_cols[i]:
#             value += 4
#         if "새벽" in timezone_for_chattings_cols[i]:
#             value += 5
#
#         processed_timezone_for_chattings_cols[i] = value
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     return scaler.fit_transform(processed_timezone_for_chattings_cols.reshape(-1, 1))
#     # return processed_timezone_for_chattings_cols.reshape(-1, 1)
#
#
# # categorical value to int
# # return int_array, dicSize
# def category_array_to_int_array(categorical_array):
#     processed_int_array = np.zeros(len(categorical_array)).astype(int)
#     dicMap = {}
#     index = 0
#     for i in range(0, len(categorical_array)):
#         if dicMap.get(categorical_array[i]) == None:
#             dicMap[categorical_array[i]] = index
#             processed_int_array[i] = index
#             index += 1
#         else:
#             processed_int_array[i] = dicMap.get(categorical_array[i])
#
#     return processed_int_array, len(dicMap.keys())
#
#
# processed_data = preprocessing()
# np.savetxt("../data/confess_success_434.csv",processed_data,delimiter=",")
