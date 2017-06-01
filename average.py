import pymongo

client = pymongo.MongoClient("mongodb://ssumtago:Tjaxkrh@expirit.co.kr/ssumtago")

db_ssumtago = client['ssumtago']
surveyResults = db_ssumtago['surveyResults']

avg1 = 0.0
avg2 = 0.0

index = 0

for i in surveyResults.find():
    if index < 20:
        avg1 += float(i["result"])
    else:
        avg2 += float(i["result"])
    index += 1


print(avg1/20)
print(avg2/20)


