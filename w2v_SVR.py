
import csv
import numpy as np
import math
import random
from sklearn import svm
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/hangruan/Downloads/GoogleNews-vectors-negative300.bin', binary=True) 

def readFile(filename): #return the data as the format of list
	with open(filename, 'r') as csvfile:
		dataStream = csv.reader(csvfile)
		orginData = list(dataStream)
		dataSet = orginData[1:]
		title = orginData[:1]
		title = title[-1]
		trainSet = dataSet[:7500]
		testSet = dataSet[7500:]
	return np.array(trainSet), np.array(testSet), np.array(title)



#featureTitle and labelTitle is. String
#return double result
def caculateWordVec(feature_title, label_title):
	similarity = repr(model.similarity(feature_title, label_title))
	return similarity


#replace the string with word similarity 
def preProcessData(trainSet, testSet, label_title):
	train_label_vec = trainSet[..., -1]
	test_label_vec = testSet[..., -1]
	train_feature_vec = trainSet[..., 0:-1]
	test_feature_vec = testSet[...,0:-1]
	map1 = {} 
	# map3{string, map4{string, float}}
	map2 = {}
	map3 = {}
	map4 = {}






	train_feature_vec1 = train_feature_vec.copy()
	test_feature_vec1 = test_feature_vec.copy()

	train_feature_vec2 = train_feature_vec.copy()
	test_feature_vec2 = test_feature_vec.copy()

	train_feature_vec3 = train_feature_vec.copy()
	test_feature_vec3 = test_feature_vec.copy()

	train_feature_vec4 = train_feature_vec.copy()
	test_feature_vec4 = test_feature_vec.copy()

	seasons = ["spring", "summer", "autumn", "winter"]
	weathers = ["sunny", "rainy", "foggy", "rainstorm"]
	for i in range(len(seasons)):
		season1 = seasons[i]
		weather1 = weathers[i]
		map3[season1] = {}
		map3[weather1] = {}
		map4[season1] = {}
		map4[weather1] = {}



	for i in range(len(seasons)):
		season1 = seasons[i]
		for j in range(len(seasons)):
			season2 = seasons[j]
			if i == j:
				map3[season1][season2] = 1
				map3[season2][season1] = 1
				map4[season1][season2] = 1
				map4[season2][season1] = 1
			else:
				if season2 in map3[season1]:
					continue
				similarity = caculateWordVec(season1, season2)
				map3[season1][season2] = similarity
				map3[season2][season1] = similarity
				map4[season1][season2] = 0
				map4[season2][season1] = 0

	for i in range(len(weathers)):
		weather1 = weathers[i]
		for j in range(len(weathers)):
			weather2 = weathers[j]
			if i == j:
				map3[weather1][weather2] = 1
				map3[weather2][weather1] = 1
				map4[weather1][weather2] = 1
				map4[weather2][weather1] = 1
			else:
				if weather2 in map3[weather1]:
					continue
				similarity = caculateWordVec(weather1, weather2)
				map3[weather1][weather2] = similarity
				map3[weather2][weather1] = similarity
				map4[weather1][weather2] = 0
				map4[weather2][weather1] = 0

	train_feature_vec3 = train_feature_vec3.tolist()
	test_feature_vec3 = test_feature_vec3.tolist()

	train_feature_vec4 = train_feature_vec4.tolist()
	test_feature_vec4 = test_feature_vec4.tolist()

	for i in range(len(train_feature_vec3)):
		season1 = train_feature_vec3[i][1]
		weather1 = train_feature_vec3[i][4]
		for j in range(len(seasons)):
			season2 = seasons[j]
			train_feature_vec3[i].append(map3[season1][season2])
			train_feature_vec4[i].append(map4[season1][season2])
		for j in range(len(weathers)):
			weather2 = weathers[j]
			train_feature_vec3[i].append(map3[weather1][weather2])
			train_feature_vec4[i].append(map4[weather1][weather2])



	for i in range(len(test_feature_vec3)):
		season1 = test_feature_vec3[i][1]
		weather1 = test_feature_vec3[i][4]
		for j in range(len(seasons)):
			season2 = seasons[j]
			test_feature_vec3[i].append(map3[season1][season2])
			test_feature_vec4[i].append(map4[season1][season2])
		for j in range(len(weathers)):
			weather2 = weathers[j]
			test_feature_vec3[i].append(map3[weather1][weather2])
			test_feature_vec4[i].append(map4[weather1][weather2])
	print(map4)
	train_feature_vec3 = np.array(train_feature_vec3)
	test_feature_vec3 = np.array(test_feature_vec3)
	train_feature_vec4 = np.array(train_feature_vec4)
	test_feature_vec4 = np.array(test_feature_vec4)

	train_feature_vec3 = np.delete(train_feature_vec3, [1,4], axis=1)
	test_feature_vec3 = np.delete(test_feature_vec3, [1,4], axis=1)
	train_feature_vec4 = np.delete(train_feature_vec4, [1,4], axis=1)
	test_feature_vec4 = np.delete(test_feature_vec4, [1,4], axis=1)

	similarity_spring = caculateWordVec("spring", label_title)
	similarity_summer = caculateWordVec("summer", label_title)
	similarity_autumn = caculateWordVec("autumn", label_title)
	similarity_winter = caculateWordVec("winter", label_title)

	similarity_sunny = caculateWordVec("sunny", label_title)
	similarity_rainy = caculateWordVec("rainy", label_title)
	similarity_foggy = caculateWordVec("foggy", label_title)
	similarity_rainstorm = caculateWordVec("rainstorm", label_title)

	map1["spring"] = similarity_spring
	map1["summer"] = similarity_summer
	map1["autumn"] = similarity_autumn
	map1["winter"] = similarity_winter
	map1["sunny"] = similarity_sunny
	map1["rainy"] = similarity_rainy
	map1["foggy"] = similarity_foggy
	map1["rainstorm"] = similarity_rainstorm

	map2["spring"] = 1
	map2["summer"] = 2
	map2["autumn"] = 3
	map2["winter"] = 4
	map2["sunny"] = 1
	map2["rainy"] = 2
	map2["foggy"] = 3
	map2["rainstorm"] = 4

	for i in range(len(train_feature_vec)):
		train_feature_vec1[i][1] = map1[train_feature_vec[i][1]]
		train_feature_vec1[i][4] = map1[train_feature_vec[i][4]]
		train_feature_vec1[i][0] = train_feature_vec[i][0].split(" ")[-1].split(":")[0]

		train_feature_vec2[i][1] = map2[train_feature_vec[i][1]]
		train_feature_vec2[i][4] = map2[train_feature_vec[i][4]]
		train_feature_vec2[i][0] = train_feature_vec[i][0].split(" ")[-1].split(":")[0]

		train_feature_vec3[i][0] = train_feature_vec[i][0].split(" ")[-1].split(":")[0]
		train_feature_vec4[i][0] = train_feature_vec[i][0].split(" ")[-1].split(":")[0]


	for i in range(len(test_feature_vec)):
		test_feature_vec1[i][1] = map1[test_feature_vec[i][1]]
		test_feature_vec1[i][4] = map1[test_feature_vec[i][4]]
		test_feature_vec1[i][0] = test_feature_vec[i][0].split(" ")[-1].split(":")[0]

		test_feature_vec2[i][1] = map2[test_feature_vec[i][1]]
		test_feature_vec2[i][4] = map2[test_feature_vec[i][4]]
		test_feature_vec2[i][0] = test_feature_vec[i][0].split(" ")[-1].split(":")[0]

		test_feature_vec3[i][0] = test_feature_vec[i][0].split(" ")[-1].split(":")[0]
		test_feature_vec4[i][0] = test_feature_vec[i][0].split(" ")[-1].split(":")[0]

	train_feature_vec1 = train_feature_vec1.astype(float)
	test_feature_vec1 = test_feature_vec1.astype(float)

	train_label_vec = train_label_vec.astype(float)
	test_label_vec = test_label_vec.astype(float)

	train_feature_vec2 = train_feature_vec2.astype(float)
	test_feature_vec2 = test_feature_vec2.astype(float)

	train_feature_vec3 = train_feature_vec3.astype(float)
	test_feature_vec3 = test_feature_vec3.astype(float)

	train_feature_vec4 = train_feature_vec4.astype(float)
	test_feature_vec4 = test_feature_vec4.astype(float)

	nomornazation(train_feature_vec1)
	nomornazation(test_feature_vec1)

	nomornazation(train_feature_vec2)
	nomornazation(test_feature_vec2)

	nomornazation(train_feature_vec3)
	nomornazation(test_feature_vec3)

	nomornazation1(train_feature_vec4)
	nomornazation1(test_feature_vec4)

	# print(train_feature_vec)
	# print(test_feature_vec)
	# print(train_label_vec)
	# print(test_label_vec)

	return train_feature_vec1, test_feature_vec1, train_feature_vec2, test_feature_vec2, train_feature_vec3, test_feature_vec3, train_feature_vec4, test_feature_vec4, test_label_vec, train_label_vec
	#return train_feature_vec3, test_feature_vec3, test_label_vec, train_label_vec,

	#1 : word2ved    2 : naive method(0 1 2 3 4)   3 : one hot 

def nomornazation(feature_vec):
	row = len(feature_vec)
	column = len(feature_vec[0])

	for j in range(column):
		Max = max(feature_vec[...,j])
		for i in range(row):
			feature_vec[i][j] = feature_vec[i][j] / Max

def nomornazation1(feature_vec):
	row = len(feature_vec)
	column = len(feature_vec[0])

	for j in range(column - 8):
		Max = max(feature_vec[...,j])
		for i in range(row):
			feature_vec[i][j] = feature_vec[i][j] / Max

def SVR1(train_feature_vec, train_label_vec, test_feature_vec, c):
	clf = svm.SVR(kernel='rbf', C=c, gamma=0.1)
	clf = clf.fit(train_feature_vec, train_label_vec) 
	test_predict_label_vec = clf.predict(test_feature_vec)
	return test_predict_label_vec

def e(test_predict_label_vec, test_label_vec):
	length = len(test_label_vec)
	e = 0.0
	for i in range(length):
		e += pow((test_predict_label_vec[i] - test_label_vec[i]),2)
	e = (e / length)
	return e


def main():
	e11 = []
	e22 = []
	e33 = []
	e44 = []
	trainSet, testSet, title = readFile("train.csv")
	label_title = title[-1]
	train_feature_vec1, test_feature_vec1, train_feature_vec2, test_feature_vec2, train_feature_vec3, test_feature_vec3, train_feature_vec4, test_feature_vec4, test_label_vec, train_label_vec = preProcessData(trainSet, testSet, label_title)
	c = 1
	for i in range(5):
		c *= 10
		test_predict_label_vec1 = SVR1(train_feature_vec1, train_label_vec, test_feature_vec1, c)
		test_predict_label_vec2 = SVR1(train_feature_vec2, train_label_vec, test_feature_vec2, c)
		test_predict_label_vec3 = SVR1(train_feature_vec3, train_label_vec, test_feature_vec3, c)
		test_predict_label_vec4 = SVR1(train_feature_vec4, train_label_vec, test_feature_vec4, c)
		e1 = e(test_predict_label_vec1, test_label_vec)
		e2 = e(test_predict_label_vec2, test_label_vec)
		e3 = e(test_predict_label_vec3, test_label_vec)
		e4 = e(test_predict_label_vec4, test_label_vec)
		e11.append(e1)
		e22.append(e2)
		e33.append(e3)
		e44.append(e4)

	# test_predict_label_vec1 = SVR1(train_feature_vec1, train_label_vec, test_feature_vec1)
	# test_predict_label_vec2 = SVR1(train_feature_vec2, train_label_vec, test_feature_vec2)
	# test_predict_label_vec3 = SVR1(train_feature_vec3, train_label_vec, test_feature_vec3)
	# test_predict_label_vec4 = SVR1(train_feature_vec4, train_label_vec, test_feature_vec4)
	
	# print("test_predict_label_vec1: ")
	# print(test_predict_label_vec1)
	# print("test_predict_label_vec2: ")
	# print(test_predict_label_vec2)
	# print("test_predict_label_vec3: ")
	# print(test_predict_label_vec3)
	# print("test_predict_label_vec4: ")
	# print(test_predict_label_vec4)
	# print("test_label_vec: ")
	# print(test_label_vec)
	# e1 = e(test_predict_label_vec1, test_label_vec)
	# e2 = e(test_predict_label_vec2, test_label_vec)
	# e3 = e(test_predict_label_vec3, test_label_vec)
	# e4 = e(test_predict_label_vec4, test_label_vec)
	print(e11)
	print(e22)
	print(e33)
	print(e44)




main()
