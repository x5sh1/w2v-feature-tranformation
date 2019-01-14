import gensim
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import copy

# train word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/x5sh1/Downloads/GoogleNews-vectors-negative300.bin', binary = True) 
print(model.wv['income'])

# read data from file
def Read_File(fileName, dataSet):
	with open(fileName, 'rb') as file:
		lines = csv.reader(file)
		temp = list(lines)
		title = temp[0]
		temp.remove(temp[0])
		for data in temp:
			if not '?' in data:
				dataSet.append(data)
		return title

# nomarlsation
def Normalisation(dataSet):
	col = len(dataSet[0])
	row = len(dataSet)
	for c in range(col):
		Max = max(dataSet[:,c])
		Min = min(dataSet[:,c])
		diff = Max - Min
		for r in range(row):
			dataSet[r][c] = (dataSet[r][c] - Min) / diff

#new method
def Feature_Representation_New2(dataSet, titles):
	result = []
	HashTable = {}
	order = {}
	for c in range(len(dataSet[0]) - 1):
		if not dataSet[0][c].isdigit():
			order[titles[c]] = set(dataSet[:,c])
			temp = order[titles[c]]
			for a in temp:
				if a not in HashTable:
					HashTable[a] = {}
				if '-' in a:
					at = a.split('-')
				else:
					at = [a]
				for b in temp:
					if a != b:
						if '-' in b:
							bt = b.split('-')
						else:
							bt = [b]
						if not (b in HashTable and a in HashTable[b]):
							HashTable[a][b] = model.n_similarity(at, bt);

	for data in dataSet:
		result.append([])
		tData = result[-1]
		for i in range(len(data) - 1):
			if data[i].isdigit():
				tData.append(float(data[i]))
			else:
				vec = order[titles[i]]
				for w in vec:
					if data[i] == w:
						tData.append(1)
					else:
						if data[i] in HashTable and w in HashTable[data[i]]:
							tData.append(HashTable[data[i]][w])
						else:
							tData.append(HashTable[w][data[i]])
		if data[-1] == '>50K':
			tData.append(1)
		else:
			tData.append(-1)
	return result

#one hot
def Feature_Representation_Ohot(dataSet, titles):
	result = []
	order = {}
	for c in range(len(dataSet[0]) - 1):
		order[titles[c]] = set(dataSet[:,c])

	for data in dataSet:
		result.append([])
		tData = result[-1]
		for i in range(len(data) - 1):
			if data[i].isdigit():
				tData.append(float(data[i]))
			else:
				vec = order[titles[i]]
				for w in vec:
					if data[i] == w:
						tData.append(1)
					else:
						tData.append(0)
		if data[-1] == '>50K':
			tData.append(1)
		else:
			tData.append(-1)
	return result

# feature representation-new method
def Feature_Representation_New1(dataSet, labelTitle, title):
	labelTitle = [labelTitle]
	# the number of feature
	col_amount = len(dataSet[0]) - 1
	# the number of data
	row_amount = len(dataSet)
	for col in range(col_amount):
		HashTable = {}
		for row in range(row_amount):
			if dataSet[row][col].isdigit():
				dataSet[row][col] = float(dataSet[row][col])
			else:
				if dataSet[row][col] in HashTable:
					dataSet[row][col] = HashTable[dataSet[row][col]]
				else:
					if '-' in dataSet[row][col]:
						HashTable[dataSet[row][col]] = model.n_similarity(dataSet[row][col].split('-'), labelTitle)
					else:
						HashTable[dataSet[row][col]] = model.n_similarity([dataSet[row][col]], labelTitle)
					dataSet[row][col] = HashTable[dataSet[row][col]]
		# print(title[col] + ': ')
		# keys = HashTable.keys()
		# for key in keys:
		# 	print(key + ': ' + repr(HashTable[key]))
		# print('------------------------------------------------------------------')
	# transfer label
	for row in range(row_amount):
		if dataSet[row][-1] == '>50K':
			dataSet[row][-1] = 1
		else:
			dataSet[row][-1] = -1

#1, 2, 3....
def Feature_Representation_Old(dataSet):
	# the number of feature
	col_amount = len(dataSet[0]) - 1
	# the number of data
	row_amount = len(dataSet)
	for col in range(col_amount):
		HashTable = {}
		pos = 1
		for row in range(row_amount):
			if dataSet[row][col].isdigit():
				dataSet[row][col] = float(dataSet[row][col])
			else:
				if dataSet[row][col] in HashTable:
					dataSet[row][col] = HashTable[dataSet[row][col]]
				else:
					HashTable[dataSet[row][col]] = pos
					pos += 1
					dataSet[row][col] = HashTable[dataSet[row][col]]
	# transfer label
	for row in range(row_amount):
		if dataSet[row][-1] == '>50K':
			dataSet[row][-1] = 1
		else:
			dataSet[row][-1] = -1

# KNN trianing 
def KNN_Train(dataSet, perfom):
	X_ = dataSet[:,:-1]
	y_ = dataSet[:,-1]
	Normalisation(X_)
	K = 16
	itra = 1
	for k in range(1, K):
		# X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.33)
		sp = int(len(X_) * 0.67)
		X_train = X_[:sp]
		y_train = y_[:sp]
		X_test = X_[sp + 1:]
		y_test = y_[sp + 1:]
		accuracy = 0
		precision = 0
		recall = 0
		f1 = 0
		for it in range(itra):
			clf = neighbors.KNeighborsClassifier(k, weights = 'uniform')
			clf.fit(X_train, y_train)
			predict_y = clf.predict(X_test)
			# accuracy 
			accuracy += metrics.accuracy_score(y_test, predict_y)
			# precision
			precision += metrics.precision_score(y_test, predict_y)
			# recall
			recall += metrics.recall_score(y_test, predict_y)
			# F1 score
			f1 += metrics.f1_score(y_test, predict_y)
			perfom['accuracy'].append(accuracy)
			perfom['precision'].append(precision)
			perfom['recall'].append(recall)
			perfom['f1'].append(f1)
	keys = perfom.keys()
	for key in keys:
		print(key)
		for ele in perfom[key]:
			print(ele)

# main function
def main():
	dataSet = []
	fileName = 'adult.csv'
	transferContainer = []
	title = Read_File(fileName, dataSet)
	labelTitle = title[-1]
	perfom = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
	# new method 1, use similarity between title of label with words
	n1_dataSet = copy.deepcopy(dataSet)
	Feature_Representation_New1(n1_dataSet, labelTitle, title)
	n1_dataSet = np.array(n1_dataSet)
	KNN_Train(n1_dataSet, perfom)
	print('----------------------------------------------------')
	#new method 2, motivited from one hot encoding
	perfom = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
	n2_dataSet = np.array(dataSet)
	n2_dataSet = Feature_Representation_New2(n2_dataSet, title)
	n2_dataSet = np.array(n2_dataSet)
	KNN_Train(n2_dataSet, perfom)
	print('----------------------------------------------------')
	#one not encoding
	perfom = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
	ohot_dataSet = np.array(dataSet)
	ohot_dataSet = Feature_Representation_Ohot(ohot_dataSet, title)
	ohot_dataSet = np.array(ohot_dataSet)
	KNN_Train(ohot_dataSet, perfom)
	print('----------------------------------------------------')
	#naive method
	perfom = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
	na_dataSet = copy.deepcopy(dataSet)
	Feature_Representation_Old(na_dataSet)
	na_dataSet = np.array(na_dataSet)
	KNN_Train(na_dataSet, perfom)
# main()