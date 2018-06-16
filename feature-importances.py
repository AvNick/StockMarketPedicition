from pandas import read_csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from TechnicalAnalysis import *
# from DataFetcher import DataFetcher
import os
from datetime import datetime
from matplotlib import pyplot as plt
from DataPreprocessor import DataPreprocessor
from ModelEvaluation import Evaluator
from multiprocessing import Process
import matplotlib.patches as mpatches
import pandas as pd
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import random

np.set_printoptions(precision=2)

def getPerformanceStats(TP, TN, FP, FN):
	#print TP, TN, FP, FN
	accuracy = ((TP + TN)/(TP + TN + FP + FN))*100

	try:
		recall =  TP/(TP+FN)
	except:
		recall = -999

	try:
		precision = TP/(TP + FP)
		print(TP)
		print(FP)
	except:
		precision = -999

	try:
		specificity = TN/(TN + FP)
	except:
		specificity = -999

	try:
		fscore = 2*(precision * recall)/(precision + recall)
	except:
		fscore = -999
	return ("%.2f" % accuracy), ("%.2f" % recall), ("%.2f" % precision), ("%.2f" % specificity), ("%.2f" % fscore)

def getData(CSVFile):

	smoother = DataPreprocessor()
	data = read_csv(CSVFile)
	data = data[::-1] # reverse
	ohclv_data = np.c_[data['Open'],
					   data['High'],
					   data['Low'],
					   data['Close'],
					   data['Volume']]

	smoothened_ohclv_data = smoother.PandaSmoother(ohclv_data)
	return  smoothened_ohclv_data, np.array(data["Close"]), list(data["Date"])

def getTechnicalIndicators(X,d):

	RSI = getRSI(X[:,3])
	StochasticOscillator = getStochasticOscillator(X)
	Williams = getWilliams(X)


	MACD = getMACD(X[:,3])
	PROC = getPriceRateOfChange(X[:,3],d)
	OBV = getOnBalanceVolume(X)

	min_len = min(len(RSI),
				  len(StochasticOscillator),
				  len(Williams),
				  len(MACD),
				  len(PROC),
				  len(OBV))

	RSI = RSI[len(RSI) - min_len:]
	StochasticOscillator = StochasticOscillator[len(StochasticOscillator) - min_len:]
	Williams = Williams[len(Williams) - min_len: ]
	MACD = MACD[len(MACD) - min_len:]
	PROC = PROC[len(PROC) - min_len:]
	OBV = OBV[len(OBV) - min_len:]


	feature_matrix = np.c_[RSI[:,0],
						   StochasticOscillator[:,0],
						   Williams[:,0],
						   MACD[:,0],
						   PROC[:,0],
						   OBV[:,0]]

	return feature_matrix

def prepareData(X,close,date,d):

	feature_matrix = getTechnicalIndicators(X,d)

	number_of_samples = feature_matrix.shape[0]
	date = date[len(date) - number_of_samples:]
	close = close[len(close) - number_of_samples:]

	#y0 = feature_matrix[:,-1][ :number_of_samples-d]
	#y1 = feature_matrix[:,-1][d:]

	y0 = close[:number_of_samples - d]
	y1 = close[d:]

	feature_matrix_1 = feature_matrix[:number_of_samples-d]
	feature_matrix_2 = feature_matrix[number_of_samples - 1000:]
	date = date[number_of_samples - 1000:]



	#closeplot = feature_matrix[:,-1][number_of_samples - 1000:]
	closeplot = close[number_of_samples - 1000:]
	y = np.sign(y1 - y0)

	feature_matrix_1 = feature_matrix_1[:, range(6)]

	return feature_matrix_1,y,feature_matrix_2[:,range(6)],closeplot,date



def plotTradingStrategy(model, xplot, closeplot, Trading_Day,date):

	colorMap = {-1.0:"r",1.0:"b",0.0:"y"}
	tradeMap = {-1.0:"Sell",1.0:"Buy",0.0:"Buy"}
	plt.figure()
	plt.plot(closeplot, c = "g")
	#x = [xplot[i] for i in xrange(0,len(xplot),Trading_Day)]
	#y = [closeplot[i] for i in xrange(0, len(closeplot),Trading_Day)]
	x = [xplot[i] for i in range(0,len(xplot),Trading_Day)]
	y = [closeplot[i] for i in range(0, len(closeplot),Trading_Day)]
	y_pred = model.predict(x)

	#c = [colorMap[y_pred[i]] for i in xrange(len(y_pred))]
	c = [colorMap[y_pred[i]] for i in range(len(y_pred))]

	#df = pd.DataFrame(np.c_[[ i+1 for i in xrange(0, len(xplot),Trading_Day)], x, y, [tradeMap[y_pred[i]] for i in xrange(len(y_pred)) ]],
	df = pd.DataFrame(np.c_[[ i+1 for i in range(0, len(xplot),Trading_Day)], x, y, [tradeMap[y_pred[i]] for i in range(len(y_pred)) ]],
   			columns = ["Day","RSI","Stochastic Oscillator","Williams","MACD","Price Rate Of Change","On Balance Volume","Close","Buy/Sell"])
	df.to_csv("AAPLBuySellTradePoints.csv",index = False)


	#plt.scatter([i for i in xrange(0,len(xplot),Trading_Day)],y, c = c)
	plt.scatter([i for i in range(0,len(xplot),Trading_Day)],y, c = c)
	#plt.xticks([i for i in xrange(0,len(xplot),Trading_Day)],[date[i] for i in xrange(0,len(xplot),Trading_Day)])
	red_patch = mpatches.Patch(color='red', label='Sell')
	blue_patch = mpatches.Patch(color = "blue", label = "Buy")
	plt.legend(handles = [red_patch,blue_patch])
	plt.xlabel("Time")
	plt.ylabel("Closing price")
	#plt.title("Trading strategy for {} days trading window".format(Trading_Day))
	plt.title("Trading indications for {} days trading window".format(Trading_Day))
	plt.savefig("TradingStrategy.png")
	plt.show(block = False)


def main(stock_symbol,Trading_Day, classifier_choice):

	# fetcher = DataFetcher()

	# fetch_result = fetcher.getHistoricalData(stock_symbol)
	# if fetch_result == -1:
	# 	raise Exception("NO INTERNET CONNECTIVITY OR INVALID STOCK SYMBOL")

	dir_name = os.path.dirname(os.path.abspath(__file__))
	filename = stock_symbol+".csv"
	CSVFile = os.path.join(dir_name,"Dataset",filename)

	ohclv_data, close, date= getData(CSVFile)

	#current_data = regression(ohclv_data)
	#ohclv_data.append(current_data)

	ohclv_data = np.array(ohclv_data)

	X,y,xplot,closeplot,dateplot = prepareData(ohclv_data, close, date, Trading_Day)

	y[y == 0] = 1

	TP = 0.0
	TN = 0.0
	FP = 0.0
	FN = 0.0
	NUM_ITER = 1
	pred_prob = []
	true_lbls = []
	scr_list = []
	feature_names = ['RSI', 'SO', 'W%R', 'MACD', 'PROC', 'OBV']

	for iteration in range (0, NUM_ITER):
		Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, random_state = 0)
		#dummies, inserted by Suryo:
		#print "Length of training set:", len(Xtrain)
		#print "Length of test set:", len(Xtest)

		if classifier_choice == 'RF':
			print('------------------------------')
			print('RF '+ ' stock_symbol ' + str(Trading_Day))
			model = RandomForestClassifier(n_estimators = 100,criterion = "gini", random_state = random.randint(1,12345678))
			model.fit(Xtrain, ytrain)
			importances = model.feature_importances_
			std = np.std([tree.feature_importances_ for tree in model.estimators_],
				axis=0)
			indices = np.argsort(importances)[::-1]
			print(indices)

			#for val in indices:
			for val in range(0, 6):
				#print(feature_names[r[0]], str(round((r[1]/total)*100, 2)), '%')
				print(feature_names[val], str(round(importances[val]*100, 2)))
				#print(feature_names[val], importances[val])
				#print(feature_names[val], '\\\\')

		elif classifier_choice == 'XGB':
			print('------------------------------')
			print('XGB '+ ' stock_symbol ' + str(Trading_Day))

			training_data = np.matrix(Xtrain)
			test_data = np.matrix(Xtest)

			param = {'learning_rate':0.00001,
			'n_estimators':100,
			'max_depth':20,
			'min_child_weight':1,
			'eta':0.0001,
			'silent':1,
			'objective':'multi:softmax',
			'num_class':2,
			'subsample':0.6,
			'gamma':0}
			num_round = 25

			"""EDIT THIS"""
			#BRING IN THE MACHINE LEARNING SWAG RIGHT HERE

			#labels_training = [x+1 for x in ytrain]
			for i in range(0, len(ytrain)):
				if ytrain[i] == -1:
					ytrain[i] = 0

			#xgb_train = xgb.DMatrix(training_data, labels_training)
			xgb_train = xgb.DMatrix(training_data, ytrain)
			xgb_test = xgb.DMatrix(test_data)

			xgb_train = xgb.DMatrix(Xtrain, ytrain)
			#print(Xtrain)
			bst = xgb.train(param, xgb_train, num_round)
			imp = bst.get_score().items()

			#print(imp)
			imp_table = []
			for val in imp:
				index, imp = val
				#print(index)
				index = int(index[1:])
				r = [index, imp]
				imp_table.append(r)

			total = 0
			for r in imp_table:
				total += r[1]


			for r in imp_table:
				print(feature_names[r[0]], str(round((r[1]/total)*100, 2)))



#COMPANIES = ['AAPL', 'AMS', 'AMZN', 'FB', 'MSFT', 'NKE', 'SNE', 'TATA', 'TWTR', 'TYO']
#COMPANIES = ['novartis', 'cipla', 'pfizer', 'roche']
COMPANIES = ['AAPL', 'FB']

TRADING_BINS = [3, 5, 10, 15, 30, 60, 90]
#CLASSIFIERS = ['RF', 'XGB']
CLASSIFIERS = ['XGB']

for classifier_choice in CLASSIFIERS:
	print("---------------------")
	print("")
	print('RESULTS FOR', classifier_choice)
	print("")

	for stock_symbol in COMPANIES:
		for Trading_Day in TRADING_BINS:
			#main(stock_symbol.upper(),Trading_Day, classifier_choice.upper())
			main(stock_symbol,Trading_Day, classifier_choice.upper())
