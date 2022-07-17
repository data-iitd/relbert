import torch
from torch import nn
from torch.nn import functional as F
import math, copy
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
import time
import sqlite3
from sqlite3 import OperationalError
import random
import joblib
import gensim
import numpy as np
import sys
from torch import optim
import pickle

from models import Attention_Database, Attention_Database_NSP
from global_vars import get_sql_path, get_dict_path, get_model_path
from init_new import loadConfig, getTokenInputIds, newGetTokenInputIds

db = "mimic"
file_path = "/home/garima/RelBERT/tmp/files/"
sql_path = "/home/garima/relevant_files/"+db+"/sql_files/"
model_path = "/home/garima/relevant_files/"+db+"/model/"

#User input
embeddingDim = int(sys.argv[1])
batch_size_argument=int(sys.argv[2])
grad_back=bool(int(sys.argv[3])) # grad_back=0 for not finetuning embedding layer else 1
tie_weights=bool(int(sys.argv[4])) # tie_weights=0 for not having same weight for output softmax layer else 1
#maxLength = 
table = sys.argv[5] #table name

def getTable(qStmt):

	if db == "mimic":
		conn = sqlite3.connect(sql_path+'mimic.db')
	else:	
		conn = sqlite3.connect(sql_path+'csc455_pre1.db')
	
	cursor = conn.cursor()

	quertStatement = qStmt
	cursor.execute(quertStatement)

	columnHeaderList = [description[0] for description in cursor.description]	

	return columnHeaderList




# ********** main() ********** 

DEBUG = False


#train NSP over joined table 
tableList = table.split(":")
table1 = tableList[0]
table2 = tableList[-1]
dbName, tableList1, columnHeaderDict = loadConfig()


if db == "imdb":
	t1ColumnHeaderList = columnHeaderDict[table1]
#t2ColumnHeaderList = columnHeaderDict[table2]


	qStmt = "Select * from " #todo: might have to change this to join 3 tables which effectively give 2 table join
	for tab in  tableList:
		qStmt += tab + ' natural join '

	qStmt = qStmt[:qStmt.rfind('natural join ')]
	qStmt += "; "
else:
	qStmt =  "select ADMITTIME , DISCHTIME , ADMISSION_TYPE , ADMISSION_LOCATION , DISCHARGE_LOCATION , INSURANCE , LANGUAGE , RELIGION , MARITAL_STATUS , ETHNICITY , DIAGNOSIS , ICD9_CODE , DRG_TYPE , DRG_CODE , DESCRIPTION , DRG_SEVERITY , DRG_MORTALITY FROM admissions join diagnoses_icd join drgcodes where admissions.subject_id = diagnoses_icd.subject_id and admissions.hadm_id = diagnoses_icd.hadm_id and admissions.subject_id = drgcodes.subject_id and admissions.hadm_id = drgcodes.hadm_id;"

	t1ColumnHeaderList = []
	t1ColumnHeaderList.append("ADMITTIME")
	t1ColumnHeaderList.append("DISCHTIME")
	t1ColumnHeaderList.append("ADMISSION_TYPE")
	t1ColumnHeaderList.append("ADMISSION_LOCATION")
	t1ColumnHeaderList.append("DISCHARGE_LOCATION")
	t1ColumnHeaderList.append("INSURANCE")
	t1ColumnHeaderList.append("LANGUAGE")
	t1ColumnHeaderList.append("RELIGION")
	t1ColumnHeaderList.append("MARITAL_STATUS")
	t1ColumnHeaderList.append("ETHNICITY")
	t1ColumnHeaderList.append("DIAGNOSIS")


'''
t1ColumnHeaderList = columnHeaderDict[table1]

qStmt = "Select * from " #todo: might have to change this to join 3 tables which effectively give 2 table join
for tab in  tableList:
	qStmt += tab + ' natural join '

qStmt = qStmt[:qStmt.rfind('natural join ')]
qStmt += "; "
'''

resultHeaderList = getTable(qStmt)

t2ColumnHeaderList = []
for col in resultHeaderList:
	if col not in t1ColumnHeaderList:
		t2ColumnHeaderList.append(col)


colHeaderList = []

colHeaderList.append('spec') #for [cls]

for col in t1ColumnHeaderList:
	colHeaderList.append(table1+"."+col)
colHeaderList.append('spec') #for [sep]

for col in t2ColumnHeaderList:
	tab = ""

	for t in range(1,len(tableList)):
		if col in columnHeaderDict[tableList[t]]:
			tab = tableList[t]
			break

	colHeaderList.append(tab+"."+col)

colHeaderList.append('spec') #Last [sep]

maxLength = len(colHeaderList)

if DEBUG:
	print("MAx seq length: "+ str(maxLength))

tokenCountDict = {}
tokenInputIds = newGetTokenInputIds(colHeaderList)
tokenInputIds.pop('table')

count = 0
for key in colHeaderList:
	tokenCountDict[count] = len(tokenInputIds[key])
	count += 1


testNSPSentences = []
testMaskSentences = []
testLabels = []
testMaskedTokenList = []
testMaskedIndexList = []

#saving testing set for later testing
testNSPSentences = torch.load(file_path+table+"_testNSP_MLMSentences.pt")
testMaskSentences = torch.load(file_path+table+"_testNSP_MLMMaskSentences.pt")
testLabels = torch.load(file_path+table+"_testNSP_MLMLabels.pt")
testMaskedTokenList = torch.load(file_path+table+"_testNSP_MLMMaskedTokenList.pt")
testMaskedIndexList = torch.load(file_path+table+"_testNSP_MLMMaskedIndexList.pt")

print("Train sentence count: "+str(len(testNSPSentences)))

#Start training

tokenInputIds = newGetTokenInputIds(colHeaderList)
tokenInputIds.pop('table')
#tokenInputIds.pop('spec')

if db == "imdb":
	model1 = Attention_Database_NSP(tokenCountDict,colHeaderList, embeddingDim, position_arg = False, loadPretrainEmbeddings = True, entityDict = tokenInputIds, nonTrainable = not(grad_back), embeddingPath = model_path+"actors_movie_sample_data_clean.bin", tie_weights=tie_weights).to(device) 
else:
	threshold="filtered_SUBJECTID_10.pkl"
	model1 = Attention_Database_NSP(tokenCountDict,colHeaderList, embeddingDim, position_arg = False, loadPretrainEmbeddings = True, entityDict = tokenInputIds, nonTrainable = not(grad_back), embeddingPath = model_path+'COMPLETE_DENORMALISED_drug_nospace_'+threshold+'_300.bin', tie_weights=tie_weights).to(device) 


model1.load_state_dict(torch.load(model_path+'nsp_mlm_'+table+'.pth'))
output = open(file_path+"test_nsp_mlm_ranking_"+table+".txt","w")


batchSize = int(batch_size_argument/4)
model1 = model1.to(device)
model1.eval()



batchCount = int(testNSPSentences.size(0)/batchSize)

if testNSPSentences.size(0)%batchSize != 0:
	batchCount +=1

for batchNum in range(batchCount):

	startIndex = batchNum * batchSize
	endIndex = min(((batchNum+1) * batchSize), testNSPSentences.size(0))	
		
	trainSeq = testNSPSentences[startIndex:endIndex]
	maskSeq = testMaskSentences[startIndex:endIndex]
	maskedTokens = testMaskedTokenList[startIndex:endIndex]
	maskedIndexes = testMaskedIndexList[startIndex:endIndex]
	senLabels = testLabels[startIndex:endIndex]
		# Just for MLM
		#predictions = model1.NSP(trainSeq,maskSeq)
		#loss = loss_function(predictions,senLabels.to(device))
		#For MLM and NSP simultaneously
	mlm_result, nsp_result = model1.MLM_NSP(trainSeq,maskSeq,maskedIndexes)
	predictions = mlm_result[torch.arange(mlm_result.size(0)), maskedTokens].view(-1,1)
	ranks = torch.sum((mlm_result > predictions), dim =-1)

	for i in range(len(maskedIndexes)):
		output.write(str(maskedIndexes[i].item())+"\t"+str(ranks[i].item())+"\n")


output.close()



