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
dict_path = "/home/garima/RelBERT/tmp/dict/"
sql_path = "/home/garima/relevant_files/"+db+"/sql_files/"
model_path = "/home/garima/relevant_files/"+db+"/model/"

#User input
embeddingDim = int(sys.argv[1])
batch_size_argument=int(sys.argv[2])
grad_back=bool(int(sys.argv[3])) # grad_back=0 for not finetuning embedding layer else 1
tie_weights=bool(int(sys.argv[4])) # tie_weights=0 for not having same weight for output softmax layer else 1
#maxLength = 
table = sys.argv[5] #table name
#negativeSampleCount = int(sys.argv[5]) #todo assign value from command line arg


def getNSP_MLMSentences(sentences, tmpColumnHeader, maskingRange, maxSentenceLength, table1, table2):

	tokenInputIds = newGetTokenInputIds(tmpColumnHeader)	
	MLMSentences = {}
	maskedSentences = []
	maskedTokensList = []
	maskedIndexList = []
	#sentenceLength = len(sentences[0].split("; "))
	seq1Length = len(sentences[0][0].split("; "))
	seq2Length = len(sentences[0][1].split("; "))
	sentenceLength = seq1Length + seq2Length + 3 #3 for 1 cls and 2 sep tokens

	print("sentence length: ", str(sentenceLength))

	tokenIdTab1 = tokenInputIds['table'][table1]
	tokenIdTab2 = tokenInputIds['table'][table2]
	tabSentence = [str(tokenIdTab1)]
	tabSentence *= int(seq1Length + 2)
	tabSentence += [str(tokenIdTab2)] * int(seq2Length + 1)
	tabSentence += ['<NONE>'] * (maxSentenceLength - len(tabSentence))

	tokenInNone = tokenInputIds['spec']['<NONE>']
	tokenInputIds.pop('table')
#	tokenInputIds.pop('spec')

	tokenCountDict = {}
	
#	for key in tokenInputIds:
	#	tokenInputIds[key]['<MASK>'] = len(tokenInputIds[key])  

	count = 0
	columnDict = {}

	for key in tmpColumnHeader:
		tokenCountDict[count] = len(tokenInputIds[key])
	#	columnDict[table+"."+col] = count
		count += 1
		
#	print("column dict: ", str(columnDict))
	
	for index in range(len(sentences)):
		
		tokenIndex = 0
		MLMSentence = []
		sentence = ['[CLS]']

		sentence += sentences[index][0].split("; ")
		sentence += ['[SEP]']
		sentence += sentences[index][1].split("; ")
		sentence += ['[SEP]']


		maskIndex = random.randint(1,maskingRange) #todo: cehck limit of the random generator
		mask = np.ones(maxSentenceLength)

		for colHeader in tmpColumnHeader:
					
			token = sentence[tokenIndex]
			tokenId = tokenInputIds[colHeader][token]

			if tokenIndex == maskIndex:
				maskedTokensList.append(tokenId)
				maskedIndexList.append(tokenIndex)
				tokenId = tokenInputIds[colHeader]["<MASK>"]
				mask[maskIndex] = 0

			elif token == "None": #For handling None tokens in the column
				mask[tokenIndex] = 0
				tokenId = tokenInputIds[colHeader]["<MASK>"]
									
			MLMSentence.append(tokenId)	
			tokenIndex +=1

		MLMSentence += [str(tokenInNone)] * (maxSentenceLength-len(MLMSentence))
		MLMSentences[index] = MLMSentence
		maskedSentences.append(mask)

	return MLMSentences, maskedSentences, maskedTokensList, maskedIndexList, tokenCountDict, tabSentence

def getNSPSentences(sentences, tmpColumnHeader, maxSentenceLength, table1, table2):

	tokenInputIds = newGetTokenInputIds(tmpColumnHeader)	
	NSPSentences = {}
	maskedSentences = []
	#maskedTokensList = []
	#maskedIndexList = []
	#sentenceLength = len(sentences[0].split("; "))
	seq1Length = len(sentences[0][0].split("; "))
	seq2Length = len(sentences[0][1].split("; "))
	sentenceLength = seq1Length + seq2Length + 3 #3 for 1 cls and 2 sep tokens

	print("sentence length: ", str(sentenceLength))

	tokenIdTab1 = tokenInputIds['table'][table1]
	tokenIdTab2 = tokenInputIds['table'][table2]
	tabSentence = [str(tokenIdTab1)]
	tabSentence *= int(seq1Length + 2)
	tabSentence += [str(tokenIdTab2)] * int(seq2Length + 1)
	tabSentence += ['<NONE>'] * (maxSentenceLength - len(tabSentence))

	tokenInNone = tokenInputIds['spec']['<NONE>']

	tokenInputIds.pop('table')
#	tokenInputIds.pop('spec')

	tokenCountDict = {}
	
#	for key in tokenInputIds:
	#	tokenInputIds[key]['<MASK>'] = len(tokenInputIds[key])  

	count = 0
	columnDict = {}

	for key in tmpColumnHeader:
		tokenCountDict[count] = len(tokenInputIds[key])
	#	columnDict[table+"."+col] = count
		count += 1
		
#	print("column dict: ", str(columnDict))
	
	for index in range(len(sentences)):
		
		tokenIndex = 0
		NSPSentence = []
		sentence = ['[CLS]']

		sentence += sentences[index][0].split("; ")
		sentence += ['[SEP]']
		sentence += sentences[index][1].split("; ")
		sentence += ['[SEP]']

		mask = np.ones(maxSentenceLength)

		for colHeader in tmpColumnHeader:
					
			token = sentence[tokenIndex]
			tokenId = tokenInputIds[colHeader][token]

			if token == "None": #For handling None tokens in the column
				mask[tokenIndex] = 0
				tokenId = tokenInputIds[colHeader]["<MASK>"]
									
			NSPSentence.append(tokenId)	
			tokenIndex +=1
		
		NSPSentence += [str(tokenInNone)] * (maxSentenceLength-len(NSPSentence))
		NSPSentences[index] = NSPSentence
		maskedSentences.append(mask)

	return NSPSentences, maskedSentences, tokenCountDict, tabSentence


def getTable(qStmt):

	if db == "imdb":
		conn = sqlite3.connect(sql_path+'csc455_pre1.db')
	else:
		conn = sqlite3.connect(sql_path+"mimic.db")	
	cursor = conn.cursor()

	quertStatement = qStmt
	cursor.execute(quertStatement)

	columnHeaderList = [description[0] for description in cursor.description]
	joinResult = []
	
	isFirst = True

	resultSetSize = 0
	for result in cursor:
		resultDict = {}
		resultSetSize +=1
	#	c = 0

		for i in range(len(columnHeaderList)):
			#if ((columnHeaderList[i] in t1ColumnList) or (columnHeaderList[i] in t2ColumnList)):
			resultDict[columnHeaderList[i]] = result[i].replace(" ","_")

#			resultDict[columnHeaderList[i]] = result[i]
			if DEBUG and isFirst:
				print("Chosen header: "+columnHeaderList[i])
			
		isFirst = False
		joinResult.append(resultDict)


	return joinResult, columnHeaderList


def convertSqlResultToSentence(joinResult, t1ColumnHeaderList, t2ColumnHeaderList, isPair):

	sentences = {} #  in case of sentnece pair: sentences[result_id] = [sentence of table 1, sentnece of table 2], otherwise will be sentneces[result_id] = sentence

	if isPair:

		for index in range(len(joinResult)):
			t1Sentence = ""
			t2Sentence = ""
			sentencePair = []

			#t1Sentence and t2Sentence will be strings that look like "col1Value; col2Value; ...; colnValue"
			for columnHeader in t1ColumnHeaderList:
				t1Sentence = t1Sentence + str(joinResult[index][columnHeader]) + "; "

			for columnHeader in t2ColumnHeaderList:
				t2Sentence = t2Sentence + str(joinResult[index][columnHeader]) + "; "
			
			#Remove the last ocurrence of "; " in t1Sentence and t2Sentence
			t1Sentence = t1Sentence[:t1Sentence.rfind("; ")]
			t2Sentence = t2Sentence[:t2Sentence.rfind("; ")]
			sentencePair.append(t1Sentence)
			sentencePair.append(t2Sentence)

			sentences[index] = sentencePair
	else:
			for index in range(len(joinResult)):
				t1Sentence = ""
				
				for columnHeader in t1ColumnHeaderList:
					t1Sentence = t1Sentence + str(joinResult[index][columnHeader]) + "; "

			#Remove the last ocurrence of ", " in t1Sentence 
				t1Sentence = t1Sentence[:t1Sentence.rfind("; ")]
				sentences[index] = t1Sentence


	return sentences

def addNegativeSamples(sentences, negativeSampleCount):

	negativeSampleIndex = len(sentences)
	negativeSentences = {}
	indexToChooseFrom = len(sentences)
	totalSentences = sentences.copy()

	for sentenceKey in sentences:
		sentence1 = sentences[sentenceKey][0]
		sentence2 = sentences[sentenceKey][1]
		chosenSampleCount = 0

		# adding negative samples corresponding to each sentence in sentences
		while chosenSampleCount < negativeSampleCount:
			
			chosenIndex = random.choice(range(indexToChooseFrom))

			if ((sentences[chosenIndex][0] != sentence1) and 
				(sentences[chosenIndex][1] != sentence2)):

				sentencePair = [sentence1, sentences[chosenIndex][1]]
				negativeSentences[negativeSampleIndex] = sentencePair
				negativeSampleIndex +=1
				chosenSampleCount +=1


	totalSentences.update(negativeSentences)

	return totalSentences

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
	


	

if DEBUG:
	print(qStmt)

result,resultHeaderList = getTable(qStmt)

t2ColumnHeaderList = []
for col in resultHeaderList:
	if col not in t1ColumnHeaderList:
		t2ColumnHeaderList.append(col)

if DEBUG:
	print("SQL result: ",str(result[0]))
	print("sQL header: ", str(resultHeaderList))

posSentencesPairs = convertSqlResultToSentence(result, t1ColumnHeaderList, t2ColumnHeaderList, True)

if DEBUG:
	print("SQL to Positive Sentence result: ",str(posSentencesPairs[0]))


positiveSentenceTheshold = len(posSentencesPairs)
#sentencePairs = addNegativeSamples(posSentencesPairs, negativeSampleCount)
sentencePairs = posSentencesPairs
labels = [1] * positiveSentenceTheshold
#labels += [0] * (len(sentencePairs) - positiveSentenceTheshold)

if DEBUG:
	print("SQL to Sentence result: ",str(sentencePairs[0]))

	print("Neg Sentence pair1: "+str(sentencePairs[positiveSentenceTheshold])+"\t"+str(labels[positiveSentenceTheshold]))
	print("Neg Sentence pair2: "+str(sentencePairs[positiveSentenceTheshold+1])+"\t"+str(labels[positiveSentenceTheshold+1]))
	print("Neg Sentence pair3: "+str(sentencePairs[positiveSentenceTheshold+2])+"\t"+str(labels[positiveSentenceTheshold+2]))
	print("Neg Sentence pair4: "+str(sentencePairs[positiveSentenceTheshold+3])+"\t"+str(labels[positiveSentenceTheshold+3]))
	print("Neg Sentence pair5: "+str(sentencePairs[positiveSentenceTheshold+4])+"\t"+str(labels[positiveSentenceTheshold+4]))

	print("SQL to Sentence result: ",str(sentencePairs[1]))

	print("Neg Sentence pair6: "+str(sentencePairs[positiveSentenceTheshold+5])+"\t"+str(labels[positiveSentenceTheshold+5]))


colHeaderList = []
maskingRange = len(t1ColumnHeaderList)
primaryTab = table1

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


if DEBUG:
	print("result headers: ",str(resultHeaderList))
	print("real header list: ",str(colHeaderList))

maxLength = len(colHeaderList)

if DEBUG:
	print("MAx seq length: "+ str(maxLength))

NSPSentencePairs, maskSentences, maskedTokenList, maskedIndexList, tokenCountDict, tabSentence = getNSP_MLMSentences(sentencePairs, colHeaderList, maskingRange, maxLength, table1, table2)

#NSPSentencePairs, maskSentences, tokenCountDict, tabSentence = getNSPSentences(sentencePairs, colHeaderList, maxLength, table1, table2)

if DEBUG:
	print(tokenCountDict)
if DEBUG:
	print("NSP sentence: ",str(NSPSentencePairs[0]))
	print("maskSentences: ", str(maskSentences[0]))
	print("Masked Token: ", str(maskedTokenList[0]))
	print("Masked Index: ", str(maskedIndexList[0]))
	print("Columns count: ", str(len(tokenCountDict.keys())))
	print("Masking range: ", str(maskingRange))

#Split and data into train,validate,test partitions

indexArray = list(range(len(NSPSentencePairs)))
random.shuffle(indexArray)
totalCount = len(NSPSentencePairs)
trainMax = int(0.7 * totalCount)
validMax = int(0.85 * totalCount)
testMax = totalCount
count = 0

trainNSPSentences = []
testNSPSentences = []
validNSPSentences = []

trainMaskSentences = []
trainLabels = []
trainMaskedTokenList = []
trainMaskedIndexList = []

validMaskSentences = []
validLabels = []
validMaskedTokenList = []
validMaskedIndexList = []

testMaskSentences = []
testLabels = []
testMaskedTokenList = []
testMaskedIndexList = []

for index in indexArray:
	if count < trainMax:
		trainNSPSentences.append(NSPSentencePairs[index])
		trainMaskSentences.append(maskSentences[index])
		trainLabels.append(labels[index])
		trainMaskedTokenList.append(maskedTokenList[index])
		trainMaskedIndexList.append(maskedIndexList[index])
	elif ((count >= trainMax) and (count < validMax)):
		validNSPSentences.append(NSPSentencePairs[index])
		validMaskSentences.append(maskSentences[index])
		validLabels.append(labels[index])

		validMaskedTokenList.append(maskedTokenList[index])
		validMaskedIndexList.append(maskedIndexList[index])
	else:
		testNSPSentences.append(NSPSentencePairs[index])
		testMaskSentences.append(maskSentences[index])
		testLabels.append(labels[index])

		testMaskedTokenList.append(maskedTokenList[index])
		testMaskedIndexList.append(maskedIndexList[index])
	count += 1


# Save the test set for later assesment
if DEBUG:
	print ("Training data:\n")
	print(str(len(trainNSPSentences)), str(len(trainMaskSentences)), str(len(trainMaskedTokenList)), str(len(trainMaskedIndexList)))

	print(str(len(trainNSPSentences)), str(len(trainMaskSentences)))

	print("Total: "+str(len(NSPSentencePairs)))
	print("Train: "+str(len(trainNSPSentences)))
	print("Validation: "+str(len(validNSPSentences)))
	print("Test: ",str(len(testNSPSentences)))

# convert all the arrays into torch.tensor
trainNSPSentences = torch.from_numpy(np.array(trainNSPSentences))
trainMaskSentences = torch.from_numpy(np.array(trainMaskSentences))
trainLabels = torch.from_numpy(np.array(trainLabels))
trainMaskedTokenList = torch.from_numpy(np.array(trainMaskedTokenList))
trainMaskedIndexList = torch.from_numpy(np.array(trainMaskedIndexList))

validNSPSentences = torch.from_numpy(np.array(validNSPSentences))
validMaskSentences = torch.from_numpy(np.array(validMaskSentences))
validLabels = torch.from_numpy(np.array(validLabels))

validMaskedTokenList  = torch.from_numpy(np.array(validMaskedTokenList))
validMaskedIndexList = torch.from_numpy(np.array(validMaskedIndexList))

testNSPSentences = torch.from_numpy(np.array(testNSPSentences))
testMaskSentences = torch.from_numpy(np.array(testMaskSentences))
testLabels = torch.from_numpy(np.array(testLabels))

testMaskedTokenList = torch.from_numpy(np.array(testMaskedTokenList))
testMaskedIndexList = torch.from_numpy(np.array(testMaskedIndexList))

#saving testing set for later testing
torch.save(testNSPSentences, file_path+table+"_testNSP_MLMSentences.pt")
torch.save(testMaskSentences,file_path+table+"_testNSP_MLMMaskSentences.pt")
torch.save(testLabels,file_path+table+"_testNSP_MLMLabels.pt")
torch.save(testMaskedTokenList,file_path+table+"_testNSP_MLMMaskedTokenList.pt")
torch.save(testMaskedIndexList,file_path+table+"_testNSP_MLMMaskedIndexList.pt")

#saving validation set
torch.save(validNSPSentences, file_path+table+"_validNSP_MLMSentences.pt")
torch.save(validMaskSentences,file_path+table+"_validNSP_MLMMaskSentences.pt")
torch.save(validLabels,file_path+table+"_validNSP_MLMLabels.pt")
torch.save(validMaskedTokenList,file_path+table+"_validNSP_MLMMaskedTokenList.pt")
torch.save(validMaskedIndexList,file_path+table+"_validNSP_MLMMaskedIndexList.pt")


#saving testing set for later testing
#torch.save(testNSPSentences, file_path+table+"_testNSPSentences.pt")
#torch.save(testMaskSentences,file_path+table+"_testNSPMaskSentences.pt")
#torch.save(testLabels,file_path+table+"_testNSPLabels.pt")
#torch.save(testMaskedTokenList,file_path+table+"_testNSPMaskedTokenList.pt")
#torch.save(testMaskedIndexList,file_path+table+"_testNSPMaskedIndexList.pt")

#saving validation set
#torch.save(validNSPSentences, file_path+table+"_validNSPSentences.pt")
#torch.save(validMaskSentences,file_path+table+"_validNSPMaskSentences.pt")
#torch.save(validLabels,file_path+table+"_validNSPLabels.pt")
#torch.save(validMaskedTokenList,file_path+table+"_validMaskedTokenList.pt")
#torch.save(validMaskedIndexList,file_path+table+"_validMaskedIndexList.pt")


#Start training

tokenInputIds = newGetTokenInputIds(colHeaderList)
tokenInputIds.pop('table')
#tokenInputIds.pop('spec')

if DEBUG:
	print("In nsp.py: tokenInputIds of spec: "+str(tokenInputIds['spec']))

if db == "imdb":
	model1 = Attention_Database_NSP(tokenCountDict,colHeaderList, embeddingDim, position_arg = False, loadPretrainEmbeddings = True, entityDict = tokenInputIds, nonTrainable = not(grad_back), embeddingPath = model_path+"actors_movie_sample_data_clean.bin", tie_weights=tie_weights).to(device) 
else: #mimic
	threshold="filtered_SUBJECTID_10.pkl"
	model1 = Attention_Database_NSP(tokenCountDict,colHeaderList, embeddingDim, position_arg = False, loadPretrainEmbeddings = True, entityDict = tokenInputIds, nonTrainable = not(grad_back), embeddingPath = model_path+'COMPLETE_DENORMALISED_drug_nospace_'+threshold+'_300.bin', tie_weights=tie_weights).to(device) 


epochs = 50
loss_function = nn.NLLLoss().to(device)
optimizer = optim.Adam(model1.parameters(), lr=0.001)
batchSize = int(batch_size_argument)
model1 = model1.to(device)
 	
for epoch in range(epochs):

	model1.train()

	batchCount = int(trainNSPSentences.size(0)/batchSize)

	if trainNSPSentences.size(0)%batchSize != 0:
		batchCount +=1

	for batchNum in range(batchCount):

		startIndex = batchNum * batchSize
		endIndex = min(((batchNum+1) * batchSize), trainNSPSentences.size(0))	
		
		trainSeq = trainNSPSentences[startIndex:endIndex]
		maskSeq = trainMaskSentences[startIndex:endIndex]
		maskedTokens = trainMaskedTokenList[startIndex:endIndex]
		maskedIndexes = trainMaskedIndexList[startIndex:endIndex]
		senLabels = trainLabels[startIndex:endIndex]
		# Just for MLM
		#predictions = model1.NSP(trainSeq,maskSeq)
		#loss = loss_function(predictions,senLabels.to(device))
		#For MLM and NSP simultaneously
		mlm_predictions, nsp_predictions = model1.MLM_NSP(trainSeq,maskSeq,maskedIndexes)
		mlm_loss = loss_function(mlm_predictions,maskedTokens.to(device))
		nsp_loss = loss_function(nsp_predictions, senLabels.to(device)) #NSP loss computation
		loss = mlm_loss + nsp_loss
		loss.backward()
		print("epoch = %d iteration = %d loss = %f"%(epoch, batchNum, loss))
		optimizer.step()
		model1.zero_grad()

	if ((epoch % 10 == 0) and (epoch != 0)):
		torch.save(model1.state_dict(), model_path+'nsp_mlm_'+str(epoch)+'_'+table+'.pth')


torch.save(model1.state_dict(), model_path+'nsp_mlm_'+table+'.pth')

#validate the learned model
'''
model1.eval()

results = model1.NSP(validNSPSentences, validMaskSentences)

predictions = results[torch.arange(results.size(0)), validLabels].view(-1,1)
ranks = torch.sum((results > predictions), dim =-1) #todo: compute accuracy

output = open(file_path+"validation_ranking_"+table,"w")

for i in range(len(validMaskedIndexList)):
	print(str(validMaskedIndexList[i].item())+"\t"+str(ranks[i].item()))
	output.write(str(validMaskedIndexList[i].item())+"\t"+str(ranks[i].item())+"\n")

output.close()
'''
