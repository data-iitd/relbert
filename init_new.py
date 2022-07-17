#Open configuration file and load correct dictionaries with database schema

#Write the configuration file
import sqlite3
from sqlite3 import OperationalError
import pickle
from global_vars import get_sql_path, get_dict_path, get_model_path
from data_init import constructTokenToInputDict, createCleanTables

def loadConfig():
	
	dbName=""
	tableList = []
	columnDict = {}

#	file = open("config_mimic.txt", 'r') #todo:define file_path
	file = open("config.txt", 'r') #todo:define file_path

	lines = file.readlines() #todo:check readlines fucntion

	for line in lines:
		line = line.strip()
		lineSplit = line.split(' = ')

		if lineSplit[0] == 'db_name':
			dbName = lineSplit[1] #Todo: add removing trailing and leading zeros
		
		elif lineSplit[0] == 'tables_name':
			tableList = lineSplit[1].split(', ')
		
		elif lineSplit[0] == 'columns_header':
			colList = lineSplit[1].split(', ')

			for col in colList:
				colSplit = col.split(":")
				tabName = colSplit[0]
				colName = colSplit[1]

				if tabName not in columnDict:
					columnDict[tabName] = []
				
				columnDict[tabName].append(colName)

	#todo: load the inputIdsMatrix  from the file stored by data_init after construction
	#tokenToInputIdDict=pickle.load(open("entity_dict.pkl","rb"))
	print("DBName: "+dbName)
	return dbName, tableList,columnDict




#db, tableList, colHeaderList = loadConfig()
#print(db)
#print(str(tableList))
#print(str(colHeaderList.values()))
# called once to construct the tokenToInputIds map
#constructTokenToInputDict(colHeaderList)


#Load the input ids dictionary -- tokenToInputIdDict

def getTokenInputIds(colHeaderDict):
	
	tokenIdsDict = {}
	inputIdsMatrix = pickle.load(open("entity_dict.pkl","rb"))

	for table in colHeaderDict:
		for col in colHeaderDict[table]:	
			tokenIdsDict[table+"."+col] = inputIdsMatrix[table+"."+col]
			tokenIdsDict[table+"."+col]['<MASK>'] = len(tokenIdsDict[table+"."+col])  

	
	tokenIdsDict['table'] = inputIdsMatrix['table']
	tokenIdsDict['spec'] = inputIdsMatrix['spec']

	return tokenIdsDict

def newGetTokenInputIds(colHeaderList):
	
	tokenIdsDict = {}
	dict_path= get_dict_path()
	inputIdsMatrix = pickle.load(open(dict_path+"entity_dict.pkl","rb"))
	print(str(inputIdsMatrix.keys()))
	for key in colHeaderList:
		if ((key != 'spec') and (key != 'table') and (key != 'en')):
			tokenIdsDict[key] = inputIdsMatrix[key]
			tokenIdsDict[key]['<MASK>'] = len(tokenIdsDict[key])  

	tokenIdsDict['table'] = inputIdsMatrix['table']
	tokenIdsDict['spec'] = inputIdsMatrix['spec']


	return tokenIdsDict

