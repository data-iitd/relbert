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
from attention_model import make_model,PositionalEncoding
from utils import  create_attention_mask_from_input_mask,create_embedding_weight, create_embedding_weight_model, create_embedding_weight_model_conc, create_embedding_weight_matrix

from global_vars import get_sql_path, get_dict_path, get_model_path

'''

class RelBERTJ(nn.Module):
	
	#    @staticmethod
#	def __init__(self,count_dict,count_col,embedding_dim=100,arg_N=4,position_arg=True,
#					load_pre_train_embeddings=False,entity_dict=None,non_trainable = False,
#					embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
	
	def __init__(self, tokensPerColumnCountDict, tableCount, embeddingDim=100, arg_N=4,
				loadPretrainEmbeddings = False, entityDict = None, nonTrainable = False,
				embeddingPath = get_model_path()+"actors_nam_data_clean.bin", embeddingMatrix = None):
			
		super(RelBERTJ, self).__init__()
		
		self.embeddingDim=embeddingDim
		self.entityDict = entityDict
		self.tokensPerColumnCountDict = tokensPerColumnCountDict
		
		if loadPretrainEmbeddings:
			self.embeddings = nn.ModuleList([
											create_embedding_weight(embeddingPath, self.entityDict[s], embeddingDim, nonTrainable) #Retruns a token to embedding lookup table
											for s in self.entityDict
											])
		else:
			self.embeddings = embeddingMatrix
			#self.embeddings = nn.ModuleList([
			#									nn.Embedding(self.perColumnTokenCountDict[columnHeader], embeddingDim)
			#									for columnHeader in self.perColumnTokenCountDict
			#								])
			#My recommendation :def create_embedding_weight_matrix(embedding_matrix,entity_dict, embedding_dim,non_trainable):
			#self.embeddings = create_embedding_weight_matrix(embedding_matrix, entity_dict, embedding_dim, false)

		if tableCount > 1:
			self.tab_emb=nn.Embedding(tableCount, embeddingDim)
		
		self.transformer=make_model( N=arg_N, d_model=embeddingDim, d_ff=4*embeddingDim, h=4)
		self.cls=nn.Linear(embeddingDim, 2)
		
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim, self.tokensPerColumnCountDict[s])
			for s in self.tokensPerColumnCountDict
		])
		
		for k in range(len(self.tokensPerColumnCountDict)):
			self.decoder[k].weight = self.embeddings[k].weight


	def initialEmbeddingAssignment(NSPSentences, NSPTableSentences, 
								   t1ColumnHeader, t2ColumnHeader):

		NSPSentences = NSPSentences.to(device)
		sentenceCount = NSPSentences.shape[0]
		sentenceLength = NSPSentences.shape[1]
		embeddingArray=[]
		columnHeaderArray = ["spec"] + t1ColumnHeader + ["spec"] + t2ColumnHeader
		columnHeaderArray += ["spec"] * (sentenceLength - len(columnHeaderList))

		for tokenIndex in range(sentenceLength):
			embeddingArray.append(self.tokenEmbeddings[columnHeaderArray[tokenIndex]](sentence[:,tokenIndex]).to(device).reshape(sentenceCount,-1,self.embeddingDim))
	
		sentenceEmbeddingMatrix = torch.cat(embeddingArray,dim=1)
		sentenceEmbeddingMatrix = (sentenceEmbeddingMatrix + self.tab_emb(NSPTableSentences)).to(device).reshape(sentenceCount,-1,self.embeddingDim)
	
		return sentenceEmbeddingMatrix


	def NSP(self, NSPEmbeddedSentences, maskSentences):

		sentenceCount = len(NSPEmbeddedSentences)
		output = self.transformer.forward(NSPEmbeddedSentences, maskSentences)
		CLSEmbedding = self.cls(output[:,0])
		
		return CLSEmbedding.reshape(sentenceCount,-1)



	def MLM(self, sentences,index,non_index):

		sentences = sentences.to(device)
		sentenceCount=sentences.shape[0]
		#B=sentences.shape[1]
		x_array=[]
		
		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentences[:,k]).to(device).reshape(sentenceCount,-1,self.embeddingDim))
		
		x=torch.cat(x_array,dim=1)
		
		if self.position_arg:
			x=self.position(x)
		
		mask = torch.ones_like(sentences).to(device)
		mask[:,index]= torch.zeros(sentences.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentences).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.decoder[index](out[:,index])
		return self.m(res.view(sentenceCount,-1))



	def forward(self, sentence,index,non_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for k in self.count_dict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.decoder[index](out[:,index])
		return self.m(res.view(BAT,-1))

'''

class Attention_Database(nn.Module):
	
	#    @staticmethod
	def __init__(self,tokensPerColumnCountDict,columnList,embeddingDim=300,arg_N=4,position_arg=True,loadPretrainEmbeddings=False,entityDict=None,nonTrainable=False,embeddingPath=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		
		super(Attention_Database, self).__init__()
		self.tokensPerColumnCountDict = tokensPerColumnCountDict
		self.entityDict = entityDict
		self.embeddingDim = embeddingDim
		self.position_arg = position_arg
		self.outLength = 0
		self.columnList = columnList

		for col in tokensPerColumnCountDict:
			if tokensPerColumnCountDict[col] > self.outLength:
				self.outLength = tokensPerColumnCountDict[col]

		if loadPretrainEmbeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embeddingPath, self.entityDict[s], embeddingDim, nonTrainable)
				for s in self.entityDict
				#for s in self.columnList
			])

		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.tokensPerColumnCountDict[s], embeddingDim)
				for s in self.tokensPerColumnCountDict
				#for s in range(len(self.columnList))

			])

		self.position = PositionalEncoding(embeddingDim, 0)
		self.transformer = make_model( N=arg_N, d_model=embeddingDim, d_ff=4*embeddingDim, h=4)
		self.W1 = nn.Linear(embeddingDim, embeddingDim).to(device)
		
		self.decoder = nn.ModuleList([
			nn.Linear(embeddingDim,self.tokensPerColumnCountDict[s])
			for s in self.tokensPerColumnCountDict
			#for s in range(len(self.columnList))
		])
		
		if tie_weights:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		
		self.cls=nn.Linear(embeddingDim, 2)
		self.m = nn.Softmax(dim=-1)


	def forward(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
	#	for k in self.tokensPerColumnCountDict:
		for k in range(len(self.columnList)):
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
	#	print("size of mask and sentence: ",str(mask.size()), str(x.size()))

		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	print("output size: "+str(out.size()))
	#	print("MAx length: "+str(self.outLength))

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1

			#if int(res.size(0)) != self.tokensPerColumnCountDict[int(index)]:
			#	print("Error!!")
			#print("res size:"+ str(res.size(0))+"\t"+str(self.tokensPerColumnCountDict[int(index)]))

			#if sum(res == 0.0).bool():
			#	print("Zero value in softmax!!\t"+str(index)+"\t"+str(sum(res == 1.0).bool()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#if sum(softMaxed == 0.0).bool():
			#	print("Zero value in softmax!!\t"+str(index)+"\t"+str(sum(softMaxed == 1.0).bool()))
			#	print("Res Max: "+str(torch.max(res))+"\tMin: "+str(torch.min(res)))

			#	for j in range(softMaxed.size(0)):
			#		if float(softMaxed[j].item()) == 0.0:
			#			print("Res val:"+str(res[j]))
				#print("Res Val: "+res[])


			
		#	print("SoftMaxed size: "+ str(softMaxed.size()))
		#	tmp = torch.from_numpy(np.array([0.0]*(self.outLength - softMaxed.size(0)))).to(device)
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
	#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
		#	print("SoftMaxed size after padding: "+ str(softMaxed.size()))
			
			softMaxed = softMaxed.view(1,-1)
		#	print("SoftMaxed size after view: "+ str(softMaxed.size()))


			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)
		#	print("After concat SoftMaxed size: "+ str(len(pred))+ str(len(pred[0])))


		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
		#print("PRed output: "+str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions




	def NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		mask = mask.unsqueeze(-1)
		print("size of mask and sentence: ",str(mask.size()), str(x.size()))
		out = self.transformer.forward(x,mask) #pass the input from the encoder

		cls_out = self.cls(out[:,0]).to(device)
		nsp_predictions = self.m(cls_out.view(BAT,-1)).to(device)


		return  nsp_predictions



	def MLM_NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	nsp_out = self.cls(out[:,0]).to(device)
	#	nsp_predictions = self.m(nsp_out.view(BAT,-1)).to(device)

		nsp_predictions = self.cls(out[:,0]).to(device)

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1
			#print("res size:", str(res.size()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#print("SoftMaxed size:", str(softMaxed.size()))
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)

			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)

		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
	#	print("PRed output: ",str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions, nsp_predictions


#This class model takes NLP snetences constructed by templates as inputs
class Attention_Database_Sentences(nn.Module):
	
	#    @staticmethod
	def __init__(self,tokensPerColumnCountDict,columnList,embeddingDim=300,arg_N=4,position_arg=True,loadPretrainEmbeddings=False,entityDict=None,nonTrainable=False,embeddingPath=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		
		super(Attention_Database_Sentences, self).__init__()
		self.tokensPerColumnCountDict = tokensPerColumnCountDict
		self.entityDict = entityDict
		self.embeddingDim = embeddingDim
		self.position_arg = position_arg
		self.outLength = 0
		self.columnList = columnList

		for col in tokensPerColumnCountDict:
			if tokensPerColumnCountDict[col] > self.outLength:
				self.outLength = tokensPerColumnCountDict[col]

		self.embeddings_spec = 	create_embedding_weight(embeddingPath,self.entityDict['en'], embeddingDim,nonTrainable)
	#	self.entityDict.pop('spec')
		print("\n\n***Length of token count dict: "+str(len(tokensPerColumnCountDict))+"\t length of columnList: "+str(len(columnList)))
	
		if loadPretrainEmbeddings: 
			self.embeddings = nn.ModuleList()

			#Construct embedding array taking into account special tokens
			for col in range(len(self.columnList)):
				key = self.columnList[col]
			
				if key == 'en':
					self.embeddings.append(self.embeddings_spec)
				else:
					self.embeddings.append(create_embedding_weight(embeddingPath,self.entityDict[key], embeddingDim,nonTrainable))	
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.tokensPerColumnCountDict[s], embeddingDim)
				for s in self.tokensPerColumnCountDict
				#for s in range(len(self.columnList))

			])
		print("Size of embedding matrix: "+str(len(self.embeddings)))
		self.position = PositionalEncoding(embeddingDim, 0)
		self.transformer = make_model( N=arg_N, d_model=embeddingDim, d_ff=4*embeddingDim, h=4)
		self.W1 = nn.Linear(embeddingDim, embeddingDim).to(device)
		
		self.decoder = nn.ModuleList([
			nn.Linear(embeddingDim,self.tokensPerColumnCountDict[s])
			for s in self.tokensPerColumnCountDict
			#for s in range(len(self.columnList))
		])
		
		if tie_weights:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		
		self.cls=nn.Linear(embeddingDim, 2)
		self.m = nn.Softmax(dim=-1)


	def forward(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
	#	for k in self.tokensPerColumnCountDict:
		for k in range(len(self.columnList)):
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
	#	print("size of mask and sentence: ",str(mask.size()), str(x.size()))

		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	print("output size: "+str(out.size()))
	#	print("MAx length: "+str(self.outLength))

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1

			#if int(res.size(0)) != self.tokensPerColumnCountDict[int(index)]:
			#	print("Error!!")
			#print("res size:"+ str(res.size(0))+"\t"+str(self.tokensPerColumnCountDict[int(index)]))

			#if sum(res == 0.0).bool():
			#	print("Zero value in softmax!!\t"+str(index)+"\t"+str(sum(res == 1.0).bool()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#if sum(softMaxed == 0.0).bool():
			#	print("Zero value in softmax!!\t"+str(index)+"\t"+str(sum(softMaxed == 1.0).bool()))
			#	print("Res Max: "+str(torch.max(res))+"\tMin: "+str(torch.min(res)))

			#	for j in range(softMaxed.size(0)):
			#		if float(softMaxed[j].item()) == 0.0:
			#			print("Res val:"+str(res[j]))
				#print("Res Val: "+res[])


			
		#	print("SoftMaxed size: "+ str(softMaxed.size()))
		#	tmp = torch.from_numpy(np.array([0.0]*(self.outLength - softMaxed.size(0)))).to(device)
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
	#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
		#	print("SoftMaxed size after padding: "+ str(softMaxed.size()))
			
			softMaxed = softMaxed.view(1,-1)
		#	print("SoftMaxed size after view: "+ str(softMaxed.size()))


			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)
		#	print("After concat SoftMaxed size: "+ str(len(pred))+ str(len(pred[0])))


		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
		#print("PRed output: "+str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions




	def NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		mask = mask.unsqueeze(-1)
		print("size of mask and sentence: ",str(mask.size()), str(x.size()))
		out = self.transformer.forward(x,mask) #pass the input from the encoder

		cls_out = self.cls(out[:,0]).to(device)
		nsp_predictions = self.m(cls_out.view(BAT,-1)).to(device)


		return  nsp_predictions



	def MLM_NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	nsp_out = self.cls(out[:,0]).to(device)
	#	nsp_predictions = self.m(nsp_out.view(BAT,-1)).to(device)

		nsp_predictions = self.cls(out[:,0]).to(device)

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1
			#print("res size:", str(res.size()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#print("SoftMaxed size:", str(softMaxed.size()))
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)

			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)

		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
	#	print("PRed output: ",str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions, nsp_predictions



class Attention_Database_NSP(nn.Module):
	
	#    @staticmethod
	def __init__(self,tokensPerColumnCountDict,columnList,embeddingDim=300,arg_N=4,position_arg=True,loadPretrainEmbeddings=False,entityDict=None,nonTrainable=False,embeddingPath=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		
		super(Attention_Database_NSP, self).__init__()
		self.tokensPerColumnCountDict = tokensPerColumnCountDict
		self.entityDict = entityDict
		self.embeddingDim = embeddingDim
		self.position_arg = position_arg
		self.outLength = 0
		self.columnList = columnList

		print("CHECK keys of entity dict: "+str(self.entityDict.keys()) + "\t"+str(self.entityDict['spec']))
		self.embeddings_spec = 	create_embedding_weight(embeddingPath,self.entityDict['spec'], embeddingDim,nonTrainable)
	#	self.entityDict.pop('spec')

		for col in tokensPerColumnCountDict:
			if tokensPerColumnCountDict[col] > self.outLength:
				self.outLength = tokensPerColumnCountDict[col]

		if loadPretrainEmbeddings: 
			self.embeddings = nn.ModuleList()

			#Construct embedding array taking into account special tokens
			for col in range(len(self.columnList)):
				key = self.columnList[col]
			
				if key == 'spec':
					self.embeddings.append(self.embeddings_spec)
				else:
					self.embeddings.append(create_embedding_weight(embeddingPath,self.entityDict[key], embeddingDim,nonTrainable))	
		
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.tokensPerColumnCountDict[s], embeddingDim)
				for s in self.tokensPerColumnCountDict
				#for s in range(len(self.columnList))

			])

		self.position = PositionalEncoding(embeddingDim, 0)
		self.transformer = make_model( N=arg_N, d_model=embeddingDim, d_ff=4*embeddingDim, h=4)
		self.W1 = nn.Linear(embeddingDim, embeddingDim).to(device)
		#self.tab_emb=nn.Embedding(count_col, embedding_dim)

		self.decoder = nn.ModuleList([
			nn.Linear(embeddingDim,self.tokensPerColumnCountDict[s])
			for s in self.tokensPerColumnCountDict
			#for s in range(len(self.columnList))
		])
		
		if tie_weights:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		
		self.cls=nn.Linear(embeddingDim, 2)
		self.m = nn.Softmax(dim=-1)



	def NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		colIndex = 0

		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		mask = mask.unsqueeze(-1)
		print("size of mask and sentence: ",str(mask.size()), str(x.size()))
		out = self.transformer.forward(x,mask) #pass the input from the encoder

		cls_out = self.cls(out[:,0]).to(device)
		nsp_predictions = self.m(cls_out.view(BAT,-1)).to(device)

			
		return  nsp_predictions



	def MLM_NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
	#	print("After size of mask and sentence: ",str(mask.size()), str(x.size()))

		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	nsp_out = self.cls(out[:,0]).to(device)
	#	nsp_predictions = self.m(nsp_out.view(BAT,-1)).to(device)

		nsp_predictions = self.cls(out[:,0]).to(device)

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1
			#print("res size:", str(res.size()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#print("SoftMaxed size:", str(softMaxed.size()))
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)

			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)

		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
	#	print("PRed output: ",str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions, nsp_predictions

	#For finetuning
	def MLM(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
	#	print("After size of mask and sentence: ",str(mask.size()), str(x.size()))

		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	nsp_out = self.cls(out[:,0]).to(device)
	#	nsp_predictions = self.m(nsp_out.view(BAT,-1)).to(device)

	#	nsp_predictions = self.cls(out[:,0]).to(device)

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1
			#print("res size:", str(res.size()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#print("SoftMaxed size:", str(softMaxed.size()))
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)

			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)

		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
	#	print("PRed output: ",str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions



	def forward(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
	#	for k in self.tokensPerColumnCountDict:
		for k in range(len(self.columnList)):
				x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
		mask = mask.unsqueeze(-1)
		out = self.transformer.forward(x,mask) #pass the input from the encoder
		i = 0
		pred = []
		
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1

			softMaxed = res.to(device)
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)
			pred.append(softMaxed)


		predictions = torch.cat(pred,dim=0)

		return predictions


#For single latent space handling


class Attention_Database_NSP_SS(nn.Module):
	
	#    @staticmethod
	def __init__(self,tokensCount,embeddingDim=300,arg_N=4,position_arg=True,loadPretrainEmbeddings=False,entityDict=None,nonTrainable=False,embeddingPath=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		
		super(Attention_Database_NSP_SS, self).__init__()
#		self.tokensPerColumnCountDict = tokensPerColumnCountDict
		self.entityDict = entityDict
		self.embeddingDim = embeddingDim
		self.position_arg = position_arg
		self.outLength = 0
	

	#	print("CHECK keys of entity dict: "+str(self.entityDict.keys()) + "\t"+str(self.entityDict['spec']))
	#	self.embeddings_spec = 	create_embedding_weight(embeddingPath,self.entityDict['spec'], embeddingDim,nonTrainable)
	#	self.entityDict.pop('spec')

#		for col in tokensPerColumnCountDict:
#			if tokensPerColumnCountDict[col] > self.outLength:
#				self.outLength = tokensPerColumnCountDict[col]
		self.outLength = tokensCount;

		if loadPretrainEmbeddings: 
                 self.embeddings = create_embedding_weight(embeddingPath,entityDict, embeddingDim,nonTrainable) #fetch learned embedding from the embedding_path		
		else:
		  self.embeddings = nn.Embedding(len(entityDict), embedding_dim)

		self.position = PositionalEncoding(embeddingDim, 0)
		self.transformer = make_model( N=arg_N, d_model=embeddingDim, d_ff=4*embeddingDim, h=4)
		self.W1 = nn.Linear(embeddingDim, embeddingDim).to(device)
		#self.tab_emb=nn.Embedding(count_col, embedding_dim)

		self.decoder = nn.Linear(embeddingDim, tokensCount)
		self.decoder.weight = self.embeddings.weight
		
		
	#	if tie_weights:
	#		for k in range(len(self.columnList)):
	#			self.decoder[k].weight = self.embeddings[k].weight
	#	else:
	#		for k in range(len(self.columnList)):
	#			self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		
		self.cls=nn.Linear(embeddingDim, 2)
		self.m = nn.Softmax(dim=-1)



	def NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		colIndex = 0

		#for k in self.columnList:
		#	x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		#x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding

		x=(self.embeddings(sentence)).to(device).reshape(BAT,-1,self.embeddingDim)
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		mask = mask.unsqueeze(-1)
		print("size of mask and sentence: ",str(mask.size()), str(x.size()))
		out = self.transformer.forward(x,mask) #pass the input from the encoder

		cls_out = self.cls(out[:,0]).to(device)
		nsp_predictions = self.m(cls_out.view(BAT,-1)).to(device)

			
		return  nsp_predictions



	def MLM_NSP(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		#for k in self.columnList:

		#	x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		#x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		x=(self.embeddings(sentence)).to(device).reshape(BAT,-1,self.embeddingDim)
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
	#	print("After size of mask and sentence: ",str(mask.size()), str(x.size()))

		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	nsp_out = self.cls(out[:,0]).to(device)
	#	nsp_predictions = self.m(nsp_out.view(BAT,-1)).to(device)

		nsp_predictions = self.cls(out[:,0]).to(device)

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder(out[i, index])
			i += 1
			#print("res size:", str(res.size()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#print("SoftMaxed size:", str(softMaxed.size()))
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)

			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)

		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
	#	print("PRed output: ",str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions, nsp_predictions

	#For finetuning
	def MLM(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		for k in self.coulmnList:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		#mask = torch.ones_like(sentence).to(device)
		#mask[:,index] = torch.zeros(sentence.shape[0]).to(device)
		#mask1 = torch.where(non_index == 0, torch.zeros_like(sentence).to(device), mask)
		#mask = torch.tensor(mask)
		mask = mask.unsqueeze(-1)
	#	print("After size of mask and sentence: ",str(mask.size()), str(x.size()))

		out = self.transformer.forward(x,mask) #pass the input from the encoder

	#	nsp_out = self.cls(out[:,0]).to(device)
	#	nsp_predictions = self.m(nsp_out.view(BAT,-1)).to(device)

	#	nsp_predictions = self.cls(out[:,0]).to(device)

		i = 0
		pred = []
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1
			#print("res size:", str(res.size()))

			#softMaxed = self.m(res).to(device)
			softMaxed = res.to(device)
			#print("SoftMaxed size:", str(softMaxed.size()))
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			#softMaxed += [0.0] * (self.outLength - len(softMaxed))
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)

			#print("After concat SoftMaxed size:", str(softMaxed.size()))
			pred.append(softMaxed)

		predictions = torch.cat(pred,dim=0)

		#res = self.decoder[index](out[:,index])
		#return self.m(res.view(BAT,-1))
	#	print("PRed output: ",str(predictions.size()))
		#return torch.from_numpy(np.array(pred))
		return predictions



	def forward(self, sentence, mask, maskedIndex):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
	#	for k in self.tokensPerColumnCountDict:
		for k in range(len(self.columnList)):
				x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
		mask = mask.unsqueeze(-1)
		out = self.transformer.forward(x,mask) #pass the input from the encoder
		i = 0
		pred = []
		
		for index in maskedIndex:
			res = self.decoder[index](out[i, index])
			i += 1

			softMaxed = res.to(device)
			neg_infinity = float('-inf')
			tmp = torch.from_numpy(np.array([neg_infinity]*(self.outLength - softMaxed.size(0)))).to(device)
			softMaxed = torch.cat((softMaxed,tmp),0)
			softMaxed = softMaxed.view(1,-1)
			pred.append(softMaxed)


		predictions = torch.cat(pred,dim=0)

		return predictions




#Later change name of the class swap it with the pervious one; this one is to give functionality 
#of MLM followed by NSP

class Attention_Database_MLM_NSP(nn.Module):
	
	#    @staticmethod
	def __init__(self,tokensPerColumnCountDict,columnList,embeddingDim=300,arg_N=4,position_arg=True,loadPretrainEmbeddings=False,entityDict=None,nonTrainable=False,embeddingPath=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True,load_pre_train_model=None):
		
		super(Attention_Database_MLM_NSP, self).__init__()
		self.tokensPerColumnCountDict = tokensPerColumnCountDict
		self.entityDict = entityDict
		self.embeddingDim = embeddingDim
		self.position_arg = position_arg
		self.outLength = 0
		self.columnList = columnList

		print("CHECK keys of entity dict: "+str(self.entityDict.keys()) + "\t"+str(self.entityDict['spec']))
		self.embeddings_spec = 	create_embedding_weight(embeddingPath,self.entityDict['spec'], embeddingDim,nonTrainable)
	#	self.entityDict.pop('spec')

		for col in tokensPerColumnCountDict:
			if tokensPerColumnCountDict[col] > self.outLength:
				self.outLength = tokensPerColumnCountDict[col]

		if load_pre_train_model is not None:
			self.embeddings = nn.ModuleList()

			#Construct embedding array taking into account special tokens
			for col in range(len(self.columnList)):
				key = self.columnList[col]
			
				if key == 'spec':
					self.embeddings.append(self.embeddings_spec)
				else:
					self.embeddings.append(nn.Embedding.from_pretrained(torch.FloatTensor(load_pre_train_model[key])))

		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.tokensPerColumnCountDict[s], embeddingDim)
				for s in self.tokensPerColumnCountDict
			])

		self.position = PositionalEncoding(embeddingDim, 0)
		self.transformer = make_model( N=arg_N, d_model=embeddingDim, d_ff=4*embeddingDim, h=4)
		self.W1 = nn.Linear(embeddingDim, embeddingDim).to(device)
		#self.tab_emb=nn.Embedding(count_col, embedding_dim)

		self.decoder = nn.ModuleList([
			nn.Linear(embeddingDim,self.tokensPerColumnCountDict[s])
			for s in self.tokensPerColumnCountDict
			#for s in range(len(self.columnList))
		])
		
		if tie_weights:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.tokensPerColumnCountDict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		
		self.cls=nn.Linear(embeddingDim, 2)
		self.m = nn.Softmax(dim=-1)



	def NSP(self, sentence, mask):

		sentence = sentence.to(device) #sentence is a np.array
		BAT = sentence.shape[0] # number of sentences
		B = sentence.shape[1]	#length of the sentences
		x_array = []
		#adding for trail
		mask = mask.to(device)
	
		colIndex = 0

		for k in self.tokensPerColumnCountDict:
			x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embeddingDim))
			#x_array.append(self.embeddings[k](sentence[:,k]).reshape(BAT,-1,self.embeddingDim))

		x = torch.cat(x_array,dim=1) #substituting tokens with their respective embedding
		
		if self.position_arg:
			x = self.position(x)
		
	#	print("Before size of mask and sentence: ",str(mask.size()), str(x.size()))

		mask = mask.unsqueeze(-1)
		print("size of mask and sentence: ",str(mask.size()), str(x.size()))
		out = self.transformer.forward(x,mask) #pass the input from the encoder

		cls_out = self.cls(out[:,0]).to(device)
		nsp_predictions = self.m(cls_out.view(BAT,-1)).to(device)

			
		return  nsp_predictions


