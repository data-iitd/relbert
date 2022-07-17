import torch
from torch import nn
from torch.nn import functional as F
import math, copy
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
from utils import create_embedding_weight_EmbDi, create_attention_mask_from_input_mask,create_embedding_weight, create_embedding_weight_model, create_embedding_weight_model_conc, create_embedding_weight_matrix

from global_vars import get_sql_path, get_dict_path, get_model_path


class Attention_Database_NSP_MLM_diff(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,count_tab,count_col,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_NSP_MLM_diff, self).__init__()
		self.entity_dict=entity_dict
		self.count_dict=count_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable) #Retruns a token to embedding lookup table
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.tab_emb=nn.Embedding(count_col, embedding_dim)
# 		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
# 		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
# 		self.activation = nn.Tanh()
		self.cls=nn.Linear(embedding_dim,2)
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		for k in range(len(self.count_dict)):
			self.decoder[k].weight = self.embeddings[k].weight

	def forward(self, sentence,col_sentence,table_sentence,non_index,mask_index,index_array):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for index_k in range(B):
			x_array.append(self.embeddings[index_array[index_k]](sentence[:,index_k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		x=(x+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		res_MLM=self.decoder[index_array[mask_index]](out[:,mask_index])
		return res.reshape(BAT,-1),res_MLM.view(BAT,-1)
	
	def forward_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index,index_array):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for index_k in range(B):
			x_array.append(self.embeddings[index_array[index_k]](sentence[:,index_k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		x=(x+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res_MLM=self.decoder[index_array[mask_index]](out[:,mask_index])
		return res_MLM.view(BAT,-1)

	def forward_NSP(self, sentence,col_sentence,table_sentence,non_index,index_array):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for index_k in range(B):
			x_array.append(self.embeddings[index_array[index_k]](sentence[:,index_k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		x=(x+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)

class Attention_Database_NSP_MLM_col_diff(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,count_tab,count_col,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_NSP_MLM_col_diff, self).__init__()
		self.entity_dict=entity_dict
		self.count_dict=count_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.tab_emb=nn.Embedding(count_col, embedding_dim)
		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
# 		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
# 		self.activation = nn.Tanh()
		self.cls=nn.Linear(embedding_dim,2)
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		for k in range(len(self.count_dict)):
			self.decoder[k].weight = self.embeddings[k].weight

	def forward(self, sentence,col_sentence,table_sentence,non_index,mask_index,index_array):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for index_k in range(B):
			x_array.append(self.embeddings[index_array[index_k]](sentence[:,index_k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		x=(x+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		res_MLM=self.decoder[index_array[mask_index]](out[:,mask_index])
		return res.reshape(BAT,-1),res_MLM.view(BAT,-1)
	
	def forward_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index,index_array):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for index_k in range(B):
			x_array.append(self.embeddings[index_array[index_k]](sentence[:,index_k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		x=(x+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res_MLM=self.decoder[index_array[mask_index]](out[:,mask_index])
		return res_MLM.view(BAT,-1)

	def forward_NSP(self, sentence,col_sentence,table_sentence,non_index,index_array):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for index_k in range(B):
			x_array.append(self.embeddings[index_array[index_k]](sentence[:,index_k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		x=(x+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)

class Attention_Database_NSP_MLM_multiple(nn.Module):
	#    @staticmethod

	def __init__(self,count_tab,count_col,count_ENT,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True,total_entity_dict=None):
		super(Attention_Database_NSP_MLM_multiple, self).__init__()
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = create_embedding_weight(embedding_path,total_entity_dict, embedding_dim,non_trainable)
		else:
			self.embeddings = nn.Embedding(count_ENT, embedding_dim)
		self.tab_emb=nn.Embedding(count_col, embedding_dim)
		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
		self.cls=nn.Linear(embedding_dim,2)
		self.decoder = nn.Linear(embedding_dim, count_ENT)
		self.decoder.weight = self.embeddings.weight
		self.m = nn.LogSoftmax(dim=-1)

	def forward(self, sentence,col_sentence,table_sentence,non_index,mask_index_array):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		for mask_index in mask_index_array:
			mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		res_MLM_array=[]
		for mask_index in mask_index_array:        
			res_MLM_array.append(self.decoder(out[:,mask_index]).view(BAT,1,-1))
		res_MLM_cat=torch.cat(res_MLM_array,axis=1)
		return res.reshape(BAT,-1),self.m(res_MLM_cat)
	
	def forward_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res_MLM=self.decoder(out[:,mask_index])
		return self.m(res_MLM.view(BAT,-1))
	
	def forward_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)


class Attention_Database_NSP_MLM(nn.Module):
	
	#    @staticmethod
	def __init__(self,count_tab,count_col,count_ENT,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
	
		super(Attention_Database_NSP_MLM, self).__init__()
		
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = create_embedding_weight(embedding_path,entity_dict, embedding_dim,non_trainable) #fetch learned embedding from the embedding_path
		else:
			self.embeddings = nn.Embedding(count_ENT, embedding_dim)

		self.tab_emb=nn.Embedding(count_col, embedding_dim)
		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
		self.cls=nn.Linear(embedding_dim,2)
		self.decoder = nn.Linear(embedding_dim, count_ENT)
		self.decoder.weight = self.embeddings.weight
		self.m = nn.LogSoftmax(dim=-1)

	def forward(self, sentence,col_sentence,table_sentence,non_index,mask_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		res_MLM=self.decoder(out[:,mask_index])
		return res.reshape(BAT,-1),self.m(res_MLM.view(BAT,-1))
	
	def forward_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0] #number of data points
		B=sentence.shape[1] # number of tokens in a sentence 
		#adding emebeddings of tokens, column ids and respective table ids
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device) #mask all toekns at index mask_index
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask) #masking all the indexes which have toekn "None"
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res_MLM=self.decoder(out[:,mask_index])
		return self.m(res_MLM.view(BAT,-1))

	def forward_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)
	
	def infer_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.interpret(x,mask1)
		return out 
	
	def infer_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.interpret(x,mask1)
		return out

class Attention_Database_NSP_MLM_vert(nn.Module):
	#    @staticmethod

	def __init__(self,count_tab,count_col,count_ENT,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_NSP_MLM_vert, self).__init__()
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = create_embedding_weight(embedding_path,entity_dict, embedding_dim,non_trainable)
		else:
			self.embeddings = nn.Embedding(count_ENT, embedding_dim)
		self.tab_emb=nn.Embedding(count_col, embedding_dim)
		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.vert_transformer=make_model( N=2, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
		self.cls=nn.Linear(embedding_dim,2)
		self.decoder = nn.Linear(embedding_dim, count_ENT)
		self.decoder.weight = self.embeddings.weight
		self.m = nn.LogSoftmax(dim=-1)

	def forward(self, sentence,col_sentence,table_sentence,non_index,mask_index,vertical_sentences,non_vertical_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		batch_size=vertical_sentences.shape[1]
		out_arr=[out[:batch_size,mask_index].reshape(batch_size,1,-1)]
		mask1_arr=[torch.ones_like(sentence[:batch_size,mask_index]).reshape(batch_size,1,-1).to(device)]
		for k in range(vertical_sentences.shape[0]):
			x=(self.embeddings(vertical_sentences[k])+self.col_emb(col_sentence[:batch_size])+self.tab_emb(table_sentence[:batch_size])).to(device).reshape(int(BAT/2),-1,self.embedding_dim)
			mask = torch.ones_like(vertical_sentences[k]).to(device)
			mask2=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask[:,mask_index]= torch.zeros(vertical_sentences[k].shape[0]).to(device)
			mask1=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask1=mask1.unsqueeze(-2)
			out_sen=self.transformer.forward(x,mask1)
			outk=out_sen[:,mask_index]
			mask1_arr.append(mask2[:,mask_index].reshape(-1,1,1))
			out_arr.append(outk.reshape(batch_size,1,-1))
		out_arr_tens=torch.cat(out_arr,dim=1)
		mask1_arr_tens=torch.cat(mask1_arr,dim=2)
		res_MLM=self.decoder(self.vert_transformer.forward(out_arr_tens,mask1_arr_tens)[:,0])
		return res.reshape(BAT,-1),self.m(res_MLM.view(batch_size,-1))

	def forward_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index,vertical_sentences,non_vertical_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		batch_size=vertical_sentences.shape[1]
		out_arr=[out[:batch_size,mask_index].reshape(batch_size,1,-1)]
		mask1_arr=[torch.ones_like(sentence[:batch_size,mask_index]).reshape(batch_size,1,-1).to(device)]
		for k in range(vertical_sentences.shape[0]):
			x=(self.embeddings(vertical_sentences[k])+self.col_emb(col_sentence[:batch_size])+self.tab_emb(table_sentence[:batch_size])).to(device).reshape(int(BAT),-1,self.embedding_dim)
			mask = torch.ones_like(vertical_sentences[k]).to(device)
			mask2=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask[:,mask_index]= torch.zeros(vertical_sentences[k].shape[0]).to(device)
			mask1=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask1=mask1.unsqueeze(-2)
			out_sen=self.transformer.forward(x,mask1)
			outk=out_sen[:,mask_index]
			mask1_arr.append(mask2[:,mask_index].reshape(-1,1,1))
			out_arr.append(outk.reshape(batch_size,1,-1))
		out_arr_tens=torch.cat(out_arr,dim=1)
		mask1_arr_tens=torch.cat(mask1_arr,dim=2)
		res_MLM=self.decoder(self.vert_transformer.forward(out_arr_tens,mask1_arr_tens)[:,0])
		return self.m(res_MLM.view(BAT,-1))

	def forward_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)

	def test_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res_MLM=self.decoder(out[:,mask_index])
		return self.m(res_MLM.view(BAT,-1))

	def infer_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.interpret(x,mask1)
		return out

class Attention_Database_NSP_MLM_vert_mean(nn.Module):
	#    @staticmethod

	def __init__(self,count_tab,count_col,count_ENT,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_NSP_MLM_vert_mean, self).__init__()
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = create_embedding_weight(embedding_path,entity_dict, embedding_dim,non_trainable)
		else:
			self.embeddings = nn.Embedding(count_ENT, embedding_dim)
		self.tab_emb=nn.Embedding(count_col, embedding_dim)
		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
		self.cls=nn.Linear(embedding_dim,2)
		self.decoder = nn.Linear(embedding_dim, count_ENT)
		self.decoder.weight = self.embeddings.weight
		self.m = nn.LogSoftmax(dim=-1)

	def forward(self, sentence,col_sentence,table_sentence,non_index,mask_index,vertical_sentences,non_vertical_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		batch_size=vertical_sentences.shape[1]
		out_arr=[out[:batch_size,mask_index].reshape(batch_size,1,-1)]
		mask1_arr=[torch.ones_like(sentence[:batch_size,mask_index]).reshape(batch_size,1,-1).to(device)]
		for k in range(vertical_sentences.shape[0]):
			x=(self.embeddings(vertical_sentences[k])+self.col_emb(col_sentence[:batch_size])+self.tab_emb(table_sentence[:batch_size])).to(device).reshape(int(BAT/2),-1,self.embedding_dim)
			mask = torch.ones_like(vertical_sentences[k]).to(device)
			mask2=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask[:,mask_index]= torch.zeros(vertical_sentences[k].shape[0]).to(device)
			mask1=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask1=mask1.unsqueeze(-2)
			out_sen=self.transformer.forward(x,mask1)
			outk=out_sen[:,mask_index]
			mask1_arr.append(mask2[:,mask_index].reshape(-1,1,1))
			out_arr.append(outk.reshape(batch_size,1,-1))
		out_arr_tens=torch.cat(out_arr,dim=1)
		out_arr_mean=torch.mean(out_arr_tens,dim=1)
		res_MLM=self.decoder(out_arr_mean)
		return res.reshape(BAT,-1),self.m(res_MLM.view(batch_size,-1))

	def forward_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index,vertical_sentences,non_vertical_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		batch_size=vertical_sentences.shape[1]
		out_arr=[out[:batch_size,mask_index].reshape(batch_size,1,-1)]
		mask1_arr=[torch.ones_like(sentence[:batch_size,mask_index]).reshape(batch_size,1,-1).to(device)]
		for k in range(vertical_sentences.shape[0]):
			x=(self.embeddings(vertical_sentences[k])+self.col_emb(col_sentence[:batch_size])+self.tab_emb(table_sentence[:batch_size])).to(device).reshape(int(BAT),-1,self.embedding_dim)
			mask = torch.ones_like(vertical_sentences[k]).to(device)
			mask2=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask[:,mask_index]= torch.zeros(vertical_sentences[k].shape[0]).to(device)
			mask1=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask1=mask1.unsqueeze(-2)
			out_sen=self.transformer.forward(x,mask1)
			outk=out_sen[:,mask_index]
			mask1_arr.append(mask2[:,mask_index].reshape(-1,1,1))
			out_arr.append(outk.reshape(batch_size,1,-1))
		out_arr_tens=torch.cat(out_arr,dim=1)
		out_arr_mean=torch.mean(out_arr_tens,dim=1)
		res_MLM=self.decoder(out_arr_mean)
		return self.m(res_MLM.view(BAT,-1))

	def forward_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)

	def test_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res_MLM=self.decoder(out[:,mask_index])
		return self.m(res_MLM.view(BAT,-1))

	def infer_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.interpret(x,mask1)
		return out

class Attention_Database_NSP_MLM_vert_max(nn.Module):
	#    @staticmethod

	def __init__(self,count_tab,count_col,count_ENT,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_NSP_MLM_vert_max, self).__init__()
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = create_embedding_weight(embedding_path,entity_dict, embedding_dim,non_trainable)
		else:
			self.embeddings = nn.Embedding(count_ENT, embedding_dim)
		self.tab_emb=nn.Embedding(count_col, embedding_dim)
		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
		self.cls=nn.Linear(embedding_dim,2)
		self.decoder = nn.Linear(embedding_dim, count_ENT)
		self.decoder.weight = self.embeddings.weight
		self.m = nn.LogSoftmax(dim=-1)

	def forward(self, sentence,col_sentence,table_sentence,non_index,mask_index,vertical_sentences,non_vertical_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		batch_size=vertical_sentences.shape[1]
		out_arr=[out[:batch_size,mask_index].reshape(batch_size,1,-1)]
		mask1_arr=[torch.ones_like(sentence[:batch_size,mask_index]).reshape(batch_size,1,-1).to(device)]
		for k in range(vertical_sentences.shape[0]):
			x=(self.embeddings(vertical_sentences[k])+self.col_emb(col_sentence[:batch_size])+self.tab_emb(table_sentence[:batch_size])).to(device).reshape(int(BAT/2),-1,self.embedding_dim)
			mask = torch.ones_like(vertical_sentences[k]).to(device)
			mask2=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask[:,mask_index]= torch.zeros(vertical_sentences[k].shape[0]).to(device)
			mask1=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask1=mask1.unsqueeze(-2)
			out_sen=self.transformer.forward(x,mask1)
			outk=out_sen[:,mask_index]
			mask1_arr.append(mask2[:,mask_index].reshape(-1,1,1))
			out_arr.append(outk.reshape(batch_size,1,-1))
		out_arr_tens=torch.cat(out_arr,dim=1)
		out_arr_max=torch.max(out_arr_tens,1)
		res_MLM=self.decoder(out_arr_max[0])
		return res.reshape(BAT,-1),self.m(res_MLM.view(batch_size,-1))

	def forward_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index,vertical_sentences,non_vertical_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		batch_size=vertical_sentences.shape[1]
		out_arr=[out[:batch_size,mask_index].reshape(batch_size,1,-1)]
		mask1_arr=[torch.ones_like(sentence[:batch_size,mask_index]).reshape(batch_size,1,-1).to(device)]
		for k in range(vertical_sentences.shape[0]):
			x=(self.embeddings(vertical_sentences[k])+self.col_emb(col_sentence[:batch_size])+self.tab_emb(table_sentence[:batch_size])).to(device).reshape(int(BAT),-1,self.embedding_dim)
			mask = torch.ones_like(vertical_sentences[k]).to(device)
			mask2=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask[:,mask_index]= torch.zeros(vertical_sentences[k].shape[0]).to(device)
			mask1=torch.where(non_vertical_index[k]==0,torch.zeros_like(vertical_sentences[k]).to(device),mask)
			mask1=mask1.unsqueeze(-2)
			out_sen=self.transformer.forward(x,mask1)
			outk=out_sen[:,mask_index]
			mask1_arr.append(mask2[:,mask_index].reshape(-1,1,1))
			out_arr.append(outk.reshape(batch_size,1,-1))
		out_arr_tens=torch.cat(out_arr,dim=1)
		out_arr_max=torch.max(out_arr_tens,1)
		res_MLM=self.decoder(out_arr_max[0])
		return self.m(res_MLM.view(BAT,-1))

	def forward_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)

	def test_MLM(self, sentence,col_sentence,table_sentence,non_index,mask_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask[:,mask_index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res_MLM=self.decoder(out[:,mask_index])
		return self.m(res_MLM.view(BAT,-1))

	def infer_NSP(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.interpret(x,mask1)
		return out

class Attention_Database_NSP(nn.Module):
	#    @staticmethod

	def __init__(self,count_tab,count_col,count_ENT,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_NSP, self).__init__()
		self.entity_dict=entity_dict
		self.embeddings = nn.Embedding(count_ENT, embedding_dim)
		self.tab_emb=nn.Embedding(count_col, embedding_dim)
		self.col_emb=nn.Embedding(count_col, embedding_dim)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.embedding_dim=embedding_dim
		self.cls=nn.Linear(embedding_dim,2)

	def forward(self, sentence,col_sentence,table_sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=(self.embeddings(sentence)+self.col_emb(col_sentence)+self.tab_emb(table_sentence)).to(device).reshape(BAT,-1,self.embedding_dim)
		mask = torch.ones_like(sentence).to(device)
		# print(mask)
		# print(mask.shape)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.cls(out[:,0])
		return res.reshape(BAT,-1)

def repackage_hidden(h):
	"""Wraps hidden states in new Tensors, to detach them from their history."""

	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

class RNNModel(nn.Module):

	def __init__(self, count_dict, entity_dict, embedding_path, rnn_type, embedding_dim=300, nhid=300, nlayers=2, dropout=0.2, tie_weights=True):
		super(RNNModel, self).__init__()
		# self.ntoken = ntoken
		self.drop = nn.Dropout(dropout)
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		self.embeddings =create_embedding_weight(embedding_path,self.entity_dict, embedding_dim)
		self.embedding_dim=embedding_dim
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(embedding_dim, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		self.decoder = nn.Linear(nhid,self.count_dict)

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			self.decoder.weight = self.embeddings.weight

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def forward(self, sentence, hidden):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		# print(sentence.shape)
		emb = self.drop(torch.transpose(self.embeddings(sentence).to(device).reshape(BAT,B,self.embedding_dim),0,1))
		# print(emb.shape)
		output, hidden = self.rnn(emb, hidden)
		output = self.drop(output)
		decoded = self.decoder(output)
		decoded = decoded.view(-1, self.count_dict)
		return F.log_softmax(decoded, dim=1), hidden


	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

class biRNNModel(nn.Module):

	def __init__(self, count_dict, entity_dict, embedding_path, rnn_type, embedding_dim=300, nhid=300, nlayers=2, dropout=0.2, tie_weights=True):
		super(biRNNModel, self).__init__()
		# self.ntoken = ntoken
		self.drop = nn.Dropout(dropout)
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		self.embeddings =create_embedding_weight(embedding_path,self.entity_dict, embedding_dim)
		self.embedding_dim=embedding_dim
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(embedding_dim, nhid, nlayers, dropout=dropout,bidirectional=True)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		self.decoder = nn.Linear(nhid*2,self.count_dict)

		# # Optionally tie weights as in:
		# # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# # https://arxiv.org/abs/1608.05859
		# # and
		# # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# # https://arxiv.org/abs/1611.01462
		# if tie_weights:
		# 	self.decoder.weight = self.embeddings.weight

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def forward(self, sentence, hidden):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		# print(sentence.shape)
		emb = self.drop(torch.transpose(self.embeddings(sentence).to(device).reshape(BAT,B,self.embedding_dim),0,1))
		# print(emb.shape)
		output, hidden = self.rnn(emb, hidden)
		forward_output, backward_output = output[:-2, :, :self.embedding_dim], output[2:, :, self.embedding_dim:]
		staggered_output = torch.cat((forward_output, backward_output), dim=-1)

		staggered_output = self.drop(staggered_output)
		decoded = self.decoder(staggered_output)
		decoded = decoded.view(-1, self.count_dict)
		return F.log_softmax(decoded, dim=1), hidden


	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(2*self.nlayers, bsz, self.nhid),
					weight.new_zeros(2*self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

class Attention_Database_GOWALLA_same(nn.Module):
	#    @staticmethod

	def __init__(self,subject_id_index,user_id_index2,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_GOWALLA_same, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.embeddings[user_id_index2].weight = self.embeddings[subject_id_index].weight
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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


class Attention_Database_GOWALLA(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_GOWALLA, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

class Attention_Database(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

	def interpret(self, sentence,non_index):
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
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.interpret(x,mask1)
		return out

class Attention_Database_EmbDi(nn.Module):
	#    @staticmethod

	def __init__(self,numeric_col,non_numeric_col,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_EmbDi, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s in numeric_col:
					embeddings_array.append(create_embedding_weight_EmbDi(embedding_path,self.entity_dict[s], embedding_dim,non_trainable,True,s))
				elif s in non_numeric_col:
					embeddings_array.append(create_embedding_weight_EmbDi(embedding_path,self.entity_dict[s], embedding_dim,non_trainable,False,s))
				else:
					print("Not found in col array")
					exit()
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

class Attention_Database_ETE(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_ETE, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.vertical_diag_transformer=make_model( N=1, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.vertical_proc_transformer=make_model( N=1, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

	def vertical_proc_forward(self, sentence,index,non_index,proc_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim)
		x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.vertical_proc_transformer.forward(x,mask1)
		res=self.decoder[proc_index](out[:,index])
		return self.m(res.view(BAT,-1))

	def vertical_diag_forward(self, sentence,index,non_index,proc_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim)
		x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.vertical_diag_transformer.forward(x,mask1)
		res=self.decoder[proc_index](out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_ETE_LSTM(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True,dropout=0.2):
		super(Attention_Database_ETE_LSTM, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		self.drop = nn.Dropout(dropout)
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.vertical_diag_transformer=getattr(nn, "LSTM")(embedding_dim, embedding_dim, 2, dropout=0.2)
		self.vertical_proc_transformer=getattr(nn, "LSTM")(embedding_dim, embedding_dim, 2, dropout=0.2)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

	def init_hidden(self, bsz):
		return (
			Variable(torch.zeros(2, bsz, self.embedding_dim).to(device)),
			Variable(torch.zeros(2, bsz, self.embedding_dim).to(device)),
		)

	def vertical_proc_forward(self, sentence, hidden,proc_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		# print(sentence.shape)
		emb = self.drop(torch.transpose(self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim),0,1))
		# print(emb.shape)
		output, hidden = self.vertical_proc_transformer(emb, hidden)
		output = self.drop(output)
		decoded = self.decoder[proc_index](output)
		decoded = decoded.view(-1, self.count_dict[proc_index])
		return F.log_softmax(decoded, dim=1), hidden

	def vertical_diag_forward(self, sentence, hidden,proc_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		# print(sentence.shape)
		emb = self.drop(torch.transpose(self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim),0,1))
		# print(emb.shape)
		output, hidden = self.vertical_diag_transformer(emb, hidden)
		output = self.drop(output)
		decoded = self.decoder[proc_index](output)
		decoded = decoded.view(-1, self.count_dict[proc_index])
		return F.log_softmax(decoded, dim=1), hidden

class Attention_Database_ETE_year(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_ETE_year, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.vertical_year_transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

	def vertical_year_forward(self, sentence,index,non_index,proc_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim)
		x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.vertical_year_transformer.forward(x,mask1)
		res=self.decoder[proc_index](out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_ETE_BiLSTM_year(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_ETE_BiLSTM_year, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.drop = nn.Dropout(0.2)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.rnn=getattr(nn, "LSTM")(embedding_dim, embedding_dim, 2, dropout=0.2,bidirectional=True)
		self.W1 = nn.Linear(2*embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

	def vertical_year_forward(self, sentence, hidden,proc_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		# print(sentence.shape)
		emb = self.drop(torch.transpose(self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim),0,1))
		# print(emb.shape)
		output, hidden = self.rnn(emb, hidden)
		forward_output, backward_output = output[:-2, :, :self.embedding_dim], output[2:, :, self.embedding_dim:]
		staggered_output = torch.cat((forward_output, backward_output), dim=-1)

		staggered_output = self.drop(staggered_output)
		decoded = self.decoder[proc_index](self.W1(staggered_output))
		decoded = decoded.view(-1, self.count_dict[proc_index])
		return F.log_softmax(decoded, dim=1), hidden

	# def vertical_year_forward(self, sentence,index,non_index,proc_index):

	# 	sentence = sentence.to(device)
	# 	BAT=sentence.shape[0]
	# 	B=sentence.shape[1]
	# 	x=self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim)
	# 	# x=self.position(x)
	# 	mask = torch.ones_like(sentence).to(device)
	# 	mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
	# 	mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
	# 	mask1=mask1.unsqueeze(-2)
	# 	out=self.vertical_year_transformer.forward(x,mask1)
	# 	res=self.decoder[proc_index](out[:,index])
	# 	return self.m(res.view(BAT,-1))

	def init_hidden(self, bsz):
		return (
			Variable(torch.zeros(2*2, bsz, self.embedding_dim).to(device)),
			Variable(torch.zeros(2*2, bsz, self.embedding_dim).to(device)),
		)

class Attention_Database_ETE_mrank(nn.Module):
	#    @staticmethod

	def __init__(self,class_label,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_ETE_mrank, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		vertical_transformer=[]
		for k in class_label:
			vertical_transformer.append(make_model( N=2, d_model=embedding_dim, d_ff=4*embedding_dim, h=4))
		self.vertical_transformer=nn.ModuleList(vertical_transformer)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

	def vertical_forward(self, sentence,index,non_index,proc_index,proc_ver_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=self.embeddings[proc_index](sentence).to(device).reshape(BAT,B,self.embedding_dim)
		x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.vertical_transformer[proc_ver_index].forward(x,mask1)
		res=self.decoder[proc_index](out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_VER_COMPLETE(nn.Module):
	#    @staticmethod

	def __init__(self,diag_embedding_path,proc_embedding_path,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_COMPLETE, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s=='ICD9_CODE_x':
# 					entity_count=len(embeddings_array)
					embeddings_array.append(create_embedding_weight_model(diag_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				elif s=='ICD9_CODE_y':
# 					entity_count=len(embeddings_array)
					embeddings_array.append(create_embedding_weight_model(proc_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

class Attention_Database_VER_COMPLETE_GOWALLA(nn.Module):
	#    @staticmethod

	def __init__(self,proc_embedding_path,diag_embedding_path,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_COMPLETE_GOWALLA, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s=='place_id':
					embeddings_array.append(create_embedding_weight_model(proc_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				elif s=='user_id':
					embeddings_array.append(create_embedding_weight_model(diag_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

class Attention_Database_VER_CONCAT_GOWALLA(nn.Module):
	#    @staticmethod

	def __init__(self,class_label_dict, proc_embedding_path,diag_embedding_path,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_CONCAT_GOWALLA, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s=='place_id':
					embeddings_array.append(create_embedding_weight_model_conc(embedding_path,proc_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				elif s=='user_id':
					embeddings_array.append(create_embedding_weight_model_conc(embedding_path,diag_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.transform_proc=nn.Linear(2*embedding_dim,embedding_dim)
		self.transform_diag=nn.Linear(2*embedding_dim,embedding_dim)
		self.decode_proc=nn.Linear(embedding_dim,2*embedding_dim)
		self.decode_diag=nn.Linear(embedding_dim,2*embedding_dim)
		self.transform_proc.weight.data=torch.transpose(self.decode_proc.weight.data,0,1)
		self.transform_diag.weight.data=torch.transpose(self.decode_diag.weight.data,0,1)
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)
		self.class_label_dict=class_label_dict

	def forward(self, sentence,index,non_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for k in self.count_dict:
			if self.class_label_dict[k]=='user_id':
				at=self.embeddings[k](sentence[:,k]).to(device)
				# print(at.shape)
				at1=self.transform_diag(at)
				# print(at1.shape)
				x_array.append(at1.reshape(BAT,-1,self.embedding_dim))
			elif self.class_label_dict[k]=='place_id':
				x_array.append(self.transform_proc(self.embeddings[k](sentence[:,k]).to(device)).reshape(BAT,-1,self.embedding_dim))
			else:
				x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		if self.class_label_dict[index]=='user_id':
			res=self.decoder[index](self.decode_diag(out[:,index]))
		elif self.class_label_dict[index]=='place_id':
			res=self.decoder[index](self.decode_proc(out[:,index]))
		else:
			res=self.decoder[index](out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_VER_CONCATENATE(nn.Module):
	#    @staticmethod

	def __init__(self,class_label_dict,diag_embedding_path,proc_embedding_path,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_CONCATENATE, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s=='ICD9_CODE_x':
# 					entity_count=len(embeddings_array)
					embeddings_array.append(create_embedding_weight_model_conc(embedding_path,diag_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				elif s=='ICD9_CODE_y':
# 					entity_count=len(embeddings_array)
					embeddings_array.append(create_embedding_weight_model_conc(embedding_path,proc_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.transform_proc=nn.Linear(2*embedding_dim,embedding_dim)
		self.transform_diag=nn.Linear(2*embedding_dim,embedding_dim)
		self.decode_proc=nn.Linear(embedding_dim,2*embedding_dim)
		self.decode_diag=nn.Linear(embedding_dim,2*embedding_dim)
		self.transform_proc.weight.data=torch.transpose(self.decode_proc.weight.data,0,1)
		self.transform_diag.weight.data=torch.transpose(self.decode_diag.weight.data,0,1)
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)
		self.class_label_dict=class_label_dict

	def forward(self, sentence,index,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for k in self.count_dict:
			if self.class_label_dict[k]=='ICD9_CODE_x':
				at=self.embeddings[k](sentence[:,k]).to(device)
				# print(at.shape)
				at1=self.transform_diag(at)
				# print(at1.shape)
				x_array.append(at1.reshape(BAT,-1,self.embedding_dim))
			elif self.class_label_dict[k]=='ICD9_CODE_y':
				x_array.append(self.transform_proc(self.embeddings[k](sentence[:,k]).to(device)).reshape(BAT,-1,self.embedding_dim))
			else:
				x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		if self.class_label_dict[index]=='ICD9_CODE_x':
			res=self.decoder[index](self.decode_diag(out[:,index]))
		elif self.class_label_dict[index]=='ICD9_CODE_y':
			res=self.decoder[index](self.decode_proc(out[:,index]))
		else:
			res=self.decoder[index](out[:,index])
		return self.m(res.view(BAT,-1))

	def interpret(self, sentence,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for k in self.count_dict:
			if self.class_label_dict[k]=='ICD9_CODE_x':
				at=self.embeddings[k](sentence[:,k]).to(device)
				# print(at.shape)
				at1=self.transform_diag(at)
				# print(at1.shape)
				x_array.append(at1.reshape(BAT,-1,self.embedding_dim))
			elif self.class_label_dict[k]=='ICD9_CODE_y':
				x_array.append(self.transform_proc(self.embeddings[k](sentence[:,k]).to(device)).reshape(BAT,-1,self.embedding_dim))
			else:
				x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.interpret(x,mask1)
		return out

class Attention_Database_VER_CONCATENATE_GOWALLA(nn.Module):
	#    @staticmethod

	def __init__(self,class_label_dict,proc_embedding_path,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_CONCATENATE_GOWALLA, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s=='place_id':
					embeddings_array.append(create_embedding_weight_model_conc(embedding_path,proc_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.transform_proc=nn.Linear(2*embedding_dim,embedding_dim)
		self.decode_proc=nn.Linear(embedding_dim,2*embedding_dim)
		self.transform_proc.weight.data=torch.transpose(self.decode_proc.weight.data,0,1)
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)
		self.class_label_dict=class_label_dict

	def forward(self, sentence,index,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for k in self.count_dict:
			if self.class_label_dict[k]=='place_id':
				x_array.append(self.transform_proc(self.embeddings[k](sentence[:,k]).to(device)).reshape(BAT,-1,self.embedding_dim))
			else:
				x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		if self.class_label_dict[index]=='place_id':
			res=self.decoder[index](self.decode_proc(out[:,index]))
		else:
			res=self.decoder[index](out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_VER_CONCATENATE_YEAR(nn.Module):
	#    @staticmethod

	def __init__(self,class_label_dict,diag_embedding_path,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_CONCATENATE_YEAR, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s=='movie_id':
# 					entity_count=len(embeddings_array)
					embeddings_array.append(create_embedding_weight_model_conc(embedding_path,diag_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		# self.transform_proc=nn.Linear(2*embedding_dim,embedding_dim)
		self.transform_diag=nn.Linear(2*embedding_dim,embedding_dim)
		# self.decode_proc=nn.Linear(embedding_dim,2*embedding_dim)
		self.decode_diag=nn.Linear(embedding_dim,2*embedding_dim)
		# self.transform_proc.weight.data=torch.transpose(self.decode_proc.weight.data,0,1)
		self.transform_diag.weight.data=torch.transpose(self.decode_diag.weight.data,0,1)
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)
		self.class_label_dict=class_label_dict

	def forward(self, sentence,index,non_index):
		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		for k in self.count_dict:
			if self.class_label_dict[k]=='movie_id':
				at=self.embeddings[k](sentence[:,k]).to(device)
				# print(at.shape)
				at1=self.transform_diag(at)
				# print(at1.shape)
				x_array.append(at1.reshape(BAT,-1,self.embedding_dim))
			else:
				x_array.append(self.embeddings[k](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		if self.class_label_dict[index]=='movie_id':
			res=self.decoder[index](self.decode_diag(out[:,index]))
		else:
			res=self.decoder[index](out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_VER_COM_YEAR(nn.Module):
	#    @staticmethod

	def __init__(self,diag_embedding_path,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_COM_YEAR, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s=='movie_id':
					embeddings_array.append(create_embedding_weight_model(diag_embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

class Attention_Database_VER_COM_ALL(nn.Module):
	#    @staticmethod

	def __init__(self,attention_model_dict,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_COM_ALL, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
				if s in attention_model_dict:
					embeddings_array.append(create_embedding_weight_model(attention_model_dict[s],self.entity_dict[s], embedding_dim,non_trainable))
				else:
					embeddings_array.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))                
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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

class Attention_Database_VER(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		self.embeddings =create_embedding_weight(embedding_path,self.entity_dict, embedding_dim,non_trainable)

		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.Linear(embedding_dim,self.count_dict)
		self.position_arg=position_arg
		if tie_weights:
			self.decoder.weight = self.embeddings.weight
		self.m = nn.LogSoftmax(dim=-1)

	def forward(self, sentence,index,non_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x=self.embeddings(sentence).to(device).reshape(BAT,B,self.embedding_dim)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(device)
		mask[:,index]= torch.zeros(sentence.shape[0]).to(device)
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		res=self.decoder(out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_BertTable(nn.Module):
	#    @staticmethod

	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_BertTable, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			emb_arr=[]
			for s in self.entity_dict:
				if s=="extra":
					emb_arr.append(nn.Embedding(self.count_dict[11], embedding_dim))
				else:
					emb_arr.append(create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable))
			self.embeddings = nn.ModuleList(emb_arr)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

	def forward(self, sentence,index,non_index):

		sentence = sentence.to(device)
		BAT=sentence.shape[0]
		B=sentence.shape[1]
		x_array=[]
		# print(len(self.embeddings))
		for k in range(B):
			if ((k%4==0) and (k!=0)): 
				x_array.append(self.embeddings[int(k/4)-1](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
			else:
				x_array.append(self.embeddings[-1](sentence[:,k]).to(device).reshape(BAT,-1,self.embedding_dim))
		x=torch.cat(x_array,dim=1)
		# print(x)
		if self.position_arg:
			x=self.position(x)
		mask = torch.ones_like(sentence).to(sentence.get_device())
		mask[:,index]= torch.zeros(sentence.shape[0]).to(sentence.get_device())
		mask1=torch.where(non_index==0,torch.zeros_like(sentence).to(device),mask)
		mask1=mask1.unsqueeze(-2)
		out=self.transformer.forward(x,mask1)
		if ((index%4==0) and (index!=0)): 
			res=self.decoder[int(index/4)-1](out[:,index])
		else:
			res=self.decoder[-1](out[:,index])
		return self.m(res.view(BAT,-1))

class Attention_Database_CLS(nn.Module):
	#    @staticmethod
	def __init__(self,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path=get_model_path()+"actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_CLS, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			self.embeddings = nn.ModuleList([
				create_embedding_weight(embedding_path,self.entity_dict[s], embedding_dim,non_trainable)
				for s in self.entity_dict
			])
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.CLS=torch.nn.Parameter(torch.tensor(np.random.normal(0,1e-3,(embedding_dim))))
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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
		mask1_inp=torch.cat((torch.ones(sentence.shape[0]).reshape(-1,1).long().to(device),mask1),dim=1)
		mask1_inp=mask1_inp.unsqueeze(-2)
		CLS_array=torch.cat([self.CLS.reshape(1,-1) for i in range(BAT)],dim=0)
		CLS_array=CLS_array.reshape(BAT,1,-1)
		x_inp=torch.cat((CLS_array.float(),x),dim=1)
		out=self.transformer.forward(x_inp,mask1_inp)
		res=self.decoder[index](out[:,0])
		return self.m(res.view(BAT,-1))

class Attention_Database_VER_COMPLETE(nn.Module):
	#    @staticmethod

	def __init__(self,embedding_dict,count_dict,embedding_dim=100,arg_N=4,position_arg=True,load_pre_train_embeddings=False,entity_dict=None,non_trainable=False,embedding_path="/home/sid/sid_folder/Query-biased-embedding-on-relational-database/actors_nam_data_clean.bin",tie_weights=True):
		super(Attention_Database_VER_COMPLETE, self).__init__()
		self.count_dict=count_dict
		self.entity_dict=entity_dict
		if load_pre_train_embeddings:
			embeddings_array=[]
			for s in self.entity_dict:
					embeddings_array.append(create_embedding_weight_matrix(embedding_dict[s],self.entity_dict[s], embedding_dim,non_trainable))              
			self.embeddings = nn.ModuleList(embeddings_array)
		else:
			self.embeddings = nn.ModuleList([
				nn.Embedding(self.count_dict[s], embedding_dim)
				for s in self.count_dict
			])
		self.position=PositionalEncoding(embedding_dim, 0)
		self.transformer=make_model( N=arg_N, d_model=embedding_dim, d_ff=4*embedding_dim, h=4)
		self.W1 = nn.Linear(embedding_dim, embedding_dim).to(device)
		self.embedding_dim=embedding_dim
		self.decoder = nn.ModuleList([
			nn.Linear(embedding_dim,self.count_dict[s])
			for s in self.count_dict
		])
		self.position_arg=position_arg
		if tie_weights:
			for k in range(len(self.count_dict)):
				self.decoder[k].weight = self.embeddings[k].weight
		else:
			for k in range(len(self.count_dict)):
				self.decoder[k].load_state_dict({"weight": self.embeddings[k].weight.data,"bias": self.decoder[k].bias.data})
		self.m = nn.LogSoftmax(dim=-1)

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