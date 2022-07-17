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

def prepare_NSP_mask_sequence_vert(vert_sent_dict,train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None): 
	if ((train_sentences.shape[1])!=3): 
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1) 
	seq1=train_sentences[:,0] 
	seq2=train_sentences[:,1] 
	seq3=train_sentences[:,2] 
	length_row=32 
	index=random.randrange(length_row)

	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	vert_sen1_total_array=[]
	vert_sen2_total_array=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
		vert_a=vert_sent_dict[tab_inv_dict[a]][tuple(seq1[i].astype("str"))]
		vert_b=vert_sent_dict[tab_inv_dict[b]][tuple(seq2[i].astype("str"))]
		# print(vert_a)
		# print(vert_b)
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		vert_sen1_arr=[]
		for k in vert_a:
			vert_sen1=CLS_sent[i]+list(k)+SEP_sent[i]
			vert_sen1=vert_sen1+['<NONE>'] * (max_seq_len - len(vert_sen1))
			vert_sen1_arr.append(vert_sen1)
		vert_sen1_total_array.append(np.array(vert_sen1_arr).astype('str').reshape(len(vert_a),1,-1))
		vert_sen2_arr=[]
		for k in vert_b:
			vert_sen2=CLS_sent[i]+['<NONE>' for i in range(len(seq1[i]))]+SEP_sent[i]+list(k)+SEP_sent[i]
			vert_sen2=vert_sen2+['<NONE>'] * (max_seq_len - len(vert_sen2))
			vert_sen2_arr.append(vert_sen2)
		vert_sen2_total_array.append(np.array(vert_sen2_arr).astype('str').reshape(len(vert_b),1,-1))
		col_sen1=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
	#         print(len(col_sen1))
		if (len(col_sen1)!=32):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
	#     print(pos_seq)
	#     seq=seq2.copy()

	total_seq=np.array(pos_seq+neg_seq).astype('str')
	vert_sen1_seq=np.concatenate(vert_sen1_total_array,axis=1)
	vert_sen2_seq=np.concatenate(vert_sen2_total_array,axis=1)
	# print(vert_sen1_seq.shape)
	# print(vert_sen2_seq.shape)
	#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
	#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
	res_vert1=np.zeros(vert_sen1_seq.shape)
	for k in range(res_vert1.shape[0]):
		for k1 in range(res_vert1.shape[1]):
			sen_arr=[]
			for x in vert_sen1_seq[k,k1,:]:
				sen_arr.append(entity_dict[x])   
			res_vert1[k,k1,:]=sen_arr
	res_vert2=np.zeros(vert_sen2_seq.shape)
	for k in range(res_vert2.shape[0]):
		for k1 in range(res_vert2.shape[1]):
			sen_arr=[]
			for x in vert_sen2_seq[k,k1,:]:
				sen_arr.append(entity_dict[x])   
			res_vert2[k,k1,:]=sen_arr
	#     # print(res)
	mask1=(total_seq!="<NONE>")
	mask_vert1=(vert_sen1_seq!="<NONE>")
	mask_vert2=(vert_sen2_seq!="<NONE>")
	# print(res_vert1.shape)
	# print(res_vert2.shape)
	#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	torch.LongTensor(lab_seq).to(device)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(res_vert1).to(device),torch.LongTensor(res_vert2).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(1*mask_vert1).to(device),torch.LongTensor(1*mask_vert2).to(device),torch.LongTensor(lab_seq).to(device),torch.LongTensor(targets_MLM).to(device),seq_None

def prepare_DRG_mask_sequence_vert(vert_sent_dict,seq,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=7,class_label_dict=None):
	seq1=seq
	#     print(seq1)
	length_row=5
	index=random.randrange(length_row)+3

	BAT=seq.shape[0]
	CLS_sent=[["[CLS]"] for i in range(seq.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	# max_seq_len=7
	vert_sen1_total_array=[]
	for i in range(seq1.shape[0]):
		a=4
		vert_a=vert_sent_dict[tab_inv_dict[a]][tuple(seq[i].astype("str"))]
		col_a=tab_col_dict[tab_inv_dict[a]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[a] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+[42] * (max_seq_len - len(sen1))
		vert_sen1_arr=[]
		for k in vert_a:
			vert_sen1=CLS_sent[i]+list(k)+SEP_sent[i]
			vert_sen1=vert_sen1+['<NONE>'] * (max_seq_len - len(vert_sen1))
			vert_sen1_arr.append(vert_sen1)
		vert_sen1_total_array.append(np.array(vert_sen1_arr).astype('str').reshape(len(vert_a),1,-1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		if (len(col_sen1)!=9):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)    
	total_seq=np.array(pos_seq).astype('str')
	vert_sen1_seq=np.concatenate(vert_sen1_total_array,axis=1)
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
	#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			if x=="None":
				sen_arr.append(entity_dict["<NONE>"])
			else:
				sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
	#     # print(res)
	res_vert1=np.zeros(vert_sen1_seq.shape)
	for k in range(res_vert1.shape[0]):
		for k1 in range(res_vert1.shape[1]):
			sen_arr=[]
			for x in vert_sen1_seq[k,k1,:]:
				sen_arr.append(entity_dict[x])   
			res_vert1[k,k1,:]=sen_arr
	
	mask1=(total_seq!="<NONE>")
	mask_vert1=(vert_sen1_seq!="<NONE>")
	#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(res_vert1).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(1*mask_vert1).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  

# same as prepare_NSP_mask_sequence_IMDB method without negative examples ofcourse
# Used in run_NSP_MLM_sample_test
def prepare_MLM_eval_sequence_IMDB(seq1,seq2,table_a,table_b,tab_col_dict,tab_inv_dict,entity_dict,index,max_seq_len=32,class_label_dict=None,max_col_len=42):
	BAT=seq1.shape[0]
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	for i in range(seq2.shape[0]):
		a=table_a
		b=table_b
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		if (len(col_sen1)!=max_seq_len):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
	total_seq=np.array(pos_seq).astype('str')
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
	mask1=(total_seq!="<NONE>")
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(targets_MLM).to(device),seq_None

def prepare_NSP_sequence_IMDB(train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None,max_col_len=42):
	
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0] # tuple of table 1
	seq2=train_sentences[:,1] # joining tuple of table 2
	seq3=train_sentences[:,2] # negative sample from table 2
#     print(train_sentences[0])
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]

	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0] # ID of table 1
		b=train_table_sentence[i,1] # ID of table 2
#         print(a)
#         print(b)
		col_a=tab_col_dict[tab_inv_dict[a]] # list of ids of columns of table 1
		col_b=tab_col_dict[tab_inv_dict[b]] # list of ids of column of table 2

		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i] # sen1 = [CLS]+tuple from table 1+[SEP]+joining tuple from table 2+[SEP]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1)) # snetence with table ids
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1)) # sentence with column ids of respective tables
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1) # sen1 padded with <NONE> token
		tab_pos_seq.append(tab_sen1)
#         print(len(col_sen1))
		if (len(col_sen1)!=max_seq_len):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	
	#constructing negative sample datapoints
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]

	for i in range(seq2.shape[0]):

		a=train_table_sentence[i,0] 
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]

		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i] #sen2 = [CLS]+tuple from table 1+[SEP]+negative sample from table 2+[SEP]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2)) 
		col_sen2=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	# collecting all the data points (negative and positive)
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),

def prepare_NSP_sequence_IMDB_complete_diff(complete_arr,col_index_array,train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None,max_col_len=42):
	seq1=train_sentences[0]
	correct_seq2=np.array(train_sentences[1])
	seq2=np.array(complete_arr)
	# seq3=train_sentences[:,2]
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	for i in range(seq2.shape[0]):
		# print(i)
		a=train_table_sentence[0]
		b=train_table_sentence[1]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1)+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1))
		tab_sen1=[a]+[a for i in range(len(seq1))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		if (len(col_sen1)!=max_seq_len):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		if i in correct_seq2:
			lab_seq.append(1)
		else:
			lab_seq.append(0)
	
	total_seq=np.array(pos_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	lab_seq=np.array(lab_seq)
	# print(total_seq.shape)
	# print(len(total_col_seq))
	# print(len(total_col_seq[0]))
	# print(len(total_tab_seq))
	# print(len(total_tab_seq[0]))
	# exit()
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[1]):
		sen_arr=[]
		for x in total_seq[:,k]:
			sen_arr.append(entity_dict[class_label_dict[col_index_array[k]]][x])   
		res[:,k]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),
#Used 
def prepare_NSP_sequence_IMDB_diff_neg_sample(col_index_array,train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None,max_col_len=42):
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0]
	seq2=train_sentences[:,1]
	seq3=train_sentences[:,2]
#     print(train_sentences[0])
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1))
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		if (len(col_sen1)!=max_seq_len):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		for seq3_unit in seq3[i]:
			sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3_unit)+SEP_sent[i]
			tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3_unit))]+[b]+[b] * (max_seq_len - len(sen2))
			col_sen2=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen2))
			sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
			neg_seq.append(sen2)
			tab_neg_seq.append(tab_sen2)
			col_neg_seq.append(col_sen2)
			lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[1]):
		sen_arr=[]
		# print(k)
		# print(col_index_array[k])
		# print(total_seq[:,k])
		for x in total_seq[:,k]:
			sen_arr.append(entity_dict[class_label_dict[col_index_array[k]]][x])   
		res[:,k]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),


def prepare_NSP_sequence_IMDB_diff(col_index_array,train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None,max_col_len=42):
	
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	
	# seq1 has sentences from table 1, seq2 has sentences from table 2 that joined to table 1 sentence, seq3 has negative sample from table 2
	seq1=train_sentences[:,0]
	seq2=train_sentences[:,1]
	seq3=train_sentences[:,2]
#     print(train_sentences[0])
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	
	# construct NSP input, i.e pair of sentences separated by SEP token
	for i in range(seq2.shape[0]):
		# extracting ids a and b of table1 and table 2 from the train_table_setence map
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
		col_a=tab_col_dict[tab_inv_dict[a]] # ids of columns of table a
		col_b=tab_col_dict[tab_inv_dict[b]] # ids of columns of table b
		#sen1 = <CLS>+lsit of tokens in seq1+<SEP>+list of tokens in seq2+<SEP>
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		#col_sen1 = sequence of column ids as present tokens present in sen1
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1))
		#tab_sen1 = sequence of table ids as present tokens present in sen1
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		# padding the sen1 with NONE tokens
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		
		if (len(col_sen1)!=max_seq_len):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	#construct negative examples with seq1 and seq3 pair of sentences
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str') #list of all the pair of sentences
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)

	for k in range(total_seq.shape[1]):
		sen_arr=[]
		# print(k)
		# print(col_index_array[k])
		# print(total_seq[:,k])
		for x in total_seq[:,k]:
			# converting tokens in sentences to the labels(ids) stored in entity_dict
			sen_arr.append(entity_dict[class_label_dict[col_index_array[k]]][x])   
		res[:,k]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),



def prepare_NSP_mask_sequence_IMDB(train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None,max_col_len=42):
	
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0] # list of correct tuples 
	seq2=train_sentences[:,1] # list of tuples joined with tuples in seq 1
	seq3=train_sentences[:,2] # negative example which would be interpreted as it is joined with the tuple of seq1
	length_row=max_seq_len
	index=random.randrange(length_row)
	
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]

	#adding all the positive pair using seq 1 and seq2 ; 
	pos_seq=[] # CLS+tokens in seq1+SEP+tokens in seq 2+SEP+<NONE>... (max length 32)
	tab_pos_seq=[] # table1 id+table1 id * length of seq1+ table1 id +table 2 id * length of seq2+table2 id + table2 id... (max length 32)
	col_pos_seq=[] # 42+ list of column ids of table1+42+ list of column ids of table2+42+42...(max length 32)
	lab_seq=[] # 1
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0] #Get id of table a
		b=train_table_sentence[i,1] #Get id of table b; tbale a is joined with table b
#         print(a)
#         print(b)
		col_a=tab_col_dict[tab_inv_dict[a]] # List of columns (ids) of table a
		col_b=tab_col_dict[tab_inv_dict[b]] #List of columns (ids) of table b
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i] # sen1 is the sentence with seq1 and seq2 of the training sentences, i.e correct pair of tuples that are actually joined
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1)) # sen1 followed by <NONE> (padding)
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
#         print(len(col_sen1))
		if (len(col_sen1)!=length_row):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)

	# construct negative samples by using seq1 and seq3	
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ]) # list of index of sentences for which the randomly chosen token at index index is not NONE
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]] # id of tokens at index index in all data points
	mask_array=["<MASK>" for i in range(total_seq.shape[0])] # list of <MASK> tokens of length that of total_seq
	total_seq[:,index]=np.array(mask_array) #masking token at index index for all the data points
#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq) # list of label of the data points -- 1 for positive examples, 0 for negative ones
	res=np.zeros(total_seq.shape) 
	# converting tokens to their ids in the data point list
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	torch.LongTensor(lab_seq).to(device)

	# index = randomly chosen index at which tokens are masked
	# res = sentences where tokens are replaced by their ids
	# total_col_seq = column sequqnces of each data point in res (see above the description of col_seq); Similar for total_tab_seq
	# 1 *mask1
	# lab_seq = label of each data point -- 0 if negative example else 1
	# targets_MLM = tokens at the masked index of the sentences
	# none_seq = list of sentence index for which the token at index index is "NONE" (check out description above)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  

def prepare_NSP_mask_sequence_vert_IMDB(vert_sent_dict,train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None,max_col_len=42): 
	if ((train_sentences.shape[1])!=3): 
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1) 
	seq1=train_sentences[:,0] 
	seq2=train_sentences[:,1] 
	seq3=train_sentences[:,2] 
	length_row=max_seq_len
	index=random.randrange(length_row)

	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	vert_sen1_total_array=[]
	vert_sen2_total_array=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
		vert_a=vert_sent_dict[tab_inv_dict[a]][tuple(np.array(seq1[i]).astype("str"))]
		vert_b=vert_sent_dict[tab_inv_dict[b]][tuple(np.array(seq2[i]).astype("str"))]
		# print(vert_a)
		# print(vert_b)
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		vert_sen1_arr=[]
		for k in vert_a:
			vert_sen1=CLS_sent[i]+list(k)+SEP_sent[i]
			vert_sen1=vert_sen1+['<NONE>'] * (max_seq_len - len(vert_sen1))
			vert_sen1_arr.append(vert_sen1)
		# print(vert_sen1_arr)
		vert_sen1_total_array.append(np.array(vert_sen1_arr).astype('str').reshape(len(vert_a),1,-1))
		vert_sen2_arr=[]
		for k in vert_b:
			vert_sen2=CLS_sent[i]+['<NONE>' for i in range(len(seq1[i]))]+SEP_sent[i]+list(k)+SEP_sent[i]
			vert_sen2=vert_sen2+['<NONE>'] * (max_seq_len - len(vert_sen2))
			vert_sen2_arr.append(vert_sen2)
		# print(vert_sen2_arr)
		vert_sen2_total_array.append(np.array(vert_sen2_arr).astype('str').reshape(len(vert_b),1,-1))
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
	#         print(len(col_sen1))
		if (len(col_sen1)!=length_row):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
	#     print(pos_seq)
	#     seq=seq2.copy()

	total_seq=np.array(pos_seq+neg_seq).astype('str')
	vert_sen1_seq=np.concatenate(vert_sen1_total_array,axis=1)
	vert_sen2_seq=np.concatenate(vert_sen2_total_array,axis=1)
	# print(vert_sen1_seq.shape)
	# print(vert_sen2_seq.shape)
	#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
	#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
	res_vert1=np.zeros(vert_sen1_seq.shape)
	for k in range(res_vert1.shape[0]):
		for k1 in range(res_vert1.shape[1]):
			sen_arr=[]
			for x in vert_sen1_seq[k,k1,:]:
				sen_arr.append(entity_dict[x])   
			res_vert1[k,k1,:]=sen_arr
	res_vert2=np.zeros(vert_sen2_seq.shape)
	for k in range(res_vert2.shape[0]):
		for k1 in range(res_vert2.shape[1]):
			sen_arr=[]
			for x in vert_sen2_seq[k,k1,:]:
				sen_arr.append(entity_dict[x])   
			res_vert2[k,k1,:]=sen_arr
	#     # print(res)
	mask1=(total_seq!="<NONE>")
	mask_vert1=(vert_sen1_seq!="<NONE>")
	mask_vert2=(vert_sen2_seq!="<NONE>")
	# print(res_vert1.shape)
	# print(res_vert2.shape)
	#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	torch.LongTensor(lab_seq).to(device)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(res_vert1).to(device),torch.LongTensor(res_vert2).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(1*mask_vert1).to(device),torch.LongTensor(1*mask_vert2).to(device),torch.LongTensor(lab_seq).to(device),torch.LongTensor(targets_MLM).to(device),seq_None

def prepare_MLM_eval_sequence_vert_IMDB(vert_sent_dict,seq1,seq2,table_a,table_b,tab_col_dict,tab_inv_dict,entity_dict,index,max_seq_len=32,class_label_dict=None,max_col_len=42):
	BAT=seq1.shape[0]
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	vert_sen1_total_array=[]
	for i in range(seq2.shape[0]):
		a=table_a
		b=table_b
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		vert_a=vert_sent_dict[tuple(np.array(seq1[i]).astype("str"))]
		while(len(vert_a)<5):
			none_a=['<NONE>'] * len(col_a)
			vert_a.append(none_a)
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		vert_sen1_arr=[]
		for k in vert_a:
			vert_sen1=CLS_sent[i]+['<NONE>' for i in range(len(seq1[i]))]+SEP_sent[i]+list(k)+SEP_sent[i]
			vert_sen1=vert_sen1+['<NONE>'] * (max_seq_len - len(vert_sen1))
			vert_sen1_arr.append(vert_sen1)
		vert_sen1_total_array.append(np.array(vert_sen1_arr).astype('str').reshape(len(vert_a),1,-1))
		col_sen1=[max_col_len]+col_a+[max_col_len]+col_b+[max_col_len]+[max_col_len] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		if (len(col_sen1)!=max_seq_len):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
	total_seq=np.array(pos_seq).astype('str')
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	vert_sen1_seq=np.concatenate(vert_sen1_total_array,axis=1)
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
	res_vert1=np.zeros(vert_sen1_seq.shape)
	for k in range(res_vert1.shape[0]):
		for k1 in range(res_vert1.shape[1]):
			sen_arr=[]
			for x in vert_sen1_seq[k,k1,:]:
				sen_arr.append(entity_dict[x])   
			res_vert1[k,k1,:]=sen_arr
	mask_vert1=(vert_sen1_seq!="<NONE>")
	mask1=(total_seq!="<NONE>")
	return torch.LongTensor(res).to(device),torch.LongTensor(res_vert1).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(1*mask_vert1).to(device),torch.LongTensor(targets_MLM).to(device),seq_None


def prepare_NSP_sequence_diff(col_index_array,train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None):
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0]
	seq2=train_sentences[:,1]
	seq3=train_sentences[:,2]
	length_row=32
	
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
#         print(len(col_sen1))
		if (len(col_sen1)!=32):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq

	
#     targets_MLM_cat=np.concatenate(targets_MLM_array,axis=1)
#     seq_none_cat=np.concatenate(seq_None_array,axis=1)
#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[1]):
		sen_arr=[]
		for x in total_seq[:,k]:
			sen_arr.append(entity_dict[class_label_dict[col_index_array[k]]][x])   
		res[:,k]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	torch.LongTensor(lab_seq).to(device)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device) 


def prepare_DRG_mask_sequence_diff(col_index_array,seq,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None):
	seq1=seq[:,2:]
#     print(seq1)
	length_row=5
	arr=[i for i in range(1,6)]
	index=random.choice(arr)
	
	BAT=seq.shape[0]
	CLS_sent=[["[CLS]"] for i in range(seq.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	max_seq_len=7
	for i in range(seq1.shape[0]):
		a=4
		col_a=tab_col_dict[tab_inv_dict[a]][2:]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[a] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		if (len(col_sen1)!=7):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)    
	total_seq=np.array(pos_seq).astype('str')
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[class_label_dict[col_index_array[index]]][x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[1]):
		sen_arr=[]
		for x in total_seq[:,k]:
			sen_arr.append(entity_dict[class_label_dict[col_index_array[k]]][x])   
		res[:,k]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  


def prepare_NSP_mask_sequence_diff(col_index_array,train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None):
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0]
	seq2=train_sentences[:,1]
	seq3=train_sentences[:,2]
	length_row=32
	a=train_table_sentence[0,0]
	b=train_table_sentence[0,1]
	col_a=tab_col_dict[tab_inv_dict[a]]
	col_b=tab_col_dict[tab_inv_dict[b]]
	arr=[i for i in range(1,len(col_a)+1)]+[i for i in range(len(col_a)+2,len(col_a)+len(col_b)+2)]
	index=random.choice(arr)	
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	for i in range(seq2.shape[0]):
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
#         print(len(col_sen1))
		if (len(col_sen1)!=32):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	targets_MLM_array=[]
	seq_None_array=[]
	total_seq[:,index]=np.array(mask_array)
	seq_None=(np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ]).reshape(-1,1))
	targets_MLM=(np.array([entity_dict[class_label_dict[col_index_array[index]]][x] for x in total_seq[:,index]]).reshape(-1,1))
#     targets_MLM_cat=np.concatenate(targets_MLM_array,axis=1)
#     seq_none_cat=np.concatenate(seq_None_array,axis=1)
#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[1]):
		sen_arr=[]
		for x in total_seq[:,k]:
			sen_arr.append(entity_dict[class_label_dict[col_index_array[k]]][x])   
		res[:,k]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	torch.LongTensor(lab_seq).to(device)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  


def prepare_NSP_mask_sequence_multiple(train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None):
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0]
	seq2=train_sentences[:,1]
	seq3=train_sentences[:,2]
	# print(seq1)
	length_row=32
	tab=[i for i in range(length_row)]
	random.shuffle(tab)
	mask_index_array=tab[:5]
	
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
#         print(a)
#         print(b)
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
#         print(len(col_sen1))
		if (len(col_sen1)!=32):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	targets_MLM_array=[]
	seq_None_array=[]
	for index in mask_index_array:
		total_seq[:,index]=np.array(mask_array)
		seq_None_array.append(np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ]).reshape(-1,1))
		targets_MLM_array.append(np.array([entity_dict[x] for x in total_seq[:,index]]).reshape(-1,1))
	targets_MLM_cat=np.concatenate(targets_MLM_array,axis=1)
	seq_none_cat=np.concatenate(seq_None_array,axis=1)
#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	torch.LongTensor(lab_seq).to(device)
	return mask_index_array,torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),torch.LongTensor(targets_MLM_cat).to(device),seq_none_cat  

def prepare_NSP_mask_sequence(train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None):
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0]
	seq2=train_sentences[:,1]
	seq3=train_sentences[:,2]
	length_row=32
	index=random.randrange(length_row)
	
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
#         print(a)
#         print(b)
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
#         print(len(col_sen1))
		if (len(col_sen1)!=32):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	torch.LongTensor(lab_seq).to(device)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  

def prepare_DRG_mask_sequence(seq,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=7,class_label_dict=None):
	seq1=seq[:,2:]
#     print(seq1)
	length_row=5
	index=random.randrange(length_row)
	
	BAT=seq.shape[0]
	CLS_sent=[["[CLS]"] for i in range(seq.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	# max_seq_len=7
	for i in range(seq1.shape[0]):
		a=4
		col_a=tab_col_dict[tab_inv_dict[a]][2:]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[a] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
		if (len(col_sen1)!=7):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)    
	total_seq=np.array(pos_seq).astype('str')
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
#     total_seq2=np.array([str_k for str_k in total_seq if (str_k[index]!="None") ])
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	return index,torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  


def prepare_NSP_sequence(train_sentences,train_table_sentence,tab_col_dict,tab_inv_dict,entity_dict,max_seq_len=32,class_label_dict=None):
	if ((train_sentences.shape[1])!=3):
		train_sentences=train_sentences.reshape(train_sentences.shape[0],3,-1)
	seq1=train_sentences[:,0]
	seq2=train_sentences[:,1]
	seq3=train_sentences[:,2]
#     print(train_sentences[0])
	BAT=len(train_sentences)
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	lab_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,1]
#         print(a)
#         print(b)
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
		col_sen1=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen1))
		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
		pos_seq.append(sen1)
		tab_pos_seq.append(tab_sen1)
#         print(len(col_sen1))
		if (len(col_sen1)!=32):
			print(train_sentences[i])
			print(col_sen1)
			print(sen1)
			print(len(sen1))
			return
		col_pos_seq.append(col_sen1)
		lab_seq.append(1)
	neg_seq=[]
	tab_neg_seq=[]
	col_neg_seq=[]
	for i in range(seq2.shape[0]):
		a=train_table_sentence[i,0]
		b=train_table_sentence[i,2]
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		sen2=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq3[i])+SEP_sent[i]
		tab_sen2=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq3[i]))]+[b]+[b] * (max_seq_len - len(sen2))
		col_sen2=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen2))
		sen2 += ['<NONE>'] * (max_seq_len - len(sen2))
		neg_seq.append(sen2)
		tab_neg_seq.append(tab_sen2)
		col_neg_seq.append(col_sen2)
		lab_seq.append(0)
#     print(pos_seq)
#     seq=seq2.copy()
	
	total_seq=np.array(pos_seq+neg_seq).astype('str')
#     print(total_seq)
	total_tab_seq=tab_pos_seq+tab_neg_seq
	total_col_seq=col_pos_seq+col_neg_seq
	lab_seq=np.array(lab_seq)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[0]):
		sen_arr=[]
		for x in total_seq[k,:]:
			sen_arr.append(entity_dict[x])   
		res[k,:]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
#     print(total_col_seq)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(lab_seq).to(device),


def create_attention_mask_from_input_mask(from_tensor, to_mask):
	batch_size = from_tensor.shape[0]
	from_seq_length =  from_tensor.shape[1]

	# to_shape = get_shape_list(to_mask, expected_rank=2)
	to_seq_length = to_mask.shape[1]
	ap=to_mask.view(batch_size, 1, to_seq_length)

	to_mask = (ap).type(torch.FloatTensor).to(device)
	# exit()
	# We don't assume that `from_tensor` is a mask (although it could be). We
	# don't actually care if we attend *from* padding tokens (only *to* padding)
	# tokens so we create a tensor of all ones.
	#
	# `broadcast_ones` = [batch_size, from_seq_length, 1]
	broadcast_ones = (torch.ones((batch_size, from_seq_length, 1))).type(torch.FloatTensor).to(device)

	# Here we broadcast along two dimensions to create the mask.
	mask = broadcast_ones * to_mask

	return mask

def prepare_sequence(seq1,entity_dict,class_label_dict):
	
	length_row=seq1.shape[1]
	index=random.randrange(length_row)
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	# print(seq2.shape)
	if (seq2.shape[0]==0):
		return -1,[],[]
	mask_array=["mask" for i in range(seq2.shape[0])]
	seq=seq2.copy()
	seq[:,index]=np.array(mask_array)
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		# print(seq[:,k])
		# print(class_label_dict[k])
		# print(entity_dict[class_label_dict[k]])
		res[:,k]=[entity_dict[class_label_dict[k]][x] for x in seq[:,k]]
	# print(res)
	mask1=(seq!="None")
	return index,torch.LongTensor(res).to(device),torch.LongTensor(1*mask1).to(device)


def prepare_nonmask_sequence(seq1,entity_dict,class_label_dict):
	
	length_row=seq1.shape[1]
	seq=seq1.copy()
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		# print(seq[:,k])
		# print(class_label_dict[k])
		# print(entity_dict[class_label_dict[k]])
		res[:,k]=[entity_dict[class_label_dict[k]][x] for x in seq[:,k]]
	# print(res)
	mask1=(seq!="None")
	return torch.LongTensor(res).to(device),torch.LongTensor(1*mask1).to(device)

def prepare_sequence_diff_sample(seq1,entity_dict,class_label_dict,sample_prob):
	
	length_row=seq1.shape[1]
	index=np.random.choice(length_row, 1, p=sample_prob)[0]
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	if (seq2.shape[0]==0):
		return -1,[],[]
	mask_array=["mask" for i in range(seq2.shape[0])]
	seq=seq2.copy()
	seq[:,index]=np.array(mask_array)
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		res[:,k]=[entity_dict[class_label_dict[k]][x] for x in seq[:,k]]
	mask1=(seq!="None")
	return index,torch.LongTensor(res).to(device),torch.LongTensor(1*mask1).to(device)

def prepare_vertical_sequence(seq1,entity_dict):
	
	length_row=seq1.shape[1]
	index=random.randrange(length_row)
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ], dtype=np.dtype('U1000'))
	# print(seq2.shape)
	if (seq2.shape[0]==0):
		return -1,[],[]
	mask_array=["mask" for i in range(seq2.shape[0])]
	seq=seq2.copy()
	# print(seq)
	# print(mask_array)
	seq[:,index]=np.array(mask_array)
	res=np.zeros(seq.shape)
	# print(seq)
	for k in range(seq.shape[1]):
		res[:,k]=[entity_dict[x] for x in seq[:,k]]
	# print(res)
	mask1=(seq!="None")
	return index,torch.LongTensor(res).to(device),torch.LongTensor(1*mask1).to(device)

def prepare_LSTM_sequence(seq1,entity_dict):
	
	seq=seq1.copy()
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		res[:,k]=[entity_dict[x] for x in seq[:,k]]

	return torch.LongTensor(res[:,:(seq.shape[1]-1)]).to(device),torch.LongTensor(res[:,1:]).to(device)

def prepare_BiLSTM_sequence(seq1,entity_dict):
	
	seq=seq1.copy()
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		res[:,k]=[entity_dict[x] for x in seq[:,k]]

	return torch.LongTensor(res).to(device),torch.LongTensor(res[:,1:-1]).to(device)

def prepare_sequence_berttable(seq1,entity_dict,class_label_dict):
	
	length_row=seq1.shape[1]
	index1=random.randrange(len(entity_dict)-1)
	index=4*index1+4
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	# print(seq2.shape)
	if (seq2.shape[0]==0):
		return -1,[],[]
	mask_array=["mask" for i in range(seq2.shape[0])]
	seq=seq2.copy()
	seq[:,index]=np.array(mask_array)
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		# print(seq[:,k])
		# print(class_label_dict[k])
		# print(entity_dict[class_label_dict[k]])
		if (not(k==0) and (k%4==0)):
			res[:,k]=[entity_dict[class_label_dict[int(k/4)-1]][x] for x in seq[:,k]]
		else:
			res[:,k]=[entity_dict["extra"][x] for x in seq[:,k]]
	# print(res)
	mask1=(seq!="None")
	return index,torch.LongTensor(res).to(device),torch.LongTensor(1*mask1).to(device)	

def prepare_sequence_test_director(seq1,entity_dict,class_label_dict,dir_id=9):
	length_row=seq1.shape[1]
	index=dir_id
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	if (seq2.shape[0]==0):
		return -1,[],[]
	mask_array=["mask" for i in range(seq2.shape[0])]
	seq=seq2.copy()
	seq[:,index]=np.array(mask_array)
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		res[:,k]=[entity_dict[class_label_dict[k]][x] for x in seq[:,k]]
	mask1=(seq!="None")
	return index,torch.LongTensor(res).to(device),torch.LongTensor(1*mask1).to(device)

def prepare_sequence_test_director_berttable(seq1,entity_dict,class_label_dict,dir_id=40):
	length_row=seq1.shape[1]
	index=dir_id
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	if (seq2.shape[0]==0):
		return -1,[],[]
	mask_array=["mask" for i in range(seq2.shape[0])]
	seq=seq2.copy()
	seq[:,index]=np.array(mask_array)
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		if (not(k==0) and (k%4==0)):
			res[:,k]=[entity_dict[class_label_dict[int(k/4)-1]][x] for x in seq[:,k]]
		else:
			res[:,k]=[entity_dict["extra"][x] for x in seq[:,k]]
	mask1=(seq!="None")
	return index,torch.LongTensor(res).to(device),torch.LongTensor(1*mask1).to(device)

def prepare_sequence_label(seq1,index,entity_dict,class_label_dict):
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	seq=seq2.copy()
	res=np.zeros(seq.shape)
	for k in range(seq.shape[1]):
		res[:,k]=[entity_dict[class_label_dict[k]][x] for x in seq[:,k]]
	return torch.LongTensor(res).to(device)

def prepare_vert_sequence1(seq1,index,entity_dict):
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	seq=seq2.copy()
	res=np.zeros(seq.shape[0])
	res=[entity_dict[x] for x in seq[:,index]]
	return torch.LongTensor(res).to(device)

def prepare_sequence1(seq1,index,entity_dict,class_label_dict):
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	seq=seq2.copy()
	res=np.zeros(seq.shape[0])
	res=[entity_dict[class_label_dict[index]][x] for x in seq[:,index]]
	return torch.LongTensor(res).to(device)

def prepare_sequence1_berttable(seq1,index,entity_dict,class_label_dict):
	seq2=np.array([str_k for str_k in seq1  if (str_k[index]!="None") ])
	seq=seq2.copy()
	res=np.zeros(seq.shape[0])
	if ((index%4==0) and (index!=0)):
		res=[entity_dict[class_label_dict[int(index/4)-1]][x] for x in seq[:,index]]
	else:
		res=[entity_dict["extra"][x] for x in seq[:,index]]
	return torch.LongTensor(res).to(device)

def create_embedding_weight(word2vec_path,entity_dict, embedding_dim=300,non_trainable=False):
	
	model=gensim.models.Word2Vec.load(word2vec_path)
	embedding_matrix=np.zeros((len(entity_dict),embedding_dim)) #it maps token id to a vector (embedding)

	if word2vec_path=="COMPLETE_DENORMALISED_drug_filtered_SUBJECTID_10.pkl_300.bin":
		for k in entity_dict:
			if k=="None":
				if "nan" in model.wv:
					embedding_matrix[entity_dict[k]]= model.wv["nan"]
				else:
					embedding_matrix[entity_dict[k]]= np.random.normal(0,1e-3,(embedding_dim))
				continue
			str_k=k.split(" ")
			embedding_matrix[entity_dict[k]]=np.zeros(embedding_dim)
			for j in str_k:
				if j not in model.wv:
					embedding_matrix[entity_dict[k]]= embedding_matrix[entity_dict[k]] + np.random.normal(0,1e-3,(embedding_dim))
				else: 
					embedding_matrix[entity_dict[k]]= embedding_matrix[entity_dict[k]] + model.wv[j]
			embedding_matrix[entity_dict[k]]=(embedding_matrix[entity_dict[k]]/len(str_k))	
	else:
		for k in entity_dict:
			if k not in model.wv:
				#print(k)
				embedding_matrix[entity_dict[k]]= np.random.normal(0,1e-3,(embedding_dim))
			else: 
				embedding_matrix[entity_dict[k]]=model.wv[k]

	emb_layer = nn.Embedding(len(entity_dict), embedding_dim)
	emb_layer.load_state_dict({"weight": torch.tensor(embedding_matrix)})
	if non_trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer

def create_embedding_weight_EmbDi(word2vec_path,entity_dict, embedding_dim=300,non_trainable=False,is_numeric=False,entity_name=""):
	model=gensim.models.Word2Vec.load(word2vec_path)
	embedding_matrix=np.zeros((len(entity_dict),embedding_dim))
	for k in entity_dict:
		if (k not in ["None","mask"]):
			if is_numeric:			
				if (((entity_name=="Month") or (entity_name=="Day"))):
					str_k="tn__"+str(entity_name).replace("_"," ").lower()+"|"+str(int(k)).replace("_"," ").lower().strip()
				elif (entity_name=="movie_id"):
					str_k="tn__"+str(entity_name).replace("_"," ").lower()+"|"+k[1:].replace("_"," ").lower().strip()
				else:
					str_k="tn__"+str(entity_name).replace("_"," ").lower()+"|"+k.replace("_"," ").lower().strip()
				if str_k not in model.wv:
					print(str_k)
					print(entity_name)
					embedding_matrix[entity_dict[k]]= np.random.normal(0,1e-3,(embedding_dim))
				else: 
					embedding_matrix[entity_dict[k]]=model.wv[str_k]
			else:
				str_k="tt__"+k.replace("_"," ").lower().strip()
				if str_k not in model.wv:
					if k!='Nan':
						str_k1="tt__"+str(entity_name).replace("_"," ").lower()+"|"+str(int(float(k))).replace("_"," ").lower().strip()
					else:
						str_k1=str_k
					if str_k1 in model.wv:
						embedding_matrix[entity_dict[k]]=model.wv[str_k1]
					else:
						#print(str_k)
						#print(entity_name)
						embedding_matrix[entity_dict[k]]= np.random.normal(0,1e-3,(embedding_dim))
				else: 
					embedding_matrix[entity_dict[k]]=model.wv[str_k]
			
		else:
			embedding_matrix[entity_dict[k]]= np.random.normal(0,1e-3,(embedding_dim))
	emb_layer = nn.Embedding(len(entity_dict), embedding_dim)
	emb_layer.load_state_dict({"weight": torch.tensor(embedding_matrix)})
	if non_trainable:
		emb_layer.weight.requires_grad = False

	return emb_layer

def create_embedding_weight_model(diag_embedding_path,entity_dict, embedding_dim,non_trainable):
	embedding_matrix=np.zeros((len(entity_dict),embedding_dim))
	embedding_matrix[:(diag_embedding_path.embeddings.weight.data.shape[0])]=diag_embedding_path.embeddings.weight.data.cpu().numpy()
	for k in range(diag_embedding_path.embeddings.weight.data.shape[0],len(entity_dict)):
		embedding_matrix[k]= np.random.normal(0,1e-3,(embedding_dim))
	emb_layer = nn.Embedding(len(entity_dict), embedding_dim)
	emb_layer.load_state_dict({"weight": torch.tensor(embedding_matrix)})
	if non_trainable:
		emb_layer.weight.requires_grad = False
	return emb_layer 

def create_embedding_weight_model_conc(word2vec_path,diag_embedding_path,entity_dict, embedding_dim,non_trainable):
	model=gensim.models.Word2Vec.load(word2vec_path)
	embedding_matrix_w2v=np.zeros((len(entity_dict),embedding_dim))
	for k in entity_dict:
		if k not in model.wv:
			embedding_matrix_w2v[entity_dict[k]]= np.random.normal(0,1e-3,(embedding_dim))
		else: 
			embedding_matrix_w2v[entity_dict[k]]=model.wv[k]
	embedding_matrix_mod=np.zeros((len(entity_dict),embedding_dim))
	embedding_matrix_mod[:(diag_embedding_path.embeddings.weight.data.shape[0])]=diag_embedding_path.embeddings.weight.data.cpu().numpy()
	for k in range(diag_embedding_path.embeddings.weight.data.shape[0],len(entity_dict)):
		embedding_matrix_mod[k]= np.random.normal(0,1e-3,(embedding_dim))
	emb_layer = nn.Embedding(len(entity_dict), 2*embedding_dim)
	emb_layer.load_state_dict({"weight": torch.tensor(np.concatenate((embedding_matrix_w2v,embedding_matrix_mod),axis=1))})
	if non_trainable:
		emb_layer.weight.requires_grad = False
	return emb_layer 

def return_embedding_weight_model(diag_embedding_path,entity_dict, embedding_dim):
	embedding_matrix=np.zeros((len(entity_dict),embedding_dim))
	embedding_matrix[:(diag_embedding_path.embeddings.weight.data.shape[0])]=diag_embedding_path.embeddings.weight.data.cpu().numpy()
	for k in range(diag_embedding_path.embeddings.weight.data.shape[0],len(entity_dict)):
		embedding_matrix[k]= np.random.normal(0,1e-3,(embedding_dim))
	return embedding_matrix

def create_embedding_weight_matrix(embedding_matrix,entity_dict, embedding_dim,non_trainable):
	emb_layer = nn.Embedding(len(entity_dict), embedding_dim)
	emb_layer.load_state_dict({"weight": torch.tensor(embedding_matrix)})
	if non_trainable:
		emb_layer.weight.requires_grad = False
	return emb_layer 

