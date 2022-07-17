import pandas as pd
import numpy as np
import gensim
import sys
import pickle
import random
random.seed(41)
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_MLM_eval_sequence(seq1,seq2,table_a,table_b,tab_col_dict,tab_inv_dict,entity_dict,index,max_seq_len=32,class_label_dict=None):
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
#     # print(res)
	mask1=(total_seq!="<NONE>")
	# print(res)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  

def get_knowns(complete_drug_dict,subject):
	ks = [complete_drug_dict[a] for a in subject]
	lens = [len(x) for x in ks]
	max_lens = max(lens)
	ks = [np.pad(x, (0, max_lens-len(x)), 'edge') for x in ks]
	result = np.array(ks)
	return torch.LongTensor(result).to(device)

def test_model(start_index,tab_col_dict,tab_inv_dict,total_entity_dict,batch_size,valid_ADM_DRGCODES_ADM_array,valid_ADM_DRGCODES_DRGCODES_array,attention_model,subject_entity_drug_dict,ADM_DRGCODES,ranks_array,only_drug_dict_array):
	# ranks_array=[]
	subject_id_index=1
	scoring_function_minimum_score=-10
	for doc_no in range(int((len(valid_ADM_DRGCODES_ADM_array)+batch_size-1)/batch_size)):
		max_id=min(((doc_no+1)*batch_size),len(valid_ADM_DRGCODES_ADM_array))
		range_id=max_id-(doc_no*batch_size)
		documents1=np.array(valid_ADM_DRGCODES_ADM_array[(doc_no*batch_size):max_id])
		documents2=np.array(valid_ADM_DRGCODES_DRGCODES_array[(doc_no*batch_size):max_id])
		training_docs,col_docs,table_docs,none_index,targets_MLM,seq_None=prepare_MLM_eval_sequence(documents1.reshape(range_id,-1),documents2.reshape(range_id,-1),start_index,4,tab_col_dict,tab_inv_dict,total_entity_dict,ADM_DRGCODES)
	#     print(col_docs.shape)
		if (len(seq_None)==0):
			continue
		res=attention_model.forward_MLM(training_docs,col_docs,table_docs,none_index,ADM_DRGCODES)
		actual_target=res[torch.arange(res.size(0)), targets_MLM].view(-1,1)
		knowns=get_knowns(subject_entity_drug_dict,training_docs[:,subject_id_index].reshape(-1).cpu().numpy())
		res.scatter_(1, knowns, scoring_function_minimum_score)
		res_new=res[:,only_drug_dict_array]
		ranks= torch.sum((res_new >= actual_target).float(), dim=-1).cpu()
		# print(ranks)
		ranks_array.append(ranks)
	return ranks_array

def test_model_vert(start_index,tab_col_dict,tab_inv_dict,total_entity_dict,batch_size,valid_ADM_DRGCODES_ADM_array,valid_ADM_DRGCODES_DRGCODES_array,attention_model,subject_entity_drug_dict,ADM_DRGCODES,ranks_array,only_drug_dict_array):
	# ranks_array=[]
	subject_id_index=1
	scoring_function_minimum_score=-10
	for doc_no in range(int((len(valid_ADM_DRGCODES_ADM_array)+batch_size-1)/batch_size)):
		max_id=min(((doc_no+1)*batch_size),len(valid_ADM_DRGCODES_ADM_array))
		range_id=max_id-(doc_no*batch_size)
		documents1=np.array(valid_ADM_DRGCODES_ADM_array[(doc_no*batch_size):max_id])
		documents2=np.array(valid_ADM_DRGCODES_DRGCODES_array[(doc_no*batch_size):max_id])
		training_docs,col_docs,table_docs,none_index,targets_MLM,seq_None=prepare_MLM_eval_sequence(documents1.reshape(range_id,-1),documents2.reshape(range_id,-1),start_index,4,tab_col_dict,tab_inv_dict,total_entity_dict,ADM_DRGCODES)
	#     print(col_docs.shape)
		if (len(seq_None)==0):
			continue
		res=attention_model.test_MLM(training_docs,col_docs,table_docs,none_index,ADM_DRGCODES)
		actual_target=res[torch.arange(res.size(0)), targets_MLM].view(-1,1)
		knowns=get_knowns(subject_entity_drug_dict,training_docs[:,subject_id_index].reshape(-1).cpu().numpy())
		res.scatter_(1, knowns, scoring_function_minimum_score)
		res_new=res[:,only_drug_dict_array]
		ranks= torch.sum((res_new >= actual_target).float(), dim=-1).cpu()
		# print(ranks)
		ranks_array.append(ranks)

	return ranks_array

def prepare_MLM_eval_sequence_vert(vert_sent_dict,seq1,seq2,table_a,table_b,tab_col_dict,tab_inv_dict,entity_dict,index,max_seq_len=32,class_label_dict=None):
	BAT=seq1.shape[0]
	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
	pos_seq=[]
	tab_pos_seq=[]
	col_pos_seq=[]
	vert_sen2_total_array=[]
	for i in range(seq2.shape[0]):
		a=table_a
		b=table_b
		col_a=tab_col_dict[tab_inv_dict[a]]
		col_b=tab_col_dict[tab_inv_dict[b]]
		vert_b=vert_sent_dict[tuple(seq2[i].astype("str"))]
		while(len(vert_b)<3):
			none_b=['<NONE>'] * len(col_b)
			vert_b.append(none_b)
# 		print(vert_b)
		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
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
	total_seq=np.array(pos_seq).astype('str')
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	vert_sen2_seq=np.concatenate(vert_sen2_total_array,axis=1)
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
#     # print(res)
	res_vert2=np.zeros(vert_sen2_seq.shape)
	for k in range(res_vert2.shape[0]):
		for k1 in range(res_vert2.shape[1]):
			sen_arr=[]
			for x in vert_sen2_seq[k,k1,:]:
				sen_arr.append(entity_dict[x])   
			res_vert2[k,k1,:]=sen_arr
	mask1=(total_seq!="<NONE>")
	mask_vert2=(vert_sen2_seq!="<NONE>")
	# print(res)
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	return torch.LongTensor(res).to(device),torch.LongTensor(res_vert2).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(1*mask_vert2).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  

def test_model_vert_complete(vert_sent_dict,start_index,tab_col_dict,tab_inv_dict,total_entity_dict,batch_size,valid_ADM_DRGCODES_ADM_array,valid_ADM_DRGCODES_DRGCODES_array,attention_model,subject_entity_drug_dict,ADM_DRGCODES,ranks_array,only_drug_dict_array):
	# ranks_array=[]
	subject_id_index=1
	scoring_function_minimum_score=-10
	for doc_no in range(int((len(valid_ADM_DRGCODES_ADM_array)+batch_size-1)/batch_size)):
		max_id=min(((doc_no+1)*batch_size),len(valid_ADM_DRGCODES_ADM_array))
		range_id=max_id-(doc_no*batch_size)
		documents1=np.array(valid_ADM_DRGCODES_ADM_array[(doc_no*batch_size):max_id])
		documents2=np.array(valid_ADM_DRGCODES_DRGCODES_array[(doc_no*batch_size):max_id])
		training_docs,training_docs_vert1,col_docs,table_docs,none_index,none_index_vert1,targets_MLM,seq_None=prepare_MLM_eval_sequence_vert(vert_sent_dict,documents1.reshape(range_id,-1),documents2.reshape(range_id,-1),start_index,4,tab_col_dict,tab_inv_dict,total_entity_dict,ADM_DRGCODES)
		if (len(seq_None)==0):
			continue
		res=attention_model.forward_MLM(training_docs,col_docs,table_docs,none_index,ADM_DRGCODES,training_docs_vert1,none_index_vert1)
		actual_target=res[torch.arange(res.size(0)), targets_MLM].view(-1,1)
		knowns=get_knowns(subject_entity_drug_dict,training_docs[:,subject_id_index].reshape(-1).cpu().numpy())
		res.scatter_(1, knowns, scoring_function_minimum_score)
		res_new=res[:,only_drug_dict_array]
		ranks= torch.sum((res_new >= actual_target).float(), dim=-1).cpu()
		ranks_array.append(ranks)
	return ranks_array

# def prepare_MLM_eval_sequence_vert_full(vert_sent_dict_real,vert_sent_dict,seq1,seq2,table_a,table_b,tab_col_dict,tab_inv_dict,entity_dict,index,max_seq_len=32,class_label_dict=None):
# 	BAT=seq1.shape[0]
# 	CLS_sent=[["[CLS]"] for i in range(seq2.shape[0])]
# 	SEP_sent=[["[SEP]"] for i in range(seq2.shape[0])]
# 	pos_seq=[]
# 	tab_pos_seq=[]
# 	col_pos_seq=[]
# 	vert_sen2_total_array=[]
# 	for i in range(seq2.shape[0]):
# 		a=table_a
# 		b=table_b
# 		col_a=tab_col_dict[tab_inv_dict[a]]
# 		col_b=tab_col_dict[tab_inv_dict[b]]
# 		vert_a=vert_sent_dict_real[tab_inv_dict[a]][tuple(seq1[i].astype("str"))]
# 		vert_b=vert_sent_dict[tuple(seq2[i].astype("str"))]
# 		while(len(vert_b)<3):
# 			none_b=['<NONE>'] * len(col_b)
# 			vert_b.append(none_b)
# # 		print(vert_b)
# 		sen1=CLS_sent[i]+list(seq1[i])+SEP_sent[i]+list(seq2[i])+SEP_sent[i]
# 		tab_sen1=[a]+[a for i in range(len(seq1[i]))]+[a]+[b for i in range(len(seq2[i]))]+[b]+[b] * (max_seq_len - len(sen1))
# 		vert_sen2_arr=[]
# 		for k in vert_b:
# 			vert_sen2=CLS_sent[i]+['<NONE>' for i in range(len(seq1[i]))]+SEP_sent[i]+list(k)+SEP_sent[i]
# 			vert_sen2=vert_sen2+['<NONE>'] * (max_seq_len - len(vert_sen2))
# 			vert_sen2_arr.append(vert_sen2)
# 		vert_sen2_total_array.append(np.array(vert_sen2_arr).astype('str').reshape(len(vert_b),1,-1))
# 		col_sen1=[42]+col_a+[42]+col_b+[42]+[42] * (max_seq_len - len(sen1))
# 		sen1 += ['<NONE>'] * (max_seq_len - len(sen1))
# 		pos_seq.append(sen1)
# 		tab_pos_seq.append(tab_sen1)
# #         print(len(col_sen1))
# 		if (len(col_sen1)!=32):
# 			print(train_sentences[i])
# 			print(col_sen1)
# 			print(sen1)
# 			print(len(sen1))
# 			return
# 		col_pos_seq.append(col_sen1)
# 	total_seq=np.array(pos_seq).astype('str')
# 	total_tab_seq=tab_pos_seq
# 	total_col_seq=col_pos_seq
# 	vert_sen2_seq=np.concatenate(vert_sen2_total_array,axis=1)
# 	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
# 	targets_MLM=[entity_dict[x] for x in total_seq[:,index]]
# 	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
# 	total_seq[:,index]=np.array(mask_array)
# 	res=np.zeros(total_seq.shape)
# 	for k in range(total_seq.shape[0]):
# 		sen_arr=[]
# 		for x in total_seq[k,:]:
# 			sen_arr.append(entity_dict[x])   
# 		res[k,:]=sen_arr
# #     # print(res)
# 	res_vert2=np.zeros(vert_sen2_seq.shape)
# 	for k in range(res_vert2.shape[0]):
# 		for k1 in range(res_vert2.shape[1]):
# 			sen_arr=[]
# 			for x in vert_sen2_seq[k,k1,:]:
# 				sen_arr.append(entity_dict[x])   
# 			res_vert2[k,k1,:]=sen_arr
# 	mask1=(total_seq!="<NONE>")
# 	mask_vert2=(vert_sen2_seq!="<NONE>")
# 	# print(res)
# 	torch.LongTensor(res).to(device)
# 	torch.LongTensor(total_col_seq).to(device)
# 	torch.LongTensor(total_tab_seq).to(device)
# 	torch.LongTensor(1*mask1).to(device)
# 	return torch.LongTensor(res).to(device),torch.LongTensor(res_vert2).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(1*mask_vert2).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  


def prepare_MLM_eval_sequence_diff(col_index_array,seq1,seq2,table_a,table_b,tab_col_dict,tab_inv_dict,entity_dict,index,max_seq_len=32,class_label_dict=None):
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
	total_seq=np.array(pos_seq).astype('str')
	total_tab_seq=tab_pos_seq
	total_col_seq=col_pos_seq
	seq_None=np.array([str_k for str_k in range(len(total_seq))  if (total_seq[str_k][index]!="None") ])
	targets_MLM=[entity_dict[class_label_dict[col_index_array[index]]][x] for x in total_seq[:,index]]
	mask_array=["<MASK>" for i in range(total_seq.shape[0])]
	total_seq[:,index]=np.array(mask_array)
	res=np.zeros(total_seq.shape)
	for k in range(total_seq.shape[1]):
		sen_arr=[]
		for x in total_seq[:,k]:
			sen_arr.append(entity_dict[class_label_dict[col_index_array[k]]][x])   
		res[:,k]=sen_arr
#     # print(res)
	mask1=(total_seq!="<NONE>")
	torch.LongTensor(res).to(device)
	torch.LongTensor(total_col_seq).to(device)
	torch.LongTensor(total_tab_seq).to(device)
	torch.LongTensor(1*mask1).to(device)
	return torch.LongTensor(res).to(device),torch.LongTensor(total_col_seq).to(device),torch.LongTensor(total_tab_seq).to(device),torch.LongTensor(1*mask1).to(device),torch.LongTensor(targets_MLM).to(device),seq_None  


def test_model_diff(class_label_dict,col_dict,start_index,tab_col_dict,tab_inv_dict,entity_dict,batch_size,valid_ADM_DRGCODES_ADM_array,valid_ADM_DRGCODES_DRGCODES_array,attention_model,subject_entity_drug_dict,ADM_DRGCODES,ranks_array,only_drug_dict_array):
	col_index_net=[col_dict["None"]]+tab_col_dict[tab_inv_dict[start_index]]+[col_dict["None"]]+tab_col_dict[tab_inv_dict[4]]+[col_dict["None"]]
	col_index_net=col_index_net+(32-len(col_index_net))*[col_dict["None"]]
	subject_id_index=1
	scoring_function_minimum_score=-10
	for doc_no in range(int((len(valid_ADM_DRGCODES_ADM_array)+batch_size-1)/batch_size)):
		max_id=min(((doc_no+1)*batch_size),len(valid_ADM_DRGCODES_ADM_array))
		range_id=max_id-(doc_no*batch_size)
		documents1=np.array(valid_ADM_DRGCODES_ADM_array[(doc_no*batch_size):max_id])
		documents2=np.array(valid_ADM_DRGCODES_DRGCODES_array[(doc_no*batch_size):max_id])
		training_docs,col_docs,table_docs,none_index,targets_MLM,seq_None=prepare_MLM_eval_sequence_diff(col_index_net,documents1.reshape(range_id,-1),documents2.reshape(range_id,-1),start_index,4,tab_col_dict,tab_inv_dict,entity_dict,ADM_DRGCODES,32,class_label_dict)
	#     print(col_docs.shape)
		if (len(seq_None)==0):
			continue
		res=attention_model.forward_MLM(training_docs,col_docs,table_docs,none_index,ADM_DRGCODES,col_index_net)
		actual_target=res[torch.arange(res.size(0)), targets_MLM].view(-1,1)
		knowns=get_knowns(subject_entity_drug_dict,training_docs[:,subject_id_index].reshape(-1).cpu().numpy())
		res.scatter_(1, knowns, scoring_function_minimum_score)
		res_new=res[:,only_drug_dict_array]
		ranks= torch.sum((res_new >= actual_target).float(), dim=-1).cpu()
		ranks_array.append(ranks)
	return ranks_array

def prepare_table(tablepath):
	data = pd.read_csv(tablepath, compression='gzip',error_bad_lines=False)
	for k in data.columns:
		if "TIME" in k:
				data=data.drop(k, axis=1)
		elif "ROW_ID" in k:
				data=data.drop(k, axis=1)
		elif "DATE" in k:
				data=data.drop(k, axis=1)
		elif "COMMENTS" in k:
				data=data.drop(k, axis=1)
		elif (k=="ORDERID" or k=="LINKORDERID"):
			data=data.drop(k, axis=1)
	data=data.drop_duplicates()
	return data

def CPT_prepare_table(tablepath):
	data = pd.read_csv(tablepath, compression='gzip',error_bad_lines=False)
	for k in data.columns:
		if "TIME" in k:
				data=data.drop(k, axis=1)
		elif "ROW_ID" in k:
				data=data.drop(k, axis=1)
		elif "DATE" in k:
				data=data.drop(k, axis=1)
		elif "COMMENTS" in k:
				data=data.drop(k, axis=1)
		elif (k=="ORDERID" or k=="LINKORDERID"):
			data=data.drop(k, axis=1)
	data=data.drop("CPT_CD", axis=1)
	data=data.drop("CPT_NUMBER", axis=1)
	data=data.drop("CPT_SUFFIX", axis=1)
	data=data.drop("TICKET_ID_SEQ", axis=1)
	data=data.drop_duplicates()
	return data

def replace_ICD_CODE(table_df,ICD9_path):
	data_ICD9=pd.read_csv(ICD9_path, compression='gzip',error_bad_lines=False)
	data_ICD9_small=data_ICD9[["ICD9_CODE","SHORT_TITLE"]]
	result = pd.merge(table_df, data_ICD9_small, on=['ICD9_CODE'])
	# result = result.drop("ICD9_CODE", axis=1)
	# result = result.drop_duplicates()
	return result

def prepare_input_table(tablepath):
	data_result = pd.read_csv(tablepath, compression='gzip',error_bad_lines=False)
	for k in data_result.columns:
		# print(k)
		j=k
		if "TIME" in k:
				data_result=data_result.drop(k, axis=1)
		elif "ROW_ID" in k:
				data_result=data_result.drop(k, axis=1)
		elif "DATE" in k:
				data_result=data_result.drop(k, axis=1)
		elif "COMMENTS" in k:
				data_result=data_result.drop(k, axis=1)
		elif (k=="ORDERID" or k=="LINKORDERID"):
			data_result=data_result.drop(k, axis=1)
		elif (k=="ORIGINALAMOUNT" or k=="ORIGINALAMOUNTUOM" or k=="ORIGINALRATE" or k=="ORIGINALRATEUOM"):
			data_result=data_result.drop(k, axis=1)
		elif j=="AMOUNT" or j=="RATE" or j=="ORIGINALAMOUNT" or j=="ORIGINALRATE":
				column_data=data_result[j][np.logical_not(np.isnan(data_result[j]))].astype("int")
				data_result[j][np.logical_not(np.isnan(data_result[j]))] = column_data
	# data_result=data_result.drop_duplicates()
	return data_result

def replace_ITEMID(data, data_ITEM_small):
	result = pd.merge(data, data_ITEM_small, on=['ITEMID'])
	result = result.drop("ITEMID", axis=1)
	# result = result.drop_duplicates()
	return result

def replace_CGID(data, data_CGID_small):
	result = pd.merge(data, data_CGID_small, on=['CGID'])
	result = result.drop("CGID", axis=1)
	result = result.drop_duplicates()
	return result

def clean_table(data, subject_index, repeated_index, threshold):
	data_small=data[[subject_index,repeated_index]].drop_duplicates()
	item_id={}
	for k in data_small.values:
		if k[1] not in item_id:
			item_id[k[1]]=[]
		item_id[k[1]].append(k[0])
	
	for t in item_id:
		if(len(item_id[t])<threshold):
			data = data[data[repeated_index] != t]
	return data

def filter_patient_data(threshold_file,data_PAT):
	threshold_dict=pickle.load(open(threshold_file,"rb"))
	data_PAT_small=data_PAT['SUBJECT_ID'].drop_duplicates()
	for k in data_PAT_small.values:
		if k not in threshold_dict:
			data_PAT=data_PAT[data_PAT['SUBJECT_ID']!=k]
	return data_PAT

def create_denormalised_table(threshold,use_def=False,datapath="/home/sid/mimic3/"):
	# pd.read_csv(datapath+"PATIENTS.csv.gz", compression='gzip',error_bad_lines=False)
	data_ADM=pd.read_csv(datapath+"ADMISSIONS.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	data_PAT=pd.read_csv(datapath+"PATIENTS.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	data_PAT=filter_patient_data(threshold,data_PAT)
	result = pd.merge(data_ADM, data_PAT, on='SUBJECT_ID')
	data_PAT=None
	data_ADM=None
	data_DIAGNOSES_ICD = pd.read_csv(datapath+"DIAGNOSES_ICD.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	# data_DIAGNOSES_ICD = data_DIAGNOSES_ICD[data_DIAGNOSES_ICD["SEQ_NUM"]<=10]
	# data_DIAGNOSES_ICD = clean_table(data_DIAGNOSES_ICD,'SUBJECT_ID',"ICD9_CODE",threshold)
	if use_def:
		data_DIAGNOSES_ICD = replace_ICD_CODE(data_DIAGNOSES_ICD, datapath+"D_ICD_DIAGNOSES.csv.gz")
	result = pd.merge(result, data_DIAGNOSES_ICD, on=['SUBJECT_ID','HADM_ID'])
	data_DIAGNOSES_ICD=None
	print(result.shape)
	data_CPT=pd.read_csv(datapath+"CPTEVENTS.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	result = pd.merge(result, data_CPT, on=['SUBJECT_ID','HADM_ID'])
	data_CPT=None
	print(result.shape)
	# data_services = prepare_table(datapath+"SERVICES.csv.gz")
	# result = pd.merge(result, data_services, on=['SUBJECT_ID','HADM_ID'])
	# print(result.shape)
	data_DRGCODES = pd.read_csv(datapath+"DRGCODES.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	# data_DRGCODES = clean_table(data_DRGCODES,'SUBJECT_ID','DRG_CODE',threshold)
	result = pd.merge(result, data_DRGCODES, on=['SUBJECT_ID','HADM_ID'])
	data_DRGCODES=None
	print(result.shape)
	data_PROCEDURES_ICD = pd.read_csv(datapath+"PROCEDURES_ICD.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	# data_PROCEDURES_ICD = clean_table(data_PROCEDURES_ICD,'SUBJECT_ID',"ICD9_CODE",threshold)
	# data_PROCEDURES_ICD = replace_ICD_CODE(data_PROCEDURES_ICD, datapath+"D_ICD_PROCEDURES.csv.gz")
	result = pd.merge(result, data_PROCEDURES_ICD, on=['SUBJECT_ID','HADM_ID'])
	data_PROCEDURES_ICD=None
	print(result.shape)
	# print(result.shape)
	# result_columns=result.columns
	# data_ICUSTAYS = prepare_table(datapath+"ICUSTAYS.csv.gz")
	# data_ICUSTAYS = data_ICUSTAYS.drop("DBSOURCE", axis=1).drop_duplicates()
	# result = pd.merge(result, data_ICUSTAYS, on=['SUBJECT_ID','HADM_ID'])
	# data_ICUSTAYS = None
	# print(result.shape)
	# data_PRESCRIPTIONS = prepare_table(datapath+"PRESCRIPTIONS.csv.gz")
	# data_PRESCRIPTIONS = data_PRESCRIPTIONS.drop("DRUG_NAME_POE", axis=1)
	# data_PRESCRIPTIONS = data_PRESCRIPTIONS.drop("DRUG_NAME_GENERIC", axis=1)
	# data_PRESCRIPTIONS = data_PRESCRIPTIONS.drop("FORMULARY_DRUG_CD", axis=1)
	# data_PRESCRIPTIONS = data_PRESCRIPTIONS.drop("GSN", axis=1)
	# data_PRESCRIPTIONS = data_PRESCRIPTIONS.drop("NDC", axis=1)
	# data_PRESCRIPTIONS = data_PRESCRIPTIONS.drop("FORM_VAL_DISP", axis=1)
	# data_PRESCRIPTIONS = data_PRESCRIPTIONS.drop("FORM_UNIT_DISP", axis=1).drop_duplicates()
	# data_PRESCRIPTIONS = clean_table(data_PRESCRIPTIONS,'ICUSTAY_ID',"DRUG",threshold)
	# result = pd.merge(result, data_PRESCRIPTIONS, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'])
	# data_PRESCRIPTIONS = None
	# print(result.shape)
	# data_INPUTCV=prepare_input_table(datapath+"INPUTEVENTS_CV.csv.gz")
	# # data_INPUTMV=prepare_input_table(datapath+"INPUTEVENTS_MV.csv.gz")
	# # data_INPUT=data_INPUTCV.append(data_INPUTMV)
	# data_CG=pd.read_csv(datapath+"CAREGIVERS.csv.gz", compression='gzip',error_bad_lines=False)
	# data_CGID_small=data_CG[["CGID","LABEL"]]
	# data_ITEM=pd.read_csv(datapath+"D_ITEMS.csv.gz", compression='gzip',error_bad_lines=False)
	# data_ITEM_small=data_ITEM[["ITEMID","LABEL"]]
	# data_INPUT = clean_table(data_INPUTCV,'ICUSTAY_ID',"ITEMID",threshold)
	# data_INPUT=replace_ITEMID(data_INPUT,data_ITEM_small)
	# data_INPUT=replace_CGID(data_INPUT,data_CGID_small)
	# print(result.columns)
	# print(data_INPUT.columns)
	# result = pd.merge(result, data_INPUT, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'])
	# data_INPUTCV = None
	# data_INPUT = None
	# print(result.shape)
	# data_OUTPUT=prepare_input_table(datapath+"OUTPUTEVENTS.csv.gz")
	# data_OUTPUT = clean_table(data_OUTPUT,'ICUSTAY_ID',"ITEMID",threshold)
	# data_OUTPUT=replace_ITEMID(data_OUTPUT,data_ITEM_small)
	# data_OUTPUT=replace_CGID(data_OUTPUT,data_CGID_small)
	# result = pd.merge(result, data_OUTPUT, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'])
	# data_OUTPUT = None
	# # print(result.shape)
	# # data_PROCEDURE=prepare_input_table(datapath+"PROCEDUREEVENTS_MV.csv.gz")
	# # data_PROCEDURE=replace_ITEMID(data_PROCEDURE,data_ITEM_small)
	# # data_PROCEDURE=replace_CGID(data_PROCEDURE,data_CGID_small)
	# # result = pd.merge(result, data_PROCEDURE, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'])
	print(result.shape)
	print(result.columns)
	return result

def filter_patient_data_vertical(threshold_dict,data_PAT):
	data_PAT_small=data_PAT['SUBJECT_ID'].drop_duplicates()
	for k in data_PAT_small.values:
		if k not in threshold_dict:
			data_PAT=data_PAT[data_PAT['SUBJECT_ID']!=k]
	return data_PAT

def create_vertical_diagnosis_sentence(subject_drug_dict,datapath="/home/sid/mimic3/"):
	data_DIA=pd.read_csv(datapath+"DIAGNOSES_ICD.csv.gz", compression='gzip',error_bad_lines=False)
	data_DIA_small=filter_patient_data_vertical(subject_drug_dict,data_DIA)
	subject_diag_dict={}
	for k in data_DIA_small.values:
		if k[1] not in subject_diag_dict:
			subject_diag_dict[k[1]]={}
		subject_diag_dict[k[1]][k[3]]=k[4]
	vertical_diag_sentences=[]
	for k in subject_diag_dict:
		current_sentence=[]
		dict1=sorted(subject_diag_dict[k])
		for j in dict1:
			current_sentence.append(str(subject_diag_dict[k][j]))
		vertical_diag_sentences.append(current_sentence)
	return vertical_diag_sentences

def create_vertical_proc_sentence(subject_drug_dict,datapath="/home/sid/mimic3/"):
	data_PRO=pd.read_csv(datapath+"PROCEDURES_ICD.csv.gz", compression='gzip',error_bad_lines=False)
	data_PRO_small=filter_patient_data_vertical(subject_drug_dict,data_PRO)
	subject_PRO_dict={}
	for k in data_PRO_small.values:
		if k[1] not in subject_PRO_dict:
			subject_PRO_dict[k[1]]={}
		if k[3]=="nan":
			print(k)
		subject_PRO_dict[k[1]][k[3]]=k[4]
	# print(subject_dia/_dict)
	vertical_PRO_sentences=[]
	for k in subject_PRO_dict:
		current_sentence=[]
		dict1=sorted(subject_PRO_dict[k])
		for j in dict1:
			current_sentence.append(str(subject_PRO_dict[k][j]))
		vertical_PRO_sentences.append(current_sentence)
	return vertical_PRO_sentences

def add_END_pos(vertical_PRO_sentences):
	corr_vertical_PRO_sentences=[]
	for k in vertical_PRO_sentences:
		k.append("<EOS>")
		corr_vertical_PRO_sentences.append(k)
	return corr_vertical_PRO_sentences

def add_END_STR(vertical_PRO_sentences):
	corr_vertical_PRO_sentences=[]
	for k in vertical_PRO_sentences:
		k.append("<EOS>")
		corr_vertical_PRO_sentences.append(["<BOS>"]+k)
	return corr_vertical_PRO_sentences

def max_pad(vertical_PRO_sentences,max_seq_len=10):
	for k in range(len(vertical_PRO_sentences)):
		vertical_PRO_sentences[k] += ['None'] * (max_seq_len - len(vertical_PRO_sentences[k]))
	return vertical_PRO_sentences

def concat_EMBDI_table(threshold,use_def=False,datapath="/home/sid/mimic3/"):
	data_ADM=pd.read_csv(datapath+"ADMISSIONS.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	data_PAT=pd.read_csv(datapath+"PATIENTS.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	data_PAT=filter_patient_data(threshold,data_PAT)
	result = pd.concat([data_ADM, data_PAT], ignore_index=True, sort=True)
	data_PAT=None
	data_ADM=None
	data_DIAGNOSES_ICD = pd.read_csv(datapath+"DIAGNOSES_ICD.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	if use_def:
		data_DIAGNOSES_ICD = replace_ICD_CODE(data_DIAGNOSES_ICD, datapath+"D_ICD_DIAGNOSES.csv.gz")
	result = pd.concat([result, data_DIAGNOSES_ICD], ignore_index=True, sort=True)
	data_DIAGNOSES_ICD=None
	print(result.shape)
	data_CPT=pd.read_csv(datapath+"CPTEVENTS.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	result = pd.concat([result, data_CPT], ignore_index=True, sort=True)
	data_CPT=None
	print(result.shape)
	data_DRGCODES = pd.read_csv(datapath+"DRGCODES.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	result = pd.concat([result, data_DRGCODES], ignore_index=True, sort=True)
	data_DRGCODES=None
	print(result.shape)
	data_PROCEDURES_ICD = pd.read_csv(datapath+"PROCEDURES_ICD.csv.gz", compression='gzip',error_bad_lines=False).drop("ROW_ID", axis=1)
	result = pd.concat([result, data_PROCEDURES_ICD], ignore_index=True, sort=True)
	data_PROCEDURES_ICD=None
	print(result.shape)
	print(result.columns)
	return result

def make_sen_pair(save_dict,table_ind1,table_ind2):
	total_sentence_pair=[]
	for k in save_dict:
		for j in range(len(save_dict[k])):
			k_neg1=random.choice(list(save_dict.keys()))
			while(k_neg1==k):
				k_neg1=random.choice(list(save_dict.keys())) 
			neg_j1=random.choice([i for i in range(0,len(save_dict[k_neg1]))])
			sen_neg1=save_dict[k_neg1][neg_j1][1]
			doc_trip1=[save_dict[k][j][0],save_dict[k][j][1],sen_neg1]
			k_neg2=random.choice(list(save_dict.keys()))
			while(k_neg2==k):
				k_neg2=random.choice(list(save_dict.keys()))
			neg_j2=random.choice([i for i in range(0,len(save_dict[k_neg2]))])
			sen_neg2=save_dict[k_neg2][neg_j2][0]
			doc_trip2=[save_dict[k][j][1],save_dict[k][j][0],sen_neg2]
			total_sentence_pair.append([doc_trip1,doc_trip2])
	random.shuffle(total_sentence_pair)
	train_sentence_pair_new=total_sentence_pair[:int(0.8*len(total_sentence_pair))]
	valid_sentence_pair_new=total_sentence_pair[int(0.8*len(total_sentence_pair)):int(0.9*len(total_sentence_pair))]
	test_sentence_pair_new=total_sentence_pair[int(0.9*len(total_sentence_pair)):]
	train_sentence_pair=[]
	valid_sentence_pair=[]
	test_sentence_pair=[]
	train_table_pair=[]
	valid_table_pair=[]
	test_table_pair=[]
	for k in train_sentence_pair_new:
		train_sentence_pair.append(k[0])
		train_table_pair.append([table_ind1,table_ind2,table_ind2])
		train_sentence_pair.append(k[1])
		train_table_pair.append([table_ind2,table_ind1,table_ind1])
	for k in valid_sentence_pair_new:
		valid_sentence_pair.append(k[0])
		valid_table_pair.append([table_ind1,table_ind2,table_ind2])
		valid_sentence_pair.append(k[1])
		valid_table_pair.append([table_ind2,table_ind1,table_ind1])
	for k in test_sentence_pair_new:
		test_sentence_pair.append(k[0])
		test_table_pair.append([table_ind1,table_ind2,table_ind2])
		test_sentence_pair.append(k[1])
		test_table_pair.append([table_ind2,table_ind1,table_ind1])

	return train_sentence_pair,valid_sentence_pair,test_sentence_pair,train_table_pair,valid_table_pair,test_table_pair

def ADM_train_NSP(table_df_total,table_df1,table_df2,table_ind1,table_ind2):
	data_ADM_PROCEDURES_ICD=table_df_total
	filter_ADM=table_df1
	filter_PROCEDURES_ICD=table_df2
	data_ADM_PROCEDURES_ICD_ADM=data_ADM_PROCEDURES_ICD[filter_ADM.columns]
	data_ADM_PROCEDURES_ICD_PROCEDURES_ICD=data_ADM_PROCEDURES_ICD[filter_PROCEDURES_ICD.columns]
	data_ADM_PROCEDURES_ICD_val=data_ADM_PROCEDURES_ICD.values
	data_ADM_PROCEDURES_ICD_ADM_val=data_ADM_PROCEDURES_ICD_ADM.values
	data_ADM_PROCEDURES_ICD_PROCEDURES_ICD_val=data_ADM_PROCEDURES_ICD_PROCEDURES_ICD.values
	save_dict={}
	for k in range(data_ADM_PROCEDURES_ICD_val.shape[0]):
		if (data_ADM_PROCEDURES_ICD_ADM_val[k][0]+"_"+data_ADM_PROCEDURES_ICD_ADM_val[k][1]) not in save_dict:
			save_dict[(data_ADM_PROCEDURES_ICD_ADM_val[k][0]+"_"+data_ADM_PROCEDURES_ICD_ADM_val[k][1])]=[]
		save_dict[(data_ADM_PROCEDURES_ICD_ADM_val[k][0]+"_"+data_ADM_PROCEDURES_ICD_ADM_val[k][1])].append([data_ADM_PROCEDURES_ICD_ADM_val[k],data_ADM_PROCEDURES_ICD_PROCEDURES_ICD_val[k]])
	return make_sen_pair(save_dict,table_ind1,table_ind2)

def PAT_train_NSP(table_df_total,table_df1,table_df2,table_ind1,table_ind2):
	data_PAT_PROCEDURES_ICD=table_df_total
	filter_PAT=table_df1
	filter_PROCEDURES_ICD=table_df2
	data_PAT_PROCEDURES_ICD_PAT=data_PAT_PROCEDURES_ICD[filter_PAT.columns]
	data_PAT_PROCEDURES_ICD_PROCEDURES_ICD=data_PAT_PROCEDURES_ICD[filter_PROCEDURES_ICD.columns]
	data_PAT_PROCEDURES_ICD_val=data_PAT_PROCEDURES_ICD.values
	data_PAT_PROCEDURES_ICD_PAT_val=data_PAT_PROCEDURES_ICD_PAT.values
	data_PAT_PROCEDURES_ICD_PROCEDURES_ICD_val=data_PAT_PROCEDURES_ICD_PROCEDURES_ICD.values
	save_dict={}
	for k in range(data_PAT_PROCEDURES_ICD_val.shape[0]):
		if (data_PAT_PROCEDURES_ICD_PAT_val[k][0]) not in save_dict:
			save_dict[(data_PAT_PROCEDURES_ICD_PAT_val[k][0])]=[]
		save_dict[(data_PAT_PROCEDURES_ICD_PAT_val[k][0])].append([data_PAT_PROCEDURES_ICD_PAT_val[k],data_PAT_PROCEDURES_ICD_PROCEDURES_ICD_val[k]])
	return make_sen_pair(save_dict,table_ind1,table_ind2)

