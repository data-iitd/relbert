import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import sys
from init_new import loadConfig
import sqlite3
from sqlite3 import OperationalError
import pickle
import random
from tokenizers import decoders


dbName, tableList1, columnHeaderDict = loadConfig()
embeddingDim = int(sys.argv[1])
batch_size = int(sys.argv[2])
table1 = sys.argv[3] #table name
epo = sys.argv[4]

tokenizer = BertTokenizer.from_pretrained("/home/garima/RelBERT/tokenizer_bert_finetune_natural_"+epo+"_"+table1+"/") #TODO: change tokenizer and model
model = BertForMaskedLM.from_pretrained("/home/garima/RelBERT/bert_finetune_natural_"+epo+"_"+table1+"/")
tokenizer.decoder = decoders.WordPiece()

#For COL tokeizned : 
print("\nInput:" +str(tokenizer.decode([2171,2095,4635,2034,18442,5691,6914, 2890])))
# For natural sentence
#print("\nInput:" +str(tokenizer.decode([3185,2001,2207,1999,1998,2003,4396,2009,2001,2856,2011,1012])))


#todo: comprehend this
class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])
      #  return len(self.encodings.input_ids)


test_dataset = torch.load('finetune_natural_test.pt') #TODO: change file name with different

DEBUG = 0
# 101, 1002, 3411, 1020, 1012, 1018, 2957,  103, 1006, 1045, 1007,  103, 102
# For movie genre filtering
genres = ["comedy", "crime", "family", "drama", "short", "romance", "adult", "action", "thriller", "documentary", "western", "music", "fantasy", "sci-fi", "horror", "war", "musical", "adventure", "animation", "mystery", "film-noir"]

new_input = {'input_ids':[], 'token_type_ids':[] , 'attention_mask':[], 'labels':[]}

for j in range(len(test_dataset)):
#    input_ids_new = test_dataset[j]['input_ids']
    input_ids_new = test_dataset[j]['labels'] # change the previos line
    label_new = test_dataset[j]['labels']
    mask_new = torch.zeros(len(input_ids_new))
    mask =  test_dataset[j]['attention_mask']
    new_input['token_type_ids'].append(test_dataset[j]['token_type_ids'].clone().detach())
    #index = input_ids_new.index(102)
    index =  (input_ids_new == 102).nonzero(as_tuple=True)[0]

# checnk for the special tokens
#mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)* (inputs.input_ids != 30522)

#102,103, 0, 30522, 2171,2095,4635,2034,18442,5691,6914, 2890 avoid them for testing, col names and COL token

    new_index =  random.randint(1,index-1)
#For COL tokenized: 
#    while (input_ids_new[new_index] in [102,103, 0, 30522, 2171,2095,4635,2034,18442,5691,6914, 2890, 6907]):

#For natural sentence
    while (input_ids_new[new_index] in [102,103, 0,3185,2001,2207,1999,1998,2003,4396,2009,2001,2856,2011,1012]):
      new_index = random.randint(1, index-1)
#3185,2001,2207,1999,1998,2003,4396,2009,2001,2856,2011,1012
    if DEBUG:
        print(new_index)
        print(str(input_ids_new)+"\t"+str(len(input_ids_new)))
        print(str(maks)+"\t"+str(len(maks)))

   # input_ids_new = label_new.clone().detach()

    input_ids_new[new_index] = 103
    mask_new[new_index] = 1

    new_input['input_ids'].append(input_ids_new.clone().detach())
    new_input['attention_mask'].append(mask_new.clone().detach())
    new_input['labels'].append(label_new.clone().detach())    

    if DEBUG:
        print(input_ids_new)
        print(mask_new)
   # test_dataset[j]['input_ids'] = input_ids_new.clone().detach()
   # test_dataset[j]['attention_mask'] = mask_new.clone().detach()

    if DEBUG:
        print("New input -ids 1: "+str(new_input['input_ids'][0]))
        print("New input masks 1: "+str(new_input['attention_mask'][0]))
        DEBUG = 0


new_test_dataset = MeditationsDataset(new_input)
new_test_dataloader = DataLoader(new_test_dataset, batch_size = batch_size, shuffle=True)

device  = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()
print("Testing is going to start!!")
optim = AdamW(model.parameters(), lr = 5e-5)

epochs = 1
DEBUG = 0

#fileHander_rank = open('bert_finetune_rank'+table1+".txt", "a")

fileHander_rank = open('bert_finetune_natural_'+str(epo)+'_rank'+table1+".txt", "a")

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(new_test_dataloader, leave=True)

    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)

        # extract loss
        loss = outputs.loss
        reference = []
        selection = []
        sorted_logit = []

        for i in range(input_ids.shape[0]):
            selection.append(torch.flatten(attention_mask[i].nonzero()).tolist())

        for i in  range(input_ids.shape[0]):
            ind = selection[i]
            reference.append(labels[i,ind])
            masked_logit = outputs.logits[i][ind]
            sorted_logit.append( np.argsort(-1*masked_logit.detach().cpu().numpy()))
        
  #      print("LEngths of selection: ",str(len(selection[0])))
  #      print("Length of reference: ",str(len(reference[0]))+"\t"+str(len(reference[1]))+"\t"+str(len(reference[2]))+"\t"+str(reference[0]))
  #      print("Length of sorted_logits: ",str(len(sorted_logit[0])))
        
        rank = []

        if DEBUG:
            print("Input: "+str(input_ids[0]))
            print("Mask: "+str(attention_mask[0]))
            print("Selection: "+str(selection[0]))
            print("Refernece: "+str(reference[0]))
            print("Logit index: "+str(sorted_logit[0]))
            DEBUG = 0

        for i in range(len(reference)):
   #         print("Logit: "+str(sorted_logit[i][0])+"\t"+str(len(sorted_logit[i])))
   #         print("\nReference: "+str(reference[i].item())+"\t"+str(len(reference[i])))
   #         print(np.where(sorted_logit[i][0] == reference[i].item())[0][0])
            actual_token = tokenizer.decode([reference[i].item()])
            filtered_pred_tokens = []
            isSpecial = False
''' for imdb type filtering
            #Filtering the predicted tokens based on their type: filtering for Year and Rank
            if (actual_token.isnumeric()):
             if (len(actual_token) == 4):
              for v in sorted_logit[i][0]:
               v1 = tokenizer.decode([v])
               if (v1.isnumeric() and len(v1) == 4):
                filtered_pred_tokens.append(v)
              isSpecial = True
             
             elif (10 <= int(actual_token) < 100):
              isSpecial = True
              for v in sorted_logit[i][0]:
               v1 = tokenizer.decode([v])
               if (v1.isnumeric() and len(v1)==2):
                filtered_pred_tokens.append(v)

            elif (actual_token in genres and selection[i][0] == 1):
             isSpecial = True
          #   print("Here!!"+"\t"+str(selection[i]))
             for v in sorted_logit[i][0]:
              v1 = tokenizer.decode([v])
              if (v1 in genres):
               filtered_pred_tokens.append(v)

            if reference[i].item() in filtered_pred_tokens:
             print("Found!!")
              
         #   print("\n"+str(reference[i].item())+"\n"+str(filtered_pred_tokens))

          #  print(str(tokenizer.decode([reference[i].item()]))+"\t"+str(isSpecial))
'''        
            if isSpecial:
             arr = np.array(filtered_pred_tokens)
           #  print(np.where(arr == reference[i].item()))
             rank.append(np.where(arr == reference[i].item())[0][0] + 1)
            else:
          #   print(np.where(filtered_pred_tokens == reference[i].item()))
             rank.append(np.where(sorted_logit[i][0] == reference[i].item())[0][0] + 1)

    #        print(str(rank[i])+"\t"+str(selection[i][0])+"\n")
            
         #   print("Ids:"+str(input_ids[i]))
      #      print("\nInput:" +str(tokenizer.decode(input_ids[i])))
            print("Output tokens: "+str(tokenizer.decode([reference[i].item()]))+"\t"+str(tokenizer.decode([sorted_logit[i][0][rank[i]-1]]))+"\t"+str(tokenizer.decode([sorted_logit[i][0][0],sorted_logit[i][0][1],sorted_logit[i][0][3]]))+"\t"+str(rank[i])+"\n")
		
            fileHander_rank.write(str(rank[i])+"\t"+str(selection[i][0])+"\t"+str(isSpecial)+"\n")
       #     exit()
 #       frank.append(rank)
 #       fselection.append(selection)
        # print relevant info to progress bar
       
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

#print(frank)

fileHander_rank.close()

#fhandler = open('test_rank.pkl', 'w') 
#pickle.dump(frank, fhandler)
#fhandler1 = open('test_column.pkl', 'w') 
#pickle.dump(fselection, fhandler1)
