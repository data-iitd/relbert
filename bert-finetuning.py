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
from tokenizers import decoders

def getTable(qStmt):

    sql_path = "/home/garima/relevant_files/imdb/sql_files/"
    conn = sqlite3.connect(sql_path+'csc455_pre1.db')
    cursor = conn.cursor()

    quertStatement = qStmt
    cursor.execute(quertStatement)

    columnHeaderList = [description[0] for description in cursor.description]
    sentences = []
	
    for result in cursor:
        sen = ""

        for i in range(len(columnHeaderList)):
            val = str(result[i])
            val = val.replace(".","")
            val = val.replace("(", "")
            val = val.replace(")", "")
            sen += columnHeaderList[i].replace("_","")
            sen += " "
            sen += val
            sen += " [COL] "

        sen =sen[:sen.rfind("[COL]")]
        sentences.append(sen)
    
    return sentences



dbName, tableList1, columnHeaderDict = loadConfig()
embeddingDim = int(sys.argv[1])
batch_size = int(sys.argv[2])
table1 = sys.argv[3] #table name

#train MLM over each table participating in join
tableList = table1.split(":")
toProject = "name, year, rank, first_name, movies_genre"

qStmt = "Select "+toProject+" from " 
for tab in  tableList:
    qStmt += tab + ' natural join '

qStmt = qStmt[:qStmt.rfind('natural join ')]
#qStmt += "; "
qStmt += "where rank != 'None'; "	#for cmovies table

print("Query: "+qStmt)

#todo: from sql query read the answers in to a list of sentences
sentences = getTable(qStmt)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer.decoder = decoders.WordPiece()

#add special token
special_tokens_dict = {'additional_special_tokens': ['[COL]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))



inputIds = []
mask = []

#print("sentences: "+str(len(sentences)))
print("A sentence looks like \n"+sentences[0])

inputs = tokenizer(sentences, return_tensors = 'pt', max_length= 200, truncation =True, padding='max_length', add_special_tokens=True)
#inputs = tokenizer(sentences, return_tensors = 'pt', max_length= 300, truncation =True, padding='max_length')


print(inputs['input_ids'][0])
print(tokenizer.decode(inputs['input_ids'][0]))
print(tokenizer.decode([2171]))
print(tokenizer.encode(['[COL]']))
inputs['labels'] = inputs.input_ids.detach().clone()

#Creating masks
rand = torch.rand(inputs.input_ids.shape)

mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0) * (inputs.input_ids != 30522)
inputs['attention_mask'] = mask_arr.clone().detach()

#Correcting labels corresponding to the masked indexes
masked_tokens = []
selection = []
for i in range(inputs.input_ids.shape[0]):
    selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

for i in  range(inputs.input_ids.shape[0]):
#    masked_tokens.append(inputs.input_ids[i,selection[i]])
    inputs.input_ids[i, selection[i]] = 103

#inputs['masked_tokens'] = torch.FloatTensor(masked_tokens)
#inputs['masked_tokens']  = masked_tokens
#print(str(inputs.keys()))
#print("Masked tokens: ",str(len(masked_tokens)))

#todo: comprehend this
class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])

dataset = MeditationsDataset(inputs)
#loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

#print("inputs: "+ str(len(inputs)))
#print("dataset size: "+str(len(dataset)))
#print("inputs.input_id: "+ str(len(dataset['input_ids'])))
#print("dataset: " +str(len(dataset['masked_tokens'])))
#print("loader: "+str(len(loader)))

train_size = int(0.7* len(dataset))
val_size = int(0.15*len(dataset))
test_size = len(dataset) - (train_size + val_size)

#print("Train size: "+str(train_size))
#print("Test size: "+str(test_size))
#print("Val size: "+str(val_size))

#train_dataset, val_dataset, test_dataset = random_split(loader,[train_size, val_size, test_size])
train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size])

filehandler = open('finetune_test.pt', 'wb') 
torch.save(test_dataset, filehandler)

filehandler1 = open('finetune_val.pt', 'wb') 
torch.save(val_dataset, filehandler1)
 
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

#train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
#val_dataloader = DataLoader(val_dataset,sampler = SequentialSampler(val_dataset), batch_size = batch_size)
#test_dataloader = DataLoader(test_dataset,sampler = SequentialSampler(test_dataset), batch_size = batch_size)


device  = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#print("\n\n***Device assigned is:"+str(device))
model.to(device)
model.train()
#print("Training is going to start!!")
optim = AdamW(model.parameters(), lr = 5e-5)
#torch.cuda.empty_cache()

#todo: for debugging
epochs = 10

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(train_dataloader, leave=True)
   # loop = tqdm(loader, leave=True)
   # print("Loop size: "+str(len(loop)))
    for batch in loop:
        #print("Batch size: "+str(len(batch)))
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        #print("Batch inputids size: "+str(len(input_ids)))
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
  #  if (epoch%10) == 0:
  #      model.save_pretrained("bert_finetune_"+str(epoch)+"_"+table1)

model.save_pretrained("bert_finetune_"+str(epochs)+"_"+table1)
tokenizer.save_pretrained("tokenizer_bert_finetune_"+str(epochs)+"_"+table1)
print("Model saved successfully!!")

'''
model.eval()
frank = []
fselection = []

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(val_dataloader, leave=True)
   # loop = tqdm(loader, leave=True)
    print("Eval Loop size: "+str(len(loop)))
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        #print("Batch inputids size: "+str(len(input_ids))+str(input_ids.shape))
        # process
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        print(outputs.loss, outputs.logits.shape)

        sorted_logit = np.argsort(-1*outputs.logits)
        
        
        
        reference = batch['masked_tokens']
        selection = []
        for i in range(input_ids.shape[0]):
            selection.append(torch.flatten(attention_mask[i].nonzero()).tolist())

        for i in  range(input_ids.shape[0]):
            reference.append(labels[i,selection[i]])
        
        rank = []
        for i in range(len(reference)):
            rank[i] = np.where(sorted_logit[i] == reference[i])[0][0] + 1

        frank.append(rank)
        fselection.append(selection)
        # print relevant info to progress bar

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

print(frank)

fhandler = open('val_rank.pkl', 'w') 
pickle.dump(frank, fhandler)
fhandler1 = open('val_column.pkl', 'w') 
pickle.dump(fselection, fhandler1)
'''
