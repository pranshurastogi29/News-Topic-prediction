import streamlit as st
from transformers import pipeline
import time
import pandas as pd
import numpy as np
import joblib
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import pickle 

class ToxicSimpleNNModel(nn.Module):

    def __init__(self, path):
        super(ToxicSimpleNNModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(in_features=self.backbone.pooler.dense.out_features*2,out_features=8)
        
    def forward(self, input_ids, attention_masks):
        seq_x, _= self.backbone(input_ids=input_ids, attention_mask=attention_masks, return_dict=False)
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)
        return self.linear(x)

def load_topic_model(base_path, model_path):
  net = ToxicSimpleNNModel(base_path)
  net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
  return net

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def Topic_generation_load(base_path, model_path):
    print('loading topic_model')
    model = load_topic_model(base_path, model_path)
    return model


def predict_topic(text, tokenizer, model):
  encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=512, 
            pad_to_max_length=True
        )
  tokens = torch.tensor(encoded['input_ids']).unsqueeze(0)
  attention_masks = torch.tensor(encoded['attention_mask']).unsqueeze(0)
  outputs = model(tokens, attention_masks)
  topics = nn.functional.sigmoid(outputs).detach().numpy()
  _ , topics = torch.topk(torch.tensor(topics), dim = 1, k = 3)
  topics = np.array(topics)
  top_dic = {'0':'business','1':'elections','2':'entertainment',
             '3':'news','4':'opinion','5':'sci-tech','6':'society',
             '7':'sport'}
  l = []
  for i in topics[0]:
    l.append(top_dic[str(i)])
  return l

with open('tokenizer.obj' , 'rb') as f: 
    tokenizer = pickle.load(f)

model = Topic_generation_load('tiny-bert' ,'model.bin')

st.title('News Topic Prediction')

if st.checkbox('Start Generate Summary'):
    text = st.text_input('Enter News here:',key=0)

st.markdown('Once done you can get the top Topics the news relate too')

if st.button('Predict Topics'):
  l = predict_topic(text, tokenizer, model)
  st.markdown('Top topics related are '+' , '.join(l))
else: pass