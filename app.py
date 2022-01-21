import streamlit as st
from transformers import pipeline
import time
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def Topic_generation_load():
    print('loading topic_model')
    model = pickle.load('model.obj')
    tokenizer = pickle.load('tokenizer.obj')
    print('topic_model loaded')
    return model , tokenizer

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_summarization_model():
    print('loading summarization model')
    summarization_pipe = pipeline('summarization')
    print('sentiment model loading')
    return summarization_pipe

def get_summarization(text, summarization, max_lenght):
  return summarization(text, max_length=max_lenght)[0]['summary_text']


def predict_topic(text, tokenizer):
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

summarization_pipe = load_summarization_model()
model , tokenizer = Topic_generation_load()

st.title('News Summary Generation and Topic Prediction')

st.markdown('Here you can enter the News in first text box and can get news summary around the subject')

text = st.text_input('Enter News here:',key=0)

if st.checkbox('Start Generate Summary'):
    st.write('uncheck the box if you are done')
    option = st.sidebar.selectbox(label='Max_Lenght',options=['20','40','50'])
    summary = get_summarization(text, summarization_pipe, int(option))
    st.write('Final summary ' + summary)
else: pass

st.markdown('Once done you can get the top Topics the news relate too')

if st.button('Predict Topics'):
  l = predict_topic(text, tokenizer)
  st.markdown('Top topics related are '+' , '.join(l))
else: pass
