from transformers import AutoModel, AutoTokenizer, BertTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import pickle

from utils import *

def main():
    dataset = pd.read_csv('ChatbotData.csv')
    
    dataset.drop_duplicates(subset=['Q'], inplace=True)
    #label 별 데이터
    dataset_zero=pd.DataFrame({"Question":dataset['Q'][:5261], "answer":dataset['A'][:5261]})
    dataset_one=pd.DataFrame({"Question":dataset['Q'][5261:8743], "answer":dataset['A'][5261:8743]})
    dataset_two=pd.DataFrame({"Question":dataset['Q'][8743:], "answer":dataset['A'][8743:]})
    
        #cls 유사도 구하는 부분
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer_cls = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_cls = AutoModel.from_pretrained(MODEL_NAME)

    #label별 question datalist 생성
    chatbot_zero_Qlist = []
    for data in dataset_zero['Question']:
        chatbot_zero_Qlist.append(data)

    chatbot_one_Qlist = []
    for data in dataset_one['Question']:
        chatbot_one_Qlist.append(data)

    chatbot_two_Qlist = []
    for data in dataset_two['Question']:
        chatbot_two_Qlist.append(data)

    #각 data tokenizing
    data_cls_zero = []
    data_cls_one = []
    data_cls_two = []

    for q in tqdm(chatbot_zero_Qlist) :
        q_cls = get_cls_token(q, model_cls, tokenizer_cls)
        data_cls_zero.append(q_cls)
    data_cls_zero = np.array(data_cls_zero).squeeze(axis=1)

    for q in tqdm(chatbot_one_Qlist) :
        q_cls = get_cls_token(q, model_cls, tokenizer_cls)
        data_cls_one.append(q_cls)
    data_cls_one = np.array(data_cls_one).squeeze(axis=1)

    for q in tqdm(chatbot_two_Qlist) :
        q_cls = get_cls_token(q, model_cls, tokenizer_cls)
        data_cls_two.append(q_cls)
    data_cls_two = np.array(data_cls_two).squeeze(axis=1)
    
    with open('./data_cls/data_cls_zero.pkl','wb') as f:
        pickle.dump(data_cls_zero,f)
        
    with open('./data_cls/data_cls_one.pkl','wb') as f:
        pickle.dump(data_cls_one,f)
        
    with open('./data_cls/data_cls_two.pkl','wb') as f:
        pickle.dump(data_cls_two,f)
        
if __name__ == "__main__":
    main()