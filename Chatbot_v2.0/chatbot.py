from transformers import AutoModel, AutoTokenizer, BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from utils import *

def main():
        #Data 불러오기
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 비속어 사전 01
    slang_list_01 = []
    f = open("./List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/ko", 'r')
    lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "")
        slang_list_01.append(line)
    f.close()

    # 비속어 사전 02
    slang_list_02 = []
    f = open("./korean-bad-words/korean-bad-words.md", 'r')
    lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "")
        slang_list_02.append(line)
    f.close()

    # 비속어 사전 03
    with open('./Gentleman/resources/badwords.json') as json_file:
        json_data = json.load(json_file)
    slang_list_03 = json_data["badwords"]

    # 비속어 사전 통합
    slang_list = slang_list_01 + slang_list_02 + slang_list_03
    slang_list = list(set(slang_list))
    
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
        
    #label별 answer datalist 생성   
    chatbot_zero_Alist = []
    for data in dataset_zero['answer']:
        chatbot_zero_Alist.append(data)

    chatbot_one_Alist = []
    for data in dataset_one['answer']:
        chatbot_one_Alist.append(data)

    chatbot_two_Alist = []
    for data in dataset_two['answer']:
        chatbot_two_Alist.append(data)

    #각 data tokenizing
    data_cls_zero = []
    data_cls_one = []
    data_cls_two = []

    with open('./data_cls/data_cls_zero.pkl','rb') as f:
        data_cls_zero = pickle.load(f)

    with open('./data_cls/data_cls_one.pkl','rb') as f:
        data_cls_one = pickle.load(f)
        
    with open('./data_cls/data_cls_two.pkl','rb') as f:
        data_cls_two = pickle.load(f)
    
    config = BertConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 3
    model = BertForSequenceClassification(config) 
    # model.load_state_dict(model_state)
    model.load_state_dict(torch.load('model_state_dict.pth'))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    #질문 입력, tokeniz
       
    print("챗봇 곰시리🐻 시작합니다")
    print("종료하려면 'quit'이나 '끝'을 입력하세요")
    
    while(True) :
        query =input("나의 질문 : ".rjust(30))
        if query == "quit" or query =="끝": break
        
        query_cls_hidden = get_cls_token(query, model_cls, tokenizer_cls)
    
        #답변 생성, 출력
        if find_Badword(query, slang_list) :
            print('욕설이 포함된 질문입니다')

        elif sentences_predict(query, model, tokenizer, device) == 0:
            cos_sim = cosine_similarity(query_cls_hidden, data_cls_zero)

            top_question = np.argmax(cos_sim)
            print('🐻 : ', chatbot_zero_Alist[top_question])

        elif sentences_predict(query, model, tokenizer, device) == 1:
            cos_sim = cosine_similarity(query_cls_hidden, data_cls_one)

            top_question = np.argmax(cos_sim)

            print('🐻 : ', chatbot_one_Alist[top_question])

        elif sentences_predict(query, model, tokenizer, device) == 2:
            cos_sim = cosine_similarity(query_cls_hidden, data_cls_two)

            top_question = np.argmax(cos_sim)

            print('🐻 : ', chatbot_two_Alist[top_question])
            
if __name__ == "__main__":
    main()