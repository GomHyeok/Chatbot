from transformers import AutoModel, AutoTokenizer, BertTokenizer
import torch
import sys
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, pipeline
from datasets import load_metric
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import os
import json

from utils import *
from dataloader import *
from model import *

    
# !git clone https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
# !git clone https://github.com/doublems/korean-bad-words
# !git clone https://github.com/organization/Gentleman/

def main():
    
    #비속어 사전 만들기
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
    
    #비속어 골라내는 함수
    def find_Badword(sent) :
        Badin = False
        for word in slang_list :
            if word in sent :
                Badin = True
        return Badin
    
    #Data 불러오기
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    seed_everything()
    
    dataset = pd.read_csv('ChatbotData.csv')
    
    dataset.drop_duplicates(subset=['Q'], inplace=True)
    #label 별 데이터
    dataset_zero=pd.DataFrame({"Question":dataset['Q'][:5261], "answer":dataset['A'][:5261]})
    dataset_one=pd.DataFrame({"Question":dataset['Q'][5261:8743], "answer":dataset['A'][5261:8743]})
    dataset_two=pd.DataFrame({"Question":dataset['Q'][8743:], "answer":dataset['A'][8743:]})
    #random data
    dataset_random = dataset.sample(frac=1).reset_index(drop=True)  # shuffling하고 index reset
    dataset = dataset_random
    
    #각 항목별 data
    train_data_Qusetion = pd.DataFrame({"document" : dataset['Q'][:7000], "label" : dataset['label'][:7000]})
    train_data_Answer = pd.DataFrame({"document" : dataset['A'][:7000], "label" : dataset['label'][:7000]})
    
    valid_data_Qusetion = pd.DataFrame({"document" : dataset['Q'][7000:9000], "label" : dataset['label'][7000:9000]})
    valid_data_Answer = pd.DataFrame({"document" : dataset['A'][7000:9000], "label" : dataset['label'][7000:9000]})
    
    test_data_Qusetion = pd.DataFrame({"document" : dataset['Q'][9000:], "label" : dataset['label'][9000:]})
    test_data_Answer = pd.DataFrame({"document" : dataset['A'][9000:], "label" : dataset['label'][9000:]})
    #공백제거
    train_data_Qusetion['document'].replace('', np.nan,inplace=True)
    valid_data_Qusetion['document'].replace('', np.nan,inplace=True)
    test_data_Qusetion['document'].replace('', np.nan,inplace=True)
    
    #model 불러오기
    MD_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MD_NAME)
    
    #tokenizer 생성
    tokenizer_train_sentences = tokenizer(
        list(train_data_Qusetion['document']),
        return_tensors="pt",
        padding = True,
        truncation=True,
        add_special_tokens=True
    )
    tokenizer_valid_sentences = tokenizer(
        list(valid_data_Qusetion['document']),
        return_tensors="pt",
        padding = True,
        truncation=True,
        add_special_tokens=True
    )
    tokenizer_test_sentences = tokenizer(
        list(test_data_Qusetion['document']),
        return_tensors="pt",
        padding = True,
        truncation=True,
        add_special_tokens=True
    )
    
    #label 분류
    train_label = train_data_Qusetion['label'].values
    test_label = test_data_Qusetion['label'].values
    valid_label = valid_data_Qusetion['label'].values
    
     #dataset 생성
    train_dataset = SingleSentDataset(tokenizer_train_sentences, train_label)
    valid_dataset = SingleSentDataset(tokenizer_valid_sentences, valid_label)
    test_dataset = SingleSentDataset(tokenizer_test_sentences, test_label)
    
    #training arguments(조건)설정
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='logs',            # directory for storing logs
        logging_steps=10,
        save_steps=300,
        save_total_limit=1,
        metric_for_best_model="accuracy"
    )
    
    #Bert model label 변경
    config = BertConfig.from_pretrained(MD_NAME)
    config.num_labels = 3
    model = BertForSequenceClassification(config) 

    metric = load_metric("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)


    model.to(device)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # training dataset
        #vocab_size=500000
        compute_metrics = compute_metrics
    )

    trainer.train()
    
    nlp_sentence_classif = pipeline('sentiment-analysis',model=model, tokenizer=tokenizer, device=0)

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
        
    #cls tokenizer
    def get_cls_token(sent) : 
      model_cls.eval()
      tokenized_sent = tokenizer_cls(
          sent,
          return_tensors = "pt",
          truncation = True,
          add_special_tokens=True,
          max_length = 128
      )
      with torch.no_grad():
        outputs = model_cls(**tokenized_sent)
      logits = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()
      return logits

    #각 data tokenizing
    data_cls_zero = []
    data_cls_one = []
    data_cls_two = []

    for q in tqdm(chatbot_zero_Qlist) :
      q_cls = get_cls_token(q)
      data_cls_zero.append(q_cls)
    data_cls_zero = np.array(data_cls_zero).squeeze(axis=1)

    for q in tqdm(chatbot_one_Qlist) :
      q_cls = get_cls_token(q)
      data_cls_one.append(q_cls)
    data_cls_one = np.array(data_cls_one).squeeze(axis=1)

    for q in tqdm(chatbot_two_Qlist) :
      q_cls = get_cls_token(q)
      data_cls_two.append(q_cls)
    data_cls_two = np.array(data_cls_two).squeeze(axis=1)
    
    #질문 입력, tokenizing
    query = '라디라펌'
    query_cls_hidden = get_cls_token(query)
    
    #답변 생성, 출력
    if find_Badword(query) :
        print('욕설이 포함된 질문입니다')

    elif sentences_predict(query, model) == 0:
        cos_sim = cosine_similarity(query_cls_hidden, data_cls_zero)

        top_question = np.argmax(cos_sim)

        print('나의 질문: ', query)
        print('저장된 답변: ', chatbot_zero_Alist[top_question])

    elif sentences_predict(query, model) == 1:
        cos_sim = cosine_similarity(query_cls_hidden, data_cls_one)

        top_question = np.argmax(cos_sim)

        print('나의 질문: ', query)
        print('저장된 답변: ', chatbot_one_Alist[top_question])

    elif sentences_predict(query, model) == 2:
        cos_sim = cosine_similarity(query_cls_hidden, data_cls_two)

        top_question = np.argmax(cos_sim)

        print('나의 질문: ', query)
        print('저장된 답변: ', chatbot_two_Alist[top_question])
    
if __name__ == "__main__":
    main()