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

#     data = {
#         "model_state" :model.state_dict()
#     }
    
#     FILE = "data.pth"
#     torch.save(data, FILE)
    torch.save(model.state_dict(), 'model_state_dict.pth')
    
if __name__ == "__main__":
    main()