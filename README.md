# Chatbot
---
## Chatbot_v1.0

### Requirements

- transformer install
```
pip install transformers
```
- 욕설 Data
```
git clone https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
git clone https://github.com/doublems/korean-bad-words	
git clone https://github.com/organization/Gentleman/
```

---
> BERT_Chatbot.ipynb를 통해 욕설 구분기능과 CLS토큰을 통한 유사도 측정을 할 수 있습니다.
---
## Chatbot_v2.0

### Train
```
python3 Chatbot_v2.0/train.py 
```
### Chatbot
```
python3 Chatbot_v2.0/chatbot.py
```

---
> Train과정을 통해 주어진 문장의 label을 구하여 v.10보다 더 정확하게 유사도를 구할 수 있고, 그 기능을 활용하여 대답을 얻을 수 있습니다.
---
