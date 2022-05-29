# 📙Chatbot

## 📘Chatbot_v1.0

### 📗요구사항
>욕설을 구분하기 위한 Data를 위해 git clone을 해야한다

- transformer 사용을 위한 철치
```
pip install transformers
```
- 욕설 Data
```
git clone https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
git clone https://github.com/doublems/korean-bad-words	
git clone https://github.com/organization/Gentleman/
```

### 📗목표
> 주어진 Data와 Pretrain된 BERT모델를 활용하여 주어진 문장에 대하여 옳바른 답변을 출력한다.

## 📘Chatbot_v2.0

### 📗요구사항
>Easy 버전과 요구사항은 같다.

### 📗목표
> 주어진 Data를 모델을 통해 학습시켜 label을 추출하고, 해당 label을 통해 답변의 범위를 줄여서 정확도를 올린다.
