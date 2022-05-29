import random
from transformers import AutoModel, AutoTokenizer, BertTokenizer
import torch
import numpy as np
import os

def seed_everything(seed: int = 4523):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["WANDB_DISABLED"] = "true"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

#predict함수
def sentences_predict(sent, model, tokenizer, device):
    model.eval()
    tokenized_sent = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )
    tokenized_sent.to(device)
    
    with torch.no_grad():# 그라디엔트 계산 비활성화
        outputs = model(
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits)
    return result


def get_cls_token(sent, model_cls, tokenizer_cls): 
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


def find_Badword(sent, slang_list):
    Badin = False
    for word in slang_list :
        if word in sent :
            Badin = True
    return Badin