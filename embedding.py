# 添加torch导入
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import jieba


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-base-zh-v1.5')

def generate_embeddings(text):
    # 统一处理文本输入为字符串
    if isinstance(text, list):
        text = " ".join(text)  # 将列表转换为字符串
    elif not isinstance(text, str):
        raise ValueError("Input must be string or list of strings")

    # 确保文本编码为UTF-8
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    
    # 添加中文分词处理（需要安装jieba）
    segmented = " ".join(jieba.cut(text))
    
    inputs = tokenizer(segmented, return_tensors="pt",  # 使用处理后的文本
                      padding=True, 
                      truncation=True,
                      max_length=512)
    
    # 使用正确的torch引用
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0].numpy()

def save_to_faiss(embeddings, index):
    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)
