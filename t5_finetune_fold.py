import json
import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback,AdamW
from transformers import DataCollatorForSeq2Seq,MT5Tokenizer, T5Tokenizer,T5ForConditionalGeneration, MT5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import torch
# from datasets import Dataset,DatasetDict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
pretrained_model  = "google/mt5-base"
max_length=1024
tokenizer=MT5Tokenizer.from_pretrained(pretrained_model, max_new_tokens=max_length, truncation=True)
model = MT5ForConditionalGeneration.from_pretrained(pretrained_model)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
class MyGenerationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = tokenizer(features, truncation=True, padding=True,max_length=max_length)
        self.labels = tokenizer(labels, truncation=True, padding=True, max_length=max_length, return_tensors="pt")["input_ids"]
        # for i in range(len(self.features["input_ids"])):
        #     print(tokenizer.decode(self.features["input_ids"][i]))
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.features.input_ids[idx])
        attention_mask = torch.tensor(self.features.attention_mask[idx])
        labels = torch.tensor(self.labels[idx]).clone().detach()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __len__(self):
        return len(self.labels)
df = pd.read_csv("./CommonCrawl/data/test/test_ckip_expansion.csv",encoding='utf-8',index_col=0)
df = df[(df["merge_label_1024"].notnull())&(df["merge_label_1024"] != "[]")]

# 5 倍交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 儲存每次交叉驗證的數據集
splits = []

for train_index, test_index in kf.split(df):
    train_valid_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    # 再次劃分train_df為訓練集和驗證集
    train_valid_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_sub_index, valid_index in train_valid_kf.split(train_valid_df):
        train_df = train_valid_df.iloc[train_sub_index]
        valid_df = train_valid_df.iloc[valid_index]
        break
    
    splits.append((train_df, valid_df, test_df))

# splits 現在包含了 5 個 (train_df, valid_df, test_df) 的元組
for i, (train_df, valid_df, test_df) in enumerate(splits):
    print(f"Fold {i+1}:")
    print(f"Train size: {train_df.shape}")
    print(f"Valid size: {valid_df.shape}")
    print(f"Test size: {test_df.shape}")
    print()
    
    train_df.to_csv(f"./CommonCrawl/data/kfold2/{i+1}/train.csv", encoding='utf-8', index=True)
    valid_df.to_csv(f"./CommonCrawl/data/kfold2/{i+1}/valid.csv", encoding='utf-8', index=True)
    test_df.to_csv(f"./CommonCrawl/data/kfold2/{i+1}/test.csv", encoding='utf-8', index=True)

# train_df = pd.read_csv("./CommonCrawl/data/test/split_train.csv",encoding='utf-8')
# valid_df = pd.read_csv("./CommonCrawl/data/test/split_valid.csv",encoding='utf-8')  
# train_df = pd.read_csv("./CommonCrawl/data/train/train_ckip_expansion.csv",encoding='utf-8')
# valid_df = pd.read_csv("./CommonCrawl/data/train/valid_ckip_expansion.csv",encoding='utf-8')
# train_df = train_df[(train_df["merge_label_1024"].notnull())&(train_df["merge_label_1024"] != "[]")]
# valid_df = valid_df[(valid_df["merge_label_1024"].notnull())&(valid_df["merge_label_1024"] != "[]")]
# print("train_df",len(train_df))
# print("valid_df",len(valid_df))
# valid_df=valid_df[:5]

    prompt = """請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。
    若無關係直接回答:無 即可
    若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係
    文章如下:
    <extra_id_0> {document}"""
    train_df['prompt'] = train_df['raw_content'].apply(lambda x: prompt.format(document=x[:max_length]))
    valid_df['prompt'] = valid_df['raw_content'].apply(lambda x: prompt.format(document=x[:max_length]))


    def generate_label_text(label):
        labels = json.loads(label)
        label_text = '<extra_id_1> 有 '
        for j in range(len(labels)):
            label_text += str(tuple(labels[j]))
            if j != len(labels) - 1:
                label_text += ','
        return label_text

    train_df['label_text'] = train_df['merge_label_1024'].apply(generate_label_text)
    valid_df['label_text'] = valid_df['merge_label_1024'].apply(generate_label_text)


    # 將DataFrame轉換為datasets.Dataset對象

    train_dataset = MyGenerationDataset(train_df['prompt'].tolist(), train_df['label_text'].tolist())
    valid_dataset = MyGenerationDataset(valid_df['prompt'].tolist(), valid_df['label_text'].tolist())

    def computemetrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print("decoded_preds:", decoded_preds)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("labels:", decoded_labels)
        
        # Compute Cosine Similarity
        pred_vectors = tokenizer.batch_encode_plus(decoded_preds, return_tensors='np', padding='max_length', truncation=True, max_length=max_length)["input_ids"]
        label_vectors = tokenizer.batch_encode_plus(decoded_labels, return_tensors='np', padding='max_length', truncation=True, max_length=max_length)["input_ids"]

        cos_sim = []
        for pred_vec, label_vec in zip(pred_vectors, label_vectors):
            similarity = cosine_similarity([pred_vec], [label_vec])[0][0]
            cos_sim.append(similarity)
        result = {"cosine_similarity": np.mean(cos_sim)}
        return {k: round(v, 4) for k, v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./CommonCrawl/data/kfold/{i+1}/",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0,
        save_total_limit=1,
        logging_dir='./logs',
        logging_steps=100,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy='steps',
        save_steps=1000,
        metric_for_best_model="cosine_similarity",
        greater_is_better=True,
        # generation_max_length=1024
    )

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=computemetrics
    )

    # 开始模型训练
    trainer.train()
