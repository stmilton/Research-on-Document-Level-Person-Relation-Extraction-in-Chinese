import itertools
import json
import random
import pandas as pd
from transformers import DataCollatorForSeq2Seq,MT5Tokenizer, T5Tokenizer,T5ForConditionalGeneration, MT5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from opencc import OpenCC

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
pretrained_model  = "google/mt5-base"
max_length=1024
tokenizer=MT5Tokenizer.from_pretrained(pretrained_model, max_length=max_length, truncation=True)
model = MT5ForConditionalGeneration.from_pretrained(pretrained_model)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
class MyGenerationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = tokenizer(features, truncation=True, padding=True,max_length=max_length)
        self.labels = tokenizer(labels, truncation=True, padding=True, return_tensors="pt")["input_ids"]
        # for i in range(len(self.features["input_ids"])):
        #     print(tokenizer.decode(self.features["input_ids"][i]))
        #     print(tokenizer.decode(self.labels[i]))
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
    
def convert_to_traditional_chinese(df, column_names, new_names=None):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    if new_names:
        for column_name,new_name in zip(column_names,new_names):
            df[new_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
        return df 
    # 将指定列的每个字符串进行繁体转换
    for column_name in column_names:
        df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 
# for i in range(5):
#     train_df = pd.read_csv(f"./CommonCrawl/data/kfold/{i+1}/train.csv",encoding='utf-8',index_col=0)
#     valid_df = pd.read_csv(f"./CommonCrawl/data/kfold/{i+1}/valid.csv",encoding='utf-8',index_col=0)

train_df = pd.read_csv("./CommonCrawl/data/train/train_ckip_expansion.csv",encoding='utf-8',index_col=0)
valid_df = pd.read_csv("./CommonCrawl/data/train/valid_ckip_expansion.csv",encoding='utf-8',index_col=0)
train_df = convert_to_traditional_chinese(train_df,["raw_content"],["trad_raw_content"])
valid_df = convert_to_traditional_chinese(valid_df,["raw_content"],["trad_raw_content"])
train_df = train_df[(train_df["merge_label_1024"].notnull())&(train_df["merge_label_1024"] != "[]")]
valid_df = valid_df[(valid_df["merge_label_1024"].notnull())&(valid_df["merge_label_1024"] != "[]")]
print(f"Train size: {train_df.shape}")
print(f"Valid size: {valid_df.shape}")


def generate_combinations(lst):
    return [tuple(sorted(comb)) for comb in itertools.combinations(lst, 2)]

def relation(df,label_col):
    id=[]
    url=[]
    title=[]
    raw_content=[]
    ner_label=[]
    re_label=[]
    # 訓練資料集，有golden entity
    ori=0
    no=0
    for idx, data in df.iterrows():
        labels= json.loads(data[label_col])
        ori+= len(labels)
        document = data["raw_content"][:max_length]
        trad_raw_content = data["trad_raw_content"][:max_length]

        already_pairs = set([(label[0],label[1]) for label in labels])
        ckip_bert_entity = json.loads(data["ckip_bert_entity"])
        ckip_bert_entity_pairs = generate_combinations(ckip_bert_entity)
        # extra_pairs=[]
        for pair in ckip_bert_entity_pairs:
            # 並非原本的entity_pairs，且未被文章截斷
            if pair not in already_pairs and pair[0] in trad_raw_content and pair[1] in trad_raw_content:
                no+=1
                labels.append([pair[0],pair[1],"沒有"])
        
        random.shuffle(labels)

        for count,label in enumerate(labels):
            id.append(f"{idx}_{count+1}")
            url.append(data['url'])
            title.append(data['title'])
            raw_content.append(data['raw_content'])
            # print(label)
            person1,person2,relation =label
            ner_label.append([person1,person2])
            re_label.append(relation)
    new_df = pd.DataFrame({
        "id":id,
        "url":url,
        "title":title,
        "raw_content":raw_content,
        "ner_label":ner_label,
        "re_label":re_label
    })
    print("4種關係",ori)
    print("沒關係",no)
    return new_df
print("train_df")
train_df = relation(train_df,"merge_label_1024")
print("valid_df")
valid_df = relation(valid_df,"merge_label_1024")
# train_df=train_df[:100]
# valid_df=valid_df[:5]
print("re_train_df",len(train_df))
print("re_valid_df",len(valid_df))

prompt = """根據以下文章，找出{person1}與{person2}中之間的關係。關係分為:親屬關係、師生關係、同事關係、其他關係、沒有關係，共5種。
文章如下:
<extra_id_0> {document}"""

def generate_prompt(row):
    # p1, p2 = eval(row["ner_label"])
    p1, p2 = row["ner_label"]
    return prompt.format(document=row['raw_content'][:max_length], person1=p1, person2=p2)

train_df['prompt'] = train_df.apply(generate_prompt,axis=1)
valid_df['prompt'] = valid_df.apply(generate_prompt,axis=1)

def generate_label_text(value):
    return f"<extra_id_1> {value}"

train_df['re_label'] = train_df['re_label'].apply(generate_label_text)
valid_df['re_label'] = valid_df['re_label'].apply(generate_label_text)

# 將DataFrame轉換為datasets.Dataset對象
train_dataset = MyGenerationDataset(train_df['prompt'].tolist(), train_df['re_label'].tolist())
valid_dataset = MyGenerationDataset(valid_df['prompt'].tolist(), valid_df['re_label'].tolist())

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
    output_dir=f"./mt5/big_data/re/",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-8,
    weight_decay=0,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=10000,
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=10000,
    save_strategy='steps',
    save_steps=10000,
    metric_for_best_model="cosine_similarity",
    greater_is_better=True,
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
