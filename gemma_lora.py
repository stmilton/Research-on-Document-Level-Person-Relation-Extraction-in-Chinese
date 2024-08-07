import json
import pandas as pd
from transformers import BitsAndBytesConfig,AutoTokenizer, TrainingArguments, AutoModelForCausalLM
import torch
from datasets import Dataset,DatasetDict
# from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, TaskType
from peft import get_peft_model
from huggingface_hub import login
from trl import SFTTrainer
from sklearn.metrics.pairwise import cosine_similarity

access_token = "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"
login(token = access_token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token, add_eos_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token, quantization_config=bnb_config, device_map={"":0})

lora_config  = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_df = pd.read_csv("./CommonCrawl/data/test/train.csv", encoding='utf-8')
print("train_df:",len(train_df))
valid_df = pd.read_csv("./CommonCrawl/data/test/valid.csv", encoding='utf-8')
print("valid_df:",len(valid_df))

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

def formatting_func(example):
    max_seq_length = 2000
    output_texts = []
    for i in range(len(example['raw_content'])):
        if len(example['raw_content'][i])>max_seq_length:
            example['raw_content'][i]=example['raw_content'][i][:max_seq_length]

        text="""<bos><start_of_turn>user
        請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事)?且兩位關係人皆必須有明確名字，只有稱謂的不算。\n
        若無關係直接回答:無 即可\n
        若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係\n
        文章如下:\n
        {raw_content}<end_of_turn>\n
        <start_of_turn>model
        有 {output}
        """
        output=''
        labels = json.loads(example['label'][i])
        for j in range(len(labels)):
            output += str(tuple(labels[j]))
            if j != len(labels)-1:
                output += ','
        text = text.format(raw_content=example['raw_content'][i],output=output)

        output_texts.append(text)
    return output_texts

# def compute_metrics(predictions):
#     generated_texts = predictions.predictions
#     references = valid_df['output'].tolist()
#     similarities = [cosine_similarity(generated_text, reference) for generated_text, reference in zip(generated_texts, references)]
#     avg_similarity = sum(similarities) / len(similarities)
#     print("avg_similarity:",avg_similarity)
#     return {"cosine_similarity": avg_similarity}

training_args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=2,
        # max_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        output_dir="/milton/sdb/ckpt/gemma",
        optim="paged_adamw_8bit",
        save_total_limit=1,
        save_steps=10,
        save_strategy = 'steps',
        eval_steps=10,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=training_args,
    peft_config=lora_config ,
    formatting_func=formatting_func,
    max_seq_length=2048,
    # compute_metrics=compute_metrics,

)
trainer.train()