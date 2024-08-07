# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login



class GemmaModel:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(self.device)
        
    def generate_text(self, messages, max_length=500, temperature=0.7,do_sample=True):
        # input_ids = self.tokenizer.encode(messages, return_tensors="pt").to(self.device)
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=max_length, temperature=temperature, do_sample=do_sample)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True) # 找到提示的索引位置
        # 返回生成的文本，去除提示部分
        prompt_index = generated_text.rfind("model")        
        return generated_text[prompt_index + len("model\n"):]
    

def main():
    checkpoint = "google/gemma-2b-it"
    my_model = GemmaModel(checkpoint)
    prompt = """
        請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。\n
        若無關係直接回答:無 即可\n
        若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係\n
        文章如下:
        習近平在北京出生並長大，是中華人民共和國開國元老習仲勳與其第二任夫人齊心的長子，也是首位出生在中華人民共和國成立後的中共最高領導人。習近平在北京接受了中小學教育，1969年，因文化大革命對家庭帶來的衝擊而被迫中止學業，作為知識青年前往陝西省延安市延川縣梁家河村參加勞動與工作，在此期間於1974年1月10日加入中國共產黨，並在後期擔任了梁家河的村黨支部書記。1975年進入清華大學化工系就讀，1979年畢業後先後任國務院辦公廳及中央軍委辦公廳秘書。1982年，離京赴河北省正定縣先後任縣委副書記、書記，開始在地方任職。1985年赴福建，先後在福建省廈門市、寧德地區、福州市任職，1999年任福建省人民政府省長，成為正部級官員。2002年起，先後任中共浙江省委書記和中共上海市委書記。2007年10月，當選為中共中央政治局常委和中共中央書記處書記，並先後兼任或當選中共中央黨校校長、國家副主席、黨和國家中央軍委副主席等職務。
        """
    messages = [
        {"role": "user", "content": prompt},
    ]
    generated_text = my_model.generate_text(messages)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()