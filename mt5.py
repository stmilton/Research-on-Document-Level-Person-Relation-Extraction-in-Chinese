# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
from transformers import MT5Tokenizer, MT5ForConditionalGeneration,T5Tokenizer,T5ForConditionalGeneration
import torch
from huggingface_hub import login

class Mt5Model:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.tokenizer = MT5Tokenizer.from_pretrained(checkpoint_path)
        self.model = MT5ForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        
       
    def generate_text(self, messages, max_length=512, temperature=0.7,do_sample=True):
        messages = self.apply_chat_template(messages)
        
        input_ids = self.tokenizer.encode(messages, return_tensors="pt").to(self.device)
        # input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=max_length, temperature=temperature, do_sample=do_sample)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True) # 找到提示的索引位置
        # 返回生成的文本，去除提示部分
        prompt_index = generated_text.rfind("<extra_id_1>")        
        return generated_text[prompt_index + len("<extra_id_1>"):]
              
        # return generated_text
    def apply_chat_template(self, messages):
        messages_text=''
        for message in messages:
            if message['role'] == 'user':
                tag = "文章如下:"
                index = message['content'].find(tag)
                if index != -1:
                    messages_text += message['content'][:index+len(tag)] + '\n<extra_id_0> ' + message['content'][index+len(tag):].strip("\n").strip()
                else:
                    messages_text += message['content']
            elif message['role'] == 'model':
                messages_text += '\n<extra_id_0> ' + message['content']

        return messages_text

def main():

    checkpoint = "/milton/sdb/ckpt/mt5/checkpoint-50000"

    my_model = Mt5Model(checkpoint)
    prompt = """請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。
若無關係直接回答:無 即可
若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係
文章如下:
首页 >> 家居 >> 蛋糕设计工作室里的女大学生
蛋糕设计工作室里的女大学生
“鼓励大学生创业”始终是毕业季的高频词，最近教育部在《做好2015年全国普通高校毕业生就业创业工作的通知》中提到，“各地各高校要把创新创业教育作为推进高等教育综合改革的重要抓手，将创新创业教育贯穿人才培养全过程，面向全体大学生开发开设创新创业教育专门课程，纳入学分管理，改进教学方法，增强实际效果。”同时，高校还要“建立弹性学制，允许在校学生休学创业。”在可以预见的未来，大学生创业一定可以为不断刷新历史的“最难就业季”降温。
刚忙完期末考试，陈思臣开始把大部分精力投入到亲手创办的蛋糕设计工作室中。这不，她又接了20多盒马卡龙的订单，这是一种用蛋白、杏仁粉、白砂糖和糖霜所做的法式甜点。与此同时，她还办起了培训课程，辅导零基础学员做蛋糕坯、搭配奶油造型、调制棒棒糖……
陈思臣是北京第二外国语学院英语系2011级本科生，还没到22岁生日。上大学以前，从未接触过蛋糕制作的陈思臣，就在去年秋天，怀揣着“蛋糕梦”，开启了创业旅途。如今，她创办了蛋糕设计工作室，申请了营业执照，当上了自己的CEO，成为世界上最年轻的“艺术蛋糕大师”。她想用自己的实际行动证明：在“蜜罐”里长大的孩子也有梦想，会靠努力找到自己想要的生活。
一台电烤箱 点燃心中的“蛋糕梦”
陈思臣从小对蛋糕有一种近乎痴迷的喜爱，时常在蛋糕房外驻足，看师傅做蛋糕。大三下学期，妈妈参加活动时，抽中了一台电烤箱。就是这样一次偶然的机会，点燃了陈思臣心中的“蛋糕梦”。
她经常在网上搜寻蛋糕制作方法，试着在家里自己烤蛋糕。最初，她以为蛋糕只能做成蛋糕店里卖的那个样子。但是突然有一天，陈思臣在博客上看到一位久居英国的女士发表了几幅艺术蛋糕作品，她被深深地震撼了。“我从来没有想到过原来蛋糕也可以做得和艺术品一样精致！”
从那以后，陈思臣开始往自己的蛋糕里注入创意，不再简单地摆放卡通模型，而是注重蛋糕的整体设计感。只要有闲暇时间，陈思臣就到网上搜寻蛋糕制作工艺，寻找大师。
陈思臣可以把蛋糕做成玻璃效果，浅蓝色蛋糕表面镶嵌着不规则多边形的纹路，看起来像是炸裂的玻璃；她还可以把蛋糕做成游乐场里旋转木马的造型，几匹白马围着小亭子奔跑。在一般人眼中，很难把这些精致的艺术作品与蛋糕联系在一起。
艺术蛋糕来源于英国，可以将蛋糕装饰出各种各样的风格，这是欧美人极其喜爱的蛋糕装饰手法，目前江浙沪一带较为盛行，但是在北京这座城市却鲜为人知。去年9月，陈思臣萌生了创办公司的想法，她想把艺术蛋糕当成事业去追求，让更多的人有所了解。
于是，陈思臣开始了密集的“拜师之旅”。
创业初期饱受争议 踏上孤独的“拜师”之旅
“这是我的一段新的路程，身为一名大学生，我情不自禁地被蛋糕的世界所吸引，开一个蛋糕设计工作室，是我的梦想。”怀揣着蛋糕梦，从没有离开爸妈单独远行的陈思臣，背起行囊，开始“探险”，踏上了难忘的拜师之旅。
毫无经验的陈思臣，完全凭借微博、博客的指引，到全国各地寻找蛋糕“大师”。陈思臣乘坐高铁来到济南，跟Joy老师学习制作一种名叫马卡龙的法式甜点。随后又抵达浙江海宁、广州、上海，学习翻糖技巧，这是制作艺术蛋糕的基础，也让陈思臣开始近距离接触“艺术蛋糕”这一行当。去年10月到12月，整整三个月，被陈思臣安排得满满的。
创业之路并不容易。“当你第一次创业的时候，你会发现：最先相信你的是陌生人，最先鼓励你的是合伙人，最先看不起你的是身边的人，最先不相信你的是你的亲人，打击你最狠的是最爱你的人……”这原本是马云说过的一句话，却在陈思臣创业初期最痛苦、最迷茫的时候，给予她莫大的精神慰藉，这句话实在让陈思臣感同身受。
陈思臣的家人并不支持她创业，认为女孩子应该找一份踏实稳定的工作，一个学翻译的大学生去搞蛋糕了，是一件很丢人的事情。班里同学对陈思臣也有误解，密集拜师求学那段时间，耽误了不少课程，这在同学眼里，是“不务正业”的表现。
面对质疑，陈思臣没有背弃初心，因为有太多人不了解艺术蛋糕，所以才会对她产生误解。然而，当陈思臣第一次拿着为客户量身定做的艺术蛋糕放到家人面前时，她的家人被陈思臣的手艺惊呆了。这时，陈思臣更加坚定了自己的创业路。
终于，去年10月底，工作室的LOGO设计出炉，陈思臣在南二环一处小区里租了一间房子，并在工商局完成营业执照申请……北京Cake Eye（糕之眸）甜品设计有限公司诞生了！
赴英国参加考试 成为最年轻的艺术蛋糕大师
Cake Eye——陈思臣为自己的工作室起了一个很好听的名字。在她看来，蛋糕就像眼睛，为她照见了一片全新的世界。同时，“Eye”的谐音是“爱”，做蛋糕，正是陈思臣所钟爱的事业。
陈思臣说，公司的定位非常明确，面向高端白领市场，专门提供艺术蛋糕定制服务，包括生日派对、婚庆典礼等，客户提供需求，他们负责完成从设计到制作的全部环节。另外，她还打算开设艺术蛋糕培训课程，包括英式翻糖、翻糖彩绘、玻璃蛋糕、零基础入门课等。下个周末，第一期培训课程就要开班了。
陈思臣为公司“招聘”了六名员工，都是她的朋友，其中有两人是二外同学。销售、讲师、宣传、原料配送……分工各有不同，公司逐渐步入了正轨。
在公司刚成立的时候，陈思臣为了提升自己和公司的实力，她专门飞往英国，参加“英国皇家蛋糕装饰艺术资格认证”考试，这是艺术蛋糕领域最权威的认证考试，获得了这项认证，意味着她就可以当之无愧地称为“艺术蛋糕设计师”。
去年11月，陈思臣在英国呆了十几天，她凭借艺术蛋糕作品“雀之圆舞曲”顺利通过考试。拿到证书后陈思臣才发现，她是所有人中年龄最小的一个。
其实，陈思臣可谓是在“蜜罐”里长大的孩子。父母都是国企高管，她从小到大衣食无忧，家人也不指望她开辟一番大事业。但她为什么还要义无反顾地选择创业之路？陈思臣说，她想证明自己，90后并非没追求、没抱负，她有属于自己的梦想，即使没有父母的帮助，也能取得成功。"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    generated_text = my_model.generate_text(messages)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()