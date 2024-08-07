# -*- coding: utf-8 -*-

# 單個純文字
import json
import re
import pandas as pd
import requests
import time
import threading

class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        super(MyThread, self).__init__(target=target, args=args)
        self._result = None

    def run(self):
        self._result = self._target(*self._args)

    def result(self):
        return self._result
    
def main():
    api_keys = [
                "AIzaSyCj4ntI2uDfylzEpFGZVoeWtRdEm6f4sHw", 
                # "AIzaSyBWyo8T7vBPCm05bsQ0ajJEZGTHC3nnbyw",   # 
                # "AIzaSyCJcz9-u17KEXqqWe5uY5KP9lxVUCAF-6I", # 04
                # "AIzaSyBrQatLob5e5XDFiYwjNxQNpPkuLoEdnxc", # 04
                # "AIzaSyAMHK3zXnLNFyS16vyNgoLG-iSXoFerizw", # 05
                # "AIzaSyCowEbcJI1BtO7x1M3kCJ0NnVgK6N96QTg", # 05
                # "AIzaSyBkCiOIW_uoWNPxAQXg0roMn8tNaZIKbMI", # 06
                # "AIzaSyBaqdSO1hlxpGLfBgv1FS58foEbTXiFNVA", # 06
                ]
    thread_num = len(api_keys)
    # unlabel_file = "./CommonCrawl/data/train/zh_head_0006.json"

    unlabel_file = r"H:\我的雲端硬碟\RelationExtration\CommonCrawl\data\train\zh_head_0001.json"

    with open(unlabel_file, encoding='utf-8') as f:
        total_length=0
        data = []
        for idx,line in enumerate(f):
            line = json.loads(line)
            data.append(line)
            total_length+=1
       
        # 切割數據
        chunk_size = total_length // thread_num
        remainder = total_length % thread_num
        threads = []
        start_index = 0
        for i in range(thread_num):
            chunk_length = chunk_size + 1 if i < remainder else chunk_size
            end_index = start_index + chunk_length
            chunk = data[start_index:end_index]
            thread = MyThread(target=gemini, args=(i+1, start_index, api_keys[i], chunk))
            thread.start()
            threads.append(thread)
            start_index = end_index

        all_url_lst=[]
        all_title=[]
        all_raw_content=[]
        all_output=[]
        all_relation=[]
        for thread in threads:
            thread.join()
            url_lst, title, raw_content, output, relation = thread.result()
            all_url_lst.extend(url_lst)
            all_title.extend(title)
            all_raw_content.extend(raw_content)
            all_output.extend(output)
            all_relation.extend(relation)

    # output_file_path = r"C:\Users\Milton\Desktop\gemini第二次標註\gemini_filter2_0006.csv"
    output_file_path = r"H:\我的雲端硬碟\RelationExtration\CommonCrawl\data\train\gemini_filter_0001.csv"

    df = pd.DataFrame(
            {
            "url":all_url_lst,
            "title":all_title,
            "raw_content":all_raw_content,
            "output":all_output,
            "relation":all_relation
            }
        )
    df.to_csv(f"{output_file_path}", encoding='utf-8-sig', index=True)


def gemini(thread_id, start_index, api_key, f):
    url = f'https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}'
    headers = {'Content-Type': 'application/json'}
    url_lst = []
    title = []
    raw_content = []
    output = []
    relation = []

    # with open(unlabel_file, encoding='utf-8') as f:
    for idx,line in enumerate(f):
        # line = json.loads(line)

        url_lst.append(line['url'])
        title.append(line['title'])
        raw_content.append(line['raw_content'])

        document = line["raw_content"]
        # 太長截斷
        if len(document) > 4000:
            document = document[:4000]
        
        data = {
            "contents": [
                {   "parts": [{"text": f"""請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、其他)，且兩位關係人皆必須有明確名字，只有稱謂的不算。
若無關係直接回答:無 即可。
若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係，小括號中必須包含2個人名實體和1個關係。
文章如下:
    [Document_start] {document} [Document_end]"""}],
                    "role":'user'
                    # "parts": [{"text": f"請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。\n若無關係直接回答:無 即可\n若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係\n文章如下:\n{document}"}]
                    # "parts": [{"text": f"請幫我找出以下文章中是否包含人與人之間的親屬關係?必須有明確人名，只有稱謂的不算。\n若無親屬關係直接回答:無 即可\n若有請回答依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有親屬關係\n{document}"}]
                }
            ],
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                }
            ],
            "generationConfig": {
                "temperature": 0, # 控制輸出的隨機性
                "maxOutputTokens": 500,
                "topP": 0.8,
                "topK": 10
            }
        }
        retry_count = 5  # 设置最大重试次数
        for _ in range(retry_count):
            try:
                response = requests.post(url, headers=headers, json=data)  
                if response.status_code == 200:
                    response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                    
    
                    if check_format(response_text):
                        print(f"thread_{thread_id} {idx+start_index+1}. {response_text}")
                        time.sleep(2)
                        break  
                    else:
                        data['contents'].append({
                                "parts": [{"text": response_text}],
                                "role":"model"
                            })
                        data['contents'].append({
                            "parts": [{"text": "請務必依照規定格式回答，若無關係直接回答:無\n若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)..，小括號中必須包含2個人名實體和1個關係"}],
                            "role":"user"
                        })
                        print(f"thread_{thread_id} 未依格式回答:{response_text}  Retrying...")
                        time.sleep(2)
                elif response.status_code == 429:
                    response_text = ""
                    print(f"thread_{thread_id} 請求頻率過高{response.status_code}  Retrying...")
                    time.sleep(5+_*5)
                else:
                    response_text = ""
                    print(f"thread_{thread_id} Request failed with status code: {response.status_code} {response.json()}  Retrying...")
                    time.sleep(5)

            except Exception as e:
                response_text = ""
                print(f"thread_{thread_id} An error occurred: {response.json()}  Retrying...")
                time.sleep(2)
        output.append(response_text)
        
        if response_text:
            if response_text[0] == "無":
                relation.append("無")
            elif response_text[0] == "有" :
                relation.append("有")
            else: 
                relation.append("無法識別")    
        else:
            relation.append("請重新嘗試")
            print(f"thread_{thread_id} 暫時無法填充....{idx+start_index+1}. ")

    return url_lst, title, raw_content, output, relation

def check_format(input_string):
    try:
        if input_string[0] == "無":
            return True
        if input_string[0] == "有":
            pattern = r'\((.*?)\)'
            re_tuples = re.findall(pattern, input_string)
            for re_tuple in re_tuples:
                re_ternary = re_tuple.split(',')
                re_ternary = [s.strip() for s in re_ternary if s.strip() != ""]
                if len(re_ternary) != 3 :
                    return False
            return True
        return False
    except Exception as e:
        return False
if __name__=='__main__':
    main()
# "parts": [{"text":f"""請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。
# 若無關係直接回答:Relations:無 即可
# 若有請依以下格式回答:
#     Relations:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係
#     Explanation:解釋原因
# 範例如下:
# TEXT:中国计划生育观察：美国之音:山东妇女怀孕6月,被强迫堕胎
#     杨林帮转自博讯
#     (博讯2013年10月05日发表)
#     来源：美国之音
#     中国山东一名怀孕6个月的妇女被以暴力手段强制堕胎。继英国“天空”卫星新闻星期五播出山东潍坊妇女刘欣雯被强制堕掉6月月大的胎儿后，美国维护女童权益的组织—女童之声星期五发表声明，详细讲述了事件过程。
#     上周五凌晨4点，刘欣雯和她的丈夫周国强在家中熟睡，包括16名男性和4名女性的20名计生委人员踢倒大门，破门而入，他们把周国强按住，同时从床上把刘欣雯拉起送进当地的一家医院，强制打针堕掉一个男胎。周国强随后花了5个小时的时间才找到妻子的下落，但是刘欣雯在被打针后，腹中的胎儿已经没有了动静。
#     这对夫妇有一个10岁的儿子。当他们发现怀上第二胎后，担心被强迫堕胎，一直隐藏怀孕。周国强说，他愿意在第二个孩子出生后付罚款。
#     中国山东省是一个以高强制堕胎率而闻名的省份。中国盲人律师陈光诚就是来自山东省，他曾经帮助数千被强行堕胎的妇女打官司。
#     女童之声创办人柴玲也来自山东。她赞扬了英国“天空”卫星电视报道这一消息的勇气，曝光了中国政府不希望被世人所知的事情。 柴玲唿吁中国政府高层关注这一令人发指的残暴行为，废除一胎化政策。
#     "" width=""500
#     女权无疆界主席瑞洁说，这对夫妇的遭遇并不是个别事件。他们的经历再次显示了中国一胎化的暴力。 她说：“中国政府会强迫怀孕9个月的妇女堕胎。这些强制堕胎有些时候如此暴力，以至于一些妇女与她们已经足月的胎儿一同死亡。 ” 瑞洁说：“强制堕胎是政府官方施行的强奸。”
#     _(网文转载)
#     (此为打印板，原文网址：
#     https://news.boxun.com/news/gb/china/2013/10/201310061849.shtml)
#     【上篇】山东腐败观察：东营市纪委通报5起违反中央八项规定精神典型问题
#     【下篇】中国计划生育观察：
#     转载请注明: 中国计划生育观察：美国之音:山东妇女怀孕6月,被强迫堕胎 | 中国民主党全委会美国委员会 +复制链接\n
# Relations:有 (刘欣雯,周国强,夫妻)
# Explanation:文章中提到刘欣雯和她的丈夫周国强在家中熟睡，可見刘欣雯與周国强為夫妻關係
# TEXT:成大材料系劉浩志團隊結合機器學習減少原子力顯微鏡量化量測誤差
#     文、圖／陳意安
#     成大材料系劉浩志教授團隊，近期以機器學習降低掃描式原子力顯微鏡量化量測不確定性的研究成果獲得國際期刊刊載
#     國立成功大學材料系劉浩志教授深耕奈米材料分析與製程技術，擁有十數項原子力顯微鏡商用探針專利。他有感於眾多學者操作該顯微鏡的困擾，與學生阮氏芳玲博士共同提出以機器學習減少原子力顯微鏡量測的不確定性，判斷準確率高達 96.8%，成果將使學者無須先備知識，就能取得極為可靠的彈性模數，探索萬千材料的無窮發展，最新成果於 2022 年 9 月獲刊《國際固體與結構雜誌》（International Journal of Solids and Structures）。
#     原子力顯微鏡可掃描物體表面形貌
#     原子力顯微鏡（Atomic Force Microscope, AFM）以懸臂樑之微細位移操控奈米探針，藉由探針跟樣品之間的原子作用力測得樣品的表面形貌、粗糙度、黏著度、彈性模數等物理性數據。劉浩志教授指出，原子力顯微鏡的優勢在於不需要將樣本進行切片或鍍金等加工處理，就可以在一般環境中執行測量，甚至當遇到樣品是活體且只存在於液態的情況，也能直接量測，例如細菌；但缺點就是只能觀測到物體表層狀態。
#     劉浩志說，原子力顯微鏡的第一篇研究論文在 1986 年才公諸於世，相對於使用歷史已達 90 年以上的電子顯微鏡，技術尚未成熟，學界的使用率也相對少，因此仍有許多發展空間。
#     國立成功大學材料系劉浩志教授深耕奈米材料分析與製程技術，擁有多種原子力顯微鏡商用探針專利
#     在累積眾多跨領域合作經驗與閱覽近 10 年 200 多篇國內外 AFM 研究論文後，劉浩志教授與當時的博士生張敬萱與阮氏芳玲發現，使用原子力顯微鏡的學者在量測出材料的彈力位移曲線（FZ 曲線）後，需要選擇後續解釋數據所需的「接觸模型」以計算出材料的彈性模數。但面對種類各異的接觸模型，學者常依靠主觀經驗或直覺選擇，忽略探針形狀造成的差異，導致最終的數值出現潛在誤差，影響應用成效；又因為探針形狀眾多且奈米針頭實際形貌在顯微鏡中無法得知，對使用者而言，精確地選對接觸模型並不容易。針對接觸模型選擇的問題，劉浩志與張敬萱於 2016 至 2018 年首先提出以 AFM 量化量測材料力學性質的準則以及修正之接觸模型，提昇 AFM 奈米力學量測的精確度。張敬萱博士畢業後即依所學專長至國家量測技術發展中心擔任研究員。
#     為進一步解決使用者痛點，劉浩志與阮氏芳玲針對不同材料的彈力位移曲線（FZ曲線）整理分類出各自適用的接觸模型，並透過機器學習減少人為判斷而產生的不確定性。
#     劉浩志教授（右）與學生阮氏芳玲博士共同提出機器學習方式減少原子力顯微鏡的不確定性
#     他們採用監督式機器學習框架（SML），先以 6500 組材料的 FZ 曲線訓練分類器，再以分類器針對從未處理過且性質較複雜的金黃色葡萄球菌的 FZ 曲線進行測試，結果準確率高達 96.8%，顯示機器學習為選擇合適的接觸模型和計算相應的材料彈性模數等力學性質提供了一個強大的工具，研究者能直接使用最合適的接觸模型分析出可靠結果，此成果於2022年5月獲得國際期刊《歐洲力學雜誌》（European Journal of Mechanics - A/Solids）刊登，且是該期刊 90 天內下載數最多的熱門文章之一。
#     隨後，劉浩志與阮氏芳玲進一步利用既有資料庫與彈性模數，以迴歸分析方式訓練分類器，日後研究者只需要量得材料的 FZ 曲線後，再經過機器自動運算，就能得到最終的彈性模數數值，無須煩惱應該選擇何種接觸模型，也省略了繁複的計算過程。這項成果受《國際固體與結構雜誌》（International Journal of Solids and Structures）青睞於2022年9月刊登，並吸引國際期刊《表面形貌：計量與特性》（Surface Topography：Metrology and Properties）編輯群邀請撰寫特稿，分享引入機器學習對掃描探針顯微鏡的未來轉機與展望。
#     劉浩志與阮氏芳玲進一步利用既有資料庫與彈性模數以迴歸分析方式訓練分類器
#     能透過機器學習解決全球使用原子力顯微鏡學者的困擾，很大部分來自於劉浩志教授本身的豐富經歷。劉浩志教授為清大材料系學士、美國史丹福大學材料系碩士與機械系設計組博士。自 2004 年起在原子力顯微鏡的領先開發商「威科儀器」（Veeco Instruments, Inc.，現為 Bruker Corporation）的前瞻開發小組擔任研究科學家與技術顧問，累積豐富商用探針研製經驗，並於 2008 年加入成功大學材料系任教至今，曾獲有國際傑出發明家國光獎章、「李國鼎科技與人文講座」金質獎等殊榮。
#     劉浩志團隊開發的顯微鏡探針是微米或奈米等級，可以量測材料的多種性質
#     「材料系本身就是跨領域的。」劉浩志表示，從材料的科學性質，如：原子怎麼排列、材料顯微結構、成分組成的基礎認知，到了解怎麼利用製程或加工調整材料的機械性質、物理性質與電性等，都是材料系的強項。而他主持的「微奈米製造與分析實驗室」亦著重「材料製程」與「材料分析」兩種技術，將材料科學工程的核心優勢跨領域應用於各種新穎材料的分析與應用，包含 3D 列印、高熵合金、掃描探針、細菌細胞及生醫材料、鈣鈦礦能源材料、鋰離子電池、機器學習輔助材料 AFM 分析等，涵蓋範圍廣泛。
#     劉浩志教授與跨領域學者分析細菌表面結構與性質，曾獲刊國際期刊封面
#     過去劉浩志教授曾與成大地科系簡錦樹教授研究嘉義布袋地底下抗砷的細菌，研究該細菌在高濃度砷含量的溶液產生的新陳代謝變化，進一步推進烏腳病相關的醫學研究；他也曾與成大醫學檢驗生物技術系蔡佩珍教授對臨床腸病毒的病毒體進行物理特性研究；他自己的團隊研究造成牙周病的變異鏈球菌多年，提出細菌胞外聚合物的分泌機制等。透過跨領域整合，一方面解決學者在開創新穎材料後不知性能優劣與否的難處，一方面也精進研究者發展高階材料分析技術，共同使相關成果能運用在醫學、半導體、光電、民生用品等各領域。
#     【補充資料​​】
#     European Journal of Mechanics - A/Solids, Volume 94
#     Machine learning approach for reducing uncertainty in AFM nanomechanical measurements through selection of appropriate contact model
#     International Journal of Solids and Structures, Volume 256
#     Machine learning framework for determination of elastic modulus without contact model fitting
#     Surface Topography: Metrology and Properties, TOPICAL REVIEW
#     Emerging machine learning strategies for diminishing measurement uncertainty in SPM nanometrology
#     維護單位: 新聞中心
# Relations:有 (劉浩志,阮氏芳玲,師生),(劉浩志,張敬萱,師生),(劉浩志,簡錦樹,同事),(劉浩志,蔡佩珍,同事)
# Explanation:文章中提及劉浩志教授與當時的博士生張敬萱與阮氏芳玲發現，可見劉浩志與阮氏芳玲為師生關係，劉浩志與張敬萱也為師生關係\n
#             另外文章中說到過去劉浩志教授曾與成大地科系簡錦樹教授研究嘉義布袋地底下抗砷的細菌，還有他也曾與成大醫學檢驗生物技術系蔡佩珍教授對臨床腸病毒的病毒體進行物理特性研究，所以可以得知劉浩志與簡錦樹為同事關係，劉浩志與蔡佩珍也為同事關係
# 文章如下:
# TEXT:{document}"""
# }]