from langdetect import detect
from warcio.archiveiterator import ArchiveIterator

def detect_language(text):
    try:
        # 尝试检测文本语言
        language = detect(text)
        return language
    except:
        # 处理检测失败的情况
        return "Unknown"

def parse_warc_file(file_path):
    count = 0
    with open(file_path, 'rb') as warc_file:
        for record in ArchiveIterator(warc_file):
            # 检查记录类型
            if record.rec_type == 'response':
                # 获取HTTP响应内容
                http_response = record.content_stream().read()
                original_url = record.rec_headers.get_header('WARC-Target-URI')
                decoded_response = http_response.decode('utf-8', errors='replace')
                # accept_language = record.rec_headers.get_header('Accept-Language')

                # 检测文本语言
                language = detect_language(decoded_response)
                
                # 打印语言信息
                # if accept_language is not None:
                if 'zh' in language:
                    count += 1
                    print(f"Language: {language}")
                    # print(f"Accept-Language: {accept_language}")
                    print(original_url)
                # break
        print(count)
parse_warc_file('./CommonCrawl/CC-MAIN-20230921073711-20230921103711-00000.warc')
