
import requests
# import html2text
from bs4 import BeautifulSoup
import re
import regex

def search_wikipedia(query):
    # 维基百科的API地址
    api_url = "https://zh.wikipedia.org/w/api.php"

    # 设置查询参数
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': query,
    }

    # 发送请求并获取响应
    response = requests.get(api_url, params=params)
    data = response.json()
    # print(data)

    # 提取搜索结果中的第一项（假设是最相关的）
    if 'query' in data and 'search' in data['query'] and data['query']['search']:
        first_result = data['query']['search'][0]

        # 获取页面ID
        page_id = first_result.get('pageid', '')
        # 使用页面ID获取完整页面内容
        wikitext_content, page_url = get_page_content_by_wikitext(page_id)
        htmltext_content = get_page_content_by_html(page_id)
    

        return page_url,wikitext_content,htmltext_content

    # 如果没有找到结果，返回提示信息
    return "No results found."

def get_page_content_by_wikitext(page_id):
    # 维基百科的API地址
    api_url = "https://zh.wikipedia.org/w/api.php"

    # 设置查询参数
    params = {
        'action': 'query',
        'format': 'json',
        'pageids': page_id,
        'prop': 'revisions|info',
        'rvprop': 'content',
        'inprop': 'url',
    }

    # 发送请求并获取响应
    response = requests.get(api_url, params=params)
    data = response.json()

    # 提取页面内容
    if 'query' in data and 'pages' in data['query']:
        page = data['query']['pages'].get(str(page_id), {})
        revisions = page.get('revisions', [])
        content = revisions[0]['*'] if revisions else ''
        page_url = page.get('fullurl', '')

        # 消歧異頁面
        if "Disambiguation" in content:
            print(content)

        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text()
        text_content = remove_braces_content(text_content)
        return text_content, page_url

    # 如果没有找到结果，返回提示信息
    return "No page content found."


def get_page_content_by_html(page_id):
    # 维基百科的API地址
    api_url = "https://zh.wikipedia.org/w/api.php"

    # 设置查询参数
    params = {
        'action': 'parse',  # 使用parse模块
        'format': 'json',
        'pageid': page_id,
        'prop': 'text',  # 请求文本内容
    }

    # 发送请求并获取响应
    response = requests.get(api_url, params=params)
    data = response.json()
    content = data['parse']['text']['*']
    
    # 消歧異頁面
    if "Disambiguation" in content:
        # print(data)
        print(content)


    soup = BeautifulSoup(content, 'html.parser')
    text_content = soup.get_text()
    # text_content = remove_braces_content(text_content)
    return text_content

    # 如果没有找到结果，返回提示信息
    return "No page content found."



def remove_braces_content(text):
    # 使用正则表达式匹配和替换{}中的内容
    pattern = r'\{((?:[^{}]|(?R))*)\}'

    cleaned_text = regex.sub(pattern, '', text)
    return cleaned_text

# 示例用法
# query = "習近平"
# page_url,wikitext_content,htmltext_content = search_wikipedia(query)
# print(page_url)
# print(htmltext_content)