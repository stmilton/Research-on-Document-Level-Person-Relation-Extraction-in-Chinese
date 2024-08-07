from googlesearch import search
import requests
from bs4 import BeautifulSoup
import time
def google_search(query, num_results=5):
    try:
        search_results = search(query, num_results=num_results, timeout=10, lang="zh", sleep_interval=5)
        final_search = []
        final_text = []
        for i, result in enumerate(search_results, start=1):
            print(f"{i}. {result}")
            # 排除維基百科
            if "wikipedia" in result:
                continue

            text_content = scrape_url(result)

            if text_content is None:
                continue
            final_search.append(result)
            final_text.append(text_content)
            # print(len(final_search))
            # print(len(final_text))
            time.sleep(2)
            return final_search[0:num_results],final_text[0:num_results]

    except Exception as e:
        print(f"An error occurred: {e}")


def scrape_url(url):
    try:
        # 发送HTTP请求并获取页面内容
        response = requests.get(url)
        # 检查请求是否成功
        if response.status_code == 200:
            # 使用BeautifulSoup解析页面内容
            soup = BeautifulSoup(response.text, 'html.parser')
            # 提取页面中的文字内容
            text_content = soup.get_text()
            # print(text_content)
            return text_content
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")


# 替换下面的query字符串为您想要搜索的内容
query = "習近平"
google_search(query)