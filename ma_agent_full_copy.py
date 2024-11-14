from autogen_agentchat.agents import CodingAssistantAgent, ToolUseAssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, StopMessageTermination
from autogen_core.components.models import OpenAIChatCompletionClient
from autogen_core.components.tools import FunctionTool
import asyncio
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import http.client
import json

OPENAI_API_KEY = 'sk-_c5WCLMLPcFi0ATazdykKeRyJu0FTgzDg1xldEsGz0T3BlbkFJ9D-4oQAkQ8e6pOETrNWVDPKrhQojXkTKKSdz6rIHYA'

# ---------------------
# 功能1：谷歌搜索并爬取网页内容
# ---------------------
def google_search_first_result(query: str, max_chars: int = 500) -> dict:
    # load_dotenv(r"C:\Users\ALIENWARE\Documents\AgentChat\.env")

    api_key = os.getenv("GOOGLE_API_KEY") # 直接用OPENAI KEY代替
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID") # 直接用GOOGLE_SEARCH_ENGINE_ID代替

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": search_engine_id, "q": query, "num": 1}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Error in API request: {response.status_code}")

    result = response.json().get("items", [])[0]

    def get_page_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            words = text.split()
            content = ""
            for word in words:
                if len(content) + len(word) + 1 > max_chars:
                    break
                content += " " + word
            return content.strip()
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"

    body = get_page_content(result["link"])

    return {
        "title": result["title"],
        "link": result["link"],
        "snippet": result["snippet"],
        "body": body
    }

# ---------------------
# 功能2：内容总结模块
# ---------------------
def summarize_content(content: str) -> str:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(content, max_length=50, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# ---------------------
# 功能3：发送邮件
# ---------------------
import http.client
import json
from typing import Any

def send_email(email_content: str, recipient_email: str) -> str:
    """
    发送邮件的主函数
    :param email_content: 发送邮件的内容
    :param recipient_email: 接收邮件的邮箱地址
    :return: 服务器的响应
    """
    from dotenv import load_dotenv
    import os

    # load_dotenv(r"C:\Users\ALIENWARE\Documents\AgentChat\.env")
    # api_key = os.getenv("EMAIL_API_KEY")

    # if not api_key:
    #    raise ValueError("API key not found. Please set EMAIL_API_KEY in .env file.")

    conn = http.client.HTTPSConnection("chat.jijyun.cn")

    # 构建请求体 payload
    payload = json.dumps({
        "instructions": f"发送邮件内容：{email_content}，到邮箱: {recipient_email}",
        "preview_only": False
    })

    # 设置请求头
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': "chat.jijyun.cn",
        'Connection': 'keep-alive'
    }

    try:
        conn.request(
            "POST",
            f"/v1/openapi/exposed/214506_1524_jjyibotID_ffd1911d7f0a44618f82ee34c4de1e00/execute/?apiKey=h8DZuBhmwnRBWeHqig8532hw1728364626",
            payload,
            headers
        )
        res = conn.getresponse()
        data = res.read()
        return data.decode("utf-8")

    except Exception as e:
        return f"Error occurred while sending email: {str(e)}"

    finally:
        conn.close()


# ---------------------
# 创建工具（Tools）
# ---------------------
google_search_tool = FunctionTool(
    google_search_first_result,
    description="Search Google for information, returns results with a snippet and body content"
)

summarize_tool = FunctionTool(
    summarize_content,
    description="Summarizes the provided text content using a pre-trained BART model."
)

send_email_tool = FunctionTool(
    send_email,
    description="Sends an email to the specified recipient with provided content."
)

# ---------------------
# 创建代理（Agents）
# ---------------------
search_agent = ToolUseAssistantAgent(
    name="Google_Search_Agent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    registered_tools=[google_search_tool],
    description="Performs Google search and extracts content.",
    system_message="Search the web and extract relevant content using your tools."
)

summarize_agent = ToolUseAssistantAgent(
    name="Summarize_Agent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    registered_tools=[summarize_tool],
    description="Summarizes content retrieved from the web.",
    system_message="Summarize the provided content effectively."
)

class EmailStopAgent(ToolUseAssistantAgent):
    async def run(self, *args, **kwargs):
        # 在发送完邮件后停止代理任务
        result = await super().run(*args, **kwargs)
        print("Email has been sent, stopping the execution.")
        # 可以直接返回，停止后续的代理任务
        return result

# 使用自定义的 EmailStopAgent
email_agent = EmailStopAgent(
    name="Email_Agent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    registered_tools=[send_email_tool],
    description="Sends summarized content to the specified email and stops the execution.",
    system_message="Send the summarized content to the provided email address and stop the execution after sending email."
)
# ---------------------
# 创建团队（Team）
# ---------------------
team = RoundRobinGroupChat([search_agent, summarize_agent, email_agent])

# ---------------------
# 主函数
# ---------------------
def run_team_query(question):
    async def main():
        result = await team.run(
            question, 
            termination_condition=StopMessageTermination()
        )
        return result

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(main())
        return result
    finally:
        loop.close()  # 确保事件循环被正确关闭
