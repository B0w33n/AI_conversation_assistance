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
# Function 1: Google Search & Scrape web content
# ---------------------
def google_search_first_result(query: str, max_chars: int = 500) -> dict:
    # load_dotenv(r"C:\Users\ALIENWARE\Documents\AgentChat\.env")

    api_key = os.getenv("GOOGLE_API_KEY") # Replace with the OPENAI KEY
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID") # Replace with the GOOGLE_SEARCH_ENGINE_ID

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
# Function 2: Content Summarization module
# ---------------------
def summarize_content(content: str) -> str:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(content, max_length=50, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# ---------------------
# Function 3: Send the email
# ---------------------
import http.client
import json
from typing import Any

def send_email(email_content: str, recipient_email: str) -> str:
    """
    Main function for sending emails
    :param email_content: content of the email
    :param recipient_email: email address for receiving mail
    :return: Server response
    """
    from dotenv import load_dotenv
    import os

    # load_dotenv(r"C:\Users\ALIENWARE\Documents\AgentChat\.env")
    # api_key = os.getenv("EMAIL_API_KEY")

    # if not api_key:
    #    raise ValueError("API key not found. Please set EMAIL_API_KEY in .env file.")

    conn = http.client.HTTPSConnection("chat.jijyun.cn")

    # Build the request body payload
    payload = json.dumps({
        "instructions": f"发送邮件内容：{email_content}，到邮箱: {recipient_email}",
        "preview_only": False
    })

    # Set the request headers
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
# Setup tools
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
# Setup agents
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
        # Stop the agent task after sending the email
        result = await super().run(*args, **kwargs)
        print("Email has been sent, stopping the execution.")
        # return directly to stop the subsequent agent tasks after sending the email
        return result

# Use the custom EmailStopAgent
email_agent = EmailStopAgent(
    name="Email_Agent",
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
    registered_tools=[send_email_tool],
    description="Sends summarized content to the specified email and stops the execution.",
    system_message="Send the summarized content to the provided email address and stop the execution after sending email."
)
# ---------------------
# Setup the team
# ---------------------
team = RoundRobinGroupChat([search_agent, summarize_agent, email_agent])

# ---------------------
# Main function
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
        loop.close()  # Ensure that the event loop is properly closed
