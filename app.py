from flask import Flask, request, jsonify, render_template
from gtts import gTTS
import requests
import os
import json
import openai
import time  # Used to generate a unique audio file name
import http.client
from bs4 import BeautifulSoup  # Used for web scraping to fetch content
from mem0 import MemoryClient
import serpapi
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Union
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer, util
import nltk
from flask_cors import CORS
from serpapi import GoogleSearch
from ma_agent_full_copy import run_team_query
nltk.download('punkt')
app = Flask(__name__)
CORS(app)





openai.api_key = "sk-_c5WCLMLPcFi0ATazdykKeRyJu0FTgzDg1xldEsGz0T3BlbkFJ9D-4oQAkQ8e6pOETrNWVDPKrhQojXkTKKSdz6rIHYA"
MEM0_API_KEY = 'm0-YyXywajmT0hsBj1SkV4SBMGhE5ymAiinz4ITqr6t'

# RAGFlow API configuration
NEW_CONVERSATION_URL = "http://34.87.214.14/v1/api/new_conversation"
COMPLETION_URL = "http://34.87.214.14/v1/api/completion"
RAGFLOW_API_KEY = "ragflow-VkZGQ1NjBjN2EzZDExZWY4MmM1MDI0Mm"
USER_ID = 'Zyp123'
USER_ID_MEM0 = 'm123'

# Set the file upload path
UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the Mem0 memory client
memory_client = MemoryClient(api_key=MEM0_API_KEY)

# Clean up old audio files in the static folder
def clear_static_audio_files():
    folder = 'static'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.endswith('.mp3'):
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error while deleting file {file_path}: {str(e)}")

def store_conversation_in_memory(conversation):
    memory_client.add(messages=conversation, user_id=USER_ID_MEM0)
    print("Conversation added to memory.")

# Create a new session and return the session ID
def create_conversation():
    headers = {
        'Authorization': f'Bearer {RAGFLOW_API_KEY}'
    }
    payload = {'user_id': USER_ID}
    try:
        response = requests.get(NEW_CONVERSATION_URL, params=payload, headers=headers)
        if response.status_code == 200:
            try:
                data = response.json()
                conversation_id = data.get('data', {}).get('id')
                if conversation_id:
                    return conversation_id
                else:
                    print("No conversation ID found in response.")
                    return None
            except ValueError:
                print("Invalid JSON response from create conversation API.")
                return None
        else:
            print(f"Failed to create conversation: {response.status_code} {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None

# Generate answers using the RAGFlow API
def query_ragflow_api(question, conversation_id):
    headers = {
        'Authorization': f'Bearer {RAGFLOW_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'conversation_id': conversation_id,
        'messages': [{"role": "user", "content": question}]
    }
    try:
        response = requests.post(COMPLETION_URL, json=payload, headers=headers)
        if response.status_code == 200:
            content = response.text.strip()
            data_parts = content.splitlines()
            valid_lines = [line.strip() for line in data_parts if line.strip()]
            if len(valid_lines) < 2:
                return "Not enough valid data parts to process"
            second_last_content = valid_lines[-2]
            if second_last_content.startswith("data:"):
                second_last_content = second_last_content[5:].strip()
            if not second_last_content:
                return "Empty JSON content"
            try:
                data = json.loads(second_last_content)
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], dict):
                    answer = data['data'].get('answer')
                    if answer:
                        return answer
                    else:
                        return "No answer provided by RAGFlow"
                else:
                    return "Unexpected data format: {}".format(data)
            except json.JSONDecodeError as ve:
                return f"JSON Decode Error: {str(ve)} - Content: {second_last_content}"
        else:
            return f"HTTP Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

# Obtain context-based answers
def get_context_aware_response(question, conversation):
    # Retrieve relevant context memories using Mem0
    relevant_memories = memory_client.search(question, user_id=USER_ID_MEM0)
    context = "\n".join([m["memory"] for m in relevant_memories])

    # If there is context memory, include it in the prompt
    if context:
        prompt = f"""You are an intelligent assistant, if there is an answer to the user's question in the previous interactions, then please summarize the relevant content to answer the question, Answers need to consider chat history.
        Previous interactions:
        {context}

        Question: {question}
        """
    else:
        prompt = f"You are an intelligent assistant, if there is no answer to the user's question in the context, then please summarize the content of the knowledge base to answer the question, please enumerate the data in the knowledge base to answer in detail. When all knowledge base content is irrelevant to the question, answer with the relevant content remembered by the mem0 memory layer. Answers need to consider chat history. Answer the user question: {question}"

    # Generate context-based answers using RAGFlow
    conversation_id = create_conversation()
    if conversation_id:
        answer = query_ragflow_api(prompt, conversation_id)
        # Save the current session to Mem0
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": answer})
        store_conversation_in_memory(conversation)
        return answer
    else:
        return "Failed to create conversation."

# Convert text to speech
def text_to_speech(text):
    # Generate a unique file name using a timestamp
    timestamp = int(time.time())
    filename = f'static/response_{timestamp}.mp3'
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    file_type = file.mimetype

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Process audio files
        if file_type.startswith('audio/'):
            with open(filepath, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                return jsonify({"content": transcript["text"]}), 200
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data received"}), 400

    question = data.get('question')
    if not question:
        return jsonify({"error": "Invalid input: 'question' is required"}), 400
    if 'email' in question:
        result = run_team_query(question)
        print(result)
        return None

    conversation = []
    answer = get_context_aware_response(question, conversation)
    if answer.startswith("Request failed") or answer.startswith("HTTP Error"):
        return jsonify({"error": answer}), 500

    try:
        audio_file = text_to_speech(answer)
        audio_url = f"/{audio_file}"
    except Exception as e:
        return jsonify({"error": f"Failed to generate audio: {str(e)}"}), 500

    return jsonify({"answer": answer, "audio_url": audio_url})

# Newly added scraping and summarization features
@app.route('/search_and_summarize', methods=['POST'])
def search_and_summarize():
    data = request.get_json()
    query = data.get('query')  # Retrieve the question sent from the frontend
    
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Perform a Google search using SerpAPI
        search = GoogleSearch({
            "q": query,
            "location": "Austin, TX",
            "hl": "en",
            "gl": "us",
            "api_key": "6e10541b32898cb0c23c891a01e1afe0dea941aa8122314229497aaa75bac227"  # Replace with the SerpAPI
        })
        results = search.get_dict()
        
        # Extract text information from the results (extract summaries of the search results)
        search_results = results.get("organic_results", [])
        if not search_results:
            return jsonify({"error": "No results found"}), 500
        
        # Combine the summaries of the search results
        text = " ".join([result.get("snippet", "") for result in search_results])
        #print(f"Extracted Text: {text[:200]}")  # Debug and output part of the content
        
        # Limit the text length to avoid issues caused by excessively long text
        text = text[:2000]

        # Use OpenAI to summarize the text
       # Use GPT-3.5-turbo to summarize the tex
        summary = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following content: {text}"}
            ]
        ).choices[0].message.content.strip()


        return jsonify({"summary": summary}), 200

    except Exception as e:
        print(f"Error: {e}")
        #traceback.print_exc()  # Print the error stack trace
        return jsonify({"error": "Failed to summarize content."}), 500

# Sending email function
class SendEmailSkill:
    def __init__(self, api_key, host="chat.jijyun.cn"):
        self.api_key = api_key
        self.host = host

    def send_email(self, email_content, recipient_email):
        conn = http.client.HTTPSConnection(self.host)
        payload = json.dumps({
            "instructions": f"发送邮件内容：{email_content}，到邮箱{recipient_email}",
            "preview_only": False
        })
        headers = {
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': self.host,
            'Connection': 'keep-alive',
            'Cookie': 'acw_tc=0bd17c5217284761475335882e64e14d015eee439fd991dc63a31a9c0dc695'
        }
        try:
            conn.request("POST", f"/v1/openapi/exposed/214506_1524_jjyibotID_ffd1911d7f0a44618f82ee34c4de1e00/execute/?apiKey={self.api_key}", payload, headers)
            res = conn.getresponse()
            data = res.read()
            return data.decode("utf-8")
        except Exception as e:
            return f"Error occurred while sending email: {str(e)}"
        finally:
            conn.close()

@app.route('/send_email', methods=['POST'])
def send_email():
    data = request.get_json()
    if not data or 'email_content' not in data or 'recipient_email' not in data:
        return jsonify({"error": "Missing required data"}), 400
    send_email_skill = SendEmailSkill(api_key="h8DZuBhmwnRBWeHqig8532hw1728364626")
    response = send_email_skill.send_email(data['email_content'], data['recipient_email'])
    return jsonify({"response": response})

smooth = SmoothingFunction().method1
scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
def tokenize_text(text):
    return nltk.word_tokenize(text)

def calculate_metrics(references: List[str], candidates: List[str]):
    bleu_score = corpus_bleu([[ref] for ref in references], candidates, smoothing_function=smooth)
    rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
    meteor_scores = [
        meteor_score([tokenize_text(ref)], tokenize_text(cand))
        for ref, cand in zip(references, candidates)
    ]
    semantic_scores = [
        util.cos_sim(bert_model.encode(ref), bert_model.encode(cand)).item()
        for ref, cand in zip(references, candidates)
    ]
    final_scores = [
        max((r['rougeL'].fmeasure + m + s) / 3, 0)  # 如果分数小于 0，则设置为 0
        for r, m, s in zip(rouge_scores, meteor_scores, semantic_scores)
    ]
    return bleu_score, final_scores


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    questions = data.get('evaluation_queries', [])
    reference_answers = data.get('ground_truths', [])

    if not questions or not reference_answers:
        return jsonify({"error": "Missing evaluation queries or ground truths"}), 400

    # create a conversation ID
    conversation_id = create_conversation()
    if not conversation_id:
        return jsonify({"error": "Failed to create conversation"}), 500

    # Pass the question and session when generating the answer ID
    generated_answers = [query_ragflow_api(q, conversation_id) for q in questions]

    composite_scores = []
    for ref, gen in zip(reference_answers, generated_answers):
        bleu, final_scores = calculate_metrics([ref['answer']], [gen])
        composite_scores.append({
            "reference_answer": ref['answer'],
            "generated_answer": gen,
            "bleu_score": bleu,
            "final_scores": final_scores
        })

    return jsonify(composite_scores)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)


if __name__ == '__main__':
    # Clean up old audio files when the application starts
    clear_static_audio_files()

    if not os.path.exists('static'):
        os.makedirs('static')
        
    app.run(host='0.0.0.0', port=5003, debug=True)
