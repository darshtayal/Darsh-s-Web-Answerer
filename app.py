# -*- coding: utf-8 -*-

#imports
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re
import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import time

# function to invoke llm
def invoke_llm(message):
    llm = ChatGroq(model='llama-3.3-70b-versatile')
    res = llm.invoke([HumanMessage(content=message)])
    return res.content

# function to invoke safety llm
def safety_checker(mess:str):
    safety_llm = ChatGroq(model='meta-llama/llama-guard-4-12b')
    try:
        res = safety_llm.invoke(mess)
    except:
        time.sleep(60)
        res = safety_llm.invoke(mess)
    finally:
        if res.content == 'safe':
            return True
        else:
            return False

# initialize GROQ API KEY
def initializer(api_key):
    os.environ['GROQ_API_KEY'] = api_key
    llm_test = ChatGroq(model='gemma2-9b-it')
    try:
        llm_test.invoke('hey')
    except:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

# function to check if the link is working
def link_checker(link):
    loader = WebBaseLoader(link)
    try:
        doc = loader.load()
        content = doc[0].page_content
    except:
        return False, 'Empty'
    else:
        return True, content

# function to clean the webpages
def clean_web_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    junk_keywords = [
        'Privacy Policy', 'Terms of Service', 'Subscribe', 'Contact us', '©',
        'Home', 'Menu', 'Sections', 'Categories', 'About Us', 'Our Team',
        'Newsletter', 'Back to top', 'Follow us', 'Sign up', 'Login',
        'Sign in', 'Support', 'Language', 'FAQ', 'Site Map', 'Feedback'
    ]
    for kw in junk_keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub('', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# function to generate summary and initialize the vectorstore database
def summary_generator(link):
    yield gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", gr.update(visible=False), ""
    status, content = link_checker(link)
    if status:
        ref_cont = clean_web_text(content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks = splitter.split_text(ref_cont)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')  
        db = Chroma.from_texts(chunks,embeddings)
        if (len(ref_cont.split(" ")) * 1.4) >= 12000:
            ref_cont = "".join(ref_cont.split(" ")[:5000])
        res = invoke_llm(f"Please summarize the following text into 2-3 lines. Give an overview of what it's about. Ignore all the unnessessary website content, focus only on text. Just generate the summary and nothing else, not even here's the summary and all. Here's the text: '{ref_cont}'")
        yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), res, gr.update(visible=False), db
    else:
        yield gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=True), ""

# function for user to ask questions and get answers based on retrieved information
def chat(query:str, vs, top_k):
    yield gr.update(visible=True), ""
    if safety_checker(query):
        info = vs.similarity_search(query)
        relevant_text = ""
        for i in range(top_k):
            relevant_text += f"{info[i].page_content}\n"
        try:
            res = invoke_llm(f"You have to answer this query: {query} based only on the following information: {relevant_text}. Reply only with the answer.")
        except:
            time.sleep(60)
            res = invoke_llm(f"You have to answer this query: {query} based only on the following information: {relevant_text}. Reply only with the answer.")
        finally:
            yield gr.update(visible=False), res
    else:
        yield gr.update(visible=False), "<SAFETY CHECKER> Sorry, this message seems harmful. Please try something else" 

# build the UI
with gr.Blocks() as ui:
    vectorstore = gr.State()
    with gr.Column(visible=True) as initialize_page:
        gr.Markdown("# Welcome to Website Q/A Project by Darsh! 🥳")
        gr.Markdown("# Just paste the link and get a summary and then ask questions directly from the website! ✨")
        free_mess = gr.Markdown("# But first we need your GROQ API KEY (it's free!) 🤩", visible = True)
        api_error_msg = gr.Markdown("# ⚠️ Your API key doesn't work. Please Try again.", visible = False)
        api_key = gr.Textbox(label='Paste your API key here:', lines=1)
        ini_submit = gr.Button("SUBMIT!")
        gr.Markdown("""\
**Here's how to get the GROQ API KEY:**  
- **Go to**: https://console.groq.com/keys  
- **Sign up for a Groq Cloud account:** If you don't already have one, visit the Groq Cloud website and sign up for a new account.  
- **Log in:** Once you have an account, log in to your Groq Cloud account.  
- **Navigate to API Keys:** Locate and click on the "API Keys" section in the left-hand navigation panel.  
- **Create a new API key:** Click the "Create API Key" button.  
- **Name your API key:** Enter a descriptive name for your API key (e.g., "My Groq API Key").  
- **Submit:** Click the "Submit" button to generate the API key.  
- **Copy and Securely Store:** Copy the key and paste it here! You're done ✅  
""")   

    with gr.Column(visible=False) as link_pro_page:
        gr.Markdown("# You're all set! 🎉 Just paste your link here to get started: 🌐", visible=True)
        inv_link = gr.Markdown("# 🛑 There was a problem accessing this link. Please check if it's broken or invalid and try again.", visible=False)
        
        link = gr.Textbox(label='Your link here:', lines=1)
        link_submit = gr.Button('SUBMIT!')
    
        gr.Markdown("""\
    ### 🔍 What happens next?
    
    1. **We fetch the website content** from the link you provide.
    2. **You get a summary** of the entire page — to get an overview of the content!
    3. **You can ask questions** directly about the website, as if you're chatting with someone who read it for you. 🤖💬
    
    ---
    
    ### 💡 Pro Tips
    
    - Make sure your link is **public and accessible**, not behind a login.
    - Stick to **informational pages** like blogs, articles, or documentation for best results.
    - You can **ask anything** related to the content — summaries, specific sections, definitions, insights, and more.
    
    ---
    
    ### 🧠 How does this work?
    
    This project uses **Large Language Models** (LLMs) + **Web scraping** + **Natural Language Understanding** to give you:
    - A clean and smart summary.
    - A chatbot that knows exactly what's in the page you gave it!
    
    ---
    
    ### 🚀 Built by Darsh  
    A student explorer passionate about AI + building useful tools for real-world problems.  
    """)


    with gr.Column(visible=False) as processing_page:
        
        processing_msg = gr.HTML("""
    <div style="display:flex; flex-direction:column; align-items:center;">
        <div class="loader" style="
            border: 8px solid #f3f3f3;
            border-top: 8px solid #3498db;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
        "></div>
        <p style="margin-top: 10px;">Processing... Please wait.</p>
    </div>
    <style>
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    </style>
    """, visible=True)

    with gr.Column(visible=False) as final_page:
        gr.HTML("""
        <div style="text-align:center; padding: 10px 0;">
            <h1 style="color:#007bff;">📄 Website Summary</h1>
            
        </div>
        """)
        
        page_summary = gr.Markdown("") 
    
        gr.HTML("<hr style='margin: 20px 0;'>")
    
        gr.Markdown("""
        ## 💬 Ask Anything About the Page  
        Type your question about the content and get an instant answer powered by AI! 🤖  
        """)
    
        question = gr.Textbox(label='Your question here:', lines=2)

        top_k = gr.Radio(choices=[1,2,3], label='top_k similarity', value=2, interactive=True)
    
        final_submit_but = gr.Button('SUBMIT!')

        loading_spinner = gr.HTML("""
<div style="text-align:center;" id="spinner">
    <div style="
        border: 6px solid #f3f3f3;
        border-top: 6px solid #007bff;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin:auto;
    "></div>
    <p style="margin-top:10px;">⏳ Thinking... AI is reading your page... 🤖📖</p>
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</div>
""", visible=False)

    
        answer = gr.TextArea(label='Answer', lines=6)
    
        gr.Markdown("---")
        gr.HTML("""
        <div style="text-align:center; color: gray; font-size: 14px;">
            💡 Built by Darsh using LLMs + web scraping + love for AI.
        </div>
        """)

    ini_submit.click(fn=initializer, inputs=[api_key], outputs=[initialize_page, api_error_msg, free_mess, link_pro_page])
    link_submit.click(fn=summary_generator, inputs=[link], outputs=[link_pro_page, processing_page, final_page, page_summary, inv_link, vectorstore] )
    final_submit_but.click(fn=chat, inputs=[question, vectorstore, top_k], outputs=[loading_spinner, answer])
    
# let's go!
ui.launch()
