from flask import Flask,render_template,request,jsonify
from flask_cors import CORS


from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 

load_dotenv()

SECRET_KEY=os.getenv('GROQ_API_KEY')
chat=ChatGroq(model="llama-3.1-8b-instant",api_key=SECRET_KEY, temperature=0.7)



def askToGroq(receive_msg):
    system=SystemMessagePromptTemplate.from_template("""
         You are an advanced **AI-powered blog content generator** designed to create **high-quality, engaging, and informative articles** based on user queries. 

### **Guidelines for Blog Generation:**
1. **Professional & Well-Structured:** Ensure the content is well-organized, with proper headings, subheadings, and a logical flow.  
2. **Engaging & Reader-Friendly:** Use a conversational yet professional tone, incorporating storytelling or real-world examples when possible.  
3. **SEO Optimized:** Naturally include relevant keywords, ensuring better visibility in search engines.  
4. **Authentic & Fact-Based:** Provide well-researched, accurate, and plagiarism-free content.  

### **Content Safety & Ethical Considerations:**
- **No Harmful or Unsafe Topics:** Avoid generating content related to violence, illegal activities, hate speech, or misinformation.  
- **Privacy & Security First:** Do not request or generate sensitive personal data.  
- **Respect Ethical Guidelines:** Ensure the content is unbiased, respectful, and suitable for a general audience.  

### **If a Query is Unrelated to Blogging:**
- Politely inform the user: *"I'm here to assist with blog content creation. Please provide a relevant topic or request."*  

Start by asking the user what topic they'd like to generate a blog on.
    
                                                     
                                                     """)
    human=HumanMessagePromptTemplate.from_template("{query}")

    chatPrompt=ChatPromptTemplate.from_messages([system,human])

    chatPromptFormatted=chatPrompt.format(query=receive_msg)
    response=chat.invoke(chatPromptFormatted)
    
    return response.content


app=Flask(__name__)
CORS(app)


@app.route("/",methods=["GET"])
def hello():
    return "<h1>Hello world</h1>"

@app.route("/blogGen",methods=["POST"])
def langRespo():
    try :
        data=request.get_json()
        query=data.get("query","No query provided")
        response=askToGroq(query)
        return jsonify({"status": "success", "message": response}), 200
    except Exception as e:
       return jsonify({"status": "error", "message": str(e)}), 500

    

if __name__=="__main__":
    app.run()