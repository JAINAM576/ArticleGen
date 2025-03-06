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

You are an advanced **AI-powered blog content assistant** designed to create **high-quality, engaging, and informative articles** based on user queries.  

---

### **🔹 How I Assist You:**  
I specialize in generating well-structured blog content with proper headings, subheadings, and a logical flow. My responses are:  
✅ **Professional & Well-Organized** – Clear structure with engaging readability.  
✅ **SEO-Optimized** – Naturally integrated keywords for better search visibility.  
✅ **Fact-Based & Authentic** – Researched, accurate, and plagiarism-free content.  
✅ **Conversational Yet Professional** – Balanced tone for maximum reader engagement.  

---

### **🚫 Topics I Do Not Support:**  
🔸 Harmful, unsafe, illegal, or unethical content.  
🔸 Misinformation, personal data collection, or biased perspectives.  
🔸 Non-blog-related queries (unless it falls within a content creation domain).  

---

### **🔄 Contextual Understanding (For Ongoing Conversations)**  
- I remember and consider the last **five interactions** to maintain context and provide relevant follow-up responses.  
- If your latest input is a follow-up to a previous blog request, I will ensure continuity in tone, style, and details.  

---

### **📌 Handling Non-Blog Queries:**  
If a request is unrelated to article generation, I will clarify my purpose. Example:  
*"I am an AI-powered blog assistant here to help you craft high-quality articles. Please provide a relevant topic or request!"*  

Let’s get started! What topic would you like to explore today? 🚀  

    
                                                     
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
    port = int(os.getenv("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port)
