from flask import Flask, request, jsonify, render_template
from groq import Groq
import os
from dotenv import load_dotenv
import pdfplumber
import io

load_dotenv()

app = Flask(__name__)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def truncate_text(text, max_chars=3000):
    """Truncate text to avoid token limits"""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    """Analyze resume against job description using Groq API"""
    try:
        # Check if files are present
        if 'job_description' not in request.files or 'resume' not in request.files:
            return jsonify({'error': 'Both job description and resume files are required'}), 400
        
        job_desc_file = request.files['job_description']
        resume_file = request.files['resume']
        
        # Validate file types
        if not job_desc_file.filename.endswith('.pdf') or not resume_file.filename.endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Extract text from PDFs
        print("Extracting text from job description...")
        job_description_text = extract_text_from_pdf(job_desc_file)
        
        print("Extracting text from resume...")
        resume_text = extract_text_from_pdf(resume_file)
        
        # Truncate texts to avoid token limits
        job_description_text = truncate_text(job_description_text, 2500)
        resume_text = truncate_text(resume_text, 2500)
        
        # Create optimized prompt for Groq
        prompt = f"""Analyze this resume against the job description. Be concise.

JOB DESCRIPTION:
{job_description_text}

RESUME:
{resume_text}

Provide:
1. Match Score (1-10) with brief reason
2. Top 3 Strengths
3. Top 3 Gaps
4. Top 3 Specific Improvements
5. 5 Keywords to Add
6. Top 3 Priority Actions"""

        # Call Groq API with current model
        print("Calling Groq API for analysis...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR professional. Provide concise, actionable resume feedback."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",  # Current active model
            temperature=0.7,
            max_tokens=1500,
            top_p=1,
            stream=False
        )
        
        # Extract response
        analysis_result = chat_completion.choices[0].message.content
        
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'job_description_length': len(job_description_text),
            'resume_length': len(resume_text)
        })
    
    except Exception as e:
        print(f"Error in analyze_resume: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot conversations using Groq API"""
    try:
        data = request.json
        message = data.get('message', '')
        conversation_history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Prepare messages for Groq
        messages = [
            {
                "role": "system",
                "content": "You are a helpful career advisor. Provide practical, concise advice about resumes, interviews, and job searches."
            }
        ]
        
        # Limit conversation history to last 4 messages to save tokens
        recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
        
        # Add conversation history
        for msg in recent_history:
            messages.append({
                "role": msg.get('role'),
                "content": msg.get('content')
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Call Groq API with current model
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",  # Current active model
            temperature=0.8,
            max_tokens=800,
            top_p=1,
            stream=False
        )
        
        response = chat_completion.choices[0].message.content
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if API key is set
    if not os.getenv('GROQ_API_KEY'):
        print("WARNING: GROQ_API_KEY not found in environment variables!")
        print("Please create a .env file with your GROQ_API_KEY")
    
    app.run(debug=True, port=5000, host='0.0.0.0')