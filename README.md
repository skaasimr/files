# AI Resume Analyzer with Groq API - Setup Guide

## 📋 Overview
This is a complete AI-powered resume analyzer that uses Groq's ultra-fast LLaMA 3.1 70B model to analyze resumes against job descriptions and provide intelligent career advice through a chatbot interface.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (HTML)                       │
│  - Resume upload interface                                   │
│  - Job description upload                                    │
│  - AI Chatbot                                               │
│  - Results display                                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ HTTP Requests (fetch API)
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                    Flask Backend (app.py)                    │
│  - /api/analyze-resume endpoint                             │
│  - /api/chat endpoint                                       │
│  - PDF text extraction (pdfplumber)                         │
│  - CORS enabled                                             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ API Calls
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                      Groq API                                │
│  - Model: llama-3.1-70b-versatile                          │
│  - Fast inference                                           │
│  - Chat completions                                         │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Step-by-Step Setup Instructions

### Step 1: Get Your Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to "API Keys" section
4. Click "Create API Key"
5. Copy your API key (it looks like: `gsk_...`)

### Step 2: Install Python Dependencies

```bash
# Navigate to the backend directory
cd backend

# Install required packages
pip install flask flask-cors groq python-dotenv pdfplumber Pillow
```

Or use the requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

1. Copy the `.env.example` file:
```bash
cp .env.example .env
```

2. Edit `.env` file and add your Groq API key:
```
GROQ_API_KEY=gsk_your_actual_api_key_here
```

### Step 4: Start the Backend Server

```bash
# Make sure you're in the backend directory
cd backend

# Run the Flask app
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Step 5: Access the Application

Open your browser and go to:
```
http://localhost:5000
```

## 🔧 How It Works

### Resume Analysis Flow

1. **User uploads files:**
   - Job description (PDF)
   - Resume (PDF)

2. **Frontend sends data:**
   ```javascript
   const formData = new FormData();
   formData.append('job_description', jobDescFile);
   formData.append('resume', resumeFile);
   
   fetch('http://localhost:5000/api/analyze-resume', {
       method: 'POST',
       body: formData
   })
   ```

3. **Backend processes:**
   - Extracts text from both PDFs using `pdfplumber`
   - Combines text into a structured prompt
   - Sends to Groq API with LLaMA 3.1 70B model

4. **Groq API analyzes:**
   - Compares resume against job requirements
   - Identifies strengths and gaps
   - Generates specific improvement suggestions
   - Recommends keywords to add
   - Provides actionable next steps

5. **Results displayed:**
   - Frontend receives formatted analysis
   - Displays in a beautiful, readable format
   - User can see detailed feedback

### Chatbot Flow

1. **User types message:**
   - Question about resume, career, interview, etc.

2. **Frontend sends:**
   ```javascript
   fetch('http://localhost:5000/api/chat', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
           message: userMessage,
           history: conversationHistory
       })
   })
   ```

3. **Backend processes:**
   - Maintains conversation context
   - Sends to Groq with career advisor system prompt
   - LLaMA 3.1 70B generates intelligent response

4. **AI responds:**
   - Personalized career advice
   - Resume optimization tips
   - Interview preparation guidance

## 📁 Project Structure

```
backend/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .env                  # Your actual API key (create this)
└── templates/
    └── index.html        # Frontend interface
```

## 🎯 API Endpoints

### 1. Analyze Resume
**Endpoint:** `POST /api/analyze-resume`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `job_description`: PDF file
  - `resume`: PDF file

**Response:**
```json
{
  "success": true,
  "analysis": "Detailed AI analysis text...",
  "job_description_length": 1234,
  "resume_length": 5678
}
```

### 2. Chat
**Endpoint:** `POST /api/chat`

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "message": "How can I improve my resume?",
  "history": [
    {"role": "user", "content": "previous message"},
    {"role": "assistant", "content": "previous response"}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "response": "Here are some tips to improve your resume..."
}
```

## 🔍 Key Features

### Resume Analyzer
- ✅ PDF text extraction
- ✅ Job description comparison
- ✅ ATS optimization suggestions
- ✅ Keyword recommendations
- ✅ Gap analysis
- ✅ Specific improvement actions

### AI Chatbot
- ✅ Career advice
- ✅ Resume tips
- ✅ Interview preparation
- ✅ Cover letter guidance
- ✅ Job search strategies
- ✅ Conversation memory

### Technical Features
- ✅ Fast inference with Groq
- ✅ LLaMA 3.1 70B model
- ✅ CORS enabled
- ✅ Error handling
- ✅ Responsive design
- ✅ Beautiful UI with Tailwind CSS

## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution:** Make sure you created a `.env` file with your API key:
```bash
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### Issue: "Failed to analyze resume"
**Solutions:**
1. Check if backend is running on port 5000
2. Verify PDF files are valid
3. Check Groq API key is correct
4. Look at backend console for error messages

### Issue: "CORS error"
**Solution:** Make sure Flask-CORS is installed:
```bash
pip install flask-cors
```

### Issue: "Cannot extract PDF text"
**Solutions:**
1. Ensure PDF is not password-protected
2. Try a different PDF
3. Check if `pdfplumber` is installed correctly

## 🎨 Customization

### Change Groq Model
In `app.py`, modify the model parameter:
```python
model="llama-3.1-70b-versatile"  # Current
# or
model="llama-3.1-8b-instant"     # Faster, less detailed
```

### Adjust Response Length
Modify `max_tokens`:
```python
max_tokens=2000,  # Longer responses
# or
max_tokens=500,   # Shorter responses
```

### Change Backend Port
In `app.py`:
```python
app.run(debug=True, port=5000)  # Change port here
```

And in `templates/index.html`:
```javascript
const API_BASE_URL = 'http://localhost:5000';  // Update here too
```

## 📊 Performance

- **Resume Analysis:** ~3-8 seconds (depending on resume length)
- **Chat Response:** ~1-3 seconds
- **Model:** LLaMA 3.1 70B (highly accurate)
- **Inference Speed:** Powered by Groq's LPU (fastest in market)

## 🔐 Security Notes

- Never commit `.env` file to Git
- Keep your Groq API key secret
- Consider adding rate limiting in production
- Validate file uploads (size, type)
- Add user authentication for production use

## 📝 Example Prompts

### For Resume Analysis:
The system automatically creates detailed prompts that include:
- Job description text
- Resume text
- Structured analysis sections
- Specific output format

### For Chatbot:
Try asking:
- "How can I improve my resume for a software engineering role?"
- "What should I include in my cover letter?"
- "How do I prepare for a technical interview?"
- "What keywords should I add for an ATS?"

## 🚀 Production Deployment

For production, consider:
1. Use a production WSGI server (gunicorn)
2. Add environment-based configuration
3. Implement rate limiting
4. Add user authentication
5. Use HTTPS
6. Add logging and monitoring
7. Deploy backend and frontend separately

Example with gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review backend console logs
3. Verify all dependencies are installed
4. Ensure API key is valid

## 🎉 You're All Set!

Your AI Resume Analyzer is now ready to use. Upload your resume and job description to get instant, AI-powered feedback!
