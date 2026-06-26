require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const { ChatGroq } = require('@langchain/groq');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');

const app = express();
app.use(cors());
app.use(express.json());

// Set up a temporary folder to capture uploaded text notes
const upload = multer({ dest: 'uploads/' });

// Initialize the Groq LLM Connection
// Replace the text below with your real API key!
const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  modelName: "llama3-8b-8192",
  temperature: 0
});

app.post('/api/chat', upload.single('file'), async (req, res) => {
  try {
    const question = req.body.question;
    const file = req.file;

    if (!file || !question) {
      return res.status(400).json({ error: "Please upload a file and ask a question." });
    }

    // 1. Read the uploaded text note
    const documentText = fs.readFileSync(file.path, 'utf-8');

    // 2. Text Splitting (Chop into small 500-character chunks)
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
    const chunks = await splitter.splitText(documentText);

    // 3. Simple Vector Context Matching
    // We filter the chunks that share keyword matches with your question
    const words = question.toLowerCase().split(' ');
    const relevantChunks = chunks.filter(chunk => 
      words.some(word => chunk.toLowerCase().includes(word))
    ).slice(0, 3);

    const contextText = relevantChunks.length > 0 ? relevantChunks.join("\n") : documentText.substring(0, 1500);

    // 4. Construct the contextual prompt for Groq
    const prompt = `You are an AI Notes Assistant. Answer the question using only the text context provided below. If you cannot find the answer, summarize the context briefly.\n\nContext:\n${contextText}\n\nQuestion: {question}`;

    // 5. Get the answer from the AI
    const response = await model.invoke(prompt);

    // Delete the temporary file from disk cleanly
    fs.unlinkSync(file.path);

    res.json({ answer: response.content });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "The server encountered an error parsing your document." });
  }
});

app.listen(5000, () => console.log("🚀 Server running live on http://localhost:5000"));