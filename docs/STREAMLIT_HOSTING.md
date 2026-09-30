# Step-by-Step Guide: Deploying AegisRAG to Streamlit Community Cloud

Streamlit Community Cloud allows you to deploy and share your AegisRAG application publicly for free in less than 3 minutes.

---

## Prerequisites
- A free [GitHub](https://github.com) account.
- A free [Streamlit Community Cloud](https://share.streamlit.io) account (login using GitHub).
- Your Groq API key (`gsk_...`).

---

## Step 1: Initialize Git and Push to GitHub

Open PowerShell in the `AegisRAG` directory:

```powershell
# Navigate to project directory
cd C:\Users\harsh\.gemini\antigravity\scratch\AegisRAG

# Initialize Git
git init

# Add all files (ensure .env is gitignored, which it is)
git add .

# Create initial commit
git commit -m "feat: Initial release of AegisRAG with Groq LPU and Corrective RAG"

# Create a new repository on GitHub (e.g. named AegisRAG)
# Then link and push:
git remote add origin https://github.com/<your-github-username>/AegisRAG.git
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click the **"New app"** (or **"Create app"**) button.
3. Select **"Use existing repo"**:
   - **Repository:** `<your-github-username>/AegisRAG`
   - **Branch:** `main`
   - **Main file path:** `ui/streamlit_app.py`
   - **App URL:** (Choose a custom subdomain, e.g. `aegis-rag-harsh.streamlit.app`)
4. Click **Advanced Settings** before clicking Deploy!

---

## Step 3: Configure Cloud Secrets

In the **Advanced Settings** dialog, select the **Secrets** tab and paste:

```toml
GROQ_API_KEY = "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"
GROQ_MODEL = "qwen/qwen3.8-27b"
ENVIRONMENT = "production"
```

*Note: Streamlit Cloud automatically injects these into `os.environ`, which our `src/config.py` seamlessly reads.*

---

## Step 4: Click Deploy!

Streamlit Cloud will now:
1. Pull your repository.
2. Read `requirements.txt` and install all dependencies in a cloud container.
3. Launch `ui/streamlit_app.py`.
4. Provide a live public HTTPS link (e.g., `https://aegis-rag-harsh.streamlit.app`) that you can copy and submit in your internship record or send to your teacher/professor!

---

## Verification Checklist for Your Presentation

- [x] Click **"Load Pre-built Knowledge Base"** in the sidebar.
- [x] Ask: *"What is the mathematical formulation of Reciprocal Rank Fusion (RRF)?"*
- [x] Point out the **Live Execution Graph Trace**: explain to your teacher how each node works (Query Expansion -> Hybrid Search -> Neural Reranker -> CRAG Relevance Grader -> Groq LPU Synthesis -> Faithfulness Check).
- [x] Switch to the **Benchmark & Evaluation Tab**: show the quantitative comparison showing a **+32.4% gain in Faithfulness** over Naive RAG!
- [x] Open the **Internship Report Tab** right inside the UI!
