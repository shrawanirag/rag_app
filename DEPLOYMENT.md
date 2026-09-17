# Deploying Notes RAG Chat

This app is ready to deploy to **Streamlit Community Cloud** (free). Steps below.

## 1. Push the project to GitHub

```bash
cd rag_app
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial RAG app"
```

Create a new repo on GitHub, then:
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Do **not** commit `secrets.toml` or your API key anywhere — `.gitignore` already excludes it.

## 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"New app"**.
3. Select your repo, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app — first
   deploy takes a few minutes (downloading the embedding model, etc.).

## 3. Add your Gemini API key as a secret (optional but recommended)

If you want the deployed app to work out-of-the-box for visitors (using *your* quota),
add your key as a Streamlit secret instead of making every visitor paste their own:

1. In the Streamlit Cloud dashboard, open your app → **Settings → Secrets**.
2. Paste:
   ```toml
   GEMINI_API_KEY = "your-actual-key-here"
   ```
3. Save. The app already reads this via `st.secrets.get("GEMINI_API_KEY", "")` and
   pre-fills the sidebar field.

If you skip this step, the app still works — each visitor just enters their own key
in the sidebar (safer if you're sharing the app publicly and don't want to cover
everyone's API usage).

## 4. Done

Your app is now live at `https://<your-app-name>.streamlit.app`. Any time you push
new commits to `main`, Streamlit Cloud redeploys automatically.

---

## Things to know before sharing it widely

- **Session state resets** — each visitor's uploaded docs and FAISS index only live
  for their browser session; nothing persists across visits or app restarts. For a
  personal/demo app this is fine. If you want persistence, you'd add a real vector DB
  (e.g. Chroma, Pinecone, Qdrant) instead of the in-memory FAISS index.
- **Free tier limits** — Streamlit Community Cloud apps sleep after inactivity and
  wake up on the next visit (a few seconds delay). Fine for personal/demo use, not for
  production traffic.
- **API key exposure** — if you add your own key as a secret, anyone who uses your
  deployed app draws from your Gemini quota. Keep this in mind if you share the link
  widely; Gemini's free tier has daily/per-minute limits.
- **Alternative hosts** — if you outgrow Streamlit Cloud, the same `app.py` (with only
  minor tweaks) can run on **Hugging Face Spaces** (also free, supports Streamlit
  natively) or as a Docker container on **Render**/**Railway**/**Fly.io** for more
  control over resources.

## Other ways to build on this

- **Persistent vector store**: swap the in-memory FAISS index for Chroma or Qdrant
  so documents don't need to be re-uploaded every session.
- **Multi-user support**: store each user's documents/index keyed by a session ID in
  a real database, rather than Streamlit's session state.
- **Better chunking**: try sentence-aware or semantic chunking (e.g. via LangChain's
  text splitters) instead of fixed word-count chunks.
- **Source citations**: track which file/page each chunk came from and show it next
  to the answer.
