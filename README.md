# AI Wardrobe Assistant

A Streamlit app for managing your wardrobe and getting AI-powered outfit recommendations.

## How to Share Your App

### Option 1: Streamlit Community Cloud (Recommended - Free & Easy)

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path to `app.py`
   - Click "Deploy"
   - Your app will be available at: `https://YOUR-APP-NAME.streamlit.app`

### Option 2: Share on Local Network

Run your app with network access:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then share your local IP address:
- Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
- Share URL: `http://YOUR_IP_ADDRESS:8501`

**Note:** Users must be on the same network.

### Option 3: Use ngrok (Temporary Public URL)

1. **Install ngrok:** https://ngrok.com/download
2. **Run your Streamlit app:**
   ```bash
   streamlit run app.py
   ```
3. **In another terminal, create tunnel:**
   ```bash
   ngrok http 8501
   ```
4. **Share the ngrok URL** (e.g., `https://abc123.ngrok.io`)

**Note:** Free ngrok URLs change each time you restart.

### Option 4: Deploy to Other Platforms

- **Heroku:** Use Procfile and requirements.txt
- **AWS:** Use Elastic Beanstalk or EC2
- **Azure:** Use Azure App Service
- **Google Cloud:** Use Cloud Run

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```


