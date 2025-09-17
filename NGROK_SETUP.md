# 🔑 Setting up ngrok for Google Colab

This guide explains how to set up ngrok authentication to create public URLs for your Portable AI Agent on Google Colab.

## 🚀 Quick Setup (Recommended)

### Step 1: Get Your ngrok Auth Token

1. **Sign up for ngrok** (it's free!):
   - Go to [https://ngrok.com/](https://ngrok.com/)
   - Click "Sign up" and create a free account
   - Verify your email address

2. **Get your auth token**:
   - After logging in, go to [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
   - Copy your personal auth token (it looks like: `2abc123def456ghi789jkl012mno345_6pqr789stu012vwx345yz`)

### Step 2: Add Token to Google Colab Secrets

1. **Open your Colab notebook**
2. **Click the 🔑 "Secrets" tab** in the left sidebar
3. **Add new secret**:
   - **Name**: `NGROK_AUTH_TOKEN`
   - **Value**: Paste your auth token from Step 1
   - **Notebook access**: Toggle ON
4. **Save the secret**

### Step 3: Run Your AI Agent

Now when you run the Colab notebook, it will automatically:
- ✅ Find your auth token from secrets
- ✅ Authenticate with ngrok
- ✅ Create a public HTTPS URL
- ✅ Allow access from any device/location

## 🛠️ Alternative Methods

### Method 1: Environment Variable (Temporary)

```python
import os
os.environ['NGROK_AUTH_TOKEN'] = 'your_token_here'
```

### Method 2: Direct Authentication

```python
from pyngrok import ngrok
ngrok.set_auth_token('your_token_here')
```

### Method 3: Manual Setup

```bash
# In a Colab cell
!ngrok config add-authtoken your_token_here
```

## 🔒 Security Best Practices

- ✅ **Use Colab Secrets**: Keep your token secure
- ✅ **Don't hardcode tokens**: Never put tokens directly in code
- ✅ **Monitor usage**: Check your ngrok dashboard regularly
- ✅ **Regenerate if needed**: You can create new tokens anytime

## 🆓 Free ngrok Limits

The free ngrok plan includes:
- **1 online tunnel** at a time
- **20 connections/minute** (plenty for personal use)
- **HTTPS secure tunnels**
- **Custom domains** (random subdomain)
- **Session-based** (URLs change each time)

## 🔧 Troubleshooting

### Error: "authentication failed"
- ✅ Check that your token is correct
- ✅ Make sure the secret name is exactly `NGROK_AUTH_TOKEN`
- ✅ Verify notebook access is enabled for the secret

### Error: "tunnel not found"
- ✅ Restart the Colab runtime
- ✅ Run the setup cells in order
- ✅ Wait for Flask server to start before creating tunnel

### Error: "account limit exceeded"
- ✅ Close other ngrok tunnels
- ✅ Check your ngrok dashboard for active tunnels
- ✅ Free accounts have limits on concurrent tunnels

### Alternative: Local Access Only

If ngrok doesn't work, the notebook will fallback to local access:
- URL: `http://localhost:5000`
- Works within Colab environment
- Not accessible from outside

## 📱 Using Your AI Agent

Once setup is complete:

1. **Get the public URL** from the notebook output
2. **Click the link** to open the chat interface
3. **Start chatting** with your AI agent
4. **Share the URL** with others (while the session is active)

## 💡 Pro Tips

- **Bookmark the URL**: While your session is active
- **Mobile friendly**: Works great on phones and tablets
- **Multiple users**: Others can access the same URL simultaneously
- **Keep cell running**: Stop the cell = URL stops working
- **Session memory**: AI remembers conversation during the session

## 🆘 Need Help?

- **ngrok Documentation**: [https://ngrok.com/docs](https://ngrop.com/docs)
- **GitHub Issues**: Report problems on the repository
- **Colab Help**: Check Google Colab documentation

---

## 🎉 That's it!

Your Portable AI Agent will now be accessible from anywhere in the world through a secure HTTPS URL. Happy chatting! 🤖✨