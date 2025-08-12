### AWS EC2 Deployment Guide — Discord Stock Bot

This guide walks you through deploying any of the bot variants in this repo to AWS EC2 and keeping it running reliably with systemd.

- Variants you can run (pick one):
  - `multi-langchain_bot.py` (token: `MULTI_LANGCHAIN_DISCORD_TOKEN`)
  - `langchain_bot.py` (token: `LANGCHAIN_DISCORD_TOKEN`)
  - `langgraph_bot.py` (token: `LANGGRAPH_DISCORD_TOKEN`)
  - `dspy_bot.py` (token: `DSPY_DISCORD_TOKEN`)

All variants support Anthropic and/or OpenAI via environment variables. The code uses `python-dotenv` to load `.env` if present.

---

### 0) Prerequisites

- AWS account, key pair (`.pem`) for SSH
- Discord bot token for the variant you’ll run (see mapping above)
- At least one LLM API key:
  - `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`
- Your local machine with SSH access to the instance

---

### 1) Launch an EC2 instance

- AMI: Ubuntu 22.04 LTS (or 24.04)
- Instance type: `t3.small` (works well; `t3.micro` may work but is tighter)
- Storage: 16–20 GB gp3
- Security group:
  - Inbound: port 22 (SSH) from your IP only
  - Outbound: allow all (needed to reach Discord and LLM APIs)
- Key pair: create/download `.pem`

---

### 2) SSH into the instance (from your local machine)

```bash
chmod 400 ~/Downloads/your-key.pem
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_INSTANCE_PUBLIC_IP
```

---

### 3) Install system dependencies (on EC2)

```bash
sudo apt update -y
sudo apt install -y python3 python3-venv python3-pip git build-essential
```

---

### 4) Transfer code to EC2

Pick one option:

- Option A — Secure copy from your local machine:
  ```bash
  # Run on your local machine (replace IP and path)
  scp -i ~/Downloads/your-key.pem -r \
    "/Users/pookansmacbookpro/Downloads/discord_bot" \
    ubuntu@YOUR_INSTANCE_PUBLIC_IP:~
  ```

- Option B — Use git (if you store the repo remotely):
  - Push your repo to GitHub (ask before pushing, as desired)
  - On EC2: `git clone https://github.com/you/discord_bot.git`

---

### 5) Create a Python virtual environment and install packages (on EC2)

```bash
cd ~/discord_bot
python3 -m venv .venv
source .venv/bin/activate
```

Create `requirements.txt` in the project root with the following content (note the `pydantic==2.5.0` pin which keeps compatibility with both LangChain and DSPy):

```bash
cat > requirements.txt << 'EOF'
discord.py==2.4.0
python-dotenv==1.0.1
yfinance==0.2.40
pandas==2.2.2
requests==2.32.3
pydantic==2.5.0

# LangChain stack (used by single, multi, and langgraph variants)
langchain==0.2.11
langchain-openai==0.1.22
langchain-anthropic==0.1.18
langgraph==0.2.32

# Optional: only if you will run dspy_bot.py
# If you include this, keep pydantic at 2.5.0
dspy==2.4.7
EOF

pip install --upgrade pip
pip install -r requirements.txt
```

If you already attempted installation and saw a dependency conflict (ResolutionImpossible) related to `pydantic` and `dspy-ai`, fix with:

```bash
source .venv/bin/activate
pip uninstall -y pydantic
pip install "pydantic==2.5.0"
pip install -r requirements.txt
```

---

### 6) Add your environment variables

You can use a single `.env` file in the project directory (loaded by `load_dotenv()`).

```bash
cat > .env << 'EOF'
# If running ONE bot, set only its token. If running ALL FOUR, include all tokens.

# Tokens (set what you need; extra tokens are OK)
MULTI_LANGCHAIN_DISCORD_TOKEN=YOUR_MULTI_TOKEN
LANGCHAIN_DISCORD_TOKEN=YOUR_SINGLE_TOKEN
LANGGRAPH_DISCORD_TOKEN=YOUR_LANGGRAPH_TOKEN
DSPY_DISCORD_TOKEN=YOUR_DSPY_TOKEN

# At least one of these (both is fine)
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
EOF
```

Token mapping by script:

- `multi-langchain_bot.py` → `MULTI_LANGCHAIN_DISCORD_TOKEN`
- `langchain_bot.py` → `LANGCHAIN_DISCORD_TOKEN`
- `langgraph_bot.py` → `LANGGRAPH_DISCORD_TOKEN`
- `dspy_bot.py` → `DSPY_DISCORD_TOKEN`

---

### 7) Test run in the foreground (on EC2)

```bash
source .venv/bin/activate
python3 multi-langchain_bot.py
```

You should see logs like:
- “✅ LangChain configured successfully with …”
- “🤖 <bot-user> has connected to Discord!”

Stop with Ctrl+C when done testing.

---

### 8) Run as a background service with systemd (recommended)

Create a secure environment file for the service:

```bash
sudo bash -c 'cat > /etc/discord-bot.env << "EOF" 
MULTI_LANGCHAIN_DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
EOF'
sudo chmod 600 /etc/discord-bot.env
```

Create the service unit (adjust `ExecStart` and the token if using a different variant):

```bash
sudo bash -c 'cat > /etc/systemd/system/discord-bot.service << "EOF"
[Unit]
Description=Discord Stock Bot (multi-langchain)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/discord_bot
EnvironmentFile=/etc/discord-bot.env
ExecStart=/home/ubuntu/discord_bot/.venv/bin/python /home/ubuntu/discord_bot/multi-langchain_bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable discord-bot
sudo systemctl start discord-bot
```

Tail logs:

```bash
sudo journalctl -u discord-bot -f | cat
```

---

### 9) Updating the bot

Re-upload or pull changes, then restart the service.

- If using `scp` from local:
  ```bash
  scp -i ~/Downloads/your-key.pem -r \
    "/Users/pookansmacbookpro/Downloads/discord_bot" \
    ubuntu@YOUR_INSTANCE_PUBLIC_IP:~
  ```

- If using git on EC2:
  ```bash
  cd ~/discord_bot && git pull
  ```

Then restart:

```bash
sudo systemctl restart discord-bot
```

---

### 10) Switching which bot runs

1) Update `/etc/discord-bot.env` to set the correct token for the target script.

2) Update the service `ExecStart`:

```bash
sudo sed -i 's/multi-langchain_bot.py/langchain_bot.py/' /etc/systemd/system/discord-bot.service
```

Or edit the file and change to one of:

- `/home/ubuntu/discord_bot/multi-langchain_bot.py`
- `/home/ubuntu/discord_bot/langchain_bot.py`
- `/home/ubuntu/discord_bot/langgraph_bot.py`
- `/home/ubuntu/discord_bot/dspy_bot.py`

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart discord-bot
```

---

### 11) Troubleshooting

- Bot not appearing online:
  - Check logs: `sudo journalctl -u discord-bot -n 200 | cat`
  - Verify the correct token env var is set for the chosen script
  - Confirm outbound internet: `curl https://discord.com/api/v10/gateway`

- “No API keys found” errors:
  - Set at least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

- Import errors for LangChain/DSPy:
  - Reinstall with `pip install -r requirements.txt`

- yfinance returns empty data:
  - Often transient; ensure outbound internet is allowed

---

### 12) Costs and operations

- `t3.small` typically costs a few dollars/month. You do not need to open inbound ports beyond SSH.
- The service auto-starts on reboot. Use `sudo systemctl status discord-bot` to check health.

---

### Appendix — Quick command summary

```bash
# On EC2
sudo apt update -y && sudo apt install -y python3 python3-venv python3-pip git build-essential
cd ~/discord_bot && python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python3 multi-langchain_bot.py  # test in foreground

# systemd
sudo systemctl daemon-reload
sudo systemctl enable discord-bot
sudo systemctl start discord-bot
sudo journalctl -u discord-bot -f | cat
```


---

### Run all four bots concurrently (single .env)

You can run all four scripts at once using one shared `.env` file in the project directory. Each script calls `load_dotenv()`, so they all read tokens and API keys from the same `.env`.

Ensure `.env` has all tokens and keys:

```bash
cat > /home/ubuntu/discord_bot/.env << 'EOF'
MULTI_LANGCHAIN_DISCORD_TOKEN=YOUR_MULTI_TOKEN
LANGCHAIN_DISCORD_TOKEN=YOUR_SINGLE_TOKEN
LANGGRAPH_DISCORD_TOKEN=YOUR_LANGGRAPH_TOKEN
DSPY_DISCORD_TOKEN=YOUR_DSPY_TOKEN

ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
EOF
```

Create four systemd services that all use the same working directory (so `.env` is picked up by `python-dotenv`):

```bash
# Multi-LangChain
sudo bash -c 'cat > /etc/systemd/system/discord-bot-multi.service << "EOF"
[Unit]
Description=Discord Stock Bot - Multi LangChain
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/discord_bot
ExecStart=/home/ubuntu/discord_bot/.venv/bin/python /home/ubuntu/discord_bot/multi-langchain_bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'

# Single LangChain
sudo bash -c 'cat > /etc/systemd/system/discord-bot-single.service << "EOF"
[Unit]
Description=Discord Stock Bot - Single LangChain
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/discord_bot
ExecStart=/home/ubuntu/discord_bot/.venv/bin/python /home/ubuntu/discord_bot/langchain_bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'

# LangGraph
sudo bash -c 'cat > /etc/systemd/system/discord-bot-graph.service << "EOF"
[Unit]
Description=Discord Stock Bot - LangGraph
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/discord_bot
ExecStart=/home/ubuntu/discord_bot/.venv/bin/python /home/ubuntu/discord_bot/langgraph_bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'

# DSPy
sudo bash -c 'cat > /etc/systemd/system/discord-bot-dspy.service << "EOF"
[Unit]
Description=Discord Stock Bot - DSPy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/discord_bot
ExecStart=/home/ubuntu/discord_bot/.venv/bin/python /home/ubuntu/discord_bot/dspy_bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF'

sudo systemctl daemon-reload
sudo systemctl enable discord-bot-multi discord-bot-single discord-bot-graph discord-bot-dspy
sudo systemctl start  discord-bot-multi discord-bot-single discord-bot-graph discord-bot-dspy
```

Logs:

```bash
sudo journalctl -u discord-bot-multi  -f | cat
sudo journalctl -u discord-bot-single -f | cat
sudo journalctl -u discord-bot-graph  -f | cat
sudo journalctl -u discord-bot-dspy   -f | cat
```

Notes:
- Make sure you have invited each bot (corresponding to each token) to your Discord server.
- All four services read from the same `.env` via `load_dotenv()` because the `WorkingDirectory` is the project folder.
- Consider `t3.medium` if running all four to ensure enough CPU/RAM.

1. Kill all bot services

sudo systemctl stop discord-bot-multi discord-bot-single discord-bot-graph discord-bot-dspy
sudo systemctl disable discord-bot-multi discord-bot-single discord-bot-graph discord-bot-dspy

2. Restart all bot services
sudo systemctl start discord-bot-multi discord-bot-single discord-bot-graph discord-bot-dspy
sudo systemctl enable --now discord-bot-multi discord-bot-single discord-bot-graph discord-bot-dspy

Check status after restart
sudo systemctl status discord-bot-multi discord-bot-single discord-bot-graph discord-bot-dspy --no-pager

Check logs

sudo journalctl -u discord-bot-multi -n 100 --no-pager | cat
sudo journalctl -u discord-bot-single -n 100 --no-pager | cat
sudo journalctl -u discord-bot-graph -n 100 --no-pager | cat
sudo journalctl -u discord-bot-dspy -n 100 --no-pager | cat
