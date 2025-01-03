#!/bin/bash

# Install dependencies
apt-get update && apt-get install -y \
    curl \
    tmux \
    python3-venv \
    python3-pip

# Install Ollama for Linux
curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/local/bin/ollama
chmod +x /usr/local/bin/ollama

# Install langchain-ollama
pip install langchain-ollama

# Create or attach to existing session
if ! tmux has-session -t myproject 2>/dev/null; then
    # Create new session
    tmux new-session -d -s myproject
    
    # Setup Python environment
    tmux send-keys -t myproject "python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" C-m
    sleep 10
    
    # Setup Ollama in horizontal split
    tmux split-window -h -t myproject
    tmux send-keys -t myproject:0.1 "ollama serve" C-m
    sleep 5
    
    # Setup Mistral in vertical split
    tmux split-window -v -t myproject:0.1
    tmux send-keys -t myproject:0.2 "ollama pull mistral" C-m
    sleep 5
    
    # Start FastAPI in main pane
    tmux select-pane -t myproject:0.0
    tmux send-keys -t myproject:0.0 "uvicorn main:app --host 0.0.0.0 --port 8000 --reload" C-m
    
    # Set layout
    tmux select-layout -t myproject main-vertical
fi

# Attach to session
tmux attach-session -t myproject