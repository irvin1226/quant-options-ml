#!/bin/bash
export PATH="/root/.local/bin:$PATH"
cd ~/inference
mkdir -p ~/inference/logs
if pgrep -f "run_dynamic.py" > /dev/null; then
    echo "[start.sh] Already running, exiting."
    exit 0
fi
if ! pgrep -f "ThetaTerminalv3.jar" > /dev/null; then
    java -jar ThetaTerminalv3.jar &
    echo "[start.sh] Waiting for ThetaData Terminal..."
    for i in $(seq 1 30); do
        sleep 2
        if curl -s http://localhost:25503 > /dev/null 2>&1; then
            echo "[start.sh] ThetaData Terminal ready."
            break
        fi
    done
fi
export $(cat .env | xargs)
uv run run_dynamic.py >> ~/inference/logs/run.log 2>&1
