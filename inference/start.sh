#!/bin/bash
export PATH="/root/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
cd ~/inference
mkdir -p ~/inference/logs

LOCKFILE=~/inference/run.lock
exec 200>"$LOCKFILE"
flock -n 200 || { echo "[start.sh] Already running, exiting."; exit 0; }
echo $$ >&200

if ! curl -s http://localhost:25503 > /dev/null 2>&1; then
    pkill -f "ThetaTerminalv3.jar" 2>/dev/null
    sleep 1
    nohup java -jar ThetaTerminalv3.jar < /dev/null >> ~/inference/logs/theta.log 2>&1 &
    echo "[start.sh] Waiting for ThetaData Terminal..."
    theta_ready=false
    for i in $(seq 1 60); do
        sleep 5
        if curl -s http://localhost:25503 > /dev/null 2>&1; then
            echo "[start.sh] ThetaData Terminal ready."
            theta_ready=true
            break
        fi
    done

    if [ "$theta_ready" = false ]; then
        echo "[start.sh] ThetaData Terminal failed to start after 5 minutes. Aborting."
        exit 1
    fi
fi

export $(cat .env | xargs)
uv run run.py >> ~/inference/logs/run.log 2>&1
