import httpx
import subprocess
import time
import pandas as pd
import fetch
import features
import infer
import execute
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HISTORY_PATH      = "history.csv"
THETADATA_URL     = "http://localhost:25503"
THETADATA_JAR     = "ThetaTerminalv3.jar"
THETADATA_LOG     = Path.home() / "inference" / "logs" / "theta.log"
ET                = ZoneInfo('America/New_York')


def _ensure_thetadata():
    try:
        httpx.get(THETADATA_URL, timeout=5)
        return True
    except Exception:
        pass

    print("[run.py] ThetaData unreachable - attempting restart...")
    subprocess.run(["pkill", "-f", THETADATA_JAR], capture_output=True)
    time.sleep(1)

    with open(THETADATA_LOG, "a") as theta_log:
        subprocess.Popen(
            ["java", "-jar", THETADATA_JAR],
            stdin=subprocess.DEVNULL,
            stdout=theta_log,
            stderr=theta_log,
        )

    for _ in range(60):
        time.sleep(5)
        try:
            httpx.get(THETADATA_URL, timeout=5)
            print("[run.py] ThetaData restarted successfully.")
            return True
        except Exception:
            pass

    print("[run.py] ThetaData failed to restart - skipping this rescan cycle.")
    return False


def main():
    print(f"[run.py] Starting {execute.TRADE_MODE} inference pipeline - {date.today()}")

    print("[run.py] Loading model artifacts...")
    gbtArtifacts, nnArtifacts = infer.load_artifacts()

    lastAtmIv      = [None]
    lastMinuteBars = [pd.DataFrame()]

    print("[run.py] Initial chain fetch...")
    chainData = fetch.get_chain()
    signals   = pd.DataFrame()

    if chainData.empty:
        print("[run.py] Pre-market: chain empty. Entering rescan-only mode.")
    else:
        print(f"[run.py] Chain loaded - {len(chainData):,} contracts after DTE filter.")

        lastMinuteBars[0] = fetch.get_spy_minute_bars()
        lastAtmIv[0]      = fetch.get_atm_implied_vol(chainData)
        featureData       = features.build_features(chainData, lastMinuteBars[0], HISTORY_PATH, lastAtmIv[0])
        signals           = infer.get_signals(featureData, gbtArtifacts, nnArtifacts)

        if signals.empty:
            print("[run.py] No signals on initial scan.")
        else:
            print(f"[run.py] {len(signals)} signal(s) on initial scan.")
            print(signals[['expiration', 'strike', 'right', 'ask', 'gbt_score', 'nn_score', 'nn_abstained']].to_string(index=False))

    def rescan():
        now = datetime.now(ET)
        print(f"[run.py] Rescan at {now.strftime('%H:%M')}...")

        thetadata_is_ready = _ensure_thetadata()
        if not thetadata_is_ready:
            return []

        chainData = fetch.get_chain()
        if chainData.empty:
            print("[run.py] Rescan: empty chain.")
            return []

        minuteBars = fetch.get_spy_minute_bars()
        atmIv      = fetch.get_atm_implied_vol(chainData)

        lastMinuteBars[0] = minuteBars
        lastAtmIv[0]      = atmIv

        featureData = features.build_features(chainData, minuteBars, HISTORY_PATH, atmIv)
        newSignals  = infer.get_signals(featureData, gbtArtifacts, nnArtifacts)

        if newSignals.empty:
            print("[run.py] Rescan: no new signals.")
            return []

        print(f"[run.py] Rescan: {len(newSignals)} signal(s).")
        print(newSignals[['expiration', 'strike', 'right', 'ask', 'gbt_score', 'nn_score', 'nn_abstained']].to_string(index=False))
        return execute.place_additional_orders(newSignals)

    print("[run.py] Starting execute loop...")
    try:
        execute.run(signals, rescan)
    finally:
        if lastAtmIv[0] is not None:
            print("[run.py] Updating daily history...")
            features.update_history(HISTORY_PATH, lastAtmIv[0], lastMinuteBars[0])
        else:
            print("[run.py] No ATM IV collected - skipping history update.")
        print("[run.py] Shutting down ThetaData Terminal...")
        subprocess.run(["pkill", "-f", THETADATA_JAR], capture_output=True)

    print("[run.py] Market is closed. Awaiting next trading day.")
    print("[run.py] Done.")


if __name__ == "__main__":
    main()
