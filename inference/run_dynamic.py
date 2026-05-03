import fetch
import features
import infer
import execute_dynamic
from datetime import date, datetime
from zoneinfo import ZoneInfo

HISTORY_PATH = "history.csv"
ET           = ZoneInfo('America/New_York')


def _is_end_of_day():
    now = datetime.now(ET)
    return now.hour == 15 and now.minute >= 50


def main():
    print(f"[run_dynamic.py] Starting inference pipeline - {date.today()}")

    print("[run_dynamic.py] Loading model artifacts...")
    gbtArtifacts, nnArtifacts = infer.load_artifacts()

    print("[run_dynamic.py] Initial chain fetch...")
    chainData = fetch.get_chain()

    if chainData.empty:
        print("[run_dynamic.py] Chain snapshot returned no data. Aborting.")
        return

    print(f"[run_dynamic.py] Chain loaded - {len(chainData):,} contracts after DTE filter.")

    minuteBars = fetch.get_spy_minute_bars()
    atmIv      = fetch.get_atm_implied_vol(chainData)
    featureData = features.build_features(chainData, minuteBars, HISTORY_PATH, atmIv)
    signals     = infer.get_signals(featureData, gbtArtifacts, nnArtifacts)

    if signals.empty:
        print("[run_dynamic.py] No signals on initial scan.")
    else:
        print(f"[run_dynamic.py] {len(signals)} signal(s) on initial scan.")
        print(signals[['expiration', 'strike', 'right', 'ask', 'gbt_score', 'nn_score', 'nn_abstained']].to_string(index=False))

    def rescan():
        now = datetime.now(ET)
        print(f"[run_dynamic.py] Rescan at {now.strftime('%H:%M')}...")

        chainData = fetch.get_chain()
        if chainData.empty:
            print("[run_dynamic.py] Rescan: empty chain.")
            return []

        minuteBars  = fetch.get_spy_minute_bars()
        atmIv       = fetch.get_atm_implied_vol(chainData)
        featureData = features.build_features(chainData, minuteBars, HISTORY_PATH, atmIv)
        newSignals  = infer.get_signals(featureData, gbtArtifacts, nnArtifacts)

        if newSignals.empty:
            print("[run_dynamic.py] Rescan: no new signals.")
            return []

        print(f"[run_dynamic.py] Rescan: {len(newSignals)} signal(s).")
        return execute_dynamic.place_additional_orders(newSignals)

    print("[run_dynamic.py] Starting execute loop...")
    execute_dynamic.run(signals, rescan)

    print("[run_dynamic.py] Updating daily history...")
    features.update_history(HISTORY_PATH, atmIv, minuteBars)

    print("[run_dynamic.py] Done.")


if __name__ == "__main__":
    main()