import fetch
import features
import infer
import execute_fixed
from datetime import date, datetime
from zoneinfo import ZoneInfo

HISTORY_PATH = "history.csv"
ET           = ZoneInfo('America/New_York')


def main():
    print(f"[run_fixed.py] Starting fixed-mode inference pipeline - {date.today()}")

    gbtArtifacts, nnArtifacts = infer.load_artifacts()

    chainData = fetch.get_chain()
    if chainData.empty:
        print("[run_fixed.py] Chain snapshot returned no data. Aborting.")
        return

    print(f"[run_fixed.py] Chain loaded - {len(chainData):,} contracts.")

    minuteBars  = fetch.get_spy_minute_bars()
    atmIv       = fetch.get_atm_implied_vol(chainData)
    featureData = features.build_features(chainData, minuteBars, HISTORY_PATH, atmIv)
    signals     = infer.get_signals(featureData, gbtArtifacts, nnArtifacts)

    if signals.empty:
        print("[run_fixed.py] No signals on initial scan.")
    else:
        print(f"[run_fixed.py] {len(signals)} signal(s) on initial scan.")
        print(signals[['expiration', 'strike', 'right', 'ask', 'gbt_score', 'nn_score', 'nn_abstained']].to_string(index=False))

    def rescan():
        now = datetime.now(ET)
        print(f"[run_fixed.py] Rescan at {now.strftime('%H:%M')}...")

        chainData = fetch.get_chain()
        if chainData.empty:
            return []

        minuteBars  = fetch.get_spy_minute_bars()
        atmIv       = fetch.get_atm_implied_vol(chainData)
        featureData = features.build_features(chainData, minuteBars, HISTORY_PATH, atmIv)
        newSignals  = infer.get_signals(featureData, gbtArtifacts, nnArtifacts)

        if newSignals.empty:
            return []

        return execute_fixed.place_additional_orders(newSignals)

    print("[run_fixed.py] Starting execute loop...")
    execute_fixed.run(signals, rescan)

    print("[run_fixed.py] Updating daily history...")
    features.update_history(HISTORY_PATH, atmIv, minuteBars)

    print("[run_fixed.py] Done.")


if __name__ == "__main__":
    main()