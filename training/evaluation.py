import numpy as np
import pandas as pd

class Metrics:

    def __init__(self, predictions: np.ndarray, true_labels: np.ndarray, threshold: float = 0.5):
        self._true_labels = true_labels
        self._threshold = threshold
        self._predicted_labels = self._apply_threshold(predictions)

    # Converts probabilities to binary decisions based on threshold.
    def _apply_threshold(self, predictions: np.ndarray):
        return (predictions >= self._threshold).astype(float)

    # Overall percentage of correct predictions across all trades
    def accuracy(self):
        return float(np.mean(self._predicted_labels == self._true_labels))

    # Of all trades model said buy, how many were actually profitable.
    def precision(self):
        predicted_positive = self._predicted_labels == 1.0
        true_positive = predicted_positive & (self._true_labels == 1.0)
        total_predicted_positive = predicted_positive.sum()
        if total_predicted_positive == 0:
            return 0.0
        return float(true_positive.sum() / total_predicted_positive)

    # Of all actually profitable trades, how many did the model catch.
    def recall(self):
        actual_positive = self._true_labels == 1.0
        true_positive = (self._predicted_labels == 1.0) & actual_positive
        total_actual_positive = actual_positive.sum()
        if total_actual_positive == 0:
            return 0.0
        return float(true_positive.sum() / total_actual_positive)

    def summary(self):
        print(f"Accuracy:\t {self.accuracy():.4f}")
        print(f"Precision:\t {self.precision():.4f}")
        print(f"Recall:\t {self.recall():.4f}")

# Simulates live trading row-by-row, managing a portfolio of open positions.
# Two position pools (standard, high-confidence) prevent HC signals from crowding out standard slots.
# Trade outcome is determined at entry from true_labels: win applies target_return, loss applies -stop_loss.
# Capital is locked at entry to prevent retroactive resizing after drawdowns.
# Commission is charged twice per trade (entry and exit leg).
# Regime-conditional sizing: when the derived threshold equals the break-even floor, the model
# could not find confident signals on the val set, indicating a low-confidence regime.
# Position size is halved in that case to limit exposure while maintaining market engagement.
class Backtester:

    def __init__(
        self,
        threshold: float,
        high_confidence_threshold: float,
        starting_capital: float,
        target_return: float,
        stop_loss: float,
        position_size: float,
        max_positions: int,
        max_high_confidence_positions: int,
        commission_per_contract: float,
        max_holding_days: int,
        breakeven_floor: float = 0.4167,
    ):
        self._threshold = threshold
        self._high_confidence_threshold = high_confidence_threshold
        self._starting_capital = starting_capital
        self._target_return = target_return
        self._stop_loss = stop_loss
        self._max_positions = max_positions
        self._max_hc_positions = max_high_confidence_positions
        self._commission = commission_per_contract
        self._max_holding_days = max_holding_days

        # Halve position size when threshold was forced to the break-even floor,
        # signaling the model lacked confident val-year signals for this regime.
        if round(threshold, 4) == round(breakeven_floor, 4):
            self._position_size = position_size * 0.5
        else:
            self._position_size = position_size

    def _close_position(self, pos, trade_return, capital, trade_log):
        capital += pos['allocated_dollars'] * trade_return
        capital -= self._commission
        trade_log.append({
            'entry_time':   pos['entry_time'],
            'entry_price':  pos['entry_price'],
            'trade_return': trade_return,
            'capital_after': capital,
            'win':          trade_return > 0,
            'high_confidence': pos['high_confidence'],
            'exit_reason':  pos.get('exit_reason', 'window'),
        })
        return capital

    # Standard backtester - fixed target_return / -stop_loss per trade.
    # Iterates predictions in order. At each tick, expired positions are closed first,
    # then the current signal is evaluated for entry. Returns final capital and a
    # trade log with one entry per closed position.
    def run(self, predictions: np.ndarray, true_labels: np.ndarray, entry_prices: np.ndarray,
            timestamps: np.ndarray, realized_returns: np.ndarray = None):
        capital   = self._starting_capital
        trade_log = []
        open_positions = []

        for i in range(len(predictions)):
            current_time = timestamps[i]

            # Close any positions whose max_holding_days window has elapsed.
            still_open = []
            for pos in open_positions:
                if current_time >= pos['close_time']:
                    capital = self._close_position(pos, pos['trade_return'], capital, trade_log)
                else:
                    still_open.append(pos)
            open_positions = still_open

            standard_used = sum(1 for p in open_positions if not p['high_confidence'])
            hc_used       = sum(1 for p in open_positions if p['high_confidence'])

            confidence        = predictions[i]
            is_high_confidence = confidence >= self._high_confidence_threshold

            if is_high_confidence:
                if hc_used < self._max_hc_positions:
                    slot_type = 'high_confidence'
                else:
                    continue
            elif confidence >= self._threshold:
                if standard_used < self._max_positions:
                    slot_type = 'standard'
                else:
                    continue
            else:
                continue

            entry_price = entry_prices[i]
            label       = true_labels[i]

            if realized_returns is not None:
                trade_return = float(realized_returns[i])
            else:
                trade_return = self._target_return if label == 1.0 else -self._stop_loss

            close_time        = current_time + np.timedelta64(self._max_holding_days, 'D')
            allocated_dollars = capital * self._position_size
            capital          -= self._commission

            open_positions.append({
                'entry_time':      current_time,
                'entry_price':     entry_price,
                'trade_return':    trade_return,
                'close_time':      close_time,
                'high_confidence': slot_type == 'high_confidence',
                'win':             trade_return > 0,
                'allocated_dollars': allocated_dollars,
            })

        for pos in open_positions:
            capital = self._close_position(pos, pos['trade_return'], capital, trade_log)

        return capital, trade_log

    # Dynamic exit backtester - uses exit model predictions to decide when to exit winners.
    # Exit predictions parquet: (entry_timestamp, current_timestamp, predicted_remaining_upside)
    # Logic per open position at each new trading day:
    #   - Look up predicted_remaining_upside for (entry_timestamp, today)
    #   - Hold if predicted_remaining > unrealized_return * HOLD_RATIO
    #   - Exit if predicted_remaining < unrealized_return * EXIT_RATIO
    #   - Emergency exit if position is up > EMERGENCY_MIN and SPY just dropped sharply
    # Falls back to realized_returns for actual trade return when exiting.
    # If no exit prediction exists (loser or missing), falls back to standard behavior.
    def run_with_exit(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        entry_prices: np.ndarray,
        timestamps: np.ndarray,
        realized_returns: np.ndarray,
        spy_returns: np.ndarray,
        exit_predictions_path: str,
        hold_ratio: float = 0.5,
        exit_ratio: float = 0.2,
        emergency_spy_drop: float = -0.005,
        emergency_min_gain: float = 0.10,
    ):
        # Load exit model predictions and build O(1) lookup dict.
        # Key: (entry_timestamp_str, date_str) -> predicted_remaining_upside
        exit_lookup = {}
        try:
            exit_df = pd.read_parquet(exit_predictions_path)
            for _, row in exit_df.iterrows():
                entry_str = str(pd.Timestamp(row['entry_timestamp']))
                date_str  = str(pd.Timestamp(row['current_timestamp']).normalize())
                exit_lookup[(entry_str, date_str)] = float(row['predicted_remaining_upside'])
            print(f"Loaded {len(exit_lookup):,} exit predictions from {exit_predictions_path}")
        except Exception as e:
            print(f"Warning: could not load exit predictions ({e}). Falling back to standard run.")
            return self.run(predictions, true_labels, entry_prices, timestamps, realized_returns)

        capital        = self._starting_capital
        trade_log      = []
        open_positions = []
        seen_dates     = {}  # entry_time_str -> last date we checked exit for this position

        for i in range(len(predictions)):
            current_time = timestamps[i]
            current_date = str(pd.Timestamp(current_time).normalize())
            spy_ret      = float(spy_returns[i]) if spy_returns is not None else 0.0

            # Check exit conditions for open positions on new trading days
            still_open = []
            for pos in open_positions:
                entry_str    = str(pd.Timestamp(pos['entry_time']))
                last_checked = seen_dates.get(entry_str)

                # Window expired - always close
                if current_time >= pos['close_time']:
                    capital = self._close_position(pos, pos['trade_return'], capital, trade_log)
                    seen_dates.pop(entry_str, None)
                    continue

                # Only evaluate exit logic once per trading day per position
                if last_checked == current_date:
                    still_open.append(pos)
                    continue

                seen_dates[entry_str] = current_date

                # Compute current unrealized return using realized_return as proxy
                # (we don't have live bid price here, so use the stored trade_return
                # which is the max realized return - conservative approximation)
                unrealized = pos['trade_return']

                # Emergency exit: SPY dropped sharply and we have meaningful gains
                if spy_ret < emergency_spy_drop and unrealized > emergency_min_gain:
                    capital = self._close_position(
                        {**pos, 'exit_reason': 'emergency'}, unrealized, capital, trade_log
                    )
                    seen_dates.pop(entry_str, None)
                    continue

                # Look up exit model prediction for this position today
                pred_remaining = exit_lookup.get((entry_str, current_date))

                if pred_remaining is not None and unrealized > 0:
                    # Exit: model thinks remaining upside < exit_ratio * what we've made
                    if pred_remaining < unrealized * exit_ratio:
                        capital = self._close_position(
                            {**pos, 'exit_reason': 'exit_model'}, unrealized, capital, trade_log
                        )
                        seen_dates.pop(entry_str, None)
                        continue

                still_open.append(pos)

            open_positions = still_open

            standard_used = sum(1 for p in open_positions if not p['high_confidence'])
            hc_used       = sum(1 for p in open_positions if p['high_confidence'])

            confidence         = predictions[i]
            is_high_confidence = confidence >= self._high_confidence_threshold

            if is_high_confidence:
                if hc_used < self._max_hc_positions:
                    slot_type = 'high_confidence'
                else:
                    continue
            elif confidence >= self._threshold:
                if standard_used < self._max_positions:
                    slot_type = 'standard'
                else:
                    continue
            else:
                continue

            entry_price  = entry_prices[i]
            trade_return = float(realized_returns[i])
            close_time   = current_time + np.timedelta64(self._max_holding_days, 'D')

            allocated_dollars = capital * self._position_size
            capital          -= self._commission

            open_positions.append({
                'entry_time':        current_time,
                'entry_price':       entry_price,
                'trade_return':      trade_return,
                'close_time':        close_time,
                'high_confidence':   slot_type == 'high_confidence',
                'win':               trade_return > 0,
                'allocated_dollars': allocated_dollars,
                'exit_reason':       'window',
            })

        for pos in open_positions:
            capital = self._close_position(pos, pos['trade_return'], capital, trade_log)

        return capital, trade_log

    def summary(self, final_capital: float, trade_log: list):
        total_trades = len(trade_log)

        wins = sum(1 for t in trade_log if t['win'])
        hc_trades = sum(1 for t in trade_log if t['high_confidence'])
        standard_trades = total_trades - hc_trades

        total_return = (final_capital - self._starting_capital) / self._starting_capital * 100
        win_rate     = wins / total_trades * 100 if total_trades > 0 else 0.0

        if total_trades > 0:
            all_returns = [t['trade_return'] for t in trade_log]
            mean_trade  = np.mean(all_returns) * 100
            win_rets    = [t['trade_return'] for t in trade_log if t['win']]
            loss_rets   = [t['trade_return'] for t in trade_log if not t['win']]
            mean_win    = np.mean(win_rets)  * 100 if win_rets  else 0.0
            mean_loss   = np.mean(loss_rets) * 100 if loss_rets else 0.0

            # Exit reason breakdown
            reasons = {}
            for t in trade_log:
                r = t.get('exit_reason', 'window')
                reasons[r] = reasons.get(r, 0) + 1
        else:
            mean_trade = mean_win = mean_loss = 0.0
            reasons = {}

        target_met     = total_return >= (self._target_return * 100)
        target_met_str = 'Yes' if target_met else 'No'
        pos_size_pct   = self._position_size * 100

        print(f"Starting Capital:\t\t${self._starting_capital:,.2f}")
        print(f"Final Capital:\t\t\t${final_capital:,.2f}")
        print(f"Total Return:\t\t\t{total_return:.2f}%")
        print(f"Total Trades:\t\t\t{total_trades}  (standard={standard_trades}, high_confidence={hc_trades})")
        print(f"Win Rate:\t\t\t{win_rate:.2f}%")
        print(f"Mean Trade Return:\t\t{mean_trade:.2f}%")
        print(f"Mean Win Return:\t\t{mean_win:.2f}%")
        print(f"Mean Loss Return:\t\t{mean_loss:.2f}%")
        print(f"Position Size:\t\t\t{pos_size_pct:.1f}%")
        print(f"Commission Paid:\t\t${self._commission * total_trades * 2:,.2f}  (${self._commission:.2f} x {total_trades} x 2 legs)")
        print(f"7% Target Met:\t\t\t{target_met_str}")
        if reasons:
            print(f"Exit reasons:\t\t\t{reasons}")