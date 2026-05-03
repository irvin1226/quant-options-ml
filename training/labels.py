import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

def _process_batch(batch):
    results = []

    for contract_key, dates, timestamps, asks, bids, eval_rows, current_dates, target_return, stop_loss, max_holding_days in batch:
        exp, strike, right = contract_key
        n = len(dates)

        dates_days = dates.astype('datetime64[D]')
        max_delta = np.timedelta64(max_holding_days, 'D')
        window_ends = np.searchsorted(dates_days, dates_days + max_delta, side='right')

        for i in range(n):
            if not eval_rows[i]:
                continue

            if dates[i] not in current_dates:
                continue

            entry_ask = asks[i]
            target_price = entry_ask * (1 + target_return)
            stop_price = entry_ask * (1 - stop_loss)

            end = window_ends[i]
            window_bids = bids[i + 1:end]

            if len(window_bids) == 0:
                results.append((exp, strike, right, timestamps[i], 0.0, 0.0))
                continue

            target_mask = window_bids >= target_price
            stop_mask = window_bids <= stop_price

            has_target = np.any(target_mask)
            has_stop = np.any(stop_mask)

            if has_target:
                first_target = np.argmax(target_mask)
            else:
                first_target = len(window_bids)

            if has_stop:
                first_stop = np.argmax(stop_mask)
            else:
                first_stop = len(window_bids)

            if has_target and first_target <= first_stop:
                # Binary label: 1 (winner) - target hit before stop.
                # After target hit, check if price subsequently drops below stop.
                # A real trader holding with a stop would be stopped out even after
                # the target fires.
                post_target_bids = window_bids[first_target + 1:]
                post_target_stop = len(post_target_bids) > 0 and np.any(post_target_bids <= stop_price)
                if post_target_stop:
                    realized = -stop_loss
                else:
                    exit_bid = window_bids[-1]
                    realized = (exit_bid - entry_ask) / entry_ask
                    realized = float(max(realized, -stop_loss))
                results.append((exp, strike, right, timestamps[i], 1.0, float(realized)))
            else:
                # Binary label: 0 (loser).
                # Realized return: fixed stop loss if stop hit, else window-end bid.
                if has_stop:
                    realized = -stop_loss
                else:
                    exit_bid = window_bids[-1]
                    realized = (exit_bid - entry_ask) / entry_ask
                    realized = float(max(realized, -stop_loss))
                results.append((exp, strike, right, timestamps[i], 0.0, float(realized)))

    return results

class LabelGenerator:

    def __init__(self, target_return: float = 0.07, stop_loss: float = 0.05, max_holding_days: int = 21):
        self._target_return    = target_return
        self._stop_loss        = stop_loss
        self._max_holding_days = max_holding_days

    # Generates labels for current month only, using next month for forward lookups.
    # Contract groups are processed in parallel across CPU cores since each group
    # is fully independent of every other group.
    def generate(self, current_month: pd.DataFrame, next_month: pd.DataFrame):
        combined = pd.concat([current_month, next_month])
        combined = combined.sort_values(by=['expiration', 'strike', 'right', 'timestamp'])

        current_month  = current_month.copy()
        current_dates  = set(current_month['date'].unique())

        groups = combined.groupby(['expiration', 'strike', 'right'])

        group_args = []

        for contract_key, group in groups:
            group = group.reset_index(drop=True)

            group_args.append((
                contract_key,
                group['date'].values.astype('datetime64[us]'),
                group['timestamp'].values,
                group['ask'].values,
                group['bid'].values,
                group['eval_row'].values,
                current_dates,
                self._target_return,
                self._stop_loss,
                self._max_holding_days,
            ))

        num_workers = max(1, multiprocessing.cpu_count() - 1)

        batch_size = max(1, len(group_args) // (num_workers * 4))
        batches = []
        for i in range(0, len(group_args), batch_size):
            batches.append(group_args[i:i + batch_size])

        with Pool(processes=num_workers) as pool:
            all_results = pool.map(_process_batch, batches)

        all_results_flat = []
        for batch_results in all_results:
            all_results_flat.extend(batch_results)

        if all_results_flat:
            labels_df = pd.DataFrame(
                all_results_flat,
                columns=['expiration', 'strike', 'right', 'timestamp', 'label', 'realized_return']
            )

            current_month = current_month.merge(
                labels_df,
                on=['expiration', 'strike', 'right', 'timestamp'],
                how='left',
            )
            current_month['label'] = current_month['label'].fillna(0.0)
            current_month['realized_return'] = current_month['realized_return'].fillna(0.0)
        else:
            current_month['label'] = 0.0
            current_month['realized_return'] = 0.0

        return current_month