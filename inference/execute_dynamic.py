import asyncio
import os
import math
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import OptionDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.enums import OptionsFeed

POSITIONS_LOG_PATH    = "positions_log.csv"
POSITION_SIZE         = 0.02
MAX_POSITIONS         = 20
TARGET_RETURN         = 0.07
STOP_LOSS             = 0.05
TRAILING_STOP_PCT     = 0.30
SPY_EMERGENCY_DROP    = -0.003
EMERGENCY_MIN_GAIN    = 0.10
SPY_POLL_INTERVAL_SEC = 60
RESCAN_INTERVAL_SEC   = 300  # 5 minutes
ET                    = ZoneInfo('America/New_York')
MONITOR_CUTOFF_HOUR   = 15
MONITOR_CUTOFF_MINUTE = 55

_trading_client  = None
_data_client     = None
_positions_data  = {}
_closing_symbols = set()
_stream          = None  # module-level so rescan loop can subscribe new symbols


def _get_clients():
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    trading    = TradingClient(api_key, secret_key, paper=True)
    data       = StockHistoricalDataClient(api_key, secret_key)
    return trading, data


def _build_occ_symbol(expiration, strike, right):
    date_str = pd.Timestamp(expiration).strftime('%y%m%d')
    right_char = 'C' if float(right) == 1.0 else 'P'
    strike_int    = int(round(float(strike) * 1000))
    strike_padded = str(strike_int).zfill(8)
    return f"SPY{date_str}{right_char}{strike_padded}"


def _load_positions_log():
    log_file = Path(POSITIONS_LOG_PATH)
    if not log_file.exists() or log_file.stat().st_size == 0:
        return pd.DataFrame(columns=['symbol', 'entry_date', 'qty', 'entry_ask', 'target_hit', 'peak_bid'])

    df = pd.read_csv(POSITIONS_LOG_PATH, parse_dates=['entry_date'])
    if 'target_hit' not in df.columns:
        df['target_hit'] = 0
    if 'peak_bid' not in df.columns:
        df['peak_bid'] = np.nan
    return df


def _save_positions_log(positions_log):
    positions_log.to_csv(POSITIONS_LOG_PATH, index=False)


def _compute_qty(ask_price, account_balance):
    dollar_allocation = POSITION_SIZE * account_balance
    contract_cost     = ask_price * 100
    qty               = math.floor(dollar_allocation / contract_cost)
    return max(qty, 1)


def _is_before_cutoff():
    now    = datetime.now(ET)
    cutoff = now.replace(hour=MONITOR_CUTOFF_HOUR, minute=MONITOR_CUTOFF_MINUTE, second=0, microsecond=0)
    return now < cutoff


def _close_expired_positions(client):
    positions_log = _load_positions_log()
    if positions_log.empty:
        return

    today = pd.Timestamp(date.today())
    positions_log['days_held'] = (today - positions_log['entry_date']).dt.days
    expired_mask = positions_log['days_held'] >= 21
    expired_rows = positions_log[expired_mask]

    if expired_rows.empty:
        return

    for row in expired_rows.itertuples(index=False):
        try:
            client.close_position(row.symbol)
            print(f"[dynamic] Day-21 close: {row.symbol} (held {row.days_held} days)")
        except Exception as e:
            print(f"[dynamic] Day-21 close failed for {row.symbol}: {e}")

    remaining = positions_log[~expired_mask].drop(columns=['days_held'])
    _save_positions_log(remaining)


def _place_orders(client, signals):
    global _positions_data

    if not signals.empty:
        account         = client.get_account()
        account_balance = float(account.portfolio_value)
        positions_log   = _load_positions_log()
        current_count   = len(positions_log)
        available_slots = MAX_POSITIONS - current_count

        if available_slots <= 0:
            print(f"[dynamic] At max positions ({MAX_POSITIONS}). No new orders.")
        else:
            new_rows = []
            signals_ranked = signals.sort_values('gbt_score', ascending=False)

            for row in signals_ranked.head(available_slots).itertuples(index=False):
                occ_symbol   = _build_occ_symbol(row.expiration, row.strike, row.right)
                already_open = not positions_log.empty and (positions_log['symbol'] == occ_symbol).any()
                if already_open:
                    continue

                qty = _compute_qty(row.ask, account_balance)
                order_request = MarketOrderRequest(
                    symbol=occ_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )

                try:
                    client.submit_order(order_request)
                    print(f"[dynamic] Bought {qty}x {occ_symbol} @ ~${row.ask:.2f} (GBT: {row.gbt_score:.4f})")
                    new_rows.append({
                        'symbol':     occ_symbol,
                        'entry_date': pd.Timestamp(date.today()),
                        'qty':        qty,
                        'entry_ask':  row.ask,
                        'target_hit': 0,
                        'peak_bid':   np.nan,
                    })
                except Exception as e:
                    print(f"[dynamic] Order failed for {occ_symbol}: {e}")

            if new_rows:
                new_entries   = pd.DataFrame(new_rows)
                positions_log = pd.concat([positions_log, new_entries], ignore_index=True)
                _save_positions_log(positions_log)

    final_log = _load_positions_log()
    _positions_data = {}
    for row in final_log.itertuples(index=False):
        peak_bid_value = float(row.peak_bid) if not pd.isna(row.peak_bid) else None
        _positions_data[row.symbol] = {
            'entry_ask':  float(row.entry_ask),
            'qty':        int(row.qty),
            'target_hit': bool(int(row.target_hit)),
            'peak_bid':   peak_bid_value,
        }


# Called from run_dynamic.py rescan() - places new orders mid-day and returns
# the list of newly added OCC symbols so the caller can subscribe them to the stream.
def place_additional_orders(signals):
    global _positions_data

    if signals.empty:
        return []

    account         = _trading_client.get_account()
    account_balance = float(account.portfolio_value)
    positions_log   = _load_positions_log()
    current_count   = len(positions_log)
    available_slots = MAX_POSITIONS - current_count

    if available_slots <= 0:
        print(f"[dynamic] At max positions ({MAX_POSITIONS}). No new orders on rescan.")
        return []

    new_symbols = []
    new_rows    = []
    signals_ranked = signals.sort_values('gbt_score', ascending=False)

    for row in signals_ranked.head(available_slots).itertuples(index=False):
        occ_symbol   = _build_occ_symbol(row.expiration, row.strike, row.right)
        already_open = occ_symbol in _positions_data or (
            not positions_log.empty and (positions_log['symbol'] == occ_symbol).any()
        )
        if already_open:
            continue

        qty = _compute_qty(row.ask, account_balance)
        order_request = MarketOrderRequest(
            symbol=occ_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

        try:
            _trading_client.submit_order(order_request)
            print(f"[dynamic] Rescan bought {qty}x {occ_symbol} @ ~${row.ask:.2f} (GBT: {row.gbt_score:.4f})")
            new_rows.append({
                'symbol':     occ_symbol,
                'entry_date': pd.Timestamp(date.today()),
                'qty':        qty,
                'entry_ask':  row.ask,
                'target_hit': 0,
                'peak_bid':   np.nan,
            })
            _positions_data[occ_symbol] = {
                'entry_ask':  row.ask,
                'qty':        qty,
                'target_hit': False,
                'peak_bid':   None,
            }
            new_symbols.append(occ_symbol)
        except Exception as e:
            print(f"[dynamic] Rescan order failed for {occ_symbol}: {e}")

    if new_rows:
        new_entries   = pd.DataFrame(new_rows)
        positions_log = pd.concat([positions_log, new_entries], ignore_index=True)
        _save_positions_log(positions_log)

    return new_symbols


async def _close_and_log(symbol, reason, exit_bid):
    if symbol in _closing_symbols:
        return
    _closing_symbols.add(symbol)
    try:
        await asyncio.to_thread(_trading_client.close_position, symbol)
        print(f"[dynamic] Closed {symbol} - {reason} (bid: ${exit_bid:.2f})")
        _positions_data.pop(symbol, None)
        positions_log = await asyncio.to_thread(_load_positions_log)
        positions_log = positions_log[positions_log['symbol'] != symbol]
        await asyncio.to_thread(_save_positions_log, positions_log)
    except Exception as e:
        print(f"[dynamic] Failed to close {symbol}: {e}")
        _closing_symbols.discard(symbol)


async def _mark_target_hit(symbol, bid_price):
    _positions_data[symbol]['target_hit'] = True
    _positions_data[symbol]['peak_bid']   = bid_price
    positions_log = await asyncio.to_thread(_load_positions_log)
    mask = positions_log['symbol'] == symbol
    positions_log.loc[mask, 'target_hit'] = 1
    positions_log.loc[mask, 'peak_bid']   = bid_price
    await asyncio.to_thread(_save_positions_log, positions_log)
    print(f"[dynamic] Target hit: {symbol} @ ${bid_price:.2f} - switching to trailing stop")


async def _quote_handler(quote):
    symbol    = quote.symbol
    bid_price = float(quote.bid_price)

    if bid_price <= 0:
        return
    if symbol not in _positions_data:
        return
    if symbol in _closing_symbols:
        return

    pos       = _positions_data[symbol]
    entry_ask = pos['entry_ask']

    if not pos['target_hit']:
        target_price = entry_ask * (1 + TARGET_RETURN)
        stop_price   = entry_ask * (1 - STOP_LOSS)

        if bid_price >= target_price:
            await _mark_target_hit(symbol, bid_price)
        elif bid_price <= stop_price:
            await _close_and_log(symbol, 'phase1_stop', bid_price)
    else:
        current_peak = pos['peak_bid']
        if current_peak is None or bid_price > current_peak:
            _positions_data[symbol]['peak_bid'] = bid_price
            current_peak = bid_price

        if bid_price <= current_peak * (1 - TRAILING_STOP_PCT):
            await _close_and_log(symbol, 'trailing_stop', bid_price)


async def _spy_emergency_checker():
    prev_spy_price = None

    while _is_before_cutoff():
        await asyncio.sleep(SPY_POLL_INTERVAL_SEC)

        try:
            request           = StockLatestBarRequest(symbol_or_symbols='SPY')
            bar_response      = await asyncio.to_thread(_data_client.get_stock_latest_bar, request)
            current_spy_price = float(bar_response['SPY'].close)
        except Exception as e:
            print(f"[dynamic] SPY price fetch failed: {e}")
            continue

        if prev_spy_price is None:
            prev_spy_price = current_spy_price
            continue

        spy_interval_return = (current_spy_price - prev_spy_price) / prev_spy_price
        prev_spy_price      = current_spy_price

        if spy_interval_return > SPY_EMERGENCY_DROP:
            continue

        print(f"[dynamic] SPY emergency drop ({spy_interval_return:.3%}) - checking positions...")

        for symbol in list(_positions_data.keys()):
            if symbol in _closing_symbols:
                continue
            pos      = _positions_data[symbol]
            peak_bid = pos['peak_bid']
            if not pos['target_hit'] or peak_bid is None:
                continue
            peak_gain = (peak_bid - pos['entry_ask']) / pos['entry_ask']
            if peak_gain >= EMERGENCY_MIN_GAIN:
                await _close_and_log(symbol, 'emergency_spy_drop', peak_bid)


# Runs every RESCAN_INTERVAL_SEC. Calls rescan_fn (sync, runs in thread) to
# fetch chain, run inference, place new orders. Subscribes any new symbols to stream.
async def _rescan_loop(rescan_fn):
    while _is_before_cutoff():
        await asyncio.sleep(RESCAN_INTERVAL_SEC)

        if not _is_before_cutoff():
            break

        try:
            new_symbols = await asyncio.to_thread(rescan_fn)
            if new_symbols and _stream is not None:
                _stream.subscribe_quotes(_quote_handler, *new_symbols)
                print(f"[dynamic] Subscribed {len(new_symbols)} new symbol(s) to stream.")
        except Exception as e:
            print(f"[dynamic] Rescan failed: {e}")


async def _cutoff_watcher():
    while _is_before_cutoff():
        await asyncio.sleep(30)
    print("[dynamic] Monitoring cutoff reached (15:55 ET) - stopping stream.")
    await _stream.stop_ws()


async def _run_stream(rescan_fn):
    global _stream

    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    _stream = OptionDataStream(api_key, secret_key, feed=OptionsFeed.INDICATIVE)
    symbols = list(_positions_data.keys())

    if symbols:
        _stream.subscribe_quotes(_quote_handler, *symbols)
    print(f"[dynamic] WebSocket subscribed to {len(symbols)} symbol(s).")

    asyncio.create_task(_cutoff_watcher())
    asyncio.create_task(_spy_emergency_checker())
    asyncio.create_task(_rescan_loop(rescan_fn))
    await _stream._run_forever()


def run(signals, rescan_fn):
    global _trading_client, _data_client

    _trading_client, _data_client = _get_clients()
    _close_expired_positions(_trading_client)
    _place_orders(_trading_client, signals)

    if len(_positions_data) == 0:
        print("[dynamic] No open positions to monitor.")
        # Still run rescan loop even if no positions at open - may get signals later
        # but we need the stream for that,so we start stream with empty subscription.

    asyncio.run(_run_stream(rescan_fn))
    print("[dynamic] Monitoring complete.")