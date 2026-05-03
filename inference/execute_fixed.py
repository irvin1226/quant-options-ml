import asyncio
import os
import math
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import OptionDataStream
from alpaca.data.enums import OptionsFeed

POSITIONS_LOG_PATH    = "positions_log_fixed.csv"
POSITION_SIZE         = 0.02
MAX_POSITIONS         = 20
TARGET_RETURN         = 0.07
STOP_LOSS             = 0.05
RESCAN_INTERVAL_SEC   = 300
ET                    = ZoneInfo('America/New_York')
MONITOR_CUTOFF_HOUR   = 15
MONITOR_CUTOFF_MINUTE = 55

_trading_client  = None
_positions_data  = {}
_closing_symbols = set()
_stream          = None


def _get_client():
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(api_key, secret_key, paper=True)


def _build_occ_symbol(expiration, strike, right):
    date_str      = pd.Timestamp(expiration).strftime('%y%m%d')
    right_char    = 'C' if float(right) == 1.0 else 'P'
    strike_int    = int(round(float(strike) * 1000))
    strike_padded = str(strike_int).zfill(8)
    return f"SPY{date_str}{right_char}{strike_padded}"


def _load_positions_log():
    log_file = Path(POSITIONS_LOG_PATH)
    if not log_file.exists() or log_file.stat().st_size == 0:
        return pd.DataFrame(columns=['symbol', 'entry_date', 'qty', 'entry_ask'])
    return pd.read_csv(POSITIONS_LOG_PATH, parse_dates=['entry_date'])


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
            print(f"[fixed] Day-21 close: {row.symbol} (held {row.days_held} days)")
        except Exception as e:
            print(f"[fixed] Day-21 close failed for {row.symbol}: {e}")

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
            print(f"[fixed] At max positions ({MAX_POSITIONS}). No new orders.")
        else:
            new_rows       = []
            signals_ranked = signals.sort_values('gbt_score', ascending=False)

            for row in signals_ranked.head(available_slots).itertuples(index=False):
                occ_symbol   = _build_occ_symbol(row.expiration, row.strike, row.right)
                already_open = not positions_log.empty and (positions_log['symbol'] == occ_symbol).any()
                if already_open:
                    continue

                qty           = _compute_qty(row.ask, account_balance)
                order_request = MarketOrderRequest(
                    symbol=occ_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )

                try:
                    client.submit_order(order_request)
                    print(f"[fixed] Bought {qty}x {occ_symbol} @ ~${row.ask:.2f} (GBT: {row.gbt_score:.4f})")
                    new_rows.append({
                        'symbol':     occ_symbol,
                        'entry_date': pd.Timestamp(date.today()),
                        'qty':        qty,
                        'entry_ask':  row.ask,
                    })
                except Exception as e:
                    print(f"[fixed] Order failed for {occ_symbol}: {e}")

            if new_rows:
                new_entries   = pd.DataFrame(new_rows)
                positions_log = pd.concat([positions_log, new_entries], ignore_index=True)
                _save_positions_log(positions_log)

    final_log       = _load_positions_log()
    _positions_data = {
        row.symbol: {'entry_ask': float(row.entry_ask), 'qty': int(row.qty)}
        for row in final_log.itertuples(index=False)
    }


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
        print(f"[fixed] At max positions ({MAX_POSITIONS}). No new orders on rescan.")
        return []

    new_symbols    = []
    new_rows       = []
    signals_ranked = signals.sort_values('gbt_score', ascending=False)

    for row in signals_ranked.head(available_slots).itertuples(index=False):
        occ_symbol   = _build_occ_symbol(row.expiration, row.strike, row.right)
        already_open = occ_symbol in _positions_data or (
            not positions_log.empty and (positions_log['symbol'] == occ_symbol).any()
        )
        if already_open:
            continue

        qty           = _compute_qty(row.ask, account_balance)
        order_request = MarketOrderRequest(
            symbol=occ_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

        try:
            _trading_client.submit_order(order_request)
            print(f"[fixed] Rescan bought {qty}x {occ_symbol} @ ~${row.ask:.2f} (GBT: {row.gbt_score:.4f})")
            new_rows.append({
                'symbol':     occ_symbol,
                'entry_date': pd.Timestamp(date.today()),
                'qty':        qty,
                'entry_ask':  row.ask,
            })
            _positions_data[occ_symbol] = {'entry_ask': row.ask, 'qty': qty}
            new_symbols.append(occ_symbol)
        except Exception as e:
            print(f"[fixed] Rescan order failed for {occ_symbol}: {e}")

    if new_rows:
        new_entries   = pd.DataFrame(new_rows)
        positions_log = pd.concat([positions_log, new_entries], ignore_index=True)
        _save_positions_log(positions_log)

    return new_symbols


async def _close_and_log(symbol, reason, bid_price):
    if symbol in _closing_symbols:
        return
    _closing_symbols.add(symbol)
    try:
        await asyncio.to_thread(_trading_client.close_position, symbol)
        print(f"[fixed] Closed {symbol} - {reason} (bid: ${bid_price:.2f})")
        _positions_data.pop(symbol, None)
        positions_log = await asyncio.to_thread(_load_positions_log)
        positions_log = positions_log[positions_log['symbol'] != symbol]
        await asyncio.to_thread(_save_positions_log, positions_log)
    except Exception as e:
        print(f"[fixed] Failed to close {symbol}: {e}")
        _closing_symbols.discard(symbol)


async def _quote_handler(quote):
    symbol    = quote.symbol
    bid_price = float(quote.bid_price)

    if bid_price <= 0:
        return
    if symbol not in _positions_data:
        return
    if symbol in _closing_symbols:
        return

    entry_ask    = _positions_data[symbol]['entry_ask']
    target_price = entry_ask * (1 + TARGET_RETURN)
    stop_price   = entry_ask * (1 - STOP_LOSS)

    if bid_price >= target_price:
        await _close_and_log(symbol, 'target', bid_price)
    elif bid_price <= stop_price:
        await _close_and_log(symbol, 'stop', bid_price)


async def _rescan_loop(rescan_fn):
    while _is_before_cutoff():
        await asyncio.sleep(RESCAN_INTERVAL_SEC)

        if not _is_before_cutoff():
            break

        try:
            new_symbols = await asyncio.to_thread(rescan_fn)
            if new_symbols and _stream is not None:
                _stream.subscribe_quotes(_quote_handler, *new_symbols)
                print(f"[fixed] Subscribed {len(new_symbols)} new symbol(s) to stream.")
        except Exception as e:
            print(f"[fixed] Rescan failed: {e}")


async def _cutoff_watcher():
    while _is_before_cutoff():
        await asyncio.sleep(30)
    print("[fixed] Monitoring cutoff reached (15:55 ET) - stopping stream.")
    await _stream.stop_ws()


async def _run_stream(rescan_fn):
    global _stream

    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    _stream  = OptionDataStream(api_key, secret_key, feed=OptionsFeed.INDICATIVE)
    symbols  = list(_positions_data.keys())

    if symbols:
        _stream.subscribe_quotes(_quote_handler, *symbols)
    print(f"[fixed] WebSocket subscribed to {len(symbols)} symbol(s).")

    asyncio.create_task(_cutoff_watcher())
    asyncio.create_task(_rescan_loop(rescan_fn))
    await _stream._run_forever()


def run(signals, rescan_fn):
    global _trading_client

    _trading_client = _get_client()
    _close_expired_positions(_trading_client)
    _place_orders(_trading_client, signals)

    asyncio.run(_run_stream(rescan_fn))
    print("[fixed] Monitoring complete.")