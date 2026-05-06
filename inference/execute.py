import asyncio
import os
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import OptionDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.enums import OptionsFeed
from execute_shared import (
    build_occ_symbol, compute_qty, is_before_cutoff, save_positions_log,
    MAX_POSITIONS, TARGET_RETURN, STOP_LOSS, RESCAN_INTERVAL_SEC,
)

TRADE_MODE = os.environ.get("TRADE_MODE", "dynamic")
IS_DYNAMIC = TRADE_MODE == "dynamic"
LOG_PREFIX = f"[{TRADE_MODE}]"

POSITIONS_LOG_PATH    = "positions_log.csv" if IS_DYNAMIC else "positions_log_fixed.csv"
TRAILING_STOP_PCT     = 0.30
SPY_EMERGENCY_DROP    = -0.003
EMERGENCY_MIN_GAIN    = 0.10
SPY_POLL_INTERVAL_SEC = 60

_trading_client  = None
_data_client     = None
_positions_data  = {}
_closing_symbols = set()
_stream          = None


def _get_clients():
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    trading    = TradingClient(api_key, secret_key, paper=True)
    data       = StockHistoricalDataClient(api_key, secret_key) if IS_DYNAMIC else None
    return trading, data


def _load_positions_log():
    log_file = Path(POSITIONS_LOG_PATH)

    if IS_DYNAMIC:
        default_columns = ['symbol', 'entry_date', 'qty', 'entry_ask', 'target_hit', 'peak_bid']
    else:
        default_columns = ['symbol', 'entry_date', 'qty', 'entry_ask']

    if not log_file.exists() or log_file.stat().st_size == 0:
        return pd.DataFrame(columns=default_columns)

    df = pd.read_csv(POSITIONS_LOG_PATH, parse_dates=['entry_date'])

    if IS_DYNAMIC:
        if 'target_hit' not in df.columns:
            df['target_hit'] = 0
        if 'peak_bid' not in df.columns:
            df['peak_bid'] = np.nan

    return df


MAX_HOLDING_DAYS = 21


def _parse_expiration(occ_symbol):
    return pd.to_datetime(occ_symbol[3:9], format='%y%m%d')


def _reconcile_positions(client):
    alpaca_positions = client.get_all_positions()

    # Filter to only SPY options opened by this system (OCC format = 18 chars: SPY + YYMMDD + C/P + 8 digits)
    alpaca_spy_options = {
        pos.symbol: pos
        for pos in alpaca_positions
        if pos.symbol.startswith('SPY') and len(pos.symbol) == 18
    }

    positions_log  = _load_positions_log()
    alpaca_symbols = set(alpaca_spy_options.keys())
    log_symbols    = set(positions_log['symbol']) if not positions_log.empty else set()

    stale_symbols    = log_symbols - alpaca_symbols
    orphaned_symbols = alpaca_symbols - log_symbols

    if not stale_symbols and not orphaned_symbols:
        return

    if stale_symbols:
        print(f"{LOG_PREFIX} Reconcile: removing {len(stale_symbols)} log entries no longer open in Alpaca.")
        positions_log = positions_log[~positions_log['symbol'].isin(stale_symbols)]

    if orphaned_symbols:
        print(f"{LOG_PREFIX} Reconcile: recovering {len(orphaned_symbols)} Alpaca position(s) missing from log.")
        new_rows = []
        for symbol in orphaned_symbols:
            pos        = alpaca_spy_options[symbol]
            expiration = _parse_expiration(symbol)
            # Anchor entry_date to expiration minus max holding window so the
            # days_held check in _close_expired_positions fires at expiration,
            # not 21 days from today.
            entry_date = expiration - pd.Timedelta(days=MAX_HOLDING_DAYS)
            new_row = {
                'symbol':     symbol,
                'entry_date': entry_date,
                'qty':        int(float(pos.qty)),
                'entry_ask':  float(pos.avg_entry_price),
            }
            if IS_DYNAMIC:
                new_row['target_hit'] = 0
                new_row['peak_bid']   = np.nan
            new_rows.append(new_row)
            print(f"{LOG_PREFIX} Reconcile: recovered {symbol} (qty={pos.qty}, avg_entry=${float(pos.avg_entry_price):.2f})")

        new_entries   = pd.DataFrame(new_rows)
        positions_log = pd.concat([positions_log, new_entries], ignore_index=True)

    save_positions_log(positions_log, POSITIONS_LOG_PATH)


def _close_expired_positions(client):
    positions_log = _load_positions_log()
    if positions_log.empty:
        return

    today = pd.Timestamp(date.today())

    positions_log['days_held'] = (today - positions_log['entry_date']).dt.days
    positions_log['dte']       = positions_log['symbol'].apply(
        lambda s: (_parse_expiration(s) - today).days
    )

    # Close if held for the full training window OR if expiration is tomorrow or sooner
    held_too_long  = positions_log['days_held'] >= MAX_HOLDING_DAYS
    near_expiry    = positions_log['dte'] <= 1
    expired_mask   = held_too_long | near_expiry
    expired_rows   = positions_log[expired_mask]

    if expired_rows.empty:
        return

    for row in expired_rows.itertuples(index=False):
        reason = f"DTE={row.dte}" if row.dte <= 1 else f"held {row.days_held} days"
        try:
            client.close_position(row.symbol)
            print(f"{LOG_PREFIX} Max-hold close: {row.symbol} ({reason})")
        except Exception as e:
            print(f"{LOG_PREFIX} Max-hold close failed for {row.symbol}: {e}")

    remaining = positions_log[~expired_mask].drop(columns=['days_held', 'dte'])
    save_positions_log(remaining, POSITIONS_LOG_PATH)


def _build_positions_data(final_log):
    if IS_DYNAMIC:
        result = {}
        for row in final_log.itertuples(index=False):
            peak_bid_value = float(row.peak_bid) if not pd.isna(row.peak_bid) else None
            result[row.symbol] = {
                'entry_ask':  float(row.entry_ask),
                'qty':        int(row.qty),
                'target_hit': bool(int(row.target_hit)),
                'peak_bid':   peak_bid_value,
            }
        return result
    else:
        return {
            row.symbol: {'entry_ask': float(row.entry_ask), 'qty': int(row.qty)}
            for row in final_log.itertuples(index=False)
        }


def _build_new_row(occ_symbol, qty, ask_price):
    row = {
        'symbol':     occ_symbol,
        'entry_date': pd.Timestamp(date.today()),
        'qty':        qty,
        'entry_ask':  ask_price,
    }
    if IS_DYNAMIC:
        row['target_hit'] = 0
        row['peak_bid']   = np.nan
    return row


def _build_position_entry(ask_price, qty):
    if IS_DYNAMIC:
        return {'entry_ask': ask_price, 'qty': qty, 'target_hit': False, 'peak_bid': None}
    else:
        return {'entry_ask': ask_price, 'qty': qty}


def _place_orders(client, signals):
    global _positions_data

    if not signals.empty:
        account         = client.get_account()
        account_balance = float(account.portfolio_value)
        positions_log   = _load_positions_log()
        available_slots = MAX_POSITIONS - len(positions_log)

        if available_slots <= 0:
            print(f"{LOG_PREFIX} At max positions ({MAX_POSITIONS}). No new orders.")
        else:
            new_rows       = []
            signals_ranked = signals.sort_values('gbt_score', ascending=False)

            for row in signals_ranked.head(available_slots).itertuples(index=False):
                occ_symbol   = build_occ_symbol(row.expiration, row.strike, row.right)
                already_open = not positions_log.empty and (positions_log['symbol'] == occ_symbol).any()
                if already_open:
                    continue

                qty           = compute_qty(row.ask, account_balance)
                order_request = MarketOrderRequest(
                    symbol=occ_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )

                try:
                    client.submit_order(order_request)
                    print(f"{LOG_PREFIX} Bought {qty}x {occ_symbol} @ ~${row.ask:.2f} (GBT: {row.gbt_score:.4f})")
                    new_rows.append(_build_new_row(occ_symbol, qty, row.ask))
                except Exception as e:
                    print(f"{LOG_PREFIX} Order failed for {occ_symbol}: {e}")

            if new_rows:
                new_entries   = pd.DataFrame(new_rows)
                positions_log = pd.concat([positions_log, new_entries], ignore_index=True)
                save_positions_log(positions_log, POSITIONS_LOG_PATH)

    _positions_data = _build_positions_data(_load_positions_log())


def place_additional_orders(signals):
    global _positions_data

    if signals.empty:
        return []

    account         = _trading_client.get_account()
    account_balance = float(account.portfolio_value)
    positions_log   = _load_positions_log()
    available_slots = MAX_POSITIONS - len(positions_log)

    if available_slots <= 0:
        print(f"{LOG_PREFIX} At max positions ({MAX_POSITIONS}). No new orders on rescan.")
        return []

    new_symbols    = []
    new_rows       = []
    signals_ranked = signals.sort_values('gbt_score', ascending=False)

    for row in signals_ranked.head(available_slots).itertuples(index=False):
        occ_symbol   = build_occ_symbol(row.expiration, row.strike, row.right)
        already_open = occ_symbol in _positions_data or (
            not positions_log.empty and (positions_log['symbol'] == occ_symbol).any()
        )
        if already_open:
            continue

        qty           = compute_qty(row.ask, account_balance)
        order_request = MarketOrderRequest(
            symbol=occ_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

        try:
            _trading_client.submit_order(order_request)
            print(f"{LOG_PREFIX} Rescan bought {qty}x {occ_symbol} @ ~${row.ask:.2f} (GBT: {row.gbt_score:.4f})")
            new_rows.append(_build_new_row(occ_symbol, qty, row.ask))
            _positions_data[occ_symbol] = _build_position_entry(row.ask, qty)
            new_symbols.append(occ_symbol)
        except Exception as e:
            print(f"{LOG_PREFIX} Rescan order failed for {occ_symbol}: {e}")

    if new_rows:
        new_entries   = pd.DataFrame(new_rows)
        positions_log = pd.concat([positions_log, new_entries], ignore_index=True)
        save_positions_log(positions_log, POSITIONS_LOG_PATH)

    return new_symbols


async def _close_and_log(symbol, reason, exit_bid):
    if symbol in _closing_symbols:
        return
    _closing_symbols.add(symbol)
    try:
        await asyncio.to_thread(_trading_client.close_position, symbol)
        print(f"{LOG_PREFIX} Closed {symbol} - {reason} (bid: ${exit_bid:.2f})")
        _positions_data.pop(symbol, None)
        positions_log = await asyncio.to_thread(_load_positions_log)
        positions_log = positions_log[positions_log['symbol'] != symbol]
        await asyncio.to_thread(save_positions_log, positions_log, POSITIONS_LOG_PATH)
    except Exception as e:
        print(f"{LOG_PREFIX} Failed to close {symbol}: {e}")
        _closing_symbols.discard(symbol)


async def _mark_target_hit(symbol, bid_price):
    _positions_data[symbol]['target_hit'] = True
    _positions_data[symbol]['peak_bid']   = bid_price
    positions_log = await asyncio.to_thread(_load_positions_log)
    mask = positions_log['symbol'] == symbol
    positions_log.loc[mask, 'target_hit'] = 1
    positions_log.loc[mask, 'peak_bid']   = bid_price
    await asyncio.to_thread(save_positions_log, positions_log, POSITIONS_LOG_PATH)
    print(f"{LOG_PREFIX} Target hit: {symbol} @ ${bid_price:.2f} - switching to trailing stop")


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

    if IS_DYNAMIC:
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
    else:
        target_price = entry_ask * (1 + TARGET_RETURN)
        stop_price   = entry_ask * (1 - STOP_LOSS)

        if bid_price >= target_price:
            await _close_and_log(symbol, 'target', bid_price)
        elif bid_price <= stop_price:
            await _close_and_log(symbol, 'stop', bid_price)


async def _spy_emergency_checker():
    prev_spy_price = None

    while is_before_cutoff():
        await asyncio.sleep(SPY_POLL_INTERVAL_SEC)

        try:
            request           = StockLatestBarRequest(symbol_or_symbols='SPY')
            bar_response      = await asyncio.to_thread(_data_client.get_stock_latest_bar, request)
            current_spy_price = float(bar_response['SPY'].close)
        except Exception as e:
            print(f"{LOG_PREFIX} SPY price fetch failed: {e}")
            continue

        if prev_spy_price is None:
            prev_spy_price = current_spy_price
            continue

        spy_interval_return = (current_spy_price - prev_spy_price) / prev_spy_price
        prev_spy_price      = current_spy_price

        if spy_interval_return > SPY_EMERGENCY_DROP:
            continue

        print(f"{LOG_PREFIX} SPY emergency drop ({spy_interval_return:.3%}) - checking positions...")

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


async def _stream_task():
    global _stream
    try:
        await _stream._run_forever()
    except Exception as e:
        print(f"{LOG_PREFIX} Stream error: {e}")
    finally:
        _stream = None
        print(f"{LOG_PREFIX} Stream task exited - will reconnect on next rescan.")


async def _start_stream(symbols):
    global _stream
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    _stream    = OptionDataStream(api_key, secret_key, feed=OptionsFeed.INDICATIVE)
    _stream.subscribe_quotes(_quote_handler, *symbols)
    asyncio.create_task(_stream_task())
    print(f"{LOG_PREFIX} WebSocket started for {len(symbols)} symbol(s).")


async def _stream_watchdog():
    while is_before_cutoff():
        await asyncio.sleep(30)
        stream_is_dead = _stream is None and bool(_positions_data)
        if stream_is_dead:
            active_symbols = list(_positions_data.keys())
            print(f"{LOG_PREFIX} Stream dead - reconnecting for {len(active_symbols)} position(s)...")
            await _start_stream(active_symbols)


async def _rescan_loop(rescan_fn):
    global _stream

    await asyncio.sleep(RESCAN_INTERVAL_SEC)

    while is_before_cutoff():
        try:
            new_symbols = await asyncio.to_thread(rescan_fn)
            if new_symbols:
                if _stream is None:
                    await _start_stream(new_symbols)
                else:
                    _stream.subscribe_quotes(_quote_handler, *new_symbols)
                    print(f"{LOG_PREFIX} Subscribed {len(new_symbols)} new symbol(s) to stream.")
        except Exception as e:
            print(f"{LOG_PREFIX} Rescan failed: {e}")

        if is_before_cutoff():
            await asyncio.sleep(RESCAN_INTERVAL_SEC)


async def _run_all(rescan_fn):
    global _stream

    symbols = list(_positions_data.keys())

    if symbols:
        await _start_stream(symbols)
    else:
        print(f"{LOG_PREFIX} No positions at start - rescan-only mode.")

    if IS_DYNAMIC:
        asyncio.create_task(_spy_emergency_checker())

    asyncio.create_task(_stream_watchdog())
    asyncio.create_task(_rescan_loop(rescan_fn))

    while is_before_cutoff():
        await asyncio.sleep(30)

    print(f"{LOG_PREFIX} Monitoring cutoff reached (15:55 ET).")
    if _stream is not None:
        try:
            await _stream.stop_ws()
        except Exception as e:
            print(f"{LOG_PREFIX} Stream stop error: {e}")


def _is_trading_day(client):
    clock = client.get_clock()
    market_is_open_now = clock.is_open
    market_opens_today = clock.next_open.date() == date.today()
    return market_is_open_now or market_opens_today


def run(signals, rescan_fn):
    global _trading_client, _data_client

    _trading_client, _data_client = _get_clients()

    if not _is_trading_day(_trading_client):
        print(f"{LOG_PREFIX} Market closed today (holiday or weekend). Exiting.")
        return

    _reconcile_positions(_trading_client)
    _close_expired_positions(_trading_client)
    _place_orders(_trading_client, signals)

    if len(_positions_data) == 0:
        print(f"{LOG_PREFIX} No open positions to monitor.")

    asyncio.run(_run_all(rescan_fn))
    print(f"{LOG_PREFIX} Monitoring complete.")
