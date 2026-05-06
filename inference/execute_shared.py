import math
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

POSITION_SIZE         = 0.02
MAX_POSITIONS         = 20
TARGET_RETURN         = 0.07
STOP_LOSS             = 0.05
RESCAN_INTERVAL_SEC   = 300
ET                    = ZoneInfo('America/New_York')
MONITOR_CUTOFF_HOUR   = 15
MONITOR_CUTOFF_MINUTE = 55


def build_occ_symbol(expiration, strike, right):
    date_str      = pd.Timestamp(expiration).strftime('%y%m%d')
    right_char    = 'C' if float(right) == 1.0 else 'P'
    strike_int    = int(round(float(strike) * 1000))
    strike_padded = str(strike_int).zfill(8)
    return f"SPY{date_str}{right_char}{strike_padded}"


def compute_qty(ask_price, account_balance):
    dollar_allocation = POSITION_SIZE * account_balance
    contract_cost     = ask_price * 100
    qty               = math.floor(dollar_allocation / contract_cost)
    return max(qty, 1)


def is_before_cutoff():
    now    = datetime.now(ET)
    cutoff = now.replace(hour=MONITOR_CUTOFF_HOUR, minute=MONITOR_CUTOFF_MINUTE, second=0, microsecond=0)
    return now < cutoff


def save_positions_log(positions_log, path):
    positions_log.to_csv(path, index=False)
