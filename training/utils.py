import os
from datetime import datetime

class Logger:

    def __init__(self, log_path: str):
        self._log_path = log_path
        if not os.path.exists(log_path):
            open(log_path, 'w').close()

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        print(entry)
        with open(self._log_path, 'a') as f:
            f.write(entry + "\n")

class ConfigManager:

    DEFAULT_CONFIG = {
        'target_return': 0.07,
        'stop_loss': 0.05,
        'max_holding_days': 21,
        'min_volume': 1,
        'max_spread_pct': 0.15,
        'min_dte': 21,
        'max_dte': 45,
        'min_open_interest': 100,

        # configs below do not need to rerun preprocess.py
        'epochs': 1000,
        'nn_learning_rate': 0.0001,
        'target_precision': 0.55,
        'high_confidence_target_precision': 0.65,
        'pos_weight': 2,
        'gbt_n_estimators': 1500,
        'gbt_learning_rate': 0.03,
        'gbt_max_depth': 6,
        'scale_pos_weight': 2,
        'random_seed': 42,

        # early stopping
        'early_stopping_patience': 75,
        'checkpoint_path': 'logs/best_nn.pt',

        # position management
        'max_positions': 20,
        'max_high_confidence_positions': 5,

        # broker fees
        'commission_per_contract': 0.65,
    }

    def __init__(self, config: dict = None):
        if config is None:
            self._config = self.DEFAULT_CONFIG.copy()
        else:
            self._config = config

    def get(self, key: str):
        if key not in self._config:
            raise KeyError(f"Config key '{key}' not found.")
        return self._config[key]

    def summary(self):
        print("Configuration:")
        for key, value in self._config.items():
            print(f"{key}: {value}")