import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path

from ...utils.data_loader import load_data
from ...utils.metrics import calculate_metrics, calculate_sharpe_ratio

def calculate_mas(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Calcula médias móveis baseado no config."""
    df = df.copy()

    # SMAs
    df['sma_fast'] = df['close'].rolling(config['strategy']['sma_fast_period']).mean()
    df['sma_mid'] = df['close'].rolling(config['strategy']['sma_mid_period']).mean()
    df['sma_slow'] = df['close'].rolling(config['strategy']['sma_slow_period']).mean()

    # EMAs
    df['ema_fast'] = df['close'].ewm(span=config['strategy']['ema_fast_period']).mean()
    df['ema_mid'] = df['close'].ewm(span=config['strategy']['ema_mid_period']).mean()
    df['ema_slow'] = df['close'].ewm(span=config['strategy']['ema_slow_period']).mean()

    # ATR para trailing
    df['atr'] = calculate_atr(df, config['strategy']['atr_period'])

    return df

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calcula ATR."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_signals(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Gera sinais baseado no signal_mode."""
    df = df.copy()
    strategy = config['strategy']

    # Filtro de referência
    if strategy['ref_filter_enabled']:
        ref_buffer = strategy['ref_buffer_pct']
        df['ref_bias'] = np.where(df['close'] > df['ref_ema'] * (1 + ref_buffer), 1,
                                  np.where(df['close'] < df['ref_ema'] * (1 - ref_buffer), -1, 0))

    # Sinais de cruzamento (simples: ema_fast sobre ema_slow)
    df['ema_cross'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    df['ema_cross_prev'] = df['ema_cross'].shift(1)
    df['signal'] = np.where((df['ema_cross'] == 1) & (df['ema_cross_prev'] == -1), 1,  # Long
                            np.where((df['ema_cross'] == -1) & (df['ema_cross_prev'] == 1), -1, 0))  # Short

    # Aplicar filtros
    if strategy['ref_filter_enabled']:
        df['signal'] = np.where(df['ref_bias'] == 1, df['signal'], 0)  # Só long em bull

    if strategy['allow_short']:
        pass  # Permite short
    else:
        df['signal'] = np.where(df['signal'] == -1, 0, df['signal'])  # Só long

    return df

def apply_trailing_stop(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Aplica trailing stop."""
    df = df.copy()
    strategy = config['strategy']

    if strategy['trailing_stop_type'] == 'percent_trailing':
        pct = strategy['percent_trailing_pct']
        df['trail_stop'] = df['close'] * (1 - pct)  # Para long; ajustar para short
    elif strategy['trailing_stop_type'] == 'atr_trailing':
        mult = strategy['atr_trail_mult']
        df['trail_stop'] = df['close'] - df['atr'] * mult
    # Outros tipos podem ser adicionados

    return df

def backtest_ema_only(df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    """Executa backtest."""
    df = calculate_mas(df, config)
    df = generate_signals(df, config)
    df = apply_trailing_stop(df, config)

    # Simulação de trades (simplificada)
    capital = config['backtest']['initial_capital']
    position = 0
    entry_price = 0
    trades = []
    equity = [capital]

    for i, row in df.iterrows():
        if row['signal'] == 1 and position == 0:  # Entrada long
            position = config['strategy']['lot_size']
            entry_price = row['close']
        elif row['signal'] == -1 and position > 0:  # Saída long
            pnl = (row['close'] - entry_price) * position
            capital += pnl
            trades.append({'entry': entry_price, 'exit': row['close'], 'pnl': pnl})
            position = 0
        # Trailing stop pode ser adicionado aqui

        equity.append(capital + position * row['close'])

    # Calcular métricas
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    metrics = calculate_metrics(trades)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)

    return {
        'config': config,
        'trades': trades,
        'equity': equity,
        'metrics': metrics
    }

def load_data_with_ref(config: Dict) -> pd.DataFrame:
    """Carrega dados principais e referência."""
    data_cfg = config['data']
    df = load_data(data_cfg['symbol'], data_cfg['timeframe'], data_cfg['days'], use_cache_only=True)

    if data_cfg.get('ref_timeframe'):
        df_ref = load_data(data_cfg['symbol'], data_cfg['ref_timeframe'], data_cfg['ref_days'], use_cache_only=True)
        df_ref['ref_ema'] = df_ref['close'].ewm(span=config['strategy']['ref_ema_period']).mean()
        # Merge com base em data aproximada
        df['Date'] = pd.to_datetime(df['Date'])
        df_ref['Date'] = pd.to_datetime(df_ref['Date'])
        df = pd.merge_asof(df.sort_values('Date'), df_ref[['Date', 'ref_ema']].sort_values('Date'), on='Date')

    return df

def run_backtest(config_path: str = 'src/strategies/ema_only/config.json') -> None:
    """Executa backtest completo."""
    with open(config_path) as f:
        config = json.load(f)

    # Carregar dados
    df = load_data_with_ref(config)

    # Backtest
    result = backtest_ema_only(df, config)

    # Salvar resultados
    outdir = Path(config['backtest']['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    output_file = outdir / f"ema_only_{config['data']['symbol']}_{config['data']['timeframe']}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Backtest concluído. Resultados salvos em {output_file}")
    print(f"Métricas: {result['metrics']}")

if __name__ == '__main__':
    run_backtest()
