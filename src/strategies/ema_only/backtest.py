import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path

from src.utils.data_loader import load_data
from src.utils.metrics import calculate_metrics, calculate_sharpe_ratio


def calculate_cci(df: pd.DataFrame, period: int) -> pd.Series:
    """Calcula CCI (Commodity Channel Index)."""
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    ma = tp.rolling(period).mean()

    def _mean_dev(x):
        x = pd.Series(x)
        return (x - x.mean()).abs().mean()

    mean_dev = tp.rolling(period).apply(_mean_dev, raw=False)
    cci = (tp - ma) / (0.015 * mean_dev)
    return cci

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

    # ATR para trailing / filtros de tendência
    df['atr'] = calculate_atr(df, config['strategy']['atr_period'])

    # CCI para filtro de força de tendência
    cci_period = config['strategy'].get('cci_period')
    if cci_period:
        df['cci'] = calculate_cci(df, int(cci_period))

    # ATR Customizado para a estratégia custom_cci_ma
    if config['strategy'].get('signal_mode') == 'custom_cci_ma':
        custom_atr_period = config['strategy'].get('custom_atr_period', 10)
        df['custom_atr'] = calculate_atr(df, custom_atr_period)
        
        # Filtro de Tendência Macro (EMA 200)
        df['ema_200'] = df['close'].ewm(span=200).mean()

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

    if strategy.get('signal_mode') == 'custom_cci_ma':
        # Lógica customizada: CCI + 2 SMAs + Filtros Dinâmicos
        mc = df['close'].rolling(strategy['custom_ma_fast']).mean()
        ml = df['close'].rolling(strategy['custom_ma_slow']).mean()
        cci = calculate_cci(df, strategy['custom_cci_period'])
        
        # Distância dinâmica: ATR * Multiplicador (default 0.5)
        atr = df.get('custom_atr', df['close']*0.01) # Fallback seguro
        dist_mult = strategy.get('custom_dist_atr_mult', 0.5)
        min_dist = atr * dist_mult
        diff = (mc - ml).abs()
        
        level = strategy['custom_cci_level']
        ema_200 = df.get('ema_200', df['close']) # Fallback
        
        # Inicializar
        df['signal'] = 0
        
        # SinalC (Long): mc > ml AND cci > Nivel AND diff > Distancia Dinâmica AND Close > EMA200
        cond_long = (mc > ml) & (cci > level) & (diff > min_dist) & (df['close'] > ema_200)
        
        # SinalV (Short): mc < ml AND cci < -Nivel AND diff > Distancia Dinâmica AND Close < EMA200
        cond_short = (mc < ml) & (cci < -level) & (diff > min_dist) & (df['close'] < ema_200)
        
        df.loc[cond_long, 'signal'] = 1
        df.loc[cond_short, 'signal'] = -1
        
    else:
        # Sinais de cruzamento (simples: ema_fast sobre ema_slow)
        df['ema_cross'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        df['ema_cross_prev'] = df['ema_cross'].shift(1)
        df['signal'] = np.where((df['ema_cross'] == 1) & (df['ema_cross_prev'] == -1), 1,  # Long
                                np.where((df['ema_cross'] == -1) & (df['ema_cross_prev'] == 1), -1, 0))  # Short

    # Filtro: evitar entradas long muito esticadas em relação à EMA rápida
    max_dist = strategy.get('max_long_entry_dist_fast_pct')
    if max_dist is not None:
        dist_fast = (df['close'] - df['ema_fast']).abs() / df['ema_fast']
        df.loc[(df['signal'] == 1) & (dist_fast > max_dist), 'signal'] = 0

    # Filtro de tendência L2: exigir que a EMA lenta tenha inclinação mínima
    lookback = strategy.get('slow_slope_lookback_bars')
    min_slope = strategy.get('min_slow_slope_for_long_pct')
    if lookback is not None and min_slope is not None and lookback > 0:
        prev_slow = df['ema_slow'].shift(int(lookback))
        slow_slope = (df['ema_slow'] - prev_slow) / prev_slow
        df.loc[(df['signal'] == 1) & (slow_slope < min_slope), 'signal'] = 0

    # Filtro CCI: exige força de tendência no oscilador
    cci_level = strategy.get('cci_level')
    if cci_level is not None and 'cci' in df.columns:
        df.loc[(df['signal'] == 1) & (df['cci'] < cci_level), 'signal'] = 0

    # Filtro ATR: exige separação mínima entre EMAs em múltiplos do ATR
    sep_mult = strategy.get('min_ema_separation_atr_mult')
    if sep_mult is not None:
        ema_sep = (df['ema_fast'] - df['ema_slow']).abs()
        df.loc[(df['signal'] == 1) & (ema_sep < sep_mult * df['atr']), 'signal'] = 0

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
    # df = apply_trailing_stop(df, config) # Trailing stop padrão desativado para lógica customizada

    # Simulação de trades
    capital = config['backtest']['initial_capital']
    position = 0.0 # Positivo = Long, Negativo = Short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    trades = []
    equity = [capital]
    
    lot_size = config['strategy']['lot_size']
    
    is_custom_mode = config['strategy'].get('signal_mode') == 'custom_cci_ma'
    target_factor = config['strategy'].get('custom_target_factor', 1.5)
    stop_factor = config['strategy'].get('custom_stop_factor', 0.9)

    for i, row in df.iterrows():
        # Lógica de Saída (TP/SL)
        if position != 0:
            pnl = 0
            exit_price = 0
            exit_reason = ""
            
            if position > 0: # Long
                if row['low'] <= stop_price:
                    exit_price = stop_price 
                    if exit_price > row['high']: exit_price = row['open'] # Gap check
                    pnl = (exit_price - entry_price) * position
                    exit_reason = "stop_loss"
                elif row['high'] >= target_price:
                    exit_price = target_price
                    pnl = (exit_price - entry_price) * position
                    exit_reason = "take_profit"
                # Saída por sinal reverso
                elif row['signal'] == -1:
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) * position
                    exit_reason = "signal_reverse"

            elif position < 0: # Short
                if row['high'] >= stop_price:
                    exit_price = stop_price
                    if exit_price < row['low']: exit_price = row['open']
                    pnl = (entry_price - exit_price) * abs(position)
                    exit_reason = "stop_loss"
                elif row['low'] <= target_price:
                    exit_price = target_price
                    pnl = (entry_price - exit_price) * abs(position)
                    exit_reason = "take_profit"
                elif row['signal'] == 1:
                    exit_price = row['close']
                    pnl = (entry_price - exit_price) * abs(position)
                    exit_reason = "signal_reverse"

            if exit_reason:
                capital += pnl
                trades.append({
                    'entry': entry_price, 
                    'exit': exit_price, 
                    'pnl': pnl, 
                    'side': 'long' if position > 0 else 'short',
                    'reason': exit_reason,
                    'date': row['Date'] if 'Date' in row else i
                })
                position = 0
                entry_price = 0
                stop_price = 0
                target_price = 0

        # Lógica de Entrada
        if position == 0:
            # Definir tamanho da posição (Fixo ou Composto)
            use_compounding = config['strategy'].get('compounding_enabled', False)
            if use_compounding:
                pct = config['strategy'].get('compounding_pct', 0.95)
                # Garante que não trade negativo se quebrou a conta
                if capital <= 0:
                    current_qty = 0
                else:
                    current_qty = (capital * pct) / row['close']
            else:
                current_qty = lot_size

            if row['signal'] == 1 and current_qty > 0: # Long
                position = current_qty
                entry_price = row['close']
                
                if is_custom_mode:
                    vol = row.get('custom_atr', row.get('atr', 0))
                    target_price = entry_price + (vol * target_factor)
                    stop_price = entry_price - (vol * stop_factor)
                else:
                    # Fallback padrão
                    target_price = entry_price * 1.5
                    stop_price = entry_price * 0.95

            elif row['signal'] == -1 and current_qty > 0: # Short
                position = -current_qty
                entry_price = row['close']
                
                if is_custom_mode:
                    vol = row.get('custom_atr', row.get('atr', 0))
                    target_price = entry_price - (vol * target_factor)
                    stop_price = entry_price + (vol * stop_factor)
                else:
                    target_price = entry_price * 0.5
                    stop_price = entry_price * 1.05

        equity.append(capital + (position * row['close'] if position > 0 else position * (2*entry_price - row['close']) if position < 0 else 0))

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
