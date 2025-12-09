import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.strategies.ema_only.rl_env import EmaEnv, RLConfig

# Carrega a configuração do projeto para obter os valores de bônus esperados
CONFIG_PATH = Path(__file__).parent.parent / 'src/strategies/ema_only/config.json'
with open(CONFIG_PATH, 'r') as f:
    config_data = json.load(f)

REWARD_CONFIG = config_data.get('rl', {}).get('reward', {})
BASE_BONUS = REWARD_CONFIG.get('entry_bonus_fast_over_slow', 0.0)
FULL_BONUS = REWARD_CONFIG.get('entry_bonus_full_trend', 0.0)

# O bônus total para nível 2 é a soma do bônus base + o bônus de tendência completa
# Se full_bonus não for definido, o código usa base_bonus novamente.
EXTRA_BONUS = FULL_BONUS if FULL_BONUS != 0.0 else BASE_BONUS
EXPECTED_LEVEL_1_BONUS = BASE_BONUS
EXPECTED_LEVEL_2_BONUS = BASE_BONUS + EXTRA_BONUS


def _create_test_env(features_data: dict) -> EmaEnv:
    """Helper para criar um ambiente de teste com dados e config mockados."""
    
    # Configuração que permite que os trades passem pelo gate sem bloqueios
    # e isola o bônus de EMA.
    rl_config = RLConfig(
        # Zera custos para não interferir no valor do bônus
        fee_pct=0.0,
        slippage_pct=0.0,
        # Parâmetros de bônus que estamos testando
        entry_bonus_fast_over_slow=BASE_BONUS,
        entry_bonus_full_trend=FULL_BONUS,
        # Desliga outros bônus/penalidades para isolar o teste
        pullback_entry_bonus=0.0,
        consensus_bonus=0.0,
        align_bonus=0.0,
        trade_penalty=0.0,
        gating_penalty=0.0,
        # Desliga experts/consenso para o gate não bloquear
        experts_master_enable=False,
        experts_enabled=None,
        # Desliga outros gates que podem interferir
        enforce_ref_bias=False,
        cross_lookback_bars=0,
        block_long_in_bear=False,
    )
    # Atributo extra usado via getattr dentro do ambiente (não faz parte do dataclass)
    rl_config.ref_slope_enabled = False
    
    # DataFrame de features com os cenários (pelo menos 2 linhas para evitar erro de índice)
    features = pd.DataFrame(features_data)
    if len(features) == 1:
        features = pd.concat([features, features.iloc[[0]].copy()], ignore_index=True)

    num_rows = len(features)

    # DataFrame de preços (close) - mantido constante para não gerar pnl
    df = pd.DataFrame({'close': [100.0] * num_rows})
    
    # Adiciona colunas dummy que o ambiente espera
    for col in ['atr_rel', 'experts_mean', 'exp_trend', 'exp_ref']:
        if col not in features.columns:
            features[col] = [0.5] * num_rows
            
    env = EmaEnv(df=df, features=features, cfg=rl_config)
    return env


@pytest.mark.parametrize(
    "scenario, action, features, expected_bonus",
    [
        (
            "Long: Bônus Nível 1 (fast > slow)", 
            1, # Ação de compra
            {
                'ema_fast': [102], 'ema_slow': [101], 'ref_ema': [101.5] # fast > slow, mas slow < ref
            },
            EXPECTED_LEVEL_1_BONUS
        ),
        (
            "Long: Bônus Nível 2 (fast > slow > ref)",
            1, # Ação de compra
            {
                'ema_fast': [103], 'ema_slow': [102], 'ref_ema': [101] # fast > slow > ref
            },
            EXPECTED_LEVEL_2_BONUS
        ),
        (
            "Long: Sem Bônus (fast < slow)",
            1, # Ação de compra
            {
                'ema_fast': [100], 'ema_slow': [101], 'ref_ema': [102] # fast < slow, gate deve falhar
            },
            0.0 # Espera-se que o gate bloqueie, mas testamos o bônus em si
        ),
        (
            "Short: Bônus Nível 1 (fast < slow)",
            2, # Ação de venda
            {
                'ema_fast': [99], 'ema_slow': [100], 'ref_ema': [99.5], # fast < slow, mas slow > ref
                'experts_mean': [0.4], 'exp_trend': [0.4], 'exp_ref': [0.4] # Para passar no gate de short
            },
            EXPECTED_LEVEL_1_BONUS
        ),
        (
            "Short: Bônus Nível 2 (fast < slow < ref)",
            2, # Ação de venda
            {
                'ema_fast': [98], 'ema_slow': [99], 'ref_ema': [100], # fast < slow < ref
                'experts_mean': [0.4], 'exp_trend': [0.4], 'exp_ref': [0.4] # Para passar no gate de short
            },
            EXPECTED_LEVEL_2_BONUS
        ),
        (
            "Short: Sem Bônus (fast > slow)",
            2, # Ação de venda
            {
                'ema_fast': [101], 'ema_slow': [100], 'ref_ema': [99], # fast > slow, gate deve falhar
                'experts_mean': [0.4], 'exp_trend': [0.4], 'exp_ref': [0.4]
            },
            0.0
        ),
    ]
)
def test_symmetric_ema_bonus_logic(scenario, action, features, expected_bonus):
    """
    Testa se o bônus de entrada com 2 níveis de EMAs é aplicado
    de forma simétrica para long e short, conforme a configuração.
    """
    print(f"Executando cenário: {scenario}")
    
    # Arrange: Cria o ambiente com os dados do cenário
    # Garante que as features para o gate de long estejam corretas
    if action == 1:
        features['experts_mean'] = [0.6]
        features['exp_trend'] = [0.6]
        features['exp_ref'] = [0.6]
    env = _create_test_env(features)

    # Act: Executa um passo no ambiente com a ação especificada
    _, reward, _, _, _ = env.step(action)

    # Assert: Verifica se a recompensa recebida é aproximadamente igual ao bônus esperado
    # O reward pode ter um pequeno resíduo do MTM PnL, por isso a tolerância.
    # O gate pode aplicar uma penalidade se a entrada for bloqueada.
    # Se o bônus esperado é 0, a recompensa deve ser 0 ou negativa (penalidade do gate).
    if expected_bonus > 0:
        assert reward == pytest.approx(expected_bonus, abs=1e-9), \
            f"Falha no cenário '{scenario}': Bônus esperado {expected_bonus}, mas recebeu {reward}"
    else:
        # Se não há bônus, o gate deve bloquear a entrada, resultando em recompensa 0 ou negativa.
        # Neste setup de teste, a penalidade de gate é 0, então o reward deve ser 0.
        assert reward <= 0, \
            f"Falha no cenário '{scenario}': Esperava-se recompensa <= 0 (gate), mas recebeu {reward}"
