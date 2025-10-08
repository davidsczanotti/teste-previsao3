from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas ao DataFrame que identificam a presença de padrões de velas
    clássicos usando a biblioteca pandas-ta.

    Os valores nas colunas são:
    - 100: Padrão de alta (Bullish)
    - -100: Padrão de baixa (Bearish)
    - 0: Nenhum padrão detectado

    Args:
        df (pd.DataFrame): DataFrame com colunas 'open', 'high', 'low', 'close'.

    Returns:
        pd.DataFrame: DataFrame original com as novas colunas de padrões.
    """
    # Lista de padrões a serem detectados.
    # Você pode adicionar ou remover padrões desta lista.
    # Veja a documentação do pandas-ta para todos os padrões disponíveis.
    patterns = [
        "engulfing",  # Engolfo de Alta/Baixa
        "hammer",  # Martelo
        "hangingman",  # Homem Enforcado
        "morningstar",  # Estrela da Manhã
        "eveningstar",  # Estrela da Noite
    ]

    # Usa a função `cdl_pattern` do pandas-ta para adicionar todos os padrões de uma vez.
    df.ta.cdl_pattern(name=patterns, append=True)

    return df
