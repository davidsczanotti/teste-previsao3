
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.cluster import KMeans

def calculate_supertrend_ai(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calcula SuperTrend AI (Clustering) inspirado no LuxAlgo.
    
    Lógica:
    1. Calcula múltiplos SuperTrends (minMult a maxMult).
    2. Avalia performance recente de cada fator (Performance Index).
    3. Agrupa fatores em 3 clusters (Best, Average, Worst) baseados na performance.
    4. Seleciona o 'target_factor' do cluster 'Best' (ou conforme config).
    5. Gera o SuperTrend final com esse fator adaptativo.
    """
    df = df.copy()
    
    # Parâmetros
    length = int(config['strategy'].get('st_length', 10))
    min_mult = int(config['strategy'].get('st_min_mult', 1))
    max_mult = int(config['strategy'].get('st_max_mult', 5))
    step = float(config['strategy'].get('st_step', 0.5))
    perf_alpha = int(config['strategy'].get('st_perf_alpha', 10))
    from_cluster = config['strategy'].get('st_from_cluster', 'Best') # Best, Average, Worst
    
    # 1. Preparação dos Fatores
    factors = np.arange(min_mult, max_mult + step, step) # Include max_mult
    
    # ATR Base
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    hl2 = (df['high'] + df['low']) / 2
    
    # Arrays para armazenar estado de cada fator
    # No Pine, isso é feito bar-by-bar. Aqui, vamos simular vetorizado onde possível, 
    # mas a natureza adaptativa (escolha do fator depende do passado) exige loop ou lógica complexa.
    
    # Como o "Target Factor" muda a cada barra baseado no cluster, precisamos de um loop principal.
    # Para otimizar, não rodaremos KMeans a cada barra. Faremos um "Batch Update" ou simplificação.
    # O Pine roda a cada barra. Para ser fiel, precisamos de loop.
    # Python Loop em 3000 barras x 15 ativos é aceitável (~45k iterações).
    
    # Estruturas de Estado para cada fator
    n_factors = len(factors)
    st_upper = np.full(n_factors, np.nan)
    st_lower = np.full(n_factors, np.nan)
    st_trend = np.zeros(n_factors, dtype=int) # 1 = Up, 0 = Down
    st_perf = np.zeros(n_factors)
    st_output = np.full(n_factors, np.nan)
    
    # Output Series
    final_trend = np.zeros(len(df))
    final_ts = np.zeros(len(df))
    
    # Pré-cálculo de Close Shifts para performance
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr_val = atr.fillna(0).values
    hl2_val = hl2.values
    
    # Performance Alpha Factor (EMA)
    alpha_perf = 2 / (perf_alpha + 1)
    
    # Loop Principal (Barra a Barra)
    # Começamos de 'length' para ter ATR válido
    for i in range(length, len(df)):
        c = close[i]
        c_prev = close[i-1]
        
        # 1. Atualizar SuperTrends Individuais e Performance
        # Vetorização sobre os fatores
        
        # Bandas Atuais
        up_vec = hl2_val[i] + atr_val[i] * factors
        dn_vec = hl2_val[i] - atr_val[i] * factors
        
        # Lógica SuperTrend Padrão (Vetorizada)
        # Se fechar acima do Upper anterior -> Trend = 1
        # Se fechar abaixo do Lower anterior -> Trend = 0
        
        # Inicialização no primeiro passo
        if i == length:
            st_upper = up_vec
            st_lower = dn_vec
            st_output = up_vec # Assume down initially or whatever
            st_trend = np.zeros(n_factors, dtype=int) # Assume down
        
        # Update Upper/Lower (Trailing logic)
        # Upper: Se close[i-1] < upper[i-1], min(up_vec, upper[i-1]), else up_vec
        mask_prev_below_upper = c_prev < st_upper
        st_upper = np.where(mask_prev_below_upper, np.minimum(up_vec, st_upper), up_vec)
        
        # Lower: Se close[i-1] > lower[i-1], max(dn_vec, lower[i-1]), else dn_vec
        mask_prev_above_lower = c_prev > st_lower
        st_lower = np.where(mask_prev_above_lower, np.maximum(dn_vec, st_lower), dn_vec)
        
        # Update Trend
        # Trend 1 if close > upper
        # Trend 0 if close < lower
        # Else keep prev
        mask_cross_up = c > st_upper
        mask_cross_down = c < st_lower
        
        new_trend = np.where(mask_cross_up, 1, np.where(mask_cross_down, 0, st_trend))
        st_trend = new_trend
        
        # Output (TS Line)
        # Se trend 1, output = lower. Se trend 0, output = upper
        current_outputs = np.where(st_trend == 1, st_lower, st_upper)
        
        # Update Performance
        # perf += alpha * ((close - close[1]) * sign(close[1] - output) - perf)
        # sign: se preço estava acima da linha, diff positivo. Se abaixo, negativo.
        # Basicamente: Se estamos long e preço sobe, bom. Se short e preço desce, bom.
        
        diff = np.sign(c_prev - st_output) # st_output é o da barra anterior aqui? No pine é `get_spt.output`.
        # No pine, `get_spt.output` é atualizado DEPOIS do calculo de performance na mesma iteração, mas usando trend atual.
        # Porem o `diff` usa `get_spt.output` ANTES da atualização?
        # Pine: 
        # diff = nz(math.sign(close[1] - get_spt.output))
        # get_spt.perf += ...
        # get_spt.output := ... (atualizado no fim)
        # Então usa o output da barra ANTERIOR.
        
        # Para i=length, st_output é init, diff pode ser 0.
        diff = np.where(diff == 0, 1, diff) # Evitar 0
        
        pnl_proxy = (c - c_prev) * diff
        st_perf = st_perf + alpha_perf * (pnl_proxy - st_perf)
        
        # Atualiza Output para a próxima
        st_output = current_outputs
        
        # 2. Clustering (K-Means 1D simplificado)
        # Não precisamos rodar KMeans completo do sklearn (pesado).
        # Vamos fazer um "Quantile Clustering" simples como fallback eficiente ou K-Means leve.
        # O Pine usa K-Means iterativo.
        # Para performance em Python loop, vamos usar Percentiles para definir centroides iniciais
        # e fazer 1 ou 2 iterações de atribuição, que é 90% do resultado.
        
        perf_data = st_perf.reshape(-1, 1)
        
        # Sklearn KMeans é overkill dentro de loop. Vamos implementar K-means 1D manual rápido.
        # Clusters: 3.
        # Init: Percentiles 25, 50, 75.
        
        centroids = np.percentile(st_perf, [25, 50, 75])
        
        # Iteração (Pine faz maxIter, mas geralmente convergem rápido em 1D)
        for _ in range(3): # 3 iterações deve bastar para estabilidade
            # Distâncias: |perf - c0|, |perf - c1|, |perf - c2|
            d0 = np.abs(st_perf - centroids[0])
            d1 = np.abs(st_perf - centroids[1])
            d2 = np.abs(st_perf - centroids[2])
            
            # Atribuição
            labels = np.argmin(np.vstack([d0, d1, d2]), axis=0)
            
            # Update Centroids
            new_centroids = np.array([st_perf[labels == k].mean() if np.any(labels==k) else centroids[k] for k in range(3)])
            
            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
            
        # Ordenar Clusters por Performance Média (Worst=0, Avg=1, Best=2)
        # Centroids já tendem a estar ordenados se inicializados por percentil, mas vamos garantir.
        sorted_idx = np.argsort(centroids)
        # Mapear labels originais para 0,1,2 ordenados
        # Ex: se centroids saiu [10, -5, 20], sorted é [1, 0, 2] (-5, 10, 20)
        # Cluster 'Worst' é o centroid[1] (-5).
        
        target_cluster_idx = sorted_idx[2] if from_cluster == 'Best' else (sorted_idx[1] if from_cluster == 'Average' else sorted_idx[0])
        
        # Identificar fatores no cluster alvo
        target_mask = (labels == target_cluster_idx)
        if not np.any(target_mask):
            # Fallback se cluster vazio (raro)
            target_factor = factors[0]
        else:
            target_factor = np.mean(factors[target_mask])
            
        # 3. Calcular SuperTrend Final (Com Target Factor)
        # "Get new supertrend" do Pine
        # Var upper, lower, os
        # O Pine mantém um ESTADO separado para esse "SuperTrend Resultante".
        
        # Precisamos de variaveis de estado fora do loop para o FINAL
        if i == length:
            final_upper = hl2_val[i] + atr_val[i] * target_factor
            final_lower = hl2_val[i] - atr_val[i] * target_factor
            final_os = 1 if c > final_upper else 0
            final_ts_val = final_lower if final_os else final_upper
        else:
            # Recupera estado anterior
            # Como estamos em loop, usamos vars locais
            # Mas espera, preciso armazenar 'final_upper', 'final_lower', 'final_os' de i-1
            # Vamos usar variaveis auxiliares fora do loop? Não, só vars locais atualizadas.
            pass # Lógica abaixo resolve
            
        # Lógica SuperTrend "Final"
        curr_up = hl2_val[i] + atr_val[i] * target_factor
        curr_dn = hl2_val[i] - atr_val[i] * target_factor
        
        if i > length:
            # Pine: upper := close[1] < upper ? math.min(up, upper) : up
            if c_prev < final_upper:
                final_upper = min(curr_up, final_upper)
            else:
                final_upper = curr_up
                
            # Pine: lower := close[1] > lower ? math.max(dn, lower) : dn
            if c_prev > final_lower:
                final_lower = max(curr_dn, final_lower)
            else:
                final_lower = curr_dn
                
            # Pine: os := close > upper ? 1 : close < lower ? 0 : os
            if c > final_upper:
                final_os = 1
            elif c < final_lower:
                final_os = 0
            # else mantém final_os anterior
            
            final_ts_val = final_lower if final_os == 1 else final_upper
        
        # Store Result
        final_trend[i] = final_os
        final_ts[i] = final_ts_val
        
    df['supertrend_ai'] = final_ts
    df['supertrend_ai_trend'] = final_trend # 1 = Bull, 0 = Bear
    
    return df
