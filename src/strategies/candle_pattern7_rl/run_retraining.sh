#!/bin/bash

# --- Script para Re-treinamento Contínuo da Estratégia candle_pattern7_rl ---

# Objetivo: Ajustar o modelo existente com os dados mais recentes do mercado.
# Este script deve ser executado periodicamente (ex: mensalmente).

echo "Iniciando o processo de re-treinamento..."

# --- Configuração ---
# Caminho para o modelo treinado que será usado como base.
# O script de treino irá carregar este modelo e continuar o aprendizado a partir dele.
# IMPORTANTE: O script de treino salva o modelo no caminho definido em `reports/`,
# então este script assume que o modelo a ser carregado está lá.
MODELO_ANTERIOR="reports/agents/candle_pattern7_rl/BTCUSDT_15m.npz"
TICKER="BTCUSDT"
INTERVALO="15m"

# Número de dias de dados a serem carregados. Idealmente, deve incluir todos os dados históricos
# para que o agente não "esqueça" os regimes de mercado antigos.
DIAS_HISTORICO=3650

# Número de episódios para o re-treinamento.
# Como estamos apenas ajustando (fine-tuning), usamos um número menor que o treino inicial.
EPISODIOS_RETREINO=150

# --- Verificação ---
# Garante que o modelo anterior existe antes de tentar o re-treinamento.
# O caminho é relativo à raiz do projeto, onde o comando `poetry run` é executado.
if [ ! -f "$MODELO_ANTERIOR" ]; then
    echo "Erro: Modelo anterior não encontrado em '$MODELO_ANTERIOR'."
    echo "Execute um treinamento inicial completo primeiro."
    exit 1
fi

echo "Usando o modelo base: $MODELO_ANTERIOR"
echo "Executando por $EPISODIOS_RETREINO episódios..."

# --- Execução do Treinamento ---
# O comando `poetry run` garante que estamos usando o ambiente Python correto.
# Usamos os mesmos parâmetros do treino original, mas adicionamos o flag `--model`
# e ajustamos o número de episódios e a taxa de exploração (epsilon).
poetry run python -m src.strategies.candle_pattern7_rl.train \
    --ticker "$TICKER" \
    --interval "$INTERVALO" \
    --days "$DIAS_HISTORICO" \
    --episodes "$EPISODIOS_RETREINO" \
    --model "$MODELO_ANTERIOR" \
    --episode_len 8192 \
    --long_only \
    --gate_on_heuristic \
    --hidden 64 \
    --lr 0.0005 \
    --min_hold_bars 3 \
    --reopen_cooldown_bars 1 \
    --action_cost_open 0.02 \
    --action_cost_close 0.02 \
    --epsilon_start 0.1 \
    --epsilon_end 0.02 \
    --bc_weight 0.05 \
    --idle_penalty 0.005 \
    --switch_penalty 0.02 \
    --reward_atr_norm

# --- Conclusão ---
if [ $? -eq 0 ]; then
    echo "Re-treinamento concluído com sucesso!"
    echo "O modelo em '$MODELO_ANTERIOR' foi atualizado com os dados mais recentes."
else
    echo "Erro durante o processo de re-treinamento."
fi
