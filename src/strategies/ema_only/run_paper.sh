#!/bin/bash
# Script para rodar o Paper Trading da estratégia EMA Only
# Execute a partir da raiz do projeto: ./src/strategies/ema_only/run_paper.sh

# Garante que estamos na raiz do projeto (assumindo que o script está em src/strategies/ema_only/)
cd "$(dirname "$0")/../../.."

# Define PYTHONPATH para incluir o diretório atual
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Iniciando Paper Trading (EMA Only)..."
echo "Ativo configurado: FXSUSDT (via config.json)"
echo "Pressione Ctrl+C para parar."

poetry run python src/strategies/ema_only/paper.py
