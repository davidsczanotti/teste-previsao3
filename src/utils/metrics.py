"""
Funções utilitárias para cálculo de métricas de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def calculate_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula métricas de performance a partir de uma lista de trades
    
    Args:
        trades: Lista de dicionários com informações dos trades
        
    Returns:
        Dicionário com as métricas calculadas
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'recovery_factor': 0.0
        }
    
    # Filtra apenas trades fechados
    closed_trades = [t for t in trades if 'pnl' in t and t['pnl'] != 0]
    
    if not closed_trades:
        return {
            'total_trades': len(trades),
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'recovery_factor': 0.0
        }
    
    # Extrai P&Ls
    pnls = [t['pnl'] for t in closed_trades]
    
    # Calcula métricas básicas
    winning_trades = [p for p in pnls if p > 0]
    losing_trades = [p for p in pnls if p <= 0]
    
    total_trades = len(closed_trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    total_profit = sum(winning_trades)
    total_loss = abs(sum(losing_trades))
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    total_pnl = sum(pnls)
    
    avg_win = total_profit / len(winning_trades) if winning_trades else 0.0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0.0
    
    max_win = max(winning_trades) if winning_trades else 0.0
    max_loss = min(losing_trades) if losing_trades else 0.0
    
    # Recovery Factor (lucro total / maior drawdown)
    recovery_factor = total_pnl / abs(max_loss) if max_loss != 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'recovery_factor': recovery_factor
    }


def calculate_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calcula métricas de drawdown a partir de uma curva de equity
    
    Args:
        equity_curve: Série pandas com os valores da equity ao longo do tempo
        
    Returns:
        Dicionário com métricas de drawdown
    """
    if equity_curve.empty:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_drawdown': 0.0,
            'drawdown_duration': 0,
            'max_drawdown_duration': 0
        }
    
    # Calcula drawdown
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve - cumulative_max
    drawdown_pct = drawdown / cumulative_max * 100
    
    max_drawdown = drawdown.min()
    max_drawdown_pct = drawdown_pct.min()
    
    # Calcula duração do drawdown
    is_drawdown = drawdown < 0
    drawdown_groups = is_drawdown.ne(is_drawdown.shift()).cumsum()
    drawdown_durations = is_drawdown.groupby(drawdown_groups).sum()
    
    max_drawdown_duration = drawdown_durations.max() if not drawdown_durations.empty else 0
    avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_drawdown': avg_drawdown,
        'drawdown_duration': len(drawdown[drawdown < 0]),
        'max_drawdown_duration': max_drawdown_duration
    }


def generate_summary_report(metrics: Dict[str, float], 
                           walk_forward_stats: Optional[Dict[str, Any]] = None) -> str:
    """
    Gera um relatório resumo das métricas
    
    Args:
        metrics: Dicionário com métricas calculadas
        walk_forward_stats: Estatísticas do walk-forward (opcional)
        
    Returns:
        String formatada com o relatório
    """
    report = []
    report.append("=" * 60)
    report.append("RELATÓRIO DE MÉTRICAS DE PERFORMANCE")
    report.append("=" * 60)
    
    # Métricas básicas
    report.append(f"Total de Trades: {metrics['total_trades']}")
    report.append(f"Trades Vencedores: {metrics['winning_trades']}")
    report.append(f"Trades Perdedores: {metrics['losing_trades']}")
    report.append(f"Taxa de Acerto: {metrics['win_rate']:.2%}")
    report.append(f"")
    
    # Métricas de P&L
    report.append(f"P&L Total: ${metrics['total_pnl']:.2f}")
    report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
    report.append(f"")
    
    # Métricas de ganho/perda
    report.append(f"Média de Ganho: ${metrics['avg_win']:.2f}")
    report.append(f"Média de Perda: ${metrics['avg_loss']:.2f}")
    report.append(f"Maior Ganho: ${metrics['max_win']:.2f}")
    report.append(f"Maior Perda: ${metrics['max_loss']:.2f}")
    report.append(f"")
    
    # Recovery Factor
    report.append(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
    report.append(f"")
    
    # Estatísticas do walk-forward (se disponíveis)
    if walk_forward_stats:
        report.append("=" * 60)
        report.append("ESTATÍSTICAS WALK-FORWARD")
        report.append("=" * 60)
        report.append(f"Períodos Totais: {walk_forward_stats['total_periods']}")
        report.append(f"Períodos Bem Sucedidos: {walk_forward_stats['successful_periods']}")
        report.append(f"Taxa de Sucesso: {walk_forward_stats['success_rate']:.2%}")
        report.append(f"P&L Total Walk-Forward: ${walk_forward_stats['total_pnl']:.2f}")
        report.append(f"P&L Médio Walk-Forward: ${walk_forward_stats['avg_pnl']:.2f}")
        report.append(f"Win Rate Médio Walk-Forward: {walk_forward_stats['avg_win_rate']:.2%}")
        report.append(f"Profit Factor Médio Walk-Forward: {walk_forward_stats['avg_profit_factor']:.2f}")
        report.append(f"Períodos Lucrativos: {walk_forward_stats['periods_with_profit']}")
        report.append(f"Períodos Prejuízo: {walk_forward_stats['periods_with_loss']}")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calcula o Sharpe Ratio
    
    Args:
        returns: Série de retornos
        risk_free_rate: Taxa livre de risco anual
        
    Returns:
        Sharpe Ratio anualizado
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    
    # Calcula retorno médio e volatilidade
    mean_return = returns.mean()
    volatility = returns.std()
    
    # Annualiza
    n_periods = 252  # Considerando 252 dias úteis por ano
    annualized_return = mean_return * n_periods
    annualized_volatility = volatility * np.sqrt(n_periods)
    
    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    return sharpe_ratio


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calcula o Sortino Ratio
    
    Args:
        returns: Série de retornos
        risk_free_rate: Taxa livre de risco anual
        
    Returns:
        Sortino Ratio anualizado
    """
    if returns.empty or returns[returns < 0].std() == 0:
        return 0.0
    
    # Calcula retorno médio
    mean_return = returns.mean()
    
    # Calcula downside deviation
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std()
    
    # Annualiza
    n_periods = 252  # Considerando 252 dias úteis por ano
    annualized_return = mean_return * n_periods
    annualized_downside = downside_deviation * np.sqrt(n_periods)
    
    # Sortino Ratio
    sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside
    
    return sortino_ratio
