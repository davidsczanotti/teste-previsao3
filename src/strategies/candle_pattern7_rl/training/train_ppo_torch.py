
from __future__ import annotations

import argparse
import os
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ..config import Candle7RlConfig
from ..env import Candle7Env, RunningNorm

# --- 1. Definições de Rede Neural em PyTorch ---

class PositionalEncoding(nn.Module):
    """Adiciona codificação posicional para informar ao Transformer a ordem da sequência."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2)) # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: Tensor, shape [batch_size, seq_len, embedding_dim] """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ActorCriticTransformer(nn.Module):
    """Política Ator-Crítico usando uma arquitetura baseada em Transformer."""
    def __init__(self, seq_input_shape: tuple[int, int], non_seq_input_dim: int, act_dim: int, hidden_size: int = 128):
        super().__init__()
        seq_len, seq_feature_dim = seq_input_shape
        
        # Camada de embedding para as features sequenciais
        self.seq_embedding = nn.Linear(seq_feature_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=seq_len)

        # Encoder do Transformer
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # MLP final que combina a saída do transformer com as features não sequenciais
        self.combined_mlp = nn.Sequential(
            self.layer_init(nn.Linear(hidden_size + non_seq_input_dim, hidden_size)),
            nn.Tanh()
        )

        # Cabeças do Ator e do Crítico
        self.actor = self.layer_init(nn.Linear(hidden_size, act_dim), std=0.01)
        self.critic = self.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def layer_init(self, layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward_transformer(self, seq_x: torch.Tensor, non_seq_x: torch.Tensor):
        seq_embedded = self.seq_embedding(seq_x)
        seq_encoded = self.pos_encoder(seq_embedded)
        transformer_out = self.transformer_encoder(seq_encoded)
        
        # Usamos a saída do último token da sequência
        transformer_last_token_out = transformer_out[:, -1, :]
        
        # Combina com as features não sequenciais
        combined_features = torch.cat([transformer_last_token_out, non_seq_x], dim=1)
        return self.combined_mlp(combined_features)

    def get_value(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        base_out = self.forward_transformer(x['sequential'], x['non_sequential'])
        return self.critic(base_out)

    def get_action_and_value(self, x: dict[str, torch.Tensor], action: Optional[torch.Tensor] = None):
        base_out = self.forward_transformer(x['sequential'], x['non_sequential'])
        logits = self.actor(base_out)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(base_out)
        return action, log_prob, entropy, value

class ActorCriticLSTM(nn.Module):
    """Política Ator-Crítico baseada em LSTM para sequência de 7 candles.

    Pipeline: Linear embedding por candle -> LSTM sobre a sequência ->
    concatena com features não sequenciais -> MLP -> cabeças actor/critic.
    """
    def __init__(
        self,
        seq_input_shape: tuple[int, int],
        non_seq_input_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        seq_len, seq_feature_dim = seq_input_shape

        self.seq_embedding = nn.Linear(seq_feature_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.combined_mlp = nn.Sequential(
            self.layer_init(nn.Linear(hidden_size + non_seq_input_dim, hidden_size)),
            nn.Tanh(),
        )

        self.actor = self.layer_init(nn.Linear(hidden_size, act_dim), std=0.01)
        self.critic = self.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def layer_init(self, layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward_lstm(self, seq_x: torch.Tensor, non_seq_x: torch.Tensor) -> torch.Tensor:
        # seq_x: [B, T, F] -> embed -> [B, T, H]
        seq_emb = self.seq_embedding(seq_x)
        out, _ = self.lstm(seq_emb)  # out: [B, T, H]
        last_out = out[:, -1, :]     # [B, H]
        combined = torch.cat([last_out, non_seq_x], dim=1)
        return self.combined_mlp(combined)

    def get_value(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        base_out = self.forward_lstm(x["sequential"], x["non_sequential"])
        return self.critic(base_out)

    def get_action_and_value(self, x: dict[str, torch.Tensor], action: Optional[torch.Tensor] = None):
        base_out = self.forward_lstm(x["sequential"], x["non_sequential"])  # [B, H]
        logits = self.actor(base_out)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(base_out)
        return action, log_prob, entropy, value


class ActorCriticMLP(nn.Module):
    """
    Política Ator-Crítico com uma arquitetura MLP (Multi-Layer Perceptron),
    replicando a estrutura da implementação NumPy.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_size, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def layer_init(self, layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
        """Inicializa os pesos da camada linear."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna o valor do estado (predição do crítico)."""
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        """
        Retorna a ação, o log da probabilidade da ação e a entropia da política,
        junto com o valor do estado.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x)
        return action, log_prob, entropy, value

# --- 2. Script Principal de Treinamento ---

def ppo_train_torch(
    cfg: Candle7RlConfig,
    model_path: Optional[str] = None,
    save: bool = True,
    run_ablation: bool = False,
    ablation_groups: Optional[list[str]] = None,
):
    """Função de treinamento PPO usando PyTorch."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Políticas que consomem sequência requerem observação estruturada
    obs_format = "structured" if cfg.policy_type in ("transformer", "lstm") else "flat"
    env = Candle7Env(
        symbol=cfg.ticker, interval=cfg.interval, days=cfg.days,
        lot_size=cfg.lot_size, fee_rate=cfg.fee_rate, slippage_bps=cfg.slippage_bps,
        action_cost_open=cfg.action_cost_open, action_cost_close=cfg.action_cost_close,
        invalid_action_penalty=cfg.invalid_action_penalty, min_hold_bars=cfg.min_hold_bars,
        reopen_cooldown_bars=cfg.reopen_cooldown_bars, max_position_bars=cfg.max_position_bars,
        long_only=cfg.long_only, m2m_weight=cfg.m2m_weight, exec_at_next_open=cfg.exec_next_open,
        switch_penalty=cfg.switch_penalty, switch_window_bars=cfg.switch_window_bars,
        episode_len=cfg.episode_len, random_start=cfg.random_start, idle_penalty=cfg.idle_penalty,
        idle_grace_bars=cfg.idle_grace_bars, idle_ramp=cfg.idle_ramp, reward_atr_norm=cfg.reward_atr_norm,
        atr_period=cfg.atr_period, gate_on_heuristic=cfg.gate_on_heuristic, obs_format=obs_format,
        include_mtf=cfg.include_mtf, mtf_timeframes=cfg.mtf_timeframes
    )

    obs_size = env.observation_size
    act_dim = env.action_size

    if cfg.policy_type == "transformer":
        agent = (
            ActorCriticTransformer(
                seq_input_shape=obs_size["sequential"],
                non_seq_input_dim=obs_size["non_sequential"],
                act_dim=act_dim,
                hidden_size=cfg.hidden_size,
            ).to(device)
        )
    elif cfg.policy_type == "lstm":
        agent = (
            ActorCriticLSTM(
                seq_input_shape=obs_size["sequential"],
                non_seq_input_dim=obs_size["non_sequential"],
                act_dim=act_dim,
                hidden_size=cfg.hidden_size,
            ).to(device)
        )
    else:  # mlp
        agent = ActorCriticMLP(obs_size, act_dim, cfg.hidden_size).to(device)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    
    # A normalização é feita no numpy, então não precisa de device
    if cfg.policy_type in ("transformer", "lstm"):
        # Normaliza apenas a parte não sequencial
        normalizer = RunningNorm(obs_size["non_sequential"])
    else:
        normalizer = RunningNorm(obs_size)

    # Carregar modelo
    if model_path:
        try:
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # type: ignore[call-arg]
            except TypeError:
                checkpoint = torch.load(model_path, map_location=device)
            agent.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Carrega o estado do normalizer
            norm_state = checkpoint.get('normalizer_state')
            if norm_state:
                normalizer.mean = norm_state['mean']
                normalizer.M2 = norm_state['M2']
                normalizer.count = norm_state['count']
            print(f"Modelo carregado de {model_path}")
        except Exception as e:
            print(f"Não foi possível carregar o modelo. Começando do zero. Erro: {e}")

    # Buffers de armazenamento
    if obs_format == 'structured':
        seq_shape = obs_size["sequential"]
        non_seq_dim = obs_size["non_sequential"]
        seq_obs_buf = torch.zeros((cfg.episode_len, *seq_shape), dtype=torch.float32).to(device)
        non_seq_obs_buf = torch.zeros((cfg.episode_len, non_seq_dim), dtype=torch.float32).to(device)
    else:
        obs_buf = torch.zeros((cfg.episode_len, obs_size), dtype=torch.float32).to(device)
    
    act_buf = torch.zeros(cfg.episode_len, dtype=torch.int64).to(device)
    logp_buf = torch.zeros(cfg.episode_len, dtype=torch.float32).to(device)
    rew_buf = torch.zeros(cfg.episode_len, dtype=torch.float32).to(device)
    done_buf = torch.zeros(cfg.episode_len, dtype=torch.float32).to(device)
    val_buf = torch.zeros(cfg.episode_len, dtype=torch.float32).to(device)

    # Loop de treinamento
    obs = env.reset(seed=cfg.seed)
    
    for ep in range(1, cfg.episodes + 1):
        for step in range(cfg.episode_len):
            # Normaliza e converte para tensor
            if obs_format == 'structured':
                normalizer.update(obs["non_sequential"])
                obs["non_sequential"] = normalizer.normalize(obs["non_sequential"])
                
                seq_tensor = torch.tensor(obs["sequential"], dtype=torch.float32).to(device).unsqueeze(0)
                non_seq_tensor = torch.tensor(obs["non_sequential"], dtype=torch.float32).to(device).unsqueeze(0)
                obs_tensor = {"sequential": seq_tensor, "non_sequential": non_seq_tensor}
                
                seq_obs_buf[step] = seq_tensor
                non_seq_obs_buf[step] = non_seq_tensor
            else:
                normalizer.update(obs)
                obs = normalizer.normalize(obs)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                obs_buf[step] = obs_tensor

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)
            
            val_buf[step] = value.flatten()
            act_buf[step] = action.flatten()
            logp_buf[step] = log_prob.flatten()

            res = env.step(action.cpu().item())
            rew_buf[step] = torch.tensor(res.reward, dtype=torch.float32).to(device)
            done_buf[step] = torch.tensor(float(res.done), dtype=torch.float32).to(device)
            
            obs = res.obs
            if res.done:
                obs = env.reset()

        # Cálculo das vantagens (GAE)
        with torch.no_grad():
            if obs_format == 'structured':
                # Normaliza a última observação antes de passar para o agente
                obs["non_sequential"] = normalizer.normalize(obs["non_sequential"])
                next_obs_tensor = {
                    "sequential": torch.tensor(obs["sequential"], dtype=torch.float32).to(device).unsqueeze(0),
                    "non_sequential": torch.tensor(obs["non_sequential"], dtype=torch.float32).to(device).unsqueeze(0)
                }
            else:
                obs = normalizer.normalize(obs)
                next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            
            next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
            advantages = torch.zeros_like(rew_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.episode_len)):
                nextnonterminal = 1.0 - done_buf[t]
                nextvalues = val_buf[t + 1] if t < cfg.episode_len - 1 else next_value
                delta = rew_buf[t] + cfg.gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        # Otimização da política
        b_inds = np.arange(cfg.episode_len)
        for epoch in range(cfg.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.episode_len, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                if obs_format == 'structured':
                    mb_obs = {
                        "sequential": seq_obs_buf[mb_inds],
                        "non_sequential": non_seq_obs_buf[mb_inds]
                    }
                else:
                    mb_obs = obs_buf[mb_inds]

                _, new_logp, entropy, new_value = agent.get_action_and_value(mb_obs, act_buf[mb_inds])
                
                logratio = new_logp - logp_buf[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = -torch.min(mb_advantages * ratio, mb_advantages * torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)).mean()
                v_loss = 0.5 * ((new_value.view(-1) - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
                optimizer.step()

        print(f"Episódio {ep}, Recompensa Total: {rew_buf.sum().item():.2f}")

    # Salvar o modelo
    if save:
        out_dir = os.path.join("reports", "agents", "candle_pattern7_rl")
        os.makedirs(out_dir, exist_ok=True)
        model_name = f"ppo_{cfg.policy_type}_{cfg.ticker}_{cfg.interval}.pt"
        save_path = os.path.join(out_dir, model_name)
        
        torch.save({
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalizer_state': {
                'mean': normalizer.mean,
                'M2': normalizer.M2,
                'count': normalizer.count
            },
            'config': asdict(cfg)
        }, save_path)
        print(f"Modelo salvo em {save_path}")

        # Optional: run ablation and save JSON
        if run_ablation:
            try:
                from .. import ablation as abl
                policy_fn, policy_type, _ = abl.load_policy(save_path)
                env_kwargs = dict(
                    symbol=cfg.ticker,
                    interval=cfg.interval,
                    days=cfg.days,
                    episode_len=cfg.episode_len,
                    random_start=False,
                    include_mtf=cfg.include_mtf,
                    mtf_timeframes=cfg.mtf_timeframes,
                    include_regime_features=cfg.include_regime_features,
                    obs_format=("structured" if cfg.policy_type in ("transformer", "lstm") else "flat"),
                )
                baseline_env = Candle7Env(**env_kwargs)
                baseline = abl.run_eval(baseline_env, policy_fn)
                groups = ablation_groups or ["seq", "non_seq_core", "non_seq_mtf", "non_seq_regime", "pos"]
                results = {"baseline": baseline}
                for g in groups:
                    env = Candle7Env(**env_kwargs, ablation_groups=[g])
                    res = abl.run_eval(env, policy_fn)
                    results[g] = res
                out_json = os.path.join(out_dir, f"ablation_{os.path.basename(save_path)}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"Ablation salvo em {out_json}")
            except Exception as e:
                print(f"WARN: Falha ao rodar ablação automática: {e}")


if __name__ == "__main__":
    # Argumentos para manter compatibilidade com o script original
    parser = argparse.ArgumentParser(description="Train PPO agent (PyTorch) on Candle7Env")
    parser.add_argument(
        "--policy",
        type=str,
        default="mlp",
        choices=["mlp", "transformer", "lstm"],
        help="Arquitetura da política (mlp | transformer | lstm).",
    )
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=365) # Menor para testes iniciais
    parser.add_argument("--episodes", type=int, default=100) # Menor para testes iniciais
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--episode_len", type=int, default=2048)
    parser.add_argument("--long_only", action="store_true")
    parser.add_argument("--model", type=str, default=None, help="Caminho para o modelo .pt a ser carregado")
    # PPO hyperparams
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    # Multi-timeframe
    parser.add_argument("--include_mtf", action="store_true", help="Inclui features multi-timeframe (1h,4h)")
    parser.add_argument(
        "--mtf_timeframes",
        type=str,
        default="1h,4h",
        help="Lista separada por vírgula de timeframes para MTF (ex: '1h,4h')",
    )
    # Regimes
    parser.add_argument("--include_regimes", action="store_true", help="Inclui one-hot de regimes (tendência/volatilidade)")
    parser.add_argument("--regime_adx_threshold", type=float, default=25.0)
    parser.add_argument("--regime_vol_multiplier", type=float, default=1.2)
    # Env shaping / execution knobs
    parser.add_argument("--gate_on_heuristic", action="store_true", help="Permite abrir somente quando heurística concorda")
    parser.add_argument("--idle_penalty", type=float, default=0.0)
    parser.add_argument("--idle_grace", type=int, default=0)
    parser.add_argument("--idle_ramp", type=float, default=0.0)
    parser.add_argument("--m2m_weight", type=float, default=0.05)
    parser.add_argument("--reward_atr_norm", action="store_true")
    parser.add_argument("--atr_period", type=int, default=14)
    parser.add_argument("--min_hold_bars", type=int, default=0)
    parser.add_argument("--reopen_cooldown_bars", type=int, default=0)
    parser.add_argument("--switch_penalty", type=float, default=0.0)
    parser.add_argument("--switch_window_bars", type=int, default=5)
    parser.add_argument("--action_cost_open", type=float, default=0.0)
    parser.add_argument("--action_cost_close", type=float, default=0.0)
    parser.add_argument("--invalid_action_penalty", type=float, default=0.0)
    parser.add_argument("--fee_rate", type=float, default=0.001)
    parser.add_argument("--slippage_bps", type=float, default=0.0)
    # Ablation
    parser.add_argument("--ablation", action="store_true", help="Roda ablação automática ao fim do treino")
    parser.add_argument(
        "--ablation_groups",
        type=str,
        default="seq,non_seq_core,non_seq_mtf,non_seq_regime,pos",
        help="Grupos de ablação separados por vírgula",
    )
    # Adicione outros argumentos do Candle7RlConfig conforme necessário
    
    args = parser.parse_args()

    cfg = Candle7RlConfig(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        long_only=bool(args.long_only),
        episodes=args.episodes,
        hidden_size=args.hidden,
        learning_rate=args.lr,
        episode_len=args.episode_len,
        # Hiperparâmetros do PPO
        ppo_epochs=4,
        minibatch_size=64,
        gamma=0.99,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        grad_clip=args.grad_clip,
        policy_type=args.policy, # Adicionado para o config
        include_mtf=bool(args.include_mtf),
        mtf_timeframes=tuple([s.strip() for s in args.mtf_timeframes.split(',') if s.strip()]),
        include_regime_features=bool(args.include_regimes),
        regime_adx_threshold=args.regime_adx_threshold,
        regime_vol_multiplier=args.regime_vol_multiplier,
        # Env shaping
        gate_on_heuristic=bool(args.gate_on_heuristic),
        idle_penalty=args.idle_penalty,
        idle_grace_bars=args.idle_grace,
        idle_ramp=args.idle_ramp,
        m2m_weight=args.m2m_weight,
        reward_atr_norm=bool(args.reward_atr_norm),
        atr_period=args.atr_period,
        min_hold_bars=args.min_hold_bars,
        reopen_cooldown_bars=args.reopen_cooldown_bars,
        switch_penalty=args.switch_penalty,
        switch_window_bars=args.switch_window_bars,
        action_cost_open=args.action_cost_open,
        action_cost_close=args.action_cost_close,
        invalid_action_penalty=args.invalid_action_penalty,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
    )
    
    groups = [s.strip() for s in args.ablation_groups.split(',') if s.strip()]
    ppo_train_torch(cfg, model_path=args.model, run_ablation=bool(args.ablation), ablation_groups=groups)
