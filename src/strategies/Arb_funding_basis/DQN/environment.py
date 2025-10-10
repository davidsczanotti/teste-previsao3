import numpy as np
import pandas as pd


class ArbFundingEnv:
    """
    Ambiente de Aprendizagem por Reforço para a estratégia de arbitragem de Funding Rate.
    """

    def __init__(self, df: pd.DataFrame, initial_capital=1000.0, fee_rate=0.00075):
        """
        Args:
            df (pd.DataFrame): DataFrame com colunas 'fundingRate', 'spot_price'.
            initial_capital (float): Capital inicial.
            fee_rate (float): Taxa de transação (ex: 0.075% = 0.00075).
        """
        self.df = df
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

        # O espaço de estado tem 3 dimensões: funding atual, funding anterior, posição atual
        self.observation_space_dim = 3
        # 3 ações: 0 (Manter), 1 (Abrir Posição), 2 (Fechar Posição)
        self.action_space_dim = 3

        self.reset()

    def reset(self):
        """Reseta o ambiente para um novo episódio."""
        self.current_step = 1  # Começamos no segundo candle para ter um 'anterior'
        self.capital = self.initial_capital
        self.position = 0  # 0 para neutro, 1 para em posição
        self.position_size_asset = 0.0
        self.entry_price = 0.0

        return self._get_observation()

    def _get_observation(self):
        """Retorna o estado atual do ambiente."""
        if self.current_step >= len(self.df):
            # Retorna um estado terminal se o episódio acabar
            return np.array([0, 0, self.position])

        funding_current = self.df.loc[self.current_step, "fundingRate"]
        funding_previous = self.df.loc[self.current_step - 1, "fundingRate"]

        return np.array([funding_current, funding_previous, self.position])

    def step(self, action: int):
        """Executa uma ação e avança um passo no tempo."""
        if self.current_step >= len(self.df) - 1:
            # Último passo, força o fechamento se estiver em posição
            if self.position == 1:
                action = 2
            else:
                # Se não, o episódio simplesmente acaba
                return self._get_observation(), 0, True, {}

        reward = 0
        done = False

        current_price = self.df.loc[self.current_step, "spot_price"]
        funding_rate = self.df.loc[self.current_step, "fundingRate"]

        # Lógica de Ações
        if action == 1 and self.position == 0:  # Abrir Posição
            self.position = 1
            self.entry_price = current_price
            self.position_size_asset = self.capital / current_price
            # Custo de abrir posição spot + perpétua
            fee = self.capital * self.fee_rate * 2
            self.capital -= fee
            reward -= fee

        elif action == 2 and self.position == 1:  # Fechar Posição
            # Custo de fechar posição spot + perpétua
            fee = self.capital * self.fee_rate * 2
            self.capital -= fee
            reward -= fee
            self.position = 0
            self.position_size_asset = 0.0
            self.entry_price = 0.0

        elif action == 0:  # Manter
            pass

        else:  # Ação inválida (ex: tentar abrir posição já em posição)
            reward = -10  # Penalidade por ação ilegal

        # Lógica de Recompensa (Funding)
        if self.position == 1:
            # O funding é pago sobre o valor nocional da posição
            notional_value = self.position_size_asset * current_price
            funding_payment = notional_value * funding_rate
            self.capital += funding_payment
            reward += funding_payment

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        # PnL do episódio para informação
        info = {"capital": self.capital, "pnl": self.capital - self.initial_capital}

        return self._get_observation(), reward, done, info

    def render(self):
        """Renderiza o estado atual (opcional)."""
        pnl = self.capital - self.initial_capital
        pos_str = "EM POSIÇÃO" if self.position == 1 else "NEUTRO"
        print(f"Passo: {self.current_step}, Capital: ${self.capital:.2f}, " f"P&L: ${pnl:.2f}, Posição: {pos_str}")
