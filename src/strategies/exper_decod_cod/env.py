import numpy as np
import pandas as pd

class TradingEnv:
    """
    Ambiente de negociação para o agente de Aprendizagem por Reforço.

    Atributos:
        df (pd.DataFrame): DataFrame com os dados históricos de mercado (OHLCV).
        initial_balance (float): O saldo inicial da conta em USD.
        lot_size (float): A quantidade de BTC a ser negociada em cada operação.
    """
    def __init__(self, df, initial_balance=1000, lot_size=0.1):
        """
        Inicializa o ambiente.

        Args:
            df (pd.DataFrame): DataFrame com os dados de mercado.
            initial_balance (float): Saldo inicial.
            lot_size (float): Tamanho do lote para cada negociação.
        """
        self.df = df
        self.initial_balance = initial_balance
        self.lot_size = lot_size

        # Espaço de ação: 0 (Aguardar), 1 (Comprar), 2 (Vender)
        self.action_space = np.array([0, 1, 2])

        self.reset()

    def reset(self):
        """
        Reseta o ambiente para o estado inicial.
        Retorna a primeira observação.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.trades = []
        
        return self._next_observation()

    def _next_observation(self):
        """
        Retorna a observação do estado atual do mercado, incluindo o preço de fechamento,
        o saldo atual e a quantidade de BTC em carteira.
        """
        # Retorna um array com o preço de fechamento atual, saldo e BTC em carteira
        obs = np.array([
            self.df.loc[self.current_step, 'close'],
            self.balance,
            self.btc_held
        ])
        return obs

    def step(self, action):
        """
        Executa uma etapa no ambiente com base na ação do agente.

        Args:
            action (int): A ação a ser tomada (0, 1 ou 2).

        Returns:
            tuple: Uma tupla contendo (observação, recompensa, done, info).
        """
        self.last_net_worth = self.net_worth
        self.current_step += 1
        
        current_price = self.df.loc[self.current_step, 'close']
        
        # Ação de Comprar
        if action == 1:
            if self.balance > self.lot_size * current_price:
                self.balance -= self.lot_size * current_price
                self.btc_held += self.lot_size
        
        # Ação de Vender
        elif action == 2:
            if self.btc_held > 0:
                self.balance += self.lot_size * current_price
                self.btc_held -= self.lot_size

        # Atualiza o patrimônio líquido
        self.net_worth = self.balance + (self.btc_held * current_price)

        # Calcula a recompensa como a variação do patrimônio
        reward = self.net_worth - self.last_net_worth
        
        # Verifica se o episódio terminou
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # info pode ser usado para debug
        info = {}

        return self._next_observation(), reward, done, info

