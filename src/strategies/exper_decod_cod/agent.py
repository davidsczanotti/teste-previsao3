import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Agent:
    """
    O Agente que irá interagir com o ambiente de negociação.
    Agora com um cérebro (rede neural) e memória, e estratégia epsilon-greedy.
    """
    def __init__(self, state_size, action_size, buffer_size=2000, learning_rate=0.001, gamma=0.95, batch_size=32, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa o Agente.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.learning_rate = learning_rate
        self.gamma = gamma  # Fator de desconto para recompensas futuras
        self.batch_size = batch_size # Tamanho do minibatch para treinamento
        
        self.model = self._build_model()
        
        # Parâmetros para a estratégia epsilon-greedy
        self.epsilon = epsilon  # Taxa de exploração inicial
        self.epsilon_decay = epsilon_decay # Taxa de decaimento da exploração
        self.epsilon_min = epsilon_min # Taxa mínima de exploração
        
        random.seed(42)

    def _build_model(self):
        """
        Constrói a rede neural que funcionará como o cérebro do agente.
        """
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # Linear, pois queremos valores Q, não probabilidades
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Armazena uma experiência na memória do agente.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Dado um estado, retorna uma ação usando a estratégia epsilon-greedy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape o estado para que o modelo possa processá-lo (adiciona dimensão de batch)
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0) # verbose=0 para não imprimir cada predição
        return np.argmax(q_values[0]) # Retorna a ação com o maior valor Q

    def learn(self):
        """
        Treina a rede neural do agente usando um minibatch de experiências da memória.
        """
        if len(self.memory) < self.batch_size:
            return # Não há experiências suficientes para treinar

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])

        # Prever os valores Q para os estados atuais e próximos estados
        q_values_current = self.model.predict(states, verbose=0)
        q_values_next = self.model.predict(next_states, verbose=0)

        X = [] # Estados para treinamento
        y = [] # Alvos Q para treinamento

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target_q = reward
            if not done:
                # Equação de Bellman: Recompensa + Fator de Desconto * max(Q(s', a'))
                target_q = reward + self.gamma * np.amax(q_values_next[i])
            
            # O alvo Q para a ação tomada é o target_q calculado
            # Para as outras ações, mantemos os valores Q previstos atualmente
            target_f = q_values_current[i].copy()
            target_f[action] = target_q
            
            X.append(state)
            y.append(target_f)
        
        # Treina o modelo com os estados e os alvos Q calculados
        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
        
        # Decai o epsilon após cada passo de aprendizado
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
