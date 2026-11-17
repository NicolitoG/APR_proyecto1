# q_learning_agent.py
from __future__ import annotations
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


class QLearningAgent:
    """
    Agente tabular de Q-Learning (Off-Policy).
    Maneja internamente la traducción entre acciones binarias físicas y los 
    índices de la tabla Q, la política epsilon-greedy y la ecuación de Bellman.
    """

    # =====================================================================
    # 1. INICIALIZACIÓN Y ESTRUCTURA DE MEMORIA
    # =====================================================================
    def __init__(
        self,
        num_satellites: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.num_satellites = num_satellites
        self.num_actions = 2 ** num_satellites  # 2^S combinaciones posibles

        # Hiperparámetros de aprendizaje
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Generador aleatorio independiente para reproducibilidad
        self.rng = np.random.default_rng(seed)

        # Tabla Q: Diccionario que mapea un estado (tupla) a un arreglo de 2^S valores.
        # Si un estado no existe, se inicializa automáticamente en ceros.
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=float)
        )

    # =====================================================================
    # 2. TRADUCTORES DE ESPACIO DE ACCIONES (Físico <-> Índice Q)
    # =====================================================================
    def _idx_to_action_array(self, action_idx: int) -> np.ndarray:
        """Convierte un índice de columna Q (ej. 3) en vector binario (ej. [1, 1])."""
        return np.array(
            [(action_idx >> s) & 1 for s in range(self.num_satellites)], 
            dtype=np.int8
        )

    def _action_array_to_idx(self, action_array: np.ndarray) -> int:
        """Convierte un vector binario físico (ej. [1, 0]) al índice de columna Q."""
        return sum(int(bit) * (2 ** i) for i, bit in enumerate(action_array))

    # =====================================================================
    # 3. SELECTOR DE ACCIÓN (Estrategia b y Estrategia π)
    # =====================================================================
    def _get_best_action_idx(self, state: tuple) -> int:
        """
        Estrategia π (Golosa): Retorna el índice de la acción con mayor valor Q.
        Si hay múltiples acciones con el mismo valor máximo (empate), elige una al azar.
        """
        q_values = self.q_table[state]
        max_val = np.max(q_values)
        best_indices = np.where(q_values == max_val)[0]
        return self.rng.choice(best_indices)

    def select_action(self, state: tuple, training: bool = True) -> np.ndarray:
        """
        Estrategia b (Comportamiento): epsilon-greedy si está en entrenamiento.
        Retorna el vector binario listo para ser enviado a env.step().
        """
        if training and self.rng.random() < self.epsilon:
            # Exploración: Elegir índice al azar
            action_idx = self.rng.integers(0, self.num_actions)
        else:
            # Explotación: Elegir el mejor índice
            action_idx = self._get_best_action_idx(state)
        
        return self._idx_to_action_array(action_idx)

    # =====================================================================
    # 4. MOTOR DE ACTUALIZACIÓN DE BELLMAN (Off-Policy)
    # =====================================================================
    def update(
        self, 
        state: tuple, 
        action_array: np.ndarray, 
        reward: float, 
        next_state: tuple
    ) -> None:
        """
        Actualiza el valor Q(s, a) usando el Error de Diferencia Temporal (TD Error).
        Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s', a') - Q(s,a)]
        """
        action_idx = self._action_array_to_idx(action_array)
        
        # Valor actual en la tabla
        current_q = self.q_table[state][action_idx]
        
        # El mejor valor posible en el siguiente estado (estrategia golosa implícita)
        max_next_q = np.max(self.q_table[next_state])
        
        # Cálculo del objetivo TD y el error
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        # Actualización
        self.q_table[state][action_idx] += self.alpha * td_error

    # =====================================================================
    # 5. GESTIÓN DEL CICLO DE VIDA (Decaimiento de Exploración)
    # =====================================================================
    def decay_epsilon(self) -> None:
        """Reduce gradualmente la tasa de exploración al final de cada episodio."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # Asegurar que no caiga por debajo del mínimo
            self.epsilon = max(self.epsilon_min, self.epsilon)

    # =====================================================================
    # 6. PERSISTENCIA (Guardado y Carga en Disco)
    # =====================================================================
    def save(self, filepath: Path | str) -> None:
        """Serializa la tabla Q a un archivo usando Pickle."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Convertimos el defaultdict a un dict normal antes de guardar
        with filepath.open("wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath: Path | str) -> None:
        """Carga una tabla Q previamente entrenada desde el disco."""
        filepath = Path(filepath)
        with filepath.open("rb") as f:
            loaded_table = pickle.load(f)
            # Actualizamos el defaultdict actual con los datos cargados
            self.q_table.update(loaded_table)