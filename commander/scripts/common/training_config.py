class TrainingConfig:
    def __init__(self):
        self.gamma = 0.99  # Discount factor
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005  # Learning rate
        self.batch_size = 64
        self.target_update = 10  # How often to update the target network
        self.memory_capacity = 50000
        self.max_gradient_norm = float('inf')