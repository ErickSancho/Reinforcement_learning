import gym
from gym.spaces.utils import flatdim
from random import randint
import numpy as np


class RLAgent:
    
    def __init__(self, env: gym.Env, policy=None, discount_factor = 0.1, learning_rate = 0.1, ratio_explotacion = 0.9):

        self.env = env

        self.steps = 100

        self.action_space = np.arange(flatdim(env.action_space))

        self.q_table_shape = (self.steps, self.steps, 3)

        # Creamos la tabla de politicas
        if policy is not None:
            self._q_table = policy
        else:
            self._q_table = np.random.uniform(0,1,self.q_table_shape)
            
        # print(env.action_space)

        # print("Q table shape", self._q_table.shape)
        
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ratio_explotacion = ratio_explotacion

    def get_next_step(self, state, env):

        state = self.discrete(state)
        
        # Damos un paso aleatorio...
        next_step = env.action_space.sample()
        
        # o tomaremos el mejor paso...
        if np.random.uniform() <= self.ratio_explotacion:
            # tomar el maximo
            idx_action = np.argmax(self._q_table[state[0],state[1]])
            
            next_step = self.action_space[idx_action]

        # print(self._q_table[state[0],state[1]])

        return next_step


    # actualizamos las politicas con las recompensas obtenidas  
    def update(self, old_state, action_taken, reward_action_taken, new_state, reached_end):
        old_state = self.discrete(old_state)
        new_state = self.discrete(new_state)

        # valor de Q para el estado anterior y action tomada
        actual_q_value = self._q_table[old_state[0], old_state[1], action_taken]

        # valor del maximo Q para el estado actual
        future_max_q_value = self._q_table[new_state[0], new_state[1]].max()

        if reached_end:
            future_max_q_value = reward_action_taken #maximum reward

        self._q_table[old_state[0], old_state[1], action_taken] = actual_q_value + self.learning_rate*(reward_action_taken + self.discount_factor*future_max_q_value -actual_q_value)
    
            
    def get_policy(self):
        return self._q_table
    
    def discrete(self, val):
        temp = np.floor(((val-self.env.observation_space.low)/(self.env.observation_space.high-self.env.observation_space.low))*self.steps)
        return tuple(temp.astype(np.int32))

    def save_model(self, file='model.npy'):
        np.save(file, self._q_table)

    def load_model(self, file='model.npy'):
        self._q_table = np.load(file)


discount_factor = 0.95
learning_rate = 0.15
ratio_explotacion = 0.8


env = gym.make('MountainCar-v0')
agent = RLAgent(env, discount_factor=discount_factor, learning_rate=learning_rate, ratio_explotacion=ratio_explotacion)



print("Starting trainning!")

rounds = 50000
max_points= -200
first_max_reached = 0
total_rw=0
steps=[]
rewards=[]

for played_games in range(0, rounds):
    env.seed(np.random.randint(10000))
    state = env.reset()
    reward, done = None, None
    
    itera=0
    total_reward=0
    while (done != True) and (itera < 500):
        old_state = np.array(state,  dtype=np.float32)
        next_action = agent.get_next_step(old_state, env)
        state, reward, done, _ = env.step(next_action)
        if rounds > 1:
            agent.update(old_state, next_action, reward, state, done)
        itera+=1
        total_reward+=reward

    
    steps.append(itera)
    rewards.append(total_reward)

    total_rw+=total_reward
    if total_reward > max_points:
        max_points=total_reward
        first_max_reached = played_games
    
    if played_games %500==0 and played_games >1:
        print("-- Partidas[", played_games, "] Avg.Puntos[", int(total_rw/played_games),"]  AVG Steps[", int(np.array(steps).mean()), "] Max Score[", max_points,"]")

    env.close()
    


import matplotlib.pyplot as plt

plt.plot(range(0,len(rewards)), rewards)
plt.xlabel("steps")
plt.ylabel("reward")
plt.show()


print("Saving model!")
agent.save_model()

print("\nTesting!")
for played_games in range(0, 10):
    state = env.reset()
    reward, done = None, None
    
    itera=0
    total_reward=0
    while (done != True) and (itera < 500):
        old_state = np.array(state,  dtype=np.float32)
        next_action = agent.get_next_step(state, env)
        state, reward, done, _ = env.step(next_action)
        itera+=1

        total_reward+=reward

        # env.render()

    print(f"Test: {played_games+1}, score: {total_reward}, steps: {itera}")


env.close()