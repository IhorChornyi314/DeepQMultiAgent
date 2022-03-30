from DeepQAgent import Agent
from gym import Env
from tensorflow import keras
import json


class MultiAgent:
    def __init__(self, agents: int, env: Env, model: keras.Model, configfile='config.json', render=False):
        configs = json.load(open(configfile))
        self._agent_number = agents
        self._render = render
        self._env = env
        self._agents = [Agent(keras.models.clone_model(model), env, configfile)]
        self._train_episodes = configs.get('train_episodes') or 300

        self._episode = 0

    def train_agents(self):
        while self._episode < self._train_episodes:
            observation = self._env.reset()
            done_all = [False] * self._agent_number
            while not all(done_all):
                for i, agent in enumerate(self._agents):
                    if self._render:
                        self._env.render()

                    action = agent.choose_action(observation)

                    new_observation, reward, done, info = self._env.step(action)
                    agent.register_new_observation(observation, action, new_observation, reward, done)
                    observation = new_observation
                    done_all[i] = done
                    if all(done_all):
                        break

        self._env.close()

