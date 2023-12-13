from game import Game
from agent import Agent

enviroment = Game()
agent = Agent(enviroment)

agent.play()