import random, sys
from copy import deepcopy
from typing import Literal
from abc import ABC, abstractmethod
from exceptions import AgentException
from heuristics import simple_score, advanced_score
from connect4 import Connect4
import math

class Agent(ABC):
    def __init__(self, my_token="o", **kwargs):
        self.my_token = my_token

    @abstractmethod
    def decide(self, connect4):
        pass

    def __str__(self):
        return f"{self.my_token} ({self.__class__.__name__})"


class RandomAgent(Agent):
    def __init__(self, my_token="o", **kwargs):
        super().__init__(my_token, **kwargs)

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")
        return random.choice(connect4.possible_drops())


class MinMaxAgent(Agent):
    def __init__(
        self, my_token="o", depth=4, heuristic: Literal["simple", "advanced"] = "simple"
    ):
        super().__init__(my_token)
        self.depth = depth
        self.heuristic = heuristic
        self.heuristic_fun = simple_score if heuristic == "simple" else advanced_score

    def __str__(self):
        return f"{self.my_token} ({self.__class__.__name__}+{self.heuristic})"

    def decide(self, connect4: Connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")

        best_move, best_score = self.minmax(connect4, depth=self.depth)
        return best_move

    def minmax(self, connect4: Connect4, depth=4, maximizing=True):
        best_move = None
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return best_move, 10000
            elif connect4.wins is None:
                return best_move, 0
            else:
                return best_move, -10000

        if depth == 0:
            return best_move, self.heuristic_fun(connect4, self.my_token)

        if maximizing:
            best_score = -math.inf
            for move in connect4.possible_drops():
                new_game = deepcopy(connect4)
                new_game.drop_token(move)
                _, score = self.minmax(new_game, depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move, best_score
        else:
            best_score = math.inf
            for move in connect4.possible_drops():
                new_game = deepcopy(connect4)
                new_game.drop_token(move)
                _, score = self.minmax(new_game, depth - 1, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_move, best_score
        


class AlphaBetaAgent(MinMaxAgent):
    def __init__(
        self, my_token="o", depth=4, heuristic: Literal["simple", "advanced"] = "simple"
    ):
        super().__init__(my_token)
        self.depth = depth
        self.heuristic = heuristic
        self.heuristic_fun = simple_score if heuristic == "simple" else advanced_score

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")

        best_move, best_score = self.alphabeta(connect4, depth=self.depth)
        return best_move

    def alphabeta(
        self,
        connect4: Connect4,
        depth=4,
        maximizing=True,
        alpha=-sys.maxsize,
        beta=sys.maxsize,
    ):
        # TODO
        best_move = None
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return best_move, 10000  
            elif connect4.wins is None:
                return best_move, 0  
            else:
                return best_move, -10000  
        if depth == 0:
            return best_move, self.heuristic_fun(connect4, self.my_token)
        if maximizing:
            best_score = -math.inf
            for move in connect4.possible_drops():
                new_game = deepcopy(connect4)
                new_game.drop_token(move)
                _, score = self.alphabeta(new_game, depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score = score
                    best_move = move  
                alpha = max(alpha, best_score)
                if score >= beta:
                    break  
            return best_move, best_score
        else:
            best_score = math.inf
            for move in connect4.possible_drops():
                new_game = deepcopy(connect4)
                new_game.drop_token(move)
                _, score = self.alphabeta(new_game, depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score = score
                    best_move = move  
                beta = min(beta, best_score)
                if score >= beta:
                    break 
            return best_move, best_score
        