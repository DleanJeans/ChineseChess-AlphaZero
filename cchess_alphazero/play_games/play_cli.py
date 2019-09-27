from collections import defaultdict
from logging import getLogger

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

from termcolor import colored
from copy import copy

logger = getLogger(__name__)

def start(config: Config, human_move_first=True):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    play = PlayWithHuman(config)
    play.start(human_move_first)

class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.human_move_first = True

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)
        self.human_move_first = human_first
        self.env.board.upside_down = human_first

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()

        while not self.env.board.is_end():
            if human_first == self.env.red_to_move:
                self.env.board.calc_chessmans_moving_list()
                is_correct_chessman = False
                is_correct_position = False
                chessman = None
                while not is_correct_chessman:
                    title = "Enter piece position: "
                    input_chessman_pos = input(title)
                    a = map(int, input_chessman_pos)
                    mx = my = None
                    if len(input_chessman_pos) == 2:
                        x, y = a
                    elif len(input_chessman_pos) == 4:
                        x, y, mx, my = a
                    
                    chessman = self.env.board.chessmans[x][y]
                    if chessman != None and chessman.is_red == self.env.board.is_red_turn:
                        piece_name = colored(f'{x}{chessman.name_cn}{y}', chessman.color)

                        print(f"Holding piece {piece_name} ", end="")
                        if not chessman.moving_list:
                            print("with no possible moves.")
                            continue
                        
                        print("with possible moves: ", end="")
                        print(" ".join(map(str, chessman.moving_list)))
                        
                        is_correct_chessman = True
                    else:
                        print(f"No piece found at position {x} {y}")
                while not is_correct_position:
                    if len(chessman.moving_list) == 1:
                        point = input_chessman_pos = chessman.moving_list[0]
                        print(f"Auto moved to ({point.x} {point.y})")
                        x, y = point.x, point.y
                    elif mx and my:
                        x, y = mx, my
                    else:
                        self.env.board.print_to_cl(holding_chessman=chessman)
                        title = "Enter move position: "
                        input_chessman_pos = input(title)
                        x, y = map(int, input_chessman_pos)

                    self.env.board.recent_position = copy(chessman.position)
                    self.env.board.recent_chessman = chessman

                    is_correct_position = chessman.move(x, y)
                    if is_correct_position:
                        self.env.board.print_to_cl()
                        self.env.board.clear_chessmans_moving_list()
            else:
                wait_text = "Waiting for AI move..."
                print(wait_text)
                action, policy = self.ai.action(self.env.get_state(), self.env.num_halfmoves)
                print('\b' * (len(wait_text) + 1), end="")
                if not self.env.red_to_move:
                    action = flip_move(action)
                if action is None:
                    print("AI surrendered!")
                    break
                
                self.env.step(action)
                print(f"AI decided to move {action}")
                self.env.board.print_to_cl()

        self.ai.close()
        print(f"The winner is {self.env.board.winner} !!!")
        self.env.board.print_record()
