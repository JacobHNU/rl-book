import numpy as np
from gym import spaces
from gym.utils import seeding

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King =10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

# 发牌函数
def draw_card(np_random):
    return int(np_random.choice(deck))

# 首轮发牌函数
def draw_hand(np_random):
    return [draw_card(np_random),draw_card(np_random)]

# 判断是否有可用Ace
def uasable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

# 计算手中牌总点数
def sum_hand()