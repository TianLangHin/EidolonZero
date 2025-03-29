import chess
import random

from typing import List 

class RandomBaselinePlayer:
    '''
    The `RandomBaselinePlayer` is a class that can be instantiated
    to randomly and uniformly choose from a certain move list.
    It must be given a seed upon instantiation, and every query to this resource
    will use the next random number in the stream as defined by this seed.
    It will not affect the global scope outside of
    temporarily overwriting the global random state during a move choice.
    '''
    def __init__(self, seed: int) -> 'RandomBaselinePlayer':
        # Remember the global random state to put this back in later
        previous_random_state = random.getstate()
        # Seed the environment with the given seed,
        # and record the random number state internally.
        random.seed(seed)
        self.random_state = random.getstate()
        # Finally, restore the global random state.
        random.setstate(previous_random_state)
    def choose_move(self, move_list: List[chess.Move]) -> chess.Move:
        # Remember the global random state to put this back later
        previous_random_state = random.getstate()
        # Continue from our previously stored state
        random.setstate(self.random_state)
        move = random.choice(move_list)
        # Store the state after the RNG has been consumed
        self.random_state = random.getstate()
        # Restore the global random state.
        random.setstate(previous_random_state)
        return move

