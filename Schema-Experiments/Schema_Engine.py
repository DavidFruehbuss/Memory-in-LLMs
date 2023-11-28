# For every schema (reward pattern) generate automatic feedback

import random

class LocationPattern:
    '''
    This class defines a periodic reward location update, with an update 
    only taking place when the previous reward has been found.
    '''
    def __init__(self, reward_location_pattern=['A', 'B', 'C', 'D']):
        self.locations = reward_location_pattern
        self.current_index = 0

    def get_current_target(self):
        return self.locations[self.current_index]

    def provide_feedback(self, agent_action):
        if agent_action == self.get_current_target():
            self._move_to_next_location()
            return 1
        else:
            return 0

    def _move_to_next_location(self):
        self.current_index = (self.current_index + 1) % len(self.locations)

def generate_episode(pattern, num_actions):
    actions = random.choices(pattern.locations, k=num_actions)
    results = []

    for action in actions:
        feedback = pattern.provide_feedback(action)
        results.append((action, feedback))

    return results

# Example Usage
# pattern = LocationPattern()
# episode = generate_episode(pattern, 10)
# for action, feedback in episode:
#     print(f"Action: {action}, Feedback: {feedback}")
