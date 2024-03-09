import random
from copy import deepcopy

ids = ["111111111", "222222222"]

import numpy as np
import itertools
from collections import Counter
import json
import copy
from itertools import product


class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = initial
        self.map = initial['map']
        self.chosen_pirate=None
        self.distances = distance_matrix(self.map)
        self.turn=initial["turns to go"]
        self.base = self.find_base(initial['map'])
        # self.shortestPathToTreasure = self.ShortestPath()
        all_states = self.build_all_states()
        # print(all_states)
        self.policy = self.policy_iteration(all_states)

    def policy_iteration(self, all_states):
        policy = {}
        value_star = {}
        actions_dict = {}
        next_state = {}
        for state in all_states:
            value_star[state] = {'t-1': 0, 't': 0}
            policy[state] = {}
            actions_dict[state] = self.actions(json.loads(state))
            treasure_options = self.get_treasure_options(state)
            marine_options = self.get_marine_options(state)
            next_state[state] = {}
            for action in actions_dict[state]:
                # json_action = json.dumps(action)
                next_state[state][action] = self.next_states(state, action, treasure_options, marine_options)
        for t in range(self.initial['turns to go']):
            for state in all_states:
                actions = actions_dict[state]
                Q = []
                for action in actions:
                    # json_action = json.dumps(action)
                    v = self.reward(action, state) + self.value_from_action(next_state[state][action], value_star)
                    Q.append([action, v])
                max_action = max(Q, key=lambda x: x[1])
                value_star[state]['t'] = max_action[1]
                policy[state][t + 1] = max_action[0]

            for state in all_states:
                value_star[state]['t-1'] = value_star[state]['t']
        return policy

    def get_marine_options(self, state):
        curr_state = json.loads(state)
        marine_options = {}

        for marine_name, marine_info in curr_state['marine_ships'].items():
            path = marine_info['path']
            index = marine_info['index']
            options = []
            if len(path) > 1:
                # Calculate probabilities based on the marine's position in the path
                if index == 0:  # If the marine is at the start of the path
                    options.append((index, 0.5))  # Stay at the same index
                    options.append((index + 1, 0.5))  # Move forward
                elif index == len(path) - 1:  # If the marine is at the end of the path
                    options.append((index, 0.5))  # Stay at the same index
                    options.append((index - 1, 0.5))  # Move backward
                else:  # If the marine is somewhere in the middle of the path
                    options.append((index + 1, 1 / 3))  # Move forward
                    options.append((index, 1 / 3))  # Stay at the same index
                    options.append((index - 1, 1 / 3))  # Move backw
            else:
                options.append((index, 1))
            marine_options[marine_name] = options

        # Generate all possible combinations of marine movements
        all_combinations = self.dict_product(marine_options)

        # Calculate the probability for each combination
        final_options = []
        for combination in all_combinations:
            prob = 1
            new_state = {'names': [], 'prob': 0}
            for marine_name, (location, p) in combination.items():
                new_state['names'].append([marine_name, location])
                prob *= p

            new_state['prob'] = prob
            final_options.append(new_state)

        return final_options

    def get_treasure_options(self, state):  # TODO: check whetherthe prob is 0 or 1

        curr_state = json.loads(state)
        treasure_dict = {}
        for treasure_name, treasure_info in curr_state['treasures'].items():
            prob_list = []
            # Probability for the treasure to move
            if len(treasure_info['possible_locations']) > 1:
                p_move = self.initial['treasures'][treasure_name]['prob_change_location']
                # Probability for the treasure to stay
                p_stay = 1 - p_move
                # Add the current location with the probability of staying
                # prob_list.append([treasure_info['location'], p_stay +])
                # Add possible new locations with the probability of moving
                for new_location in self.initial['treasures'][treasure_name]['possible_locations']:
                    if new_location == tuple(treasure_info['location']):
                        prob_list.append([new_location, p_stay + p_move / (
                            len(self.initial['treasures'][treasure_name]['possible_locations']))])
                    else:
                        prob_list.append([new_location, p_move / (
                            len(self.initial['treasures'][treasure_name]['possible_locations']))])
                    # else:  # Adjust probability for staying at the same location
                    #     prob_list[-1][1] += p_move / len(self.initial['treasures'][treasure_name]['possible_locations'])
                treasure_dict[treasure_name] = prob_list
            else:
                treasure_dict[treasure_name] = [[treasure_info["location"], 1]]
        options = []
        x = self.dict_product(treasure_dict)
        for item in x:
            prob = 1
            prob_dict = {}
            name_list = []
            for name, info in item.items():
                name_list.append([name, info[0]])
                prob *= info[1]
            prob_dict['names'] = name_list
            prob_dict['prob'] = prob
            options.append(prob_dict)

        return options

    def act(self, state):
        turns_to_go = state['turns to go']
        if len(state['pirate_ships'])>1:
            curr_state = {'pirate_ships': {self.chosen_pirate: state['pirate_ships'][self.chosen_pirate]},
                           'treasures': state['treasures'],
                           'marine_ships': state['marine_ships']}
        else:
            curr_state = {'marine_ships': state['marine_ships'], 'pirate_ships': state['pirate_ships'],
                      'treasures': state['treasures']}

        state_json = json.dumps(curr_state, sort_keys=True)
        action = self.policy[state_json][turns_to_go]

        if action[0] == 'reset':
            return 'reset'
        if action[0] == 'terminate':
            return 'terminate'
        if len(self.initial['pirate_ships']) > 1:
            for pirate_name, pirate_info in self.initial["pirate_ships"].items():
                if pirate_name != self.chosen_pirate:
                    other_action = [ list(act) for act in action]
                    other_action[0][1]=pirate_name
                    other_action = [tuple(act) for act in other_action]
                    other_action=tuple(other_action)
                    action+=other_action
        self.turn-=1
        return action

    def reward(self, action, state):
        # todo: find a way to go through marine without reward in a shorter way to treasure so the pirate will have enoght turns to bring more treasure home
        curr_state = json.loads(state)

        v = 0

        if action[0] == 'reset':
            return -500

        if action[0] == 'terminate':
            return -1000000
        for act in action:

            # if act[0] == 'sail':
            #     closest_treasure = min(self.shortestPathToTreasure[act[1]],
            #                            key=lambda treasure_name: self.shortestPathToTreasure[act[1]][treasure_name])
            #
            #     if curr_state['pirate_ships'][act[1]]['capacity'] == 2:
            #         min_location = min(
            #             self.get_adjacent_sea_cells(curr_state["treasures"][closest_treasure]["location"]),
            #             key=lambda cell: self.distances[tuple(curr_state['pirate_ships'][act[1]]['location'])][cell]
            #         )
            #         # print(min_location)
            #         if self.distances[tuple(curr_state['pirate_ships'][act[1]]['location'])][min_location] < \
            #                 self.distances[act[2]][min_location]:
            #             v += 0.0001
            #
            #     else:
            #         if self.distances[tuple(curr_state['pirate_ships'][act[1]]['location'])][self.base] < \
            #                     self.distances[act[2]][self.base]:
            #             v += 0.0001

            if act[0] == 'deposit':
                # v += 4*(2 - int(curr_state['pirate_ships'][act[1]]['capacity']))
                v+=100

            #
            if act[0] == 'collect' :
                shortest_path_to_base = self.distances[tuple(curr_state["pirate_ships"][act[1]]["location"])][self.base]
                remaining_turns = self.turn
                # Check if there are enough turns to deposit the treasure after collecting it
                if remaining_turns >= shortest_path_to_base:
                    v += 50  # Assign a reward for collecti

            for ship in curr_state['marine_ships']:
                if curr_state['pirate_ships'][act[1]]['location'] == \
                        curr_state['marine_ships'][ship]['path'][curr_state['marine_ships'][ship]["index"]] \
                        and curr_state['pirate_ships'][act[1]]['capacity'] < 2:
                    v -= 100
                # if len

            # #
            # if curr_state['turns to go']/self.shortestPathToTreasure[act[1]][closest_treשsure]
        return v


    def ShortestPath(self):
        score = {}
        for pirate_name, pirate_info in self.initial['pirate_ships'].items():
            the_map = self.initial['map']
            # if self.ok:
            #     for pirate_name1, pirate_info1 in self.initial['pirate_ships'].items():
            #         if pirate_name != pirate_name1:
            #             the_map[pirate_info1['location'][0]][pirate_info1['location'][1]] = 'I'
            distance_dict = distance_matrix(the_map)
            for treasure_name, treasure_info in self.initial['treasures'].items():
                location = treasure_info['location']
                min_location = min(
                    self.get_adjacent_sea_cells(location),
                    key=lambda cell: distance_dict[pirate_info['location']][cell]
                )
                treasure_distance_from_pirate = distance_dict[pirate_info['location']][min_location]
                treasure_distance_from_base = distance_dict[min_location][self.base] * (
                        1 - treasure_info['prob_change_location'])
                error = 0
                for loc in treasure_info['possible_locations']:
                    treasure_distance_from_base += distance_dict[location][loc] * (
                            treasure_info['prob_change_location'] / len(treasure_info['possible_locations']))
                    if distance_dict[location][loc] == -1:
                        error += (len(the_map) + len(the_map[0]) - 1) * (
                                treasure_info['prob_change_location'] / len(treasure_info['possible_locations']))
                if treasure_distance_from_pirate == -1 or treasure_distance_from_base == -1:
                    continue
                distance = treasure_distance_from_pirate + treasure_distance_from_base + error
                var = 0
                possible_locations = set(treasure_info['possible_locations'])
                # possible_goals.add(treasure_info['l'])
                mean_dest = np.mean(np.array(list(possible_locations)), axis=0)
                p_move = treasure_info['prob_change_location']
                # Probability for the treasure to stay
                p_stay = 1 - p_move
                for loc in treasure_info['possible_locations']:
                    if loc == treasure_info['location']:
                        var = (p_stay + p_move / len(treasure_info['possible_locations'])) * (
                                np.array(treasure_info['location']) - mean_dest) ** 2
                    else:
                        var += (p_stay / len(treasure_info['possible_locations'])) * (
                                np.array(loc) - mean_dest) ** 2
                var = sum(var)
                score[pirate_name] = {treasure_name: distance + var}
                # min(base_distances[(x, y)] for (x, y) in self.get_adjacent_sea_cells(treasure_location))

        return score

    def next_states(self, state, action, treasure_options, marine_options):
        curr_state = json.loads(state)
        if action[0] == 'terminate':
            return []
        if action[0] == 'reset':
            if len(self.initial["pirate_ships"])>1:
              reset_state = {'pirate_ships': {self.chosen_pirate :self.initial['pirate_ships'][self.chosen_pirate]}, 'treasures': self.initial['treasures'],
                           'marine_ships': self.initial['marine_ships']}
            else:
                reset_state = {'pirate_ships': self.initial['pirate_ships'],
                               'treasures': self.initial['treasures'],
                               'marine_ships': self.initial['marine_ships']}


            reset_state_json = json.dumps(reset_state, sort_keys=True)
            return [[reset_state_json, 1]]
        next_states = []
        for act in action:
            if act[0] == 'sail':
                curr_state['pirate_ships'][act[1]]['location'] = (act[2])
            if act[0] == 'collect' and curr_state['pirate_ships'][act[1]]['capacity'] > 0:
                curr_state['pirate_ships'][act[1]]['capacity'] -= 1
                # curr_state['treasures'][act[2]]['location'] = curr_state['pirate_ships'][act[1]]['location']

            if act[0] == 'deposit' and curr_state['pirate_ships'][act[1]]['capacity'] < 2:
                curr_state['pirate_ships'][act[1]]['capacity'] = 2
                # curr_state['treasures'][act[2]]['location'] = self.initial['treasures'][act[2]]['location']


        for Pship in curr_state["pirate_ships"]:
            for Mship in curr_state["marine_ships"]:

                index = curr_state["marine_ships"][Mship]["index"]

                if curr_state["pirate_ships"][Pship]["location"] == \
                        curr_state["marine_ships"][Mship]["path"][index]:
                    curr_state['pirate_ships'][Pship]['capacity'] = 2

        for option1 in treasure_options:


            for option2 in marine_options:
                for treasure_name, new_location in option1['names']:
                    curr_state['treasures'][treasure_name]['location'] = (new_location)
                # Update the locations of the marine ships according to the current option
                for marine_name, new_location in option2['names']:
                    curr_state['marine_ships'][marine_name]['index'] = (new_location)
                # Convert the updated state to a JSON string for consistency
                state_json = json.dumps(curr_state, sort_keys=True)
                # Append the new state along with the probability of this option
                next_states.append([state_json, option1["prob"] * option2['prob']])
        return next_states

    def value_from_action(self, next_states, value_star):
        v = 0
        for next_state in next_states:
            # print( value_star[next_state[0]]['t-1'])
            v += next_state[1] * value_star[next_state[0]]['t-1']

        return v

    def actions(self, state):
        # Define the actions function based on the current state and game rules

        actions = []
        for ship_name, ship_info in state['pirate_ships'].items():
            ship_actions = self.get_ship_actions(state, ship_name, ship_info)
            actions.append(ship_actions)
        all_actions = [element for element in itertools.product(*actions)]

        all_actions.append(tuple(['reset']))
        all_actions.append(tuple(['terminate']))
        return all_actions

    def get_ship_actions(self, state, ship_name, ship_info):
        ship_actions = []

        ship_x, ship_y = ship_info["location"]
        capacity = ship_info['capacity']
        if capacity < 2 and self.map[ship_x][ship_y] == 'B':
            ship_actions.append(('deposit', ship_name))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = ship_x + dx, ship_y + dy
            if 0 <= new_x < len(self.map) and 0 <= new_y < len(self.map[0]) and self.map[new_x][new_y] != 'I':
                ship_actions.append(('sail', ship_name, (new_x, new_y)))

        for treasure_name, treasure_info in state['treasures'].items():
            treasure_x, treasure_y = treasure_info["location"]
            if (ship_x, ship_y) in [(treasure_x + dx, treasure_y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]] \
                    and 2 >= ship_info['capacity'] >= 1:
                ship_actions.append(('collect', ship_name, treasure_name))

        ship_actions.append(('wait', ship_name))

        return ship_actions

    def build_all_states(self):
        # Define the build_all_states function based on the game rules and initial state
        pirates_list = []

        if len(self.initial['pirate_ships'])>1:

            pirate_name=list(self.initial['pirate_ships'].keys())[0]

            # pirate_info=self.initial['pirate_ships'][pirate_name]
            self.chosen_pirate=pirate_name
            states_per_pirate = []
            for i in range(len(self.map)):
                for j in range(len(self.map[0])):
                    if self.map[i][j] != 'I':
                        for capacity in range(0, 3):
                            state = {pirate_name: {'capacity': capacity, 'location': (i, j)}}
                            states_per_pirate.append(state)
            pirates_list.append(states_per_pirate)
        else:
            for pirate_name, pirate_info in self.initial['pirate_ships'].items():
                states_per_pirate = []
                for i in range(len(self.map)):
                    for j in range(len(self.map[0])):
                        if self.map[i][j] != 'I':
                            for capacity in range(0, 3):
                                state = {pirate_name: {'capacity': capacity, 'location': (i, j)}}
                                states_per_pirate.append(state)

                pirates_list.append(states_per_pirate)

        treasures_list = []
        for treasure_name, treasure_info in self.initial['treasures'].items():
            states_per_treasure = []
            current_location = self.initial["treasures"][treasure_name]["location"]
            # possible_locations = [
            #     loc for loc in self.initial["treasures"][treasure_name]["possible_locations"]
            #     if loc != current_location
            # ]
            for location in self.initial["treasures"][treasure_name]["possible_locations"]:
                state = {treasure_name: {'location': location,
                                         'possible_locations': self.initial["treasures"][treasure_name][
                                             "possible_locations"],
                                         'prob_change_location': self.initial["treasures"][treasure_name][
                                             "prob_change_location"]}}
                states_per_treasure.append(state)
            treasures_list.append(states_per_treasure)

        # for _ in range(len(self.initial['treasures'].keys())):
        #     combination = {}
        #     for state in treasures_list:
        #         combination.update(json.loads(state))
        #     treasures_list.append(combination)
        # treasures=[element for element in itertools.product( *treasures_list)]
        marine_ships_list = []
        for marine_name, marine_info in self.initial['marine_ships'].items():
            states_per_marine = []
            for path_index in range(len(marine_info['path'])):
                state = {marine_name: {'index': path_index, "path": marine_info['path']}}
                states_per_marine.append(state)
            marine_ships_list.append(states_per_marine)
        # marines=[element for element in itertools.product( *marine_ships_list)]
        # if len(self.initial["pirate_ships"])>1:
        pirates_list = [element for element in itertools.product(*pirates_list)]
        # if len(self.initial["treasures"]) > 1:
        treasures_list = [element for element in itertools.product(*treasures_list)]
        # if len(self.initial["marine_ships"]) > 1:
        marine_ships_list = [element for element in itertools.product(*marine_ships_list)]
        states = [element for element in itertools.product(pirates_list, treasures_list, marine_ships_list)]
        # print(states)
        final_states = []
        pirate_count = len(self.initial['pirate_ships'])
        treasure_count = len(self.initial['treasures'])
        marine_count = len(self.initial['marine_ships'])
        final_states = []

        for i, state in enumerate(states):
            # state=json.loads(state)
            pirate_dict = {}
            treasure_dict = {}
            marine_dict = {}
            locations = []
            flag = True
            for treasure in state[1]:
                treasure_dict[list(treasure.keys())[0]] = list(treasure.values())[0]
                locations.append(treasure_dict[list(treasure.keys())[0]]['location'])
            # locations = dict(Counter(locations))
            # pirate_locations = []
            for pirate_ship in state[0]:
                # pirate_locations.append(list(pirate_ship.values())[0]['location'])
                name = list(pirate_ship.keys())[0]
                pirate_dict[name] = list(pirate_ship.values())[0]
            #     if name in locations:
            #         if (self.initial['pirate_ships'][name]['capacity'] - locations[name]) != pirate_dict[name][
            #             'capacity']:
            #             flag = False
            #     elif self.initial['pirate_ships'][name]['capacity'] != pirate_dict[name]['capacity']:
            #         flag = False
            for marine_ship in state[2]:
                marine_dict[list(marine_ship.keys())[0]] = list(marine_ship.values())[0]
            #
            # if len(set(pirate_locations)) != len(pirate_locations):
            #     flag = False

            if flag:
                state = {'pirate_ships': pirate_dict, 'treasures': treasure_dict, 'marine_ships': marine_dict}
                final_states.append(json.dumps(state, sort_keys=True))

            # for i in range(pirate_count):
            #     pirate_ships.update(json.loads(state[i]))
            #     # print(" priate",state[i])
            # for i in range(treasure_count):
            #     treasures.update(json.loads(state[i+pirate_count]))
            #     # print(" treasure",state[i+pirate_count])
            #
            # for i in range(marine_count):
            #     marine_ships.update(json.loads(state[i + pirate_count + treasure_count]))
            #     # print(" marine", state[i + pirate_count+treasure_count])

            # # Check for collisions between pirate ships and marine ships
            # pirate_locations = [pirate_ship['location'] for pirate_ship in pirate_ships.values()]
            # marine_locations = [marine_ship['path'][marine_ship['index']] for marine_ship in marine_ships.values()]
            # if any(location in marine_locations for location in pirate_locations):
            #     continue  # Skip this state as it involves a collision

        # initial_state_json = json.dumps({
        #     'pirate_ships': self.initial['pirate_ships'],
        #     'treasures': self.initial['treasures'],
        #     'marine_ships': self.initial['marine_ships']
        # }, sort_keys=True)
        # if initial_state_json not in final_states:
        #     final_states.append(initial_state_json)
        text = "Hello, world!"
        # with open("text.txt", "w") as file:
        #     for s in final_states:
        #         file.write(s)
        # print(final_states)
        return final_states

    def dict_product(self, d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))

    def get_possible_new_locations(self, current_location):
        x, y = current_location
        possible_new_locations = []
        # Check up
        if x > 0 and self.map[x - 1][y] != 'I':
            possible_new_locations.append((x - 1, y))
        # Check down
        if x < len(self.map) - 1 and self.map[x + 1][y] != 'I':
            possible_new_locations.append((x + 1, y))
        # Check left
        if y > 0 and self.map[x][y - 1] != 'I':
            possible_new_locations.append((x, y - 1))
        # Check right
        if y < len(self.map[0]) - 1 and self.map[x][y + 1] != 'I':
            possible_new_locations.append((x, y + 1))
        return possible_new_locations

    def get_adjacent_cells(self, location):
        """
        Returns a list of adjacent cell coordinates to the given location.
        """
        # print(location)
        (x, y) = location
        adjacent_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        return [(x, y) for x, y in adjacent_cells if 0 <= x < len(self.map) and 0 <= y < len(self.map[0])]

    def get_adjacent_sea_cells(self, location):
        """
        Returns a list of adjacent sea cell coordinates to the given location.
        """
        adjacent_cells = self.get_adjacent_cells(location)
        sea_cells = [(x, y) for x, y in adjacent_cells if self.map[x][y] != 'I']
        return sea_cells

    def find_base(self, map):
        for i, row in enumerate(map):
            for j, cell in enumerate(row):
                if cell == 'B':
                    return (i, j)  # Return the coordinates of the base
        return None


class PirateAgent:

    def __init__(self, initial):
        self.initial = initial
        self.ok = True
        self.map = self.initial['map']
        self.policy = {}
        self.base = self.find_base(self.map)

        # Initialize other necessary attributes
        state = self.choose_optimal()
        self.optimal_state = self.get_optimal_state(state)
        # if self.build_all_states(self.optimal_state) < 4000 and self.initial['turns to go'] <= 150:
        #     treasure_list = list(self.initial['treasures'].keys())
        #     if len(treasure_list) > 1:
        #         new_treasure = self.closest_treasure()
        #         self.optimal_state['treasures'][new_treasure] = self.initial['treasures'][new_treasure]
        optimal_pirate = OptimalPirateAgent( self.optimal_state)
        self.policy[state["pirate_ships"]] = optimal_pirate.policy

    def find_base(self, map):
        for i, row in enumerate(map):
            for j, cell in enumerate(row):
                if cell == 'B':
                    return (i, j)  # Return the coordinates of the base
        return None

    def closest_treasure(self):
        min_dist = []
        distance_map = distance_matrix(self.optimal_state['map'])
        treasure_list = list(self.initial['treasures'].keys())
        original_treasure = list(self.optimal_state['treasures'].keys())[0]
        treasure_list.remove(original_treasure)
        for treasure in treasure_list:
            min_dist.append((treasure, distance_map[self.optimal_state['treasures'][original_treasure]['location']][
                self.initial['treasures'][treasure]['location']]))

        min_dist = min(min_dist, key=lambda x: x[1])
        return min_dist[0]

    def get_optimal_state(self, state):

        pirate = state['pirate_ships']
        treasures = state['treasures']
        marine = self.initial["marine_ships"]
        the_map = self.initial['map']
        treasures_dict = {}
        for treasure in treasures:
            treasures_dict[treasure] = self.initial['treasures'][treasure]

        optimal_state = {
            'optimal': True,
            'infinite': False,
            'map': the_map,
            'pirate_ships': {pirate: self.initial['pirate_ships'][pirate]},
            'treasures': treasures_dict,
            "marine_ships": marine,
            'turns to go': self.initial['turns to go']
        }

        return optimal_state

    def calc_score(self):
        score = []
        for pirate_name, pirate_info in self.initial['pirate_ships'].items():
            the_map = self.initial['map']
            # if self.ok:
            #     for pirate_name1, pirate_info1 in self.initial['pirate_ships'].items():
            #         if pirate_name != pirate_name1:
            #             the_map[pirate_info1['location'][0]][pirate_info1['location'][1]] = 'I'
            distance_dict = distance_matrix(the_map)
            for treasure_name, treasure_info in self.initial['treasures'].items():
                location = treasure_info['location']
                min_location = min(
                    self.get_adjacent_sea_cells(location),
                    key=lambda cell: distance_dict[pirate_info['location']][cell]
                )
                treasure_distance_from_pirate = distance_dict[pirate_info['location']][min_location]
                treasure_distance_from_base = distance_dict[min_location][self.base] * (
                        1 - treasure_info['prob_change_location'])
                error = 0
                for loc in treasure_info['possible_locations']:
                    treasure_distance_from_base += distance_dict[location][loc] * (
                            treasure_info['prob_change_location'] / len(treasure_info['possible_locations']))
                    if distance_dict[location][loc] == -1:
                        error += (len(the_map) + len(the_map[0]) - 1) * (
                                treasure_info['prob_change_location'] / len(treasure_info['possible_locations']))
                if treasure_distance_from_pirate == -1 or treasure_distance_from_base == -1:
                    continue
                distance = treasure_distance_from_pirate + treasure_distance_from_base + error
                possible_locations = set(treasure_info['possible_locations'])
                # possible_goals.add(treasure_info['l'])
                mean_dest = np.mean(np.array(list(possible_locations)), axis=0)
                p_move = treasure_info['prob_change_location']
                # Probability for the treasure to stay
                p_stay = 1 - p_move
                for loc in treasure_info['possible_locations']:
                    if loc == treasure_info['location']:
                        var = (p_stay + p_move / len(treasure_info['possible_locations'])) * (
                                np.array(treasure_info['location']) - mean_dest) ** 2
                    else:
                        var += (p_stay / len(treasure_info['possible_locations'])) * (
                                np.array(loc) - mean_dest) ** 2
                var = sum(var)
                score.append([pirate_name, [treasure_name], (distance + var)])
                # min(base_distances[(x, y)] for (x, y) in self.get_adjacent_sea_cells(treasure_location))

        return score

    def choose_optimal(self):
        score = self.calc_score()
        if len(score) == 0:
            self.ok = False
            score = self.calc_score()
            if len(score) == 0:
                min_score = [list(self.initial['pirate_ships'].keys())[0], list(self.initial['treasures'].keys())[0]]
                # Create a new state with the chosen pirate ship, treasure, and marine ships
                return {'pirate_ships': min_score[0],
                        'treasures': min_score[1],
                        'marine_ships': self.initial['marine_ships']}

        min_score = min(score, key=lambda x: x[2])
        # Create a new state with the chosen pirate ship, treasure, and marine ships
        return {'pirate_ships': min_score[0], 'treasures': min_score[1]}

    def act(self, state):
        actions = []
        optimal_pirate = list(self.optimal_state['pirate_ships'].keys())[0]
        curr_state = {'marine_ships': state["marine_ships"],
                      'pirate_ships': {optimal_pirate: state['pirate_ships'][optimal_pirate]}, 'treasures': {}}
        for treasure_name in self.optimal_state['treasures'].keys():
            curr_state['treasures'][treasure_name] = state['treasures'][treasure_name]
        curr_state = json.dumps(curr_state, sort_keys=True)
        action = (self.policy[optimal_pirate][curr_state][state['turns to go']])
        if action == ('reset',):
            return 'reset'
        if action == ('terminate',):
            return 'terminate'
        if len(self.initial['pirate_ships']) > 1:
            name_list=[optimal_pirate]
            real_action = copy.deepcopy(action)
            for pirate_name, pirate_info in self.initial["pirate_ships"].items():
                if pirate_name != optimal_pirate and pirate_name not in name_list:
                    other_action = [list(act) for act in action]
                    other_action[0][1] = pirate_name
                    name_list.append( pirate_name)
                    other_action = [tuple(act) for act in other_action]
                    other_action = tuple(other_action)
                    real_action += other_action
            return real_action

        return action

    def build_all_states(self, initial):
        # Define the build_all_states function based on the game rules and initial state
        pirates_list = []
        for pirate_name, pirate_info in initial['pirate_ships'].items():
            states_per_pirate = []
            for i in range(len(self.map)):
                for j in range(len(self.map[0])):
                    if self.map[i][j] != 'I':
                        for capacity in range(0, 3):
                            state = {pirate_name: {'capacity': capacity, 'location': (i, j)}}
                            states_per_pirate.append(state)

            pirates_list.append(states_per_pirate)

        # pirates= [element for element in itertools.product( *pirates_list)]

        treasures_list = []
        for treasure_name, treasure_info in initial['treasures'].items():
            states_per_treasure = []
            current_location = initial["treasures"][treasure_name]["location"]
            # possible_locations = [
            #     loc for loc in self.initial["treasures"][treasure_name]["possible_locations"]
            #     if loc != current_location
            # ]
            for location in initial["treasures"][treasure_name]["possible_locations"]:
                state = {treasure_name: {'location': (location),
                                         'possible_locations': initial["treasures"][treasure_name][
                                             "possible_locations"],
                                         'prob_change_location': initial["treasures"][treasure_name][
                                             "prob_change_location"]}}
                states_per_treasure.append(state)
            treasures_list.append(states_per_treasure)

        # for _ in range(len(self.initial['treasures'].keys())):
        #     combination = {}
        #     for state in treasures_list:
        #         combination.update(json.loads(state))
        #     treasures_list.append(combination)
        # treasures=[element for element in itertools.product( *treasures_list)]
        marine_ships_list = []
        for marine_name, marine_info in initial['marine_ships'].items():
            states_per_marine = []
            for path_index in range(len(marine_info['path'])):
                state = {marine_name: {'index': path_index, "path": marine_info['path']}}
                states_per_marine.append(state)
            marine_ships_list.append(states_per_marine)
        # marines=[element for element in itertools.product( *marine_ships_list)]
        # if len(self.initial["pirate_ships"])>1:
        pirates_list = [element for element in itertools.product(*pirates_list)]
        # if len(self.initial["treasures"]) > 1:
        treasures_list = [element for element in itertools.product(*treasures_list)]
        # if len(self.initial["marine_ships"]) > 1:
        marine_ships_list = [element for element in itertools.product(*marine_ships_list)]
        # print(treasures_list)
        states = [element for element in itertools.product(pirates_list, treasures_list, marine_ships_list)]
        final_states = []
        pirate_count = len(initial['pirate_ships'])
        treasure_count = len(initial['treasures'])
        marine_count = len(initial['marine_ships'])
        final_states = []

        for i, state in enumerate(states):
            # state=json.loads(state)
            pirate_dict = {}
            treasure_dict = {}
            marine_dict = {}
            locations = []
            flag = True
            for treasure in state[1]:
                treasure_dict[list(treasure.keys())[0]] = list(treasure.values())[0]
                locations.append(treasure_dict[list(treasure.keys())[0]]['location'])
            # locations = dict(Counter(locations))
            # pirate_locations = []
            for pirate_ship in state[0]:
                # pirate_locations.append(list(pirate_ship.values())[0]['location'])
                name = list(pirate_ship.keys())[0]
                pirate_dict[name] = list(pirate_ship.values())[0]
            #     if name in locations:
            #         if (self.initial['pirate_ships'][name]['capacity'] - locations[name]) != pirate_dict[name][
            #             'capacity']:
            #             flag = False
            #     elif self.initial['pirate_ships'][name]['capacity'] != pirate_dict[name]['capacity']:
            #         flag = False
            for marine_ship in state[2]:
                marine_dict[list(marine_ship.keys())[0]] = list(marine_ship.values())[0]
            #
            # if len(set(pirate_locations)) != len(pirate_locations):
            #     flag = False

            if flag:
                state = {'pirate_ships': pirate_dict, 'treasures': treasure_dict, 'marine_ships': marine_dict}
                final_states.append(state)

            # for i in range(pirate_count):
            #     pirate_ships.update(json.loads(state[i]))
            #     # print(" priate",state[i])
            # for i in range(treasure_count):
            #     treasures.update(json.loads(state[i+pirate_count]))
            #     # print(" treasure",state[i+pirate_count])
            #
            # for i in range(marine_count):
            #     marine_ships.update(json.loads(state[i + pirate_count + treasure_count]))
            #     # print(" marine", state[i + pirate_count+treasure_count])

            # # Check for collisions between pirate ships and marine ships
            # pirate_locations = [pirate_ship['location'] for pirate_ship in pirate_ships.values()]
            # marine_locations = [marine_ship['path'][marine_ship['index']] for marine_ship in marine_ships.values()]
            # if any(location in marine_locations for location in pirate_locations):
            #     continue  # Skip this state as it involves a collision

        initial_state_json = json.dumps({
            'pirate_ships': initial['pirate_ships'],
            'treasures': initial['treasures'],
            'marine_ships': initial['marine_ships']
        }, sort_keys=True)
        if initial_state_json not in final_states:
            final_states.append(initial_state_json)
        text = "Hello, world!"
        # with open("text.txt", "w") as file:
        #     for s in final_states:
        #         file.write(s)
        return len(final_states)

    def get_adjacent_cells(self, location):
        """
        Returns a list of adjacent cell coordinates to the given location.
        """
        (x, y) = location
        adjacent_cells = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        return [(x, y) for x, y in adjacent_cells if 0 <= x < len(self.map) and 0 <= y < len(self.map[0])]

    def get_adjacent_sea_cells(self, location):
        """
        Returns a list of adjacent sea cell coordinates to the given location.
        """
        adjacent_cells = self.get_adjacent_cells(location)
        sea_cells = [(x, y) for x, y in adjacent_cells if self.map[x][y] != 'I']
        return sea_cells


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma
        # self.agent = OptimalPirateAgent(initial)
        # self.agent.policy_iteration()


    def act(self, state):

        raise NotImplemented

    def value(self, state):
        raise NotImplemented

    def get_optimal_state(self, state):

        pirate = state['pirate_ships']
        treasures = state['treasures']
        marine = self.initial["marine_ships"]
        the_map = self.initial['map']
        treasures_dict = {}
        for treasure in treasures:
            treasures_dict[treasure] = self.initial['treasures'][treasure]

        optimal_state = {
            'optimal': True,
            'infinite': False,
            'map': the_map,
            'pirate_ships': {pirate: self.initial['pirate_ships'][pirate]},
            'treasures': treasures_dict,
            "marine_ships": marine,
            'turns to go': self.initial['turns to go']
        }

        return optimal_state


class QItem:
    def __init__(self, row, col, dist):
        self.row = row
        self.col = col
        self.dist = dist

    def __repr__(self):
        return f"QItem({self.row}, {self.col}, {self.dist})"


def minDistance(grid, s, d):
    source = QItem(0, 0, 0)
    source.row = s[0]
    source.col = s[1]
    # To maintain location visit status
    visited = [[False for _ in range(len(grid[0]))]
               for _ in range(len(grid))]

    # applying BFS on matrix cells starting from source
    queue = []
    queue.append(source)
    visited[source.row][source.col] = True
    while len(queue) != 0:
        source = queue.pop(0)

        if source.row == d[0] and source.col == d[1]:
            return source.dist

        # moving up
        if isValid(source.row - 1, source.col, grid, visited):
            queue.append(QItem(source.row - 1, source.col, source.dist + 1))
            visited[source.row - 1][source.col] = True

        # moving down
        if isValid(source.row + 1, source.col, grid, visited):
            queue.append(QItem(source.row + 1, source.col, source.dist + 1))
            visited[source.row + 1][source.col] = True

        # moving left
        if isValid(source.row, source.col - 1, grid, visited):
            queue.append(QItem(source.row, source.col - 1, source.dist + 1))
            visited[source.row][source.col - 1] = True

        # moving right
        if isValid(source.row, source.col + 1, grid, visited):
            queue.append(QItem(source.row, source.col + 1, source.dist + 1))
            visited[source.row][source.col + 1] = True

    return -1


# checking where move is valid or not

def isValid(x, y, grid, visited):
    if ((x >= 0 and y >= 0) and
            (x < len(grid) and y < len(grid[0])) and
            (grid[x][y] != 'I') and (visited[x][y] == False)):
        return True
    return False


def distance_matrix(the_map):
    distance_dict = {}
    row_range = np.arange(len(the_map))
    col_range = np.arange(len(the_map[0]))
    coordinates = list(product(row_range, col_range))
    for point1 in coordinates:
        distance_dict[point1] = {}
        for point2 in coordinates:
            distance = minDistance(the_map, point1, point2)
            distance_dict[point1][point2] = distance

    return distance_dict
