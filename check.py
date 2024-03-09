import random
import networkx as nx
from ex2 import InfinitePirateAgent, PirateAgent, ids, OptimalPirateAgent
from additional_inputs import additional_inputs
from inputs import small_inputs
import logging
import time
from copy import deepcopy

RESET_PENALTY = 2
DROP_IN_DESTINATION_REWARD = 4
INIT_TIME_LIMIT = 30000
TURN_TIME_LIMIT = 0.1
MARINE_COLLISION_PENALTY = 1


def initiate_agent(state):
    """
    initiate the agent with the given state
    """
    if state['optimal']:
        if state['infinite']:
            return InfinitePirateAgent(state, state['gamma'])
        return OptimalPirateAgent(state)

    return PirateAgent(state)


class EndOfGame(Exception):
    """
    Exception to be raised when the game is over
    """
    pass


class PirateStochasticProblem:

    def __init__(self, an_input):
        """
        initiate the problem with the given input
        """
        self.sum_score=0
        self.initial_state = deepcopy(an_input)
        self.state = deepcopy(an_input)
        self.graph = self.build_graph()
        start = time.perf_counter()
        self.agent = initiate_agent(deepcopy(self.state))
        end = time.perf_counter()
        print("constructor  time:",end - start)
        if end - start > INIT_TIME_LIMIT:
            logging.critical("timed out on constructor")
            raise TimeoutError
        self.score = 0

    def run_round(self):
        """
        run a round of the game
        """
        if 'turns to go' not in self.state:
            print("infinite game - skipping")
            self.terminate_execution()
        while self.state["turns to go"]:
            start = time.perf_counter()
            action = self.agent.act(deepcopy(self.state))
            # print(action)
            end = time.perf_counter()
            if end - start > TURN_TIME_LIMIT:
                logging.critical(f"timed out on an action")
                raise TimeoutError
            if not self.is_action_legal(action):

                logging.critical(f"You returned an illegal action!")
                # print(action)
                raise RuntimeError
            self.result(action)
            # print(se)
        self.terminate_execution()

    def is_action_legal(self, action):
        """
        check if the action is legal
        """
        def _is_sail_action_legal(sail_action):
            ship_name = sail_action[1]
            if ship_name not in self.state['pirate_ships'].keys():
                return False
            l1 = self.state['pirate_ships'][ship_name]["location"]
            l2 = sail_action[2]
            return l2 in list(self.graph.neighbors(l1))

        def _is_collect_action_legal(collect_action):
            ship_name = collect_action[1]
            treasure_name = collect_action[2]
            # check near position
            y, x = self.state['treasures'][treasure_name]['location']
            if self.state['pirate_ships'][ship_name]['location'] not in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                return False
            # check ship capacity
            if self.state['pirate_ships'][ship_name]['capacity'] <= 0:
                return False
            return True

        def _is_deposit_action_legal(deposit_action):
            ship_name = deposit_action[1]
            # check in base
            y, x = self.state['pirate_ships'][ship_name]['location']
            if self.state['map'][y][x] != 'B':
                return False

            return True

        def _is_action_mutex(global_action):
            assert type(
                global_action) == tuple, "global action must be a tuple"
            # one action per ship
            if len(set([a[1] for a in global_action])) != len(global_action):
                return True

            return False

        if action == "reset":
            return True
        if action == "terminate":
            return True
        if len(action) != len(self.state["pirate_ships"].keys()):
            logging.error(f"You had given {len(action)} atomic commands, while there are {len(self.state['pirate_ships'])}"
                          f" pirate ships in the problem!")
            return False
        for atomic_action in action:
            # illegal move action
            if atomic_action[0] == 'sail':
                if not _is_sail_action_legal(atomic_action):
                    logging.error(f"sail action {atomic_action} is illegal!")
                    return False
            # illegal pick action
            elif atomic_action[0] == 'collect':
                if not _is_collect_action_legal(atomic_action):
                    logging.error(
                        f"Collect action {atomic_action} is illegal!")
                    return False
            # illegal drop action
            elif atomic_action[0] == 'deposit':
                if not _is_deposit_action_legal(atomic_action):
                    logging.error(f"Drop action {atomic_action} is illegal!")
                    return False

            elif atomic_action[0] != 'wait':
                return False
        # check mutex action
        if _is_action_mutex(action):
            logging.error(f"Actions {action} are mutex!")
            return False

        return True

    def result(self, action):
        """"
        update the state according to the action
        """
        self.apply(action)
        if action != "reset":
            self.environment_step()
        self.check_collision_with_marines()

    def apply(self, action):
        """
        apply the action to the state
        """
        if action == "reset":
            self.reset_environment()
            return
        if action == "terminate":
            self.terminate_execution()
        for atomic_action in action:
            self.apply_atomic_action(atomic_action)

    def apply_atomic_action(self, atomic_action):
        """
        apply an atomic action to the state
        """
        ship_name = atomic_action[1]
        if atomic_action[0] == 'sail':
            self.state['pirate_ships'][ship_name]['location'] = atomic_action[2]
            return
        elif atomic_action[0] == 'collect':
            self.state['pirate_ships'][ship_name]['capacity'] -= 1
            return
        elif atomic_action[0] == 'deposit':
            self.score += (2 - self.state['pirate_ships'][ship_name]
                           ['capacity']) * DROP_IN_DESTINATION_REWARD
            self.state['pirate_ships'][ship_name]['capacity'] = 2
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def environment_step(self):
        """
        update the state of environment randomly
        """
        for t in self.state['treasures']:
            treasure_stats = self.state['treasures'][t]
            if random.random() < treasure_stats['prob_change_location']:
                # change destination
                treasure_stats['location'] = random.choice(
                    treasure_stats['possible_locations'])

        for marine in self.state['marine_ships']:
            marine_stats = self.state["marine_ships"][marine]
            index = marine_stats["index"]
            if len(marine_stats["path"]) == 1:
                continue
            if index == 0:
                marine_stats["index"] = random.choice([0, 1])
            elif index == len(marine_stats["path"])-1:
                marine_stats["index"] = random.choice([index, index-1])
            else:
                marine_stats["index"] = random.choice(
                    [index-1, index, index+1])
        self.state["turns to go"] -= 1
        return

    def check_collision_with_marines(self):
        marine_locations = []
        for marine_stats in self.state["marine_ships"].values():
            index = marine_stats["index"]
            marine_locations.append(marine_stats["path"][index])

        for ship_stats in self.state["pirate_ships"].values():
            if ship_stats["location"] in marine_locations:
                ship_stats["capacity"] = 2
                self.score -= MARINE_COLLISION_PENALTY

    def reset_environment(self):
        """
        reset the state of the environment
        """
        self.state["pirate_ships"] = deepcopy(
            self.initial_state["pirate_ships"])
        self.state["treasures"] = deepcopy(self.initial_state["treasures"])
        self.state["marine_ships"] = deepcopy(
            self.initial_state["marine_ships"])
        self.state["turns to go"] -= 1
        self.score -= RESET_PENALTY
        return

    def terminate_execution(self):
        """
        terminate the execution of the problem
        """
        print(f"End of game, your score is {self.score}!")
        print(f"-----------------------------------")
        raise EndOfGame

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.initial_state['map']), len(
            self.initial_state['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.initial_state['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g


def main():
    """
    main function
    """
    # random.seed(10)
    print(f"IDS: {ids}")
    for an_input in small_inputs:
        sum_score = 0
        num_iterations = 10
        for i in range(num_iterations):
            try:
                my_problem = PirateStochasticProblem(an_input)
                my_problem.run_round()

            except EndOfGame:
                sum_score += my_problem.score
                continue

        print("mean_score", sum_score / num_iterations)


    for an_input in additional_inputs:
        print(an_input)
        sum_score = 0
        num_iterations = 10
        for i in range(num_iterations):
            try:
                my_problem = PirateStochasticProblem(an_input)
                my_problem.run_round()

            except EndOfGame:
                sum_score+=my_problem.score
                continue

        print("mean_score", sum_score / num_iterations)


if __name__ == '__main__':
    main()
