small_inputs = [
    {
        "optimal": True,
        "infinite": True,
        "gamma": 0.9,
        "map": [
            ['S', 'S', 'I', 'S'],
            ['S', 'S', 'I', 'S'],
            ['B', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S']
        ],
        "pirate_ships": {'pirate_ship_2': {"location": (2, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.1}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1)]}},
    },
    {
        "optimal": True,
        "infinite": False,
        "map": [
            ['B', 'S', 'S', 'S', 'I'],
            ['I', 'S', 'I', 'S', 'I'],
            ['S', 'S', 'I', 'S', 'S'],
            ['S', 'I', 'S', 'S', 'S'],
            ['S', 'S', 'S', 'S', 'I']
        ],
        "pirate_ships": {'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(2, 3), (2, 3)]}},
        "turns to go":100
    }
]
