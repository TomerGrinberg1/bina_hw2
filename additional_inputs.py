additional_inputs = [
    # an infinite game with 5 X 5 map and one pirate ship, 1 treasure and one marine ship
    {
        "optimal": True,
        "infinite": True,
        "gamma": 0.9,
        "map": [['B', 'S', 'S', 'S', 'I'],
                ['I', 'S', 'I', 'S', 'I'],
                ['S', 'S', 'I', 'S', 'S'],
                ['S', 'I', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I']],

        "pirate_ships": {'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}},
        "marine_ships": {'marine_1': {"index": 1,
                                      "path": [(2, 3), (2, 3)]}},
    },
    {
        "optimal": True,
        "infinite": False,
        "map": [['S', 'S', 'I', 'S'],
                ['S', 'S', 'I', 'S'],
                ['B', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S']],
        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.1},
                      'treasure_2': {"location": (3, 2),
                                     "possible_locations": ((0, 2), (3, 2)),
                                     "prob_change_location": 0.1}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (2, 1), (2, 2), (2, 1)]}},
        "turns to go": 100
    },
    #new test
    {
        "optimal": True,
        "infinite": False,
        "map": [
            ['S', 'S', 'I', 'S', 'S', 'S'],
            ['S', 'S', 'S', 'S', 'I', 'S'],
            ['B', 'S', 'S', 'S', 'I', 'I']

        ],
        "pirate_ships": {
            "pirate_ship_1": {"location": (2, 0), "capacity": 2},
            "pirate_ship_2": {"location": (2, 0), "capacity": 2}

        },
        "treasures": {
            'treasure_1': {"location": (0, 2), "possible_locations": ((0, 2), (1,4)), "prob_change_location": 0.2},
            'treasure_2': {"location": (2, 5), "possible_locations": ((2, 5), (2,4)), "prob_change_location": 0.1}
        },
        "marine_ships": {
            'marine_1': {"index": 0, "path": [(1, 1), (1, 2), (2, 2), (2, 1)]},
            'marine_2': {"index": 0, "path": [(2, 1)]}
        },
        "turns to go": 100
    },



    # a finite large game - not optimal
    {
        "optimal": False,
        "infinite": False,
        "map": [['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['B', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I', 'S', 'S', 'S', 'S', 'I']],

        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2},
                         'pirate_bob': {"location": (2, 0),
                                        "capacity": 2},
                         'bob the pirate': {"location": (2, 0),
                                            "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.2},
                      'treasure_2': {"location": (2, 2),
                                     "possible_locations": ((0, 2), (2, 2), (3, 2)),
                                     "prob_change_location": 0.1},
                      'treasure_3': {"location": (3, 8),
                                     "possible_locations": ((3, 8), (3, 2), (5, 4)),
                                     "prob_change_location": 0.3},
                      'magical treasure': {"location": (5, 9),
                                           "possible_locations": ((5, 9), (5, 4)),
                                           "prob_change_location": 0.4}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (2, 1)]},
                         "larry the marine": {"index": 0,
                                              "path": [(5, 6), (4, 6), (4, 7)]},
                         },
        "turns to go": 100
    }

    # a finite game with 4 X 4 map and one pirate ship, 2 treasures and one marine ship
]
