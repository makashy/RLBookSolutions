import numpy
from random import randint


def world(x,y, num_holes, num_arrows):
    world_map = numpy.zeros((x,y))
    j = 0    
    while j < num_arrows:
        x_arrows = randint(0, x-1)
        y_arrows = randint(0, y-1)
        if world_map[x_arrows, y_arrows] == 0:
            world_map[x_arrows, y_arrows] =2
            if x_arrows-1 >= 0:
                world_map[x_arrows-1, y_arrows] = 2.5
            if x_arrows+1 < x-1:
                world_map[x_arrows+1, y_arrows] = 2.5
            if y_arrows-1 >= 0:
                world_map[x_arrows, y_arrows-1] = 2.5
            if y_arrows+1 < y-1:
                world_map[x_arrows, y_arrows+1] = 2.5
        else:
            continue
        j += 1
    i = 0
    while i < num_holes:
        x_hole = randint(0, x-1)
        y_hole = randint(0, y-1)
        if world_map[x_hole, y_hole] == 0:
            world_map[x_hole, y_hole] = 1
            if x_hole-1 >= 0:
                world_map[x_hole-1, y_hole] = 1.5
            if x_hole+1 < x-1:
                world_map[x_hole+1, y_hole] = 1.5
            if y_hole-1 >= 0:
                world_map[x_hole, y_hole-1] = 1.5
            if y_hole+1 < y-1:
                world_map[x_hole, y_hole+1] = 1.5
        else:
            continue
        i += 1
    x_wumpus = randint(0, x-1)
    y_wumpus = randint(0, y-1)
    if world_map[x_wumpus, y_wumpus] == 0:
        world_map[x_wumpus, y_wumpus] = 3
        if x_wumpus-1 >= 0:
            world_map[x_wumpus-1, y_wumpus] = 3.5
        if x_wumpus+1 < x-1:
            world_map[x_wumpus+1, y_wumpus] = 3.5
        if y_wumpus-1 >= 0:
            world_map[x_wumpus, y_wumpus-1] = 3.5
        if y_wumpus+1 < y-1:
            world_map[x_wumpus, y_wumpus+1] = 3.5
    else:
        print("failed...\n retrying...")
        return world(x,y, num_holes, num_arrows)
    world_map[0,0] = 5
    #print(world_map)
    return world_map

m = world(10,10,5,3)
print(m)
