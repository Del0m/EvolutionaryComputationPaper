import numpy as np
import random
from deap import creator
from deap import base
from deap import tools
from deap import algorithms
import time
import math
from Dataloader.dataloader import load_nslkdd as load_data
from sklearn import tree

# Variables
CXPB, MUTPB, NGEN = 0.8, 1, 100
TOURNSIZE = 3
IND_SIZE = 41
POPCOUNT = 100

# Simulated Annealing Parameters
TEMP = 1.0 # gets a 10% deduction every generation via cooling rate
COOLING_RATE = 0.9 # applied once a generation
ITER_PER_OFFSPRING = 3

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bit, n=41)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1/41)

answer = [0, 0, 0]
data = load_data(binary_labels=True, scale=True)

def accuracy(individual):
    mask = np.asarray(individual, dtype=bool)

    if not mask.any():
        return 0.0
    
    X_train_selection = data["X_train"].to_numpy()[:, mask]
    X_test_selection = data["X_test"].to_numpy()[:, mask]

    y_train = data["y_train"].to_numpy()
    y_test = data["y_test"].to_numpy()

    classifier = tree.DecisionTreeClassifier(random_state=30)
    classifier.fit(X_train_selection, y_train)

    return classifier.score(X_test_selection, y_test)

def fitness_evaluation(individual):
    acc = accuracy(individual)
    features = np.count_nonzero(individual)
    feature_ratio = features / 41
    return 0.8 * acc + 0.2 * (1 - feature_ratio)

def simulated_annealing(offspring, TEMP):
    
    for child in offspring:
        for i in range(ITER_PER_OFFSPRING):
            neighbor = toolbox.clone(child)
            randbit = random.randint(0, 40)
            neighbor[randbit] = 1 - neighbor[randbit]

            if sum(neighbor) <= 0:
                continue

            diff = fitness_evaluation(neighbor) - fitness_evaluation(child)
            
            if fitness_evaluation(neighbor) > fitness_evaluation(child):
                child[:] = neighbor
            elif random.random() < math.exp(diff / TEMP):
                child[:] = neighbor
                


def main():
    random.seed(30)
    pop = toolbox.population(n=100)

    for i in pop:
        i.fitness.values = fitness_evaluation(i),

    hof = tools.HallOfFame(1)

    for gen in range(NGEN):
        hof.update(pop)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    
        for child in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(child)
                del child.fitness.values

        simulated_annealing(offspring, TEMP)

        for child in offspring:
            child.fitness.values = fitness_evaluation(child),

        pop[:] = offspring

        worst_index = min(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
        pop[worst_index] = hof[0]   

        TEMP = TEMP * COOLING_RATE


if __name__ == "__main__":
    main()