import numpy as np
import random
from deap import creator
from deap import base
from deap import tools
from deap import algorithms
import time
import math
from sklearn import tree
from Dataloader.dataloader import load_nslkdd

# Variables
CXPB, MUTPB, NGEN = 0.8, 1, 50
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

data = load_nslkdd(binary_labels=True, scale=True)

def accuracy(individual):
    mask = np.asarray(individual, dtype=bool)
    if not mask.any():
        return 0.0

    X_train_full = data["X_train"].to_numpy()[:, mask]
    X_test_subset = data["X_test"].to_numpy()[:, mask]
    y_train_full = data["y_train"].to_numpy()
    y_test = data["y_test"].to_numpy()

    sample_size = len(y_train_full) // 10
    rng = np.random.RandomState(42)
    indices = rng.choice(len(y_train_full), sample_size, replace=False)

    clf = tree.DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_full[indices], y_train_full[indices])

    return clf.score(X_test_subset, y_test)

fitness_cache = {}

def fitness_evaluation(individual):
    key = tuple(individual)
    if key in fitness_cache:
        return fitness_cache[key]
    acc = accuracy(individual)
    feature_ratio = sum(individual) / 41
    result = 0.8 * acc + 0.2 * (1 - feature_ratio)
    fitness_cache[key] = result
    return result

def simulated_annealing(offspring, TEMP):  
    for child in offspring:
        for i in range(ITER_PER_OFFSPRING):
            neighbor = toolbox.clone(child)
            randbit = random.randint(0, 40)
            neighbor[randbit] = 1 - neighbor[randbit]

            if sum(neighbor) <= 0:
                continue

            diff = fitness_evaluation(neighbor) - fitness_evaluation(child)
            
            if diff > 0:
                child[:] = neighbor
            elif random.random() < math.exp(diff / TEMP):
                child[:] = neighbor
                

def standard_ga(seed):
    fitness_cache.clear()
    random.seed(seed)
    pop = toolbox.population(n=100)

    for i in pop:
        i.fitness.values = fitness_evaluation(i),

    hof = tools.HallOfFame(1)
    log = []

    for gen in range(NGEN):
        print(f"  Standard GA — Gen {gen+1}/{NGEN}", end="\r")
        hof.update(pop)

        fits = [ind.fitness.values[0] for ind in pop]
        best_ind = hof[0]
        log.append({"gen": gen, "best_fitness": max(fits), "avg_fitness": np.mean(fits), "best_accuracy": accuracy(best_ind), "best_features": sum(best_ind)})
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

        for child in offspring:
            child.fitness.values = fitness_evaluation(child),

        pop[:] = offspring

        worst_index = min(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
        pop[worst_index] = hof[0]   

    return hof[0], log

def hybrid_ga_with_sa(seed):
    fitness_cache.clear()
    random.seed(seed)
    temp = TEMP
    pop = toolbox.population(n=100)

    for i in pop:
        i.fitness.values = fitness_evaluation(i),

    hof = tools.HallOfFame(1)
    log = []

    for gen in range(NGEN):
        print(f"  Hybrid GA-SA — Gen {gen+1}/{NGEN}", end="\r")
        hof.update(pop)

        fits = [ind.fitness.values[0] for ind in pop]
        best_ind = hof[0]
        log.append({"gen": gen, "best_fitness": max(fits), "avg_fitness": np.mean(fits), "best_accuracy": accuracy(best_ind), "best_features": sum(best_ind)})

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

        simulated_annealing(offspring, temp)

        for child in offspring:
            child.fitness.values = fitness_evaluation(child),

        pop[:] = offspring

        worst_index = min(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
        pop[worst_index] = hof[0]   

        temp = temp * COOLING_RATE

    return hof[0], log

def main():
    ga_results = []
    hybrid_results = []
    ga_logs = []
    hybrid_logs = []
    ga_accuracies = []
    hybrid_accuracies = []
    ga_features = []
    hybrid_features = []

    start = time.time()

    for seed in range(1, 11):
        print(f"\n=== Run {seed}/10 ===")

        print("Running Standard GA...")
        best_ga, ga_log = standard_ga(seed)

        print("Running Hybrid GA-SA...")
        best_hybrid, hybrid_log = hybrid_ga_with_sa(seed)

        ga_results.append(best_ga.fitness.values[0])
        hybrid_results.append(best_hybrid.fitness.values[0])
        ga_accuracies.append(accuracy(best_ga))
        hybrid_accuracies.append(accuracy(best_hybrid))
        ga_features.append(sum(best_ga))
        hybrid_features.append(sum(best_hybrid))
        ga_logs.append(ga_log)
        hybrid_logs.append(hybrid_log)

        print(f"GA: {ga_results[-1]:.4f}  GA-SA: {hybrid_results[-1]:.4f}")

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    print(f"\nStandard GA  — Avg Fitness: {np.mean(ga_results):.4f}  Std: {np.std(ga_results):.4f}")
    print(f"Hybrid GA-SA — Avg Fitness: {np.mean(hybrid_results):.4f}  Std: {np.std(hybrid_results):.4f}")
    print(f"\nStandard GA  — Avg Accuracy: {np.mean(ga_accuracies):.4f}  Avg Features: {np.mean(ga_features):.1f}")
    print(f"Hybrid GA-SA — Avg Accuracy: {np.mean(hybrid_accuracies):.4f}  Avg Features: {np.mean(hybrid_features):.1f}")

    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(hybrid_results, ga_results)
    print(f"\nWilcoxon rank-sum test: p-value = {p_value:.6f}")
    if p_value < 0.05:
        print("Result: Statistically significant difference")
    else:
        print("Result: No statistically significant difference")

    import matplotlib.pyplot as plt

    ga_best_per_gen = np.mean([[g["best_fitness"] for g in log] for log in ga_logs], axis=0)
    hybrid_best_per_gen = np.mean([[g["best_fitness"] for g in log] for log in hybrid_logs], axis=0)

    plt.figure()
    plt.plot(range(NGEN), ga_best_per_gen, label="Standard GA")
    plt.plot(range(NGEN), hybrid_best_per_gen, label="Hybrid GA-SA")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Convergence: Standard GA vs Hybrid GA-SA")
    plt.legend()
    plt.savefig("convergence.png", dpi=300)
    plt.show()

    plt.figure()
    plt.boxplot([ga_accuracies, hybrid_accuracies], labels=["Standard GA", "Hybrid GA-SA"])
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy Across 10 Runs")
    plt.savefig("accuracy_boxplot.png", dpi=300)
    plt.show()

    plt.figure()
    plt.boxplot([ga_features, hybrid_features], labels=["Standard GA", "Hybrid GA-SA"])
    plt.ylabel("Features Selected")
    plt.title("Feature Count Across 10 Runs")
    plt.savefig("features_boxplot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
