from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def calculate_fitness(individual):
    return fitness(items, knapsack_max_capacity, individual)

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1
tournament_size = 2
start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for d in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    #wybor rodzicow
    weights = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    print(weights, "\n\n")
    selected_parents = []
    for c in range(n_selection):
        tournament_contestants = random.choices(range(len(population)), k=tournament_size)
        best_fitness = -float('inf')
        winner = None
        for i in tournament_contestants:
            individual = population[i]
            individual_fitness = weights[i]
            if individual_fitness > best_fitness:
                best_fitness = individual_fitness
                winner = individual
        selected_parents.append(winner)

    #tworzenie kolejnego pokolenia
    half = len(items) // 2
    children = []

    for i in range(0, len(selected_parents) - 1, 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1] 
        # c1 początek od p1 koniec od p2       
        child1 = parent1[:half] + parent2[half:]
        children.append(child1)

        # c2 początek od p2 koniec od p1
        child2 = parent2[:half] + parent1[half:]
        children.append(child2)


    # mutacja
    for child in children:
        mutated_gen = random.randint(0, len(child) - 1)
        child[mutated_gen] = not child[mutated_gen]


    population_sorted = sorted(population, key=calculate_fitness, reverse=True)
    elite_individuals = population_sorted[:n_elite]

    # aktualizacja populacji
    population = elite_individuals + children 


    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
