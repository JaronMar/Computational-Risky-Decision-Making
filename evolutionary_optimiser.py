from abc import ABC, abstractmethod
import random
import numpy as np
import ray

random.seed()

class Individual(ABC):
    count = {}

    def __init__(self, value=None, init_params=None):
        if value is not None:
            self.value = value
        else:
            self.value = self._random_init(init_params)

        self.fitness = None
        self.id = len(self.count)
        self.count[len(self.count)] = 1

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def _random_init(self, init_params):
        pass

    @staticmethod
    def mutation_helper(mutate_params, original_value, parameter_index):
        perturbation = np.random.normal(0, mutate_params['rate'], mutate_params['dim'])[0]
        mutated_value = original_value + perturbation
        if mutated_value < mutate_params["lower bounds"][parameter_index]:
            mutated_value = mutate_params["lower bounds"][parameter_index]
        if mutated_value > mutate_params["upper bounds"][parameter_index]:
            mutated_value = mutate_params["upper bounds"][parameter_index]
        return mutated_value

    @staticmethod
    def random_agent():
        numeracy = random.random()
        cognition = random.random()
        sensitivity = random.uniform(-3, 3)
        return [cognition, numeracy, sensitivity]


class Population:

    def __init__(self, size, fitness, individual_class, init_params):
        self.fitness = fitness
        self.individuals = [individual_class(init_params=init_params) for _ in range(size)]

        @ray.remote
        def get_fitness(individual):
            return self.fitness(individual)

        fitness_futures = [get_fitness.remote(individual) for individual in self.individuals]
        fitness = ray.get(fitness_futures)
        for i in range(len(fitness)):
            self.individuals[i].fitness = fitness[i]

        self.individuals.sort(key=lambda x: x.fitness)

    def replace(self, new_individuals):
        size = len(self.individuals)

        @ray.remote
        def get_fitness(individual):
            return self.fitness(individual)

        self.individuals.extend(new_individuals)
        fitness_futures = [get_fitness.remote(individual) for individual in self.individuals]
        fitness = ray.get(fitness_futures)

        for i in range(len(self.individuals)):
            self.individuals[i].fitness = fitness[i]

        self.individuals.sort(key=lambda x: x.fitness)
        self.individuals = self.individuals[-size:]

    def get_parents(self, n_offsprings):
        mothers = self.individuals[-2 * n_offsprings::2]
        fathers = self.individuals[-2 * n_offsprings + 1::2]

        return mothers, fathers


class Evolution:
    def __init__(self, pool_size, fitness, individual_class, n_offsprings, pair_params, mutate_params, init_params):
        self.pair_params = pair_params
        self.mutate_params = mutate_params
        self.pool = Population(pool_size, fitness, individual_class, init_params)
        self.n_offsprings = n_offsprings


    def step(self):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)
        offsprings = []

        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params)
            offsprings.append(offspring)

        self.pool.replace(offsprings)

class FixedParameterDecision(Individual):
    def pair(self, other, pair_params):
        parents = [self.value.copy(), other.value.copy()]
        child_parameters = ((np.asarray(parents[0]) + np.asarray(parents[1]))/2).tolist()
        return FixedParameterDecision(child_parameters)

    def mutate(self, mutate_params):
        for parameter_index in range(len(self.value)-1):
            self.value[parameter_index] = self.mutation_helper(mutate_params, self.value[parameter_index], parameter_index)

    def _random_init(self, init_params):
        return self.random_agent()