import numpy as np
from gnas.search_space.individual import Individual, MultipleBlockIndividual


def flip_max_value(current_value, max_value, p):
    flip = np.floor(np.random.rand(current_value.shape[0]) + p).astype('int')
    sign = (2 * (np.round(np.random.rand(current_value.shape[0])) - 0.5)).astype('int')
    new_dna = current_value + flip * sign
    new_dna[new_dna > max_value] = 0
    new_dna[new_dna < 0] = max_value[new_dna < 0]
    return new_dna


def gaussian_max_value(current_value, max_value, p_gaussian):
    mutation_mask = np.random.rand(current_value.shape[0]) < p_gaussian
    mutation_values = np.random.normal(0, 1, current_value.shape[0])
    mutation_values = np.round(mutation_values).astype(int)
    
    current_value += mutation_values * mutation_mask
    current_value = np.clip(current_value, 0, max_value)
    
    return current_value
    
def uniform_random_mutation(current_value, max_value, p_uniform):
    mutation_mask = np.random.rand(current_value.shape[0]) < p_uniform
    mutation_values = np.random.randint(0, max_value + 1, current_value.shape[0])
    
    current_value += mutation_values * mutation_mask
    current_value = np.clip(current_value, 0, max_value)
    
    return current_value

    
def _individual_combined_mutation(individual_a, p_flip, p_gaussian) -> Individual:
    max_values = individual_a.ss.get_max_values_vector(index=individual_a.index)
    new_iv = []

    for m, iv in zip(max_values, individual_a.iv):
        # Flip mutation with probability p_flip
        iv_flip = flip_max_value(iv, m, p_flip)
        
        # Random uniform mutation with probability p_gaussian
        iv_gaussian = uniform_random_mutation(iv_flip, m, p_gaussian)

        new_iv.append(iv_gaussian)

    return individual_a.update_individual(new_iv)

def individual_combined_mutation(individual_a, p_flip, p_gaussian):
    if isinstance(individual_a, Individual):
        return _individual_combined_mutation(individual_a, p_flip, p_gaussian)
    else:
        return MultipleBlockIndividual([_individual_combined_mutation(inv, p_flip, p_gaussian) for inv in individual_a.individual_list])
        

def adaptive_flip_max_value(current_value, max_value, p, adaptive_rate=0):
    # Adaptive mutation rate to control exploration-exploitation trade-off
    adaptive_p = p + adaptive_rate * np.random.randn(current_value.shape[0])
    adaptive_p = np.clip(adaptive_p, 0, 1)

    # Local neighborhood exploration
    neighborhood = np.random.randn(current_value.shape[0]) * adaptive_p * 0.1
    new_dna = current_value + np.round(neighborhood).astype('int')

    # Bound the values to be within the allowed range
    new_dna[new_dna > max_value] = 0
    new_dna[new_dna < 0] = max_value[new_dna < 0]

    return new_dna

def _individual_adaptive_flip_mutation(individual_a, p, adaptive_rate=0) -> Individual:
    max_values = individual_a.ss.get_max_values_vector(index=individual_a.index)
    new_iv = []
    for m, iv in zip(max_values, individual_a.iv):
        new_iv.append(adaptive_flip_max_value(iv, m, p, adaptive_rate))
    return individual_a.update_individual(new_iv)

def individual_adaptive_flip_mutation(individual_a, p, adaptive_rate=0):
    if isinstance(individual_a, Individual):
        return _individual_adaptive_flip_mutation(individual_a, p, adaptive_rate)
    else:
        return MultipleBlockIndividual([_individual_adaptive_flip_mutation(inv, p, adaptive_rate) for inv in individual_a.individual_list])

def _individual_flip_mutation(individual_a, p) -> Individual:
    max_values = individual_a.ss.get_max_values_vector(index=individual_a.index)
    new_iv = []
    for m, iv in zip(max_values, individual_a.iv):
        new_iv.append(flip_max_value(iv, m, p))
    return individual_a.update_individual(new_iv)


def individual_flip_mutation(individual_a, p):
    if isinstance(individual_a, Individual):
        return _individual_flip_mutation(individual_a, p)
    else:
        return MultipleBlockIndividual([_individual_flip_mutation(inv, p) for inv in individual_a.individual_list])