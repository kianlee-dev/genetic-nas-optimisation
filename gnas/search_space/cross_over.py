import numpy as np
from gnas.search_space.individual import Individual, MultipleBlockIndividual

def _individual_uniform_crossover(individual_a: Individual, individual_b: Individual):
    n = individual_a.get_length()
    selection = np.random.randint(0, 2, n)
    i = 0
    iv_a = []
    iv_b = []
    for a, b in zip(individual_a.iv, individual_b.iv):
        current_selection = selection[i:i + len(a)]
        iv_a.append(a * current_selection + b * (1 - current_selection))
        iv_b.append(a * (1 - current_selection) + b * current_selection)
        i += len(a)
    return Individual(iv_a, individual_a.mi, individual_a.ss, index=individual_a.index), \
           Individual(iv_b, individual_b.mi, individual_b.ss, index=individual_b.index)


def _individual_block_crossover(individual_a: Individual, individual_b: Individual):
    n = individual_a.get_n_op()
    selection = np.random.randint(0, 2, n)
    iv_a = []
    iv_b = []
    for i, (a, b) in enumerate(zip(individual_a.iv, individual_b.iv)):
        # current_selection = selection[i]
        iv_a.append(a * selection[i] + b * (1 - selection[i]))
        iv_b.append(a * (1 - selection[i]) + b * selection[i])

    return Individual(iv_a, individual_a.mi, individual_a.ss, index=individual_a.index), \
           Individual(iv_b, individual_b.mi, individual_b.ss, index=individual_b.index)


def individual_uniform_crossover(individual_a, individual_b, p_c):
    if np.random.rand(1) < p_c:
        if isinstance(individual_a, Individual):
            return _individual_uniform_crossover(individual_a, individual_b)
        else:
            pairs = [_individual_uniform_crossover(inv_a, inv_b) for inv_a, inv_b in
                     zip(individual_a.individual_list, individual_b.individual_list)]
            return MultipleBlockIndividual([p[0] for p in pairs]), MultipleBlockIndividual([p[1] for p in pairs])
    else:
        return individual_a, individual_b


def individual_block_crossover(individual_a, individual_b, p_c):
    if np.random.rand(1) < p_c:
        if isinstance(individual_a, Individual):
            return _individual_block_crossover(individual_a, individual_b)
        else:
            pairs = [_individual_block_crossover(inv_a, inv_b) for inv_a, inv_b in
                     zip(individual_a.individual_list, individual_b.individual_list)]
            return MultipleBlockIndividual([p[0] for p in pairs]), MultipleBlockIndividual([p[1] for p in pairs])
    else:
        return individual_a, individual_b

def blend_a(a: Individual, b: Individual, max_a, alpha=0.5, beta=0.1):
    blended_a = []
    for i in range(len(a)):
        d_i = np.abs(a[i] - b[i])
    
        if a[i] <= b[i]:
            lower_bound_a = a[i] - alpha * d_i
            upper_bound_a = b[i] + beta * d_i
            u_a = int(np.round(np.random.uniform(lower_bound_a, upper_bound_a)))
            u_a = np.maximum(0, np.minimum(u_a, max_a[i]))
            blended_a.append(int(u_a))
        else:
            lower_bound_a = b[i] - beta * d_i
            upper_bound_a = b[i] + alpha * d_i
            u_a = int(np.round(np.random.uniform(lower_bound_a, upper_bound_a)))
            u_a = np.maximum(0, np.minimum(u_a, max_a[i]))
            blended_a.append(int(u_a))
    return blended_a
    
def blend_b(a: Individual, b: Individual, max_b, alpha=0.5, beta=0.1):
    blended_b = []
    for i in range(len(a)):
            d_i = np.abs(a[i] - b[i])
    
            if a[i] <= b[i]:
                lower_bound_b = a[i] - alpha * d_i
                upper_bound_b = b[i] + beta * d_i
                u_b = int(np.round(np.random.uniform(lower_bound_b, upper_bound_b)))
                u_b = np.maximum(0, np.minimum(u_b, max_b[i]))
                blended_b.append(int(u_b))
            else:
                lower_bound_b = b[i] - beta * d_i
                upper_bound_b = b[i] + alpha * d_i
                u_b = int(np.round(np.random.uniform(lower_bound_b, upper_bound_b)))
                u_b = np.maximum(0, np.minimum(u_b, max_b[i]))
                blended_b.append(int(u_b))
    return blended_b
    
def _individual_blend_crossover(individual_a: Individual, individual_b: Individual, alpha=0.5, beta=0.1):
    n = individual_a.get_n_op()
    selection = np.random.randint(0, 2, n)
    iv_a = []
    iv_b = []
    max_a = individual_a.ss.get_max_values_vector(index=individual_a.index)
    max_b = individual_b.ss.get_max_values_vector(index=individual_b.index)

    for i, (a, b) in enumerate(zip(individual_a.iv, individual_b.iv)):
        if (selection[i] == 0):
            iv_a.append(a)
            iv_b.append(b)
        else:
            iv_a.append(np.asarray(blend_a(a,b,max_a[i])))
            iv_b.append(np.asarray(blend_b(a,b,max_b[i])))

    return Individual(iv_a, individual_a.mi, individual_a.ss, index=individual_a.index), \
           Individual(iv_b, individual_b.mi, individual_b.ss, index=individual_b.index)

def individual_blend_crossover(individual_a, individual_b, p_c):
    if np.random.rand(1) < p_c:
        if isinstance(individual_a, Individual):
            return _individual_blend_crossover(individual_a, individual_b)
        else:
            pairs = [_individual_blend_crossover(inv_a, inv_b) for inv_a, inv_b in
                     zip(individual_a.individual_list, individual_b.individual_list)]
            return MultipleBlockIndividual([p[0] for p in pairs]), MultipleBlockIndividual([p[1] for p in pairs])
    else:
        return individual_a, individual_b