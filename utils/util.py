"""
util.py

This file contains utility functions supporting the simulations described in 
the paper "Learning, sleep replay and consolidation of contextual fear memories: 
A neural network model" (Werne et al.).

Functions include:

- Pattern comparison metrics (`recall_metric`, `calculate_overlap`)
- Pattern generation utilities (`gen_random_pattern`, `gen_random_plausible_pattern`, 
  `gen_random_simple_pattern`, `gen_simple_pattern`)
- Model training procedures (`online_learning`)
- Mathematical helpers for synaptic homeostasis (`homeostasis_allee_effect`)
- Miscellaneous functions for probabilistic partitioning (`partition_n_into_k`)

"""

import numpy as np
from scipy.stats import dirichlet_multinomial, dirichlet, multinomial
from model.phase import Phase

##############################################
# Pattern comparison functions
##############################################

def recall_metric(a, b):
    """Compute a distance measure (Greve et al., 2010) between two patterns."""
    return (1/2) * (1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def calculate_overlap(output_A, output_B, sparsity):
    assert len(output_A) == len(output_B)
    output_A = np.array(output_A)
    output_B = np.array(output_B)
    num_units = len(output_A)
    num_most_active_units = int(sparsity * num_units)
    # Get binary vectors representing the most active units
    binary_A = np.zeros(num_units)
    binary_B = np.zeros(num_units)
    for i in range(num_most_active_units):
        most_active_index_A = np.argmax(output_A)
        most_active_index_B = np.argmax(output_B)
        binary_A[most_active_index_A] = 1
        binary_B[most_active_index_B] = 1
        output_A[most_active_index_A] = -1
        output_B[most_active_index_B] = -1
    # Calculate the overlap
    num_shared_units = 0
    for i in range(num_units):
        if binary_A[i] == 1 and binary_B[i] == 1:
            num_shared_units += 1
    return num_shared_units / num_most_active_units


##############################################
# General math functions
##############################################

def partition_n_into_k(n, k, alpha=None):
    """
    Partition n into k categories according to a Dirichlet-multinomial distribution.
    """
    if alpha is None:
        # Note: Increasing alpha lowers the variance of the distribution - it becomes tighter around the (uniform, in this case) mean.
        #       The alpha vector can be interpreted as a pseudo-count of prior observations. The drawn "proportions" correspond to the
        #       "true" proportions of the categories.
        alpha = 20 * np.ones(k)

    # Step 1: Sample proportions from the Dirichlet distribution
    proportions = dirichlet.rvs(alpha)[0]
    
    # Step 2: Use these proportions to allocate the total count n among the categories
    sample = multinomial.rvs(n, proportions)
    print(sample)
    return sample


##############################################
# Functions for generating random patterns
##############################################

def gen_random_pattern(num_units, sparsity):
    """Generate a random pattern of support values to clamp the network to.

    Keyword arguments:
    num_units --    number of units in the pattern
    sparsity --     (fixed) fraction of units with strong support
    """
    result = {}
    # Choose a random set of sparsity * num_units units out of num_units
    num_strong_units = int(sparsity * num_units)
    strong_unit_indices = np.random.choice(num_units, num_strong_units, replace=False)
    # Count the number of strong units between 0 and 10, between 10 and 20, ...
    for i in range(0, num_units, 10):
        count = np.sum((strong_unit_indices >= i) & (strong_unit_indices < i + 10))
        # Get those indices
        strong_indices_in_hc = strong_unit_indices[(strong_unit_indices >= i) & (strong_unit_indices < i + 10)]
        # And the weak ones
        weak_indices_in_hc = np.setdiff1d(np.arange(i, i + 10), strong_indices_in_hc)
        # Assign the strength to the result
        if count == 0:
            for j in weak_indices_in_hc:
                result[j] = 0.1
        else:
            strong_unit_proportion = np.random.uniform(0.93, 0.97)
            for j in strong_indices_in_hc:
                result[j] = strong_unit_proportion / count
            weak_unit_proportion = 1 - strong_unit_proportion
            for j in weak_indices_in_hc:
                result[j] = weak_unit_proportion / (10 - count)
    
    return result

def gen_random_plausible_pattern(num_units, num_hcs):
    """Generate a random pattern of support values to clamp the network to.
       Is better than the above function in that the generated outputs could actually 
       be attained by the (cortex) network without any intervention."""
    assert num_units % num_hcs == 0

    result = {}

    units_per_hc = num_units // num_hcs
    for k in range(num_hcs):
        # Draw (units_per_hc) values from a Poisson distribution
        random_supports = np.random.poisson(10, units_per_hc)
        # Normalize the values via softmax
        random_outputs = np.exp(random_supports) / np.sum(np.exp(random_supports))
        for i in range(units_per_hc):
            result[k * units_per_hc + i] = random_outputs[i]

    return result

def gen_random_simple_pattern(num_units, num_hcs):
    """This function is like gen_random_plausible_pattern, but it only has one active unit per hypercolumn.
        Under the BCPNN interpretation of hypercolumns as attributes and units as values, 
        this corresponds to an input that is not probabilistic -- every attribute value is known."""
         
    assert num_units % num_hcs == 0

    result = {}

    units_per_hc = num_units // num_hcs
    for k in range(num_hcs):
        # Draw a random unit to be active in this hypercolumn
        random_unit = np.random.randint(units_per_hc)
        for i in range(units_per_hc):
            if i == random_unit:
                result[k * units_per_hc + i] = 1.0
            else:
                result[k * units_per_hc + i] = 0.0

    return result


def gen_simple_pattern(num_units, num_hcs, units_per_hc, index_per_hc):
    pattern = np.full(num_units, 0.015)
    strength = 0.865
    for i in range(num_hcs):
        pattern[i * units_per_hc + index_per_hc] = np.random.uniform(strength - 0.05, strength + 0.05)
    return {i: pattern[i] for i in range(num_units)}

##############################################
# Model training/updating/simulation functions
##############################################

def online_learning(model, num_steps=1800, context_lifetime=100, lower_US_bound=0.35, upper_US_bound=0.70):
    if hasattr(model, "AMY_P"):
        bacon_transition_phase(model, Phase.PERCEPTION)
    else:
        amy_transition_phase(model, Phase.PERCEPTION)

    online_learning_len = num_steps
    contexts_seen = 0
    for i in range(0, online_learning_len, context_lifetime):
        print("Online learning", i)
        rnd_pattern = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        for j in range(context_lifetime // 6):
            us_input = np.random.uniform(lower_US_bound, upper_US_bound)
            model.update(rnd_pattern, us_input)
        for j in range(context_lifetime // 6, context_lifetime):
            model.update(rnd_pattern, 0.0)
        contexts_seen += 1

        if contexts_seen % 5 == 0:
            amy_transition_phase(model, Phase.SLEEP)
            for i in range(165):
                model.update()
            amy_transition_phase(model, Phase.PERCEPTION)

    for net in [model.EC_IN, model.HIP, model.EC_OUT, model.CTX, model.AMY_C, model.AMY_U]:
        net.current_step = 0
        net.log = {}
    if hasattr(model, "AMY_P"):
        for net in [model.AMY_P, model.AMY_I]:
            net.current_step = 0
            net.log = {}
    else:
        for net in [model.BA_N, model.BA_P, model.BA_I]:
            net.current_step = 0
            net.log = {}
    
def homeostasis_allee_effect(strengths, growth_rate, carrying_capacity, extinction_threshold, min_update=-0.1, max_update=0.1, verbose=False):
    x = strengths
    r = growth_rate
    K = carrying_capacity
    A = extinction_threshold
    dx_dt = r * x * (1 - x / K) * (x/A - 1) # cubic growth model, with strong Allee effect

    # Apply lower and upper bounds to the update
    dx_dt = np.clip(dx_dt, min_update, max_update)

    # print min and max of x, as well as of dx_dt
    if verbose:
        print("min x: ", np.min(x), "max x: ", np.max(x))
        print("min dx_dt: ", np.min(dx_dt), "max dx_dt: ", np.max(dx_dt))

    return dx_dt

def amy_transition_phase(model, phase):
    """
    Transition the model to a new phase by updating the parameters of the networks.
    """

    model.phase = phase
    default_parameters = model.default_parameters
    params = default_parameters[phase]

    # Update network parameters
    for net_name, net_params in params.items(): # Names PFC, HIP, CTX, FB
        if net_name == 'FB':  # Handle feedback connections separately
            for fb_net_name, fb_net_params in net_params.items(): # Names CTX, HIP
                network = getattr(model, fb_net_name) # CTX or HIP
                for fb_name, fb_params in fb_net_params.items(): # Names PFC, HIP
                    fb = network.feedback_connections[fb_name] # The FeedbackConnection object
                    for param, value in fb_params.items(): # Names fb_gain, tau_fb
                        setattr(fb, param, value) 
        else:
            network = getattr(model, net_name)
            for param, value in net_params.items():
                setattr(network, param, value)