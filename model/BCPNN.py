"""
Bayesian Confidence Propagation Neural Network (BCPNN)

This file implements the BCPNN module as described in the (manuscript for the) paper:
"Learning, sleep replay and consolidation of contextual fear memories: A neural network model." (Werne et al.).

Equations defining the dynamics of net input, activity normalization, Hebbian learning,
and inhibitory adaptation are adopted from "Memory consolidation from seconds to weeks: 
a three-stage neural network model with autonomous reinstatement dynamics" by Fiebig & Lansner (2013),
which, in turn, is based on a framework originally proposed by Lansner & Ekeberg (1989).


Author: Lars Werne, 2024-2025
"""

import numpy as np


class BCPNNOptimized:
    def __init__(self, model, num_hcs, units_per_hc, sparsity, tau_assoc, assoc_gain, tau_adapt, adapt_gain, recall_detect_thrsh=0.093):
        self.model = model
        self.N = num_hcs * units_per_hc # Number of units in the module
        self.num_hcs = num_hcs # Number of hypercolumns into which units are grouped
        self.units_per_hc = units_per_hc # Number of units per hypercolumn

        self.tau_L = tau_assoc # Time constant for the auto-associative (Hebbian) learning rule
        self.g_L = assoc_gain # Gain of excitatory, recurrent synapses
        self.tau_A = tau_adapt # Time constant for the inhibitory adaptation rule (None if not used)
        self.g_A = adapt_gain # Gain of inhibitory adaptation

        self.recall_detection_threshold = recall_detect_thrsh 
        # If the "distance" (see util.py) between the module's current state and a target pattern is below this threshold,
        # the pattern is considered to be "successfully recalled".

        self.feedback_connections = {} # dict follows structure {source region name: FeedbackConnection}

        self.log = {}
        self.current_step = 0

        self.supports = np.zeros(self.N) # current net input to the units
        self.output = 0.01 * np.ones(self.N) # current activity of the units 
        self.assoc_biases = np.zeros(self.N) # current auto-associative biases of the units
        self.adapt_biases = np.zeros(self.N) # current adaptation biases of the units

        lambda_0 = self.model.lambda_0 # Small noise parameter preventing excessive growth of synaptic weights
        self.W = 0.01 * np.ones((self.N, self.N)) # (Excitatory) synaptic weights between units
        self.Lambda_unit = lambda_0 * np.ones(self.N) # Per-unit variable impacting synaptic weights
        self.Lambda_conn = (lambda_0 ** 2) * np.ones((self.N, self.N)) # Per-synapse variable impacting synaptic weights

        self.V = 0.01 * np.ones((self.N, self.N)) + np.eye(self.N) # (Inhibitory) "cell adaptation" weights between units
        self.mu_unit = lambda_0 * np.ones(self.N) # Per-unit variable impacting synaptic weights
        self.mu_conn = (lambda_0 ** 2) * np.ones((self.N, self.N)) # Per-synapse variable impacting synaptic weights

    def update_supports(self):
        autoassoc_input = self.assoc_biases # (N,)
        for k in range(self.num_hcs):
            hc_indices = range(k * self.units_per_hc, (k + 1) * self.units_per_hc)
            W_hc = self.W[hc_indices, :] # (units_per_hc, N)
            pi_hc = self.output[hc_indices] # (units_per_hc,)  
            hc_sum = 1e-30 * np.ones(self.N) + np.dot(W_hc.T, pi_hc) # (N,)
            autoassoc_input += np.log(hc_sum) # (N,)

        adapt_input = 0
        if self.tau_A is not None:
            adapt_input = self.adapt_biases
            for k in range(self.num_hcs):
                hc_indices = range(k * self.units_per_hc, (k + 1) * self.units_per_hc)
                V_hc = self.V[hc_indices, :]
                pi_hc = self.output[hc_indices]
                hc_sum = 1e-30 * np.ones(self.N) + np.dot(V_hc.T, pi_hc)
                adapt_input += np.log(hc_sum)
        
        fb_input = np.zeros(self.N)
        for conn in self.feedback_connections.values():
            fb_input += conn.compute_input()

        dh_dt = self.g_L * autoassoc_input + self.g_A * adapt_input + fb_input - self.supports if self.tau_A is not None else self.g_L * autoassoc_input + fb_input - self.supports
        self.supports += dh_dt

    def update_outputs(self):
        exp_supports = np.exp(self.supports)
        exp_sums = np.zeros_like(self.supports)
        for k in range(self.num_hcs):
            hc_indices = range(k * self.units_per_hc, (k + 1) * self.units_per_hc)
            exp_sums[hc_indices] = np.sum(exp_supports[hc_indices])
            if any(exp_sums[hc_indices]) == 0:
                assert all(exp_supports[hc_indices] == 0)
                exp_sums[hc_indices] = 1
                exp_supports[hc_indices] = 1 / self.units_per_hc
        self.output = exp_supports / exp_sums
        
    def update_Lambda_unit(self):
        lambda_0 = self.model.lambda_0
        self.Lambda_unit += (1 / self.tau_L) * ((1 - lambda_0) * self.output + lambda_0 - self.Lambda_unit)

    def update_Lambda_conn(self):
        lambda_0 = self.model.lambda_0
        self.Lambda_conn += (1 / self.tau_L) * ((1 - lambda_0 ** 2) * np.outer(self.output, self.output) + lambda_0 ** 2 - self.Lambda_conn)

    def update_assoc_biases(self):
        self.assoc_biases = np.log(self.Lambda_unit)

    def update_W(self):
        self.W = self.Lambda_conn / np.outer(self.Lambda_unit, self.Lambda_unit)

    def update_mu_unit(self):
        lambda_0 = self.model.lambda_0
        self.mu_unit += (1 / self.tau_A) * ((1 - lambda_0) * self.output + lambda_0 - self.mu_unit)

    def update_mu_conn(self):
        lambda_0 = self.model.lambda_0
        self.mu_conn += (1 / self.tau_A) * ((1 - lambda_0 ** 2) * np.outer(self.output, self.output) + lambda_0 ** 2 - self.mu_conn)

    def update_adapt_biases(self):
        self.adapt_biases = np.log(self.mu_unit)

    def update_V(self):
        self.V = self.mu_conn / np.outer(self.mu_unit, self.mu_unit)

    def update(self):
        # General
        self.update_supports()
        self.update_outputs()
        # Auto-associative
        self.update_Lambda_unit()
        self.update_Lambda_conn()
        self.update_assoc_biases()
        self.update_W()
        # Cell adaptation
        if self.tau_A is not None:
            self.update_mu_unit()
            self.update_mu_conn()
            self.update_adapt_biases()
            self.update_V()
        
    def update_feedback(self):
        for connection in self.feedback_connections.values():
            connection.update()

    def log_activity(self):
        self.log[self.current_step] = {i: self.output[i] for i in range(self.N)}
        self.current_step += 1