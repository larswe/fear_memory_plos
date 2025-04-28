"""
Inter-Modular Feedback Connections

This file implements inter-modular connections as described in the manuscript for the paper:
"Learning, sleep replay and consolidation of contextual fear memories: A neural network model." (Werne et al.).

The logic for transmitting excitatory inputs between modules, including Hebbian learning and
feedback input calculations, is inspired by the framework proposed by Fiebig & Lansner (2014). 

Author: Lars Werne, 2024-2025
"""

import numpy as np

class FeedbackConnection:
    def __init__(self, source, target, tau_fb, fb_gain, bespoke_lambda_0=None):
        self.source = source # source module object
        self.target = target # target module object
        self.tau_fb = tau_fb # timescale of feedback Hebbian learning
        self.fb_gain = fb_gain # feedback input gain parameter

        self.lambda_0 = self.source.model.lambda_0 if bespoke_lambda_0 is None else bespoke_lambda_0

        # Initialization of synaptic weights and activity estimates
        lambda_0_init = 0.11 # NOTE: We use this hard-coded value for the initialization of BAN->BAP and BAN->BAI synapse weights. This is done purely for historical reasons related to model tuning.
        self.W = 0.01 + 0.07 * np.random.beta(1, 5, size=(source.N, target.N))
        self.Lambda_unit_source = lambda_0_init * np.ones(source.N)
        self.Lambda_unit_target = lambda_0_init * np.ones(target.N)
        self.Lambda_conn = (lambda_0_init ** 2) * np.zeros((source.N, target.N))

        self.bias_target = np.log(self.Lambda_unit_target)

    def update(self):
        self.update_fb_unit_rate_estimates()
        self.update_fb_connection_estimates()
        self.update_bias()
        self.update_fb_weights()

    def update_fb_unit_rate_estimates(self):
        # Update marginal activity estimates of source and target units
        self.Lambda_unit_source += ((self.source.output - self.Lambda_unit_source) / self.tau_fb)
        self.Lambda_unit_target += ((self.target.output - self.Lambda_unit_target) / self.tau_fb)

    def update_fb_connection_estimates(self):
        # Update joint activity estimates between source-target unit pairs
        self.Lambda_conn += ((np.outer(self.source.output, self.target.output) - self.Lambda_conn) / self.tau_fb)

    def update_bias(self):
        # Update bias term for target units based on marginal activity estimates
        self.bias_target = np.log(self.Lambda_unit_target)

    def update_fb_weights(self):
        # Compute updated inter-modular weights according to the Hebbian learning rule
        lambda_0 = self.lambda_0
        i_rate_padded = (1 - lambda_0) * self.Lambda_unit_source + lambda_0
        j_rate_padded = (1 - lambda_0) * self.Lambda_unit_target + lambda_0
        ij_rate_padded = (1 - lambda_0 ** 2) * self.Lambda_conn + lambda_0 ** 2
        self.W = ij_rate_padded / np.outer(i_rate_padded, j_rate_padded)

    def adjust_Lambda_conn(self): # For manual adjustments to W
        lambda_0 = self.lambda_0
        i_rate_padded = (1 - lambda_0) * self.Lambda_unit_source + lambda_0
        j_rate_padded = (1 - lambda_0) * self.Lambda_unit_target + lambda_0
        self.Lambda_conn = (self.W * np.outer(i_rate_padded, j_rate_padded) - (lambda_0 ** 2)) / (1 - lambda_0 ** 2)

    def compute_input_to_index(self, j):
        # Compute feedback input to a specific target unit
        fb_input = 1e-30 + np.dot(self.W[:, j], self.source.output)
        return self.fb_gain * np.log(fb_input)

    def compute_input(self):
        # Compute vectorized feedback inputs for all target units
        fb_input = 1e-30 * np.ones(self.target.N) + np.dot(self.W.T, self.source.output)
        return self.fb_gain * np.log(fb_input)