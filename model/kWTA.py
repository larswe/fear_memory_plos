"""
k-Winner-Takes-All (kWTA) Neural Network Module

This file implements the kWTA module described in the manuscript:
"Learning, sleep replay and consolidation of contextual fear memories: A neural network model." (Werne et al.).

The kWTA architecture is inspired by the network presented in:
"Memory consolidation from seconds to weeks: a three-stage neural network model with autonomous reinstatement dynamics"
by Fiebig & Lansner (2014).

Unlike the BCPNN, the kWTA network does not organize its units into hypercolumns. Instead,
it globally enforces sparse activity by allowing only the top-k most active units to be active at each step,
simulating a "hard-WTA" constraint through binary neuronal outputs.

Author: Lars Werne, 2024-2025
"""

import numpy as np


class kWTAOptimized:
    def __init__(self, model, num_units, sparsity, tau_assoc, assoc_gain, tau_adapt, adapt_gain, recall_detect_thrsh=0.093):
        self.model = model
        self.N = num_units  # Total number of units

        self.sparsity = sparsity  # Fraction of units active each timestep
        self.tau_L = tau_assoc  # Timescale for Hebbian learning
        self.g_L = assoc_gain  # Gain factor for excitatory (Hebbian) connections
        self.tau_A = tau_adapt  # Timescale for inhibitory adaptation (None if unused)
        self.g_A = adapt_gain  # Gain factor for inhibitory connections

        self.recall_detection_threshold = recall_detect_thrsh
        # Determines if a stored memory is considered "successfully recalled" based on the activity pattern distance

        self.feedback_connections = {}  # Structure: {source region name: FeedbackConnection}

        self.log = {}
        self.current_step = 0

        self.supports = np.zeros(self.N)  # Net input to units
        self.output = 0.01 * np.ones(self.N)  # Binary output (1 if active, 0 otherwise)
        self.assoc_biases = np.zeros(self.N)  # Auto-associative biases for units
        self.adapt_biases = np.zeros(self.N)  # Inhibitory adaptation biases for units

        lambda_0 = self.model.lambda_0  # Small noise parameter preventing excessive synaptic weight growth
        self.W = 0.01 * np.ones((self.N, self.N))  # Excitatory synaptic weights between units
        self.Lambda_unit = lambda_0 * np.ones(self.N)  # Per-unit learning variables
        self.Lambda_conn = (lambda_0 ** 2) * np.ones((self.N, self.N))  # Per-synapse learning variables

        self.V = 0.01 * np.ones((self.N, self.N)) + np.eye(self.N)  # Inhibitory (adaptation) weights
        self.mu_unit = lambda_0 * np.ones(self.N)  # Adaptation variables per unit
        self.mu_conn = (lambda_0 ** 2) * np.ones((self.N, self.N))  # Adaptation variables per connection

    def update_supports(self):
        autoassoc_inputs = self.assoc_biases
        network_sum = 1e-30 * np.ones(self.N) + np.dot(self.W.T, self.output)
        autoassoc_inputs += np.log(network_sum)

        adapt_inputs = self.adapt_biases
        network_sum = 1e-30 * np.ones(self.N) + np.dot(self.V.T, self.output)
        adapt_inputs += np.log(network_sum)

        fb_input = np.zeros(self.N)
        for conn in self.feedback_connections.values():
            fb_input += conn.compute_input()

        g_A = self.g_A if self.g_A is not None else 0
        dh_dt = self.g_L * autoassoc_inputs + g_A * adapt_inputs + fb_input - self.supports
        self.supports += dh_dt

    def update_outputs(self):
        # Enforce sparsity: activate only the top-k units based on their net inputs
        k = int(self.sparsity * self.N)
        threshold_indices = np.argpartition(self.supports, -k)[-k:]
        self.output = np.zeros(self.N)
        self.output[threshold_indices] = 1

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
        # General updates
        self.update_supports()
        self.update_outputs()
        # Auto-associative (Hebbian) updates
        self.update_Lambda_unit()
        self.update_Lambda_conn()
        self.update_assoc_biases()
        self.update_W()
        # Inhibitory cell-adaptation updates
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