"""
'Binary' Module

This file implements the Binary Module as described in the manuscript:
"Learning, sleep replay and consolidation of contextual fear memories: A neural network model." (Werne et al., 2024-2025).

The Binary Module is a simplified neural module used specifically to model valence-encoding neural populations (BA_P and BA_I) that -- in our model -- do not auto-associatively store their own activity patterns. 
Instead, they are recruited (activated) by externally driven activity patterns via plastic synapses from other modules. 
Thus, units in a Binary Module operate as threshold-based neurons activated when the net input exceeds a fixed threshold.

Author: Lars Werne, 2024-2025
"""

import numpy as np

class BinaryModule:
    def __init__(self, model, num_units, firing_threshold=0.75):
        self.model = model
        self.N = num_units  # Number of units in the module
        self.firing_threshold = firing_threshold  # Activation threshold for unit firing

        # Dictionary of feedback connections: {source region name: FeedbackConnection}
        self.feedback_connections = {}

        # Activity log for analysis
        self.log = {}
        self.current_step = 0

        # Initialize net input (supports) and outputs (binary activity)
        self.supports = np.zeros(self.N)
        self.output = np.zeros(self.N)

    def update_supports(self):
        """Updates the net input ('supports') of each unit based on feedback from other modules."""
        fb_input = np.zeros(self.N)
        for conn in self.feedback_connections.values():
            fb_input += conn.compute_input()

        if np.all(fb_input == 0):
            print("No feedback input")

        # Simple linear update toward feedback input
        dh_dt = fb_input - self.supports
        self.supports += dh_dt

    def update_outputs(self):
        """Binary activation: units fire (output = 1) if net input exceeds threshold."""
        self.output = (self.supports > self.firing_threshold).astype(int)

    def update(self):
        """Performs a single update step: updating net input and outputs."""
        self.update_supports()
        self.update_outputs()

    def update_feedback(self):
        """Updates all feedback connections from external modules."""
        for connection in self.feedback_connections.values():
            connection.update()

    def log_activity(self):
        """Logs the current binary output activity of each unit at the current step."""
        self.log[self.current_step] = {i: self.output[i] for i in range(self.N)}
        self.current_step += 1
