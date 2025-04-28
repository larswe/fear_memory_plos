"""
ThresholdCell

This class implements a single neuron whose activity is determined by applying a sigmoid activation function to its net excitatory input received from other modules. The neuron does not have intrinsic connections; all input arrives through inter-modular feedback connections.

The neuron's dynamics follow:
\[
\frac{dh(t)}{dt} = \text{Feedback Input}(t) - h(t)
\]

with activity output computed as:
\[
\pi(t) = \frac{1}{1 + e^{-s(h(t)-c)}}
\]

where:
- \(h(t)\) is the net input (support),
- \(c\) is the center (midpoint) of the sigmoid,
- \(s\) is the slope of the sigmoid.

Parameters:
- center (float): Midpoint of the sigmoid function.
- slope (float): Controls the steepness of the sigmoid activation function.
- recall_detection_threshold (float): Threshold determining when an input pattern is considered successfully recalled.

Author: Lars Werne, 2024-2025
"""


import numpy as np

class ThresholdCell:
    def __init__(self, model, center, slope, recall_detection_threshold=0.499):
        self.model = model
        self.center = center
        self.slope = slope
        self.recall_detection_threshold = recall_detection_threshold

        self.N = 1
        self.feedback_connections = {}

        self.log = {}
        self.current_step = 0

        self.support = 0
        self.output = 0.01

    def update_support(self):
        fb_input = np.zeros(1)
        for conn in self.feedback_connections.values():
            fb_input += conn.compute_input()
        dh_dt = fb_input - self.support
        print("fb_input: ", fb_input)
        print("self.support: ", self.support)
        # if dh_dt is subscriptable, take its first element
        if hasattr(dh_dt, "__getitem__"):
            dh_dt = dh_dt[0]
        self.support += dh_dt / self.model.tau_C

    def update_output(self):
        self.output = 1 / (1 + np.exp(-self.slope * (self.support - self.center)))

    def update(self):
        self.update_support()
        self.update_output()

    def update_feedback(self):
        for connection in self.feedback_connections.values():
            connection.update()

    def log_activity(self):
        self.log[self.current_step] = self.output
        self.current_step += 1