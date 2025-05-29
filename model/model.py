"""
AmygdalaEngrams Model

This file defines the AmygdalaEngrams model described in:

"Werne et al., Learning, sleep replay and consolidation of contextual fear memories: A neural network model."

The AmygdalaEngrams model simulates associative learning and consolidation processes in hippocampal-amygdala-cortical circuits, explicitly incorporating the role of sleep-dependent memory replay and synaptic homeostasis. It consists of several interacting modules:

Key Modules:
- SENSORY_CORTEX (BCPNN): Encodes sensory inputs.
- EC_IN, EC_OUT (Binary Modules): Model entorhinal cortex input-output pathways.
- HIP (kWTA): Simulates hippocampal engrams for fast learning, used for recent memory recall.
- CTX (BCPNN): Simulates slow-learning cortical engrams, strengthened over time by hippocampal reactivation and used for remote memory recall.
- BA_N (kWTA): Represents stable, associative memory indices in the basal amygdala.
- BA_P, BA_I (Binary Modules): Represent valence-coding populations in the amygdala, encoding fear and safety, respectively.
- AMY_C, AMY_U, AMY_A (Threshold Cells): Represent current fear response, US (unconditioned stimulus) input, and stress level (running average of recent US exposure), respectively.

The network dynamics evolve according to phases of Perception, Sleep, and Recall, each with distinct parameter regimes defined at the end of this file.

For a detailed step-by-step explanation of the update sequence, please refer to Appendix A2 ("Full Update Cycle of the Model") in the manuscript.

Authors: Lars Werne, 2024–2025
"""


from .BCPNN import *
from .kWTA import *
from .binary_module import *
from .threshold_cell import *
from .feedback import FeedbackConnection
from .phase import Phase
import numpy as np

from utils.util import *

inf = 999999999999999 # A very large number, used as a learning time constant for synapses not currently subject to Hebbian learning.

class AmygdalaEngrams:

    def __init__(self, background_activity=0.025, dt=10):

        # Default parameter values, used to make the model switch between phases.
        self.default_parameters = {
            Phase.PERCEPTION: {
                'EC_IN': {'firing_threshold': 0.75},
                'HIP': {'tau_L': 350, 'g_L': 0, 'tau_A': inf, 'g_A': 0.0},
                'EC_OUT': {'firing_threshold': 2.75},
                'CTX': {'tau_L': 32000, 'g_L': 0},
                'AMY_C': {'center': 8.0, 'slope': 0.3},
                'BA_N': {'tau_L_slow': 40000, 'tau_L_fast': 1000, 'tau_L': 40000, 'g_L': 0.0},
                'BA_P': {'firing_threshold': 3.05},
                'BA_I': {'firing_threshold': 4.35},
                'FB': {
                    'EC_OUT': {
                        'HIP': {'fb_gain': 0.0, 'tau_fb': 600},
                    },
                    'HIP': {},
                    'CTX': {
                        'HIP': {'fb_gain': 0.0, 'tau_fb': 200},
                    },
                    'BA_N': {
                        'HIP': {'fb_gain': 1.0, 'tau_fb_fast': 200, 'tau_fb_slow': 40000, 'tau_fb': 40000},
                        'CTX': {'fb_gain': 0.0, 'fb_gain_base': 0.0, 'tau_fb': inf, 'tau_fb_fast': 1000, 'tau_fb_slow': inf}
                    },
                    'BA_P': {
                        'BA_N': {'fb_gain': 1.0, 'tau_fb_fast': 1000, 'tau_fb_slow': inf, 'tau_fb': inf}
                    },
                    'BA_I': {
                        'BA_N': {'fb_gain': 1.0, 'tau_fb_fast': 2500, 'tau_fb_slow': inf, 'tau_fb': inf}
                    }
                }
            },
            Phase.SLEEP: {
                'EC_IN': {'firing_threshold': 0.75},
                'HIP': {'tau_L': inf, 'g_L': 1.0, 'tau_A': 1200, 'g_A': -0.85},
                'EC_OUT': {'firing_threshold': 2.75},
                'CTX': {'tau_L': 26000, 'g_L': 0},
                'AMY_C': {'center': 8.0, 'slope': 0.3},
                'BA_N': {'tau_L_slow': inf, 'tau_L_fast': inf, 'tau_L': inf, 'g_L': 0.0},
                'BA_P': {'firing_threshold': 3.05},
                'BA_I': {'firing_threshold': 4.35},
                'FB': {
                    'EC_OUT': {
                        'HIP': {'fb_gain': 1.0, 'tau_fb': inf},
                    },
                    'HIP': {},
                    'CTX': {
                        'HIP': {'fb_gain': 1.0, 'tau_fb': inf},
                    },
                    'BA_N': {
                        'HIP': {'fb_gain': 1.0, 'tau_fb_fast': inf, 'tau_fb_slow': inf, 'tau_fb': inf},
                        'CTX': {'fb_gain': 0.0, 'fb_gain_base': 0.0, 'tau_fb': 30000, 'tau_fb_fast': 1000, 'tau_fb_slow': 30000}
                    },
                    'BA_P': {
                        'BA_N': {'fb_gain': 1.0, 'tau_fb_fast': inf, 'tau_fb_slow': inf, 'tau_fb': inf}
                    },
                    'BA_I': {
                        'BA_N': {'fb_gain': 1.0, 'tau_fb_fast': inf, 'tau_fb_slow': inf, 'tau_fb': inf}
                    }
                }
            },
            Phase.RECALL: {
                'EC_IN': {'firing_threshold': 0.75},
                'HIP': {'tau_L': inf, 'g_L': 1, 'tau_A': inf, 'g_A': 0.0},
                'EC_OUT': {'firing_threshold': 2.75},
                'CTX': {'tau_L': inf, 'g_L': 1},
                'AMY_C': {'center': 8.0, 'slope': 0.3},
                'BA_N': {'tau_L_slow': inf, 'tau_L_fast': inf, 'tau_L': inf, 'g_L': 0.05},
                'BA_P': {'firing_threshold': 3.05},
                'BA_I': {'firing_threshold': 4.35},
                'FB': {
                    'EC_OUT': {
                        'HIP': {'fb_gain': 1.0, 'tau_fb': inf},
                    },
                    'HIP': {},
                    'CTX': {
                        'HIP': {'fb_gain': 0.0, 'tau_fb': inf},
                    },
                    'BA_N': {
                        'HIP': {'fb_gain': 1.0, 'tau_fb_fast': inf, 'tau_fb_slow': inf, 'tau_fb': inf},
                        'CTX': {'fb_gain': 0.0, 'tau_fb': inf}
                    },
                    'BA_P': {
                        'BA_N': {'fb_gain': 1.0, 'tau_fb_fast': inf, 'tau_fb_slow': inf, 'tau_fb': inf},
                    },
                    'BA_I': {
                        'BA_N': {'fb_gain': 1.0, 'tau_fb_fast': inf, 'tau_fb_slow': inf, 'tau_fb': inf}
                    }
                }
            },
        }

        self.lambda_0 = background_activity
        self.dt = dt

        self.phase = Phase.PERCEPTION

        self.prediction_error = 0.0

        self.P_cell_recruitability_phase = None
        self.I_cell_recruitability_phase = None

        self.validity_score = 0.0 # Whether EC_IN pattern matches EC_OUT pattern
        self.B_amy = 0.5 # Threshold above which HIP determines AMY activity

        self.maintenance_rate = 0.005

        # AMY parameters for within-session extinction
        self.tau_within = 250 # Time constant for within-session extinction
        self.Lambda_AMY_U_default = 0.3
        self.Lambda_AMY_U = self.Lambda_AMY_U_default # Range (0, 1)

        self.A_thresh = 0.9 # Threshold at which "extreme stress effect" is activated (SEFL)
        self.T_ext_P_min = 0.10
        self.T_ext_P_default = 0.20
        self.T_ext_P_recovery = 1/20000.0
        self.T_ext_P = self.T_ext_P_default
        self.P_rec_norm_default = 2.0
        self.P_rec_norm_min = 1.75
        self.P_rec_norm_recovery = 1/8000.0
        self.P_rec_norm = self.P_rec_norm_default

        self.SENSORY_CORTEX = BCPNNOptimized(self, 50, 10, 0.1, None, None, None, None, None)
        self.EC_IN = BinaryModule(self, 500, 0.75)
        self.HIP = kWTAOptimized(self, 350, 0.04, 400, 1, 400, -0.8, recall_detect_thrsh=0.15)
        self.EC_OUT = BinaryModule(self, 500, 0.75)
        self.CTX = BCPNNOptimized(self, 50, 10, 0.1, 18500, 1, None, None, recall_detect_thrsh=0.15)
        # Amygdala
        self.BA_N = kWTAOptimized(self, 500, 0.1, 100000, 1, None, None, recall_detect_thrsh=0.15)
        self.BA_P = BinaryModule(self, 250, 2.5)
        self.BA_I = BinaryModule(self, 250, 3.5)
        self.AMY_C = ThresholdCell(self, center=6.0, slope=0.05)
        self.AMY_U = ThresholdCell(self, center=0.5, slope=10.0)
        self.AMY_A = ThresholdCell(self, center=0.5, slope=10.0)
        self.AMY_A.output = self.Lambda_AMY_U_default

        # ONLY FOR LOGGING PURPOSES
        self.BA_P_PRE_US = BinaryModule(self, 250, 4.0)
        self.BA_I_PRE_US = BinaryModule(self, 250, 2.5)

        self.EC_OUT.feedback_connections['HIP'] = FeedbackConnection(self.HIP, self.EC_OUT, 200, 1.0)
        self.CTX.feedback_connections['HIP'] = FeedbackConnection(self.HIP, self.CTX, 10000, 1.0)
        self.BA_N.feedback_connections['CTX'] = FeedbackConnection(self.CTX, self.BA_N, 100000, 0.0)
        self.BA_N.feedback_connections['HIP'] = FeedbackConnection(self.HIP, self.BA_N, 100000, 1.0)
        self.BA_P.feedback_connections['BA_N'] = FeedbackConnection(self.BA_N, self.BA_P, 100000, 1.0, bespoke_lambda_0=0.025)
        self.BA_I.feedback_connections['BA_N'] = FeedbackConnection(self.BA_N, self.BA_I, 100000, 1.0, bespoke_lambda_0=0.025)

        # FF projection from sensory cortex to EC in is 1-to-1.
        # EC performs recoding from a "softmax" representation to a binary representation.
        assert(self.EC_IN.N == self.SENSORY_CORTEX.N)
        self.sensory_cortex_ff_to_ecin = np.eye(self.EC_IN.N) # TODO - Randomly permute this matrix

        # FF projection from sensory cortex to CTX is 1-to-1.
        assert(self.CTX.N == self.SENSORY_CORTEX.N)
        self.sensory_cortex_ff_to_ctx = 8 * np.eye(self.CTX.N) # TODO - Randomly permute this matrix

        # self.sensory_cortex_ff_to_ba = np.random.uniform(0, 3, size=(self.BA_N.N, self.SENSORY_CORTEX.N))
        # Select BA_N.N random indices out of range(SENSORY_CORTEX.N) -- set those to 1, the rest to 0
        random_indices = np.random.choice(self.SENSORY_CORTEX.N, self.BA_N.N, replace=False)
        self.sensory_cortex_ff_to_ba = np.zeros((self.BA_N.N, self.SENSORY_CORTEX.N))
        for i in range(self.BA_N.N):
            self.sensory_cortex_ff_to_ba[i, random_indices[i]] = 1.0

        # FF projection EC_in to EC_out is 1-to-1. On representation creation, the connection
        # in the input layer is copied to the output layer, so that the HIP engram learns to 
        # activate the units corresponding to those inputs that activated it.
        assert(self.EC_IN.N == self.EC_OUT.N)
        self.ecin_ff_to_ecout = np.eye(self.EC_IN.N) # NOTE - DO NOT permute this matrix

        # FF projection from EC_in to HIP is random.
        self.ecin_ff_to_hip = np.random.uniform(0, 1, size=(self.HIP.N, self.EC_IN.N))

        self.p_ff_to_c = 1.0
        self.i_ff_to_c = -1.0
        self.u_ff_to_p = 3.0
        self.c_ff_to_p = -3.0
        self.u_ff_to_i = -4.25
        self.c_ff_to_i = 4.25

        self.i_ff_to_p = -0.00
        self.p_ff_to_i = -0.00
        self.a_ff_to_p = -0.0 # neg. as AMY_A has output neg. acute_AMY_U_bias
        self.a_ff_to_i = 0.0

        self.fear_activation = np.zeros(self.BA_P.N)
        self.safety_activation = np.zeros(self.BA_I.N)

    def update(self, sensory_input=None, US_input=None):
        assert (sensory_input is not None) or (self.phase != Phase.PERCEPTION)

        if sensory_input is not None:
            assert len(sensory_input) == self.SENSORY_CORTEX.N
            self.SENSORY_CORTEX.output = sensory_input

            sensory_activity = np.array([sensory_input[i] for i in sensory_input.keys()])
            ecin_input = np.dot(self.sensory_cortex_ff_to_ecin, sensory_activity)
            for i in range(self.EC_IN.N):
                self.EC_IN.supports[i] = ecin_input[i]
            self.EC_IN.update_outputs()

            if self.phase == Phase.PERCEPTION:
                ecout_input = np.dot(self.ecin_ff_to_ecout, self.EC_IN.output)
                for i in range(self.EC_OUT.N):
                    self.EC_OUT.supports[i] = ecout_input[i]
                self.EC_OUT.update_outputs()

            hip_input = np.dot(self.ecin_ff_to_hip, self.EC_IN.output)
            for i in range(self.HIP.N):
                self.HIP.supports[i] = hip_input[i]
            self.HIP.update_outputs()

            ctx_input = np.dot(self.sensory_cortex_ff_to_ctx, sensory_activity)
            for i in range(self.CTX.N):
                self.CTX.supports[i] = ctx_input[i]
            self.CTX.update_outputs()

            ba_input = np.dot(self.sensory_cortex_ff_to_ba, sensory_activity)
            for i in range(self.BA_N.N):
                self.BA_N.supports[i] = ba_input[i]
            self.BA_N.update_outputs()

            for net in [self.HIP, self.CTX, self.BA_N]:
                net.update_Lambda_unit()
                net.update_Lambda_conn()
                net.update_assoc_biases()
                net.update_W()
            self.HIP.update_mu_unit()
            self.HIP.update_mu_conn()
            self.HIP.update_adapt_biases()
            self.HIP.update_V()
        else:
            for net in [self.HIP, self.EC_OUT, self.CTX]:
                net.update()
            self.update_validity_score()
            if self.validity_score > self.B_amy or self.phase == Phase.SLEEP:
                self.BA_N.feedback_connections['HIP'].fb_gain = 1.0
                self.BA_N.feedback_connections['CTX'].fb_gain = 0.0
            else:
                self.BA_N.feedback_connections['HIP'].fb_gain = 0.0
                self.BA_N.feedback_connections['CTX'].fb_gain = 1.0
            self.BA_N.update()

        """
        NOTE: Initially suppose that P/I cells are inhibited/excited by the AMY_C activity 
              / US prediction from the previous time step. 
              
              For one, this means that a sudden context switch with US delivery will count
              as surprising for the purposes of AMY learning. Potentially this matters, as it 
              would turn distressing flashbacks in PTSD into meaningful learning events,
              which would sustain the fear association of the trauma memory.

              But also, note that US delivery on step t would activate P-cells on step t. 
              This would activate AMY_C on step t. 
              This would activate I-cells on step t+1, as the P-cell activity evoked for the 
              purposes of learning would seem like a strong US prediction.

              No I-cells should be recruited based on P cells activated by US delivery
              rather than existing context fear.
              Hence we do this:
              On time step t, compute activities of P and I based on 
              N-cell input. Compute C-cell activity based on these P and I activities.
              Then use this prediction and the US input to update the P and I activities and 
              use these updated activities for the purposes of learning.

              This is cheeky because interactions happen within a time step, but it may be the 
              simplest solution for now.

              This also means that a US delivery doesn't automatically activate a maximal 
              fear response (though it may be increased because of "acute AMY sensitization").
        """

        if self.P_cell_recruitability_phase is None:
            starting_mask = np.random.binomial(1, 0.05, size=self.BA_P.N)
            self.P_cell_recruitability_phase = np.zeros(self.BA_P.N)
            self.P_cell_recruitability_phase[starting_mask == 1] = np.random.uniform(0, np.pi, size=np.sum(starting_mask))
            self.P_cell_recruitability_phase[starting_mask == 0] = np.random.uniform(-np.pi, 0, size=self.BA_P.N - np.sum(starting_mask))
        else:
            accelerator = np.array([1 if self.fear_activation[i] > 0.5 else 0.0 for i in range(self.BA_P.N)])
            self.P_cell_recruitability_phase += ((np.pi / 50.0) * accelerator)
            self.P_cell_recruitability_phase[self.P_cell_recruitability_phase > np.pi] -= 2 * np.pi
            random_push = np.array([0 if self.fear_activation[i] > 0.5 else 4 for i in range(self.BA_P.N)])
            random_push_resetter = np.random.choice([0, 1], size=self.BA_P.N, p=[15/16, 1/16])
            random_push = random_push * random_push_resetter
            self.P_cell_recruitability_phase += ((np.pi / 50.0) * random_push)
            self.P_cell_recruitability_phase[self.P_cell_recruitability_phase > np.pi] -= 2 * np.pi
        self.fear_activation = (1 + np.sin(self.P_cell_recruitability_phase)) / self.P_rec_norm
        # Add some noise to the activation
        random_noise = np.random.normal(0, 0.25, size=self.BA_P.N)
        # self.fear_activation = np.minimum(self.fear_activation + random_noise, self.fear_activation + 0.8)
        self.fear_activation = np.maximum(self.fear_activation + random_noise, 0)

        if self.I_cell_recruitability_phase is None:
            starting_mask = np.random.binomial(1, 0.05, size=self.BA_I.N)
            self.I_cell_recruitability_phase = np.zeros(self.BA_I.N)
            self.I_cell_recruitability_phase[starting_mask == 1] = np.random.uniform(0, np.pi, size=np.sum(starting_mask))
            self.I_cell_recruitability_phase[starting_mask == 0] = np.random.uniform(np.pi, 2 * np.pi, size=self.BA_I.N - np.sum(starting_mask))
        else:
            A = 1/2
            accelerator = np.array([A if self.safety_activation[i] > 0.5 else 0.0 for i in range(self.BA_I.N)])
            self.I_cell_recruitability_phase += ((np.pi / 100.0) * accelerator)
            self.I_cell_recruitability_phase[self.I_cell_recruitability_phase > np.pi] -= 2 * np.pi
            random_push = np.array([0 if self.safety_activation[i] > 0.5 else 4 * A for i in range(self.BA_I.N)])
            random_push_resetter = np.random.choice([0, 1], size=self.BA_I.N, p=[15/16, 1/16])
            random_push = random_push * random_push_resetter
            self.I_cell_recruitability_phase += ((np.pi / 100.0) * random_push)
            self.I_cell_recruitability_phase[self.I_cell_recruitability_phase > np.pi] -= 2 * np.pi
        self.safety_activation = (1 + np.sin(self.I_cell_recruitability_phase)) / 2.0
        # Add some noise to the activation
        random_noise = np.random.normal(0, 0.2, size=self.BA_I.N)
        self.safety_activation = np.minimum(self.safety_activation + random_noise, self.safety_activation + 0.8)
        self.safety_activation = np.maximum(self.safety_activation, 0)

        """
        NOTE: This mechanism is fair enough. E.g. US omission shouldn't immediately abolish the fear 
              response, but some learning should take place by strengthening synapses onto 
              I-cells -- which wouldn't be active yet. What we here call BA_P_PRE_US and BA_I_PRE_US
              would be the observed P- and I-cell activities in reality, which after all are assumed to 
              determine the AMY_C activity. 

              The BA_P and BA_I output we here set for Hebbian learning according to the Fiebig model
              may, in reality, correspond to factors determining which postsynaptic cells are selected for 
              transmitter-induced plasticity (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4411307/).

              In Bacon model, the plasticity at the P and I cell is not determined by the postsynaptic
              activity, presumably due to the same considerations.

              cf. https://www.sciencedirect.com/science/article/pii/B9780128151341000052.
              "In addition to synaptic plasticity, it is well appreciated that changes in intrinsic excitability
              regulate amygdala function. ... Intrinsic excitability directly influences the induction of LTP and
              LTD. It is not a static property but can be influenced by a wide range of factors. Changes in (it) can
              be global or restricted to micro-compartments in neurons (i.e. small portion of a dendrite). 

              Also in that paper: Notes on fear-coding cells in BLA. 
              "The engram hypothesis posits that a small distributed subset of neurons in LAn serves as a 
              long-term repository for CS/US associations. In part, (it) stems from repeated discoveries that only
              a small fraction of principal neurons (20-30%) appeared to change their firing rates after FC.

              Note on this: Should we differentiate between extinction-susceptible and -resistant P-cells? see An, Hong, Choi, 2012.


        """
        self.BA_P.update_supports()
        self.BA_P.supports += np.sum(self.i_ff_to_p * self.BA_I.output)
        self.BA_P.supports += self.a_ff_to_p * self.AMY_A.output
        self.BA_I.update_supports()
        self.BA_I.supports += np.sum(self.p_ff_to_i * self.BA_P.output)
        self.BA_I.supports += self.a_ff_to_i * self.AMY_A.output
        self.BA_P.update_outputs()
        self.BA_I.update_outputs()

        # print mean P and I activity
        #if self.phase == Phase.PERCEPTION:
        #    print("----- Before US delivery -----")
        #    print("P activity: ", np.mean(self.BA_P.output))
        #    print("I activity: ", np.mean(self.BA_I.output))
        #    print("P support: ", np.mean(self.BA_P.supports), "max: ", np.max(self.BA_P.supports))
        #    print("I support: ", np.mean(self.BA_I.supports), "max: ", np.max(self.BA_I.supports))
        self.AMY_C.support = np.sum(self.p_ff_to_c * self.BA_P.output) + np.sum(self.i_ff_to_c * self.BA_I.output)
        self.AMY_C.update_output()

        # Update amygdala
        if US_input is not None:
            self.AMY_U.output = US_input
        else:
            self.AMY_U.output = 0.0

        prediction_error = self.AMY_C.output - self.AMY_U.output
        self.prediction_error = prediction_error
        self.BA_P_PRE_US.output = self.BA_P.output
        self.BA_I_PRE_US.output = self.BA_I.output        


        # Prepare amygdala learning
        #if self.phase == Phase.PERCEPTION:
        #    print("Prediction error: ", prediction_error)
        if prediction_error > 0.3 or prediction_error < -0.3:
            self.BA_N.tau_L = self.BA_N.tau_L_fast
            self.BA_N.feedback_connections['HIP'].tau_fb = self.BA_N.feedback_connections['HIP'].tau_fb_fast
            self.BA_N.feedback_connections['CTX'].tau_fb = self.BA_N.feedback_connections['CTX'].tau_fb_fast
        else:
            self.BA_N.tau_L = self.BA_N.tau_L_slow
            self.BA_N.feedback_connections['HIP'].tau_fb = self.BA_N.feedback_connections['HIP'].tau_fb_slow
            self.BA_N.feedback_connections['CTX'].tau_fb = self.BA_N.feedback_connections['CTX'].tau_fb_slow
        if prediction_error < -0.05: # US was stronger than predicted
            self.BA_P.feedback_connections['BA_N'].tau_fb = self.BA_P.feedback_connections['BA_N'].tau_fb_fast / -prediction_error if self.phase == Phase.PERCEPTION else self.BA_P.feedback_connections['BA_N'].tau_fb_fast
            self.BA_I.feedback_connections['BA_N'].tau_fb = self.BA_I.feedback_connections['BA_N'].tau_fb_slow 
        elif prediction_error > 0.05: # US was weaker than predicted
            self.BA_P.feedback_connections['BA_N'].tau_fb = self.BA_P.feedback_connections['BA_N'].tau_fb_slow
            self.BA_I.feedback_connections['BA_N'].tau_fb = self.BA_I.feedback_connections['BA_N'].tau_fb_fast / prediction_error if self.phase == Phase.PERCEPTION else self.BA_I.feedback_connections['BA_N'].tau_fb_fast
        else:
            self.BA_P.feedback_connections['BA_N'].tau_fb = self.BA_P.feedback_connections['BA_N'].tau_fb_slow
            self.BA_I.feedback_connections['BA_N'].tau_fb = self.BA_I.feedback_connections['BA_N'].tau_fb_slow

        self.update_acute_AMY_bias()

        if self.phase == Phase.PERCEPTION:
            self.BA_I.supports += self.safety_activation * (self.u_ff_to_i * self.AMY_U.output + self.c_ff_to_i * self.AMY_C.output)
            self.BA_I.update_outputs()

            self.BA_P.supports += self.fear_activation * (self.u_ff_to_p * self.AMY_U.output + self.c_ff_to_p * self.AMY_C.output)
            self.BA_P.update_outputs()

        for net in [self.SENSORY_CORTEX, self.EC_IN, self.HIP, self.EC_OUT, self.CTX, self.BA_N, self.BA_P, self.BA_I, self.AMY_C, self.AMY_U, self.BA_P_PRE_US, self.BA_I_PRE_US]:
            net.update_feedback()
            net.log_activity()
        self.AMY_A.log_activity()

        """
        --- Sleep homeostasis protocol ---
        """

        if self.AMY_A.output > self.A_thresh:
            self.T_ext_P = self.T_ext_P_min
            self.P_rec_norm = self.P_rec_norm_min
        # Recovery from SEFL effects
        if self.T_ext_P < self.T_ext_P_default:
            self.T_ext_P += self.T_ext_P_recovery
            print("T_ext_P after recovery step: ", self.T_ext_P)
        if self.T_ext_P > self.T_ext_P_default:
            self.T_ext_P = self.T_ext_P_default
        if self.P_rec_norm < self.P_rec_norm_default:
            self.P_rec_norm += self.P_rec_norm_recovery
        if self.P_rec_norm > self.P_rec_norm_default:
            self.P_rec_norm = self.P_rec_norm_default
        
        if self.phase == Phase.SLEEP:
            
            recruitment_level_P = 0.45
            extinction_threshold_P = self.T_ext_P

            ba_active_mask = np.array([self.BA_N.output[i] > 0.5 for i in range(self.BA_N.N)])
            ba_p_feedback = self.BA_P.feedback_connections['BA_N']
            ba_p_W = ba_p_feedback.W # shape (#N, #P)
            ba_p_W_active = ba_p_W[ba_active_mask, :] # shape (#active N, #P)
            # Update feedback Weights from active BA_N to P cells
            dx_dt = homeostasis_allee_effect(ba_p_W_active, self.maintenance_rate, recruitment_level_P, extinction_threshold_P, verbose=True if self.BA_N.current_step % 100 == 0 else False)
            self.BA_P.feedback_connections['BA_N'].W[ba_active_mask, :] += dx_dt
            self.BA_P.feedback_connections['BA_N'].W[self.BA_P.feedback_connections['BA_N'].W < 0] = 0
            self.BA_P.feedback_connections['BA_N'].adjust_Lambda_conn()

            # Do the same for I-cells
            recruitment_level_I = 2.0
            extinction_threshold_I = 1.5

            ba_i_feedback = self.BA_I.feedback_connections['BA_N']
            ba_i_W = ba_i_feedback.W # shape (#N, #I)
            ba_i_W_active = ba_i_W[ba_active_mask, :] # shape (#active N, #I)
            # Update feedback Weights from active BA_N to I cells
            dx_dt = homeostasis_allee_effect(ba_i_W_active, self.maintenance_rate, recruitment_level_I, extinction_threshold_I, verbose=True if self.BA_N.current_step % 100 == 0 else False)
            self.BA_I.feedback_connections['BA_N'].W[ba_active_mask, :] += dx_dt
            self.BA_I.feedback_connections['BA_N'].W[self.BA_I.feedback_connections['BA_N'].W < 0] = 0
            self.BA_I.feedback_connections['BA_N'].adjust_Lambda_conn()

    def update_validity_score(self):
        ec_in_pattern = np.array([self.EC_IN.output[i] for i in range(len(self.EC_IN.output))])
        ec_out_pattern = np.array([self.EC_OUT.output[i] for i in range(len(self.EC_OUT.output))])
        recall = np.sum(np.logical_and(ec_out_pattern, ec_in_pattern)) / np.sum(ec_in_pattern) if np.sum(ec_in_pattern) > 0 else 0
        precision = np.sum(np.logical_and(ec_out_pattern, ec_in_pattern)) / np.sum(ec_out_pattern) if np.sum(ec_out_pattern) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        self.validity_score = f1

    def assess_env_change(self):
        """
        Assess whether the environment has changed on the current time step.
        This can be done by comparing the current sensory input to the past EC_in pattern
        and thus has to happen immediately after the sensory input is updated and before
        it is passed on to EC.
        """
        sensory_pattern = np.array([self.SENSORY_CORTEX.output[i] for i in range(len(self.SENSORY_CORTEX.output))])
        ec_in_pattern = np.array([self.EC_IN.output[i] for i in range(len(self.EC_IN.output))])
        recall = np.sum(np.logical_and(ec_in_pattern, sensory_pattern)) / np.sum(sensory_pattern) if np.sum(sensory_pattern) > 0 else 0
        precision = np.sum(np.logical_and(ec_in_pattern, sensory_pattern)) / np.sum(ec_in_pattern) if np.sum(ec_in_pattern) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return f1

    def update_acute_AMY_bias(self):
        self.AMY_A.output = self.AMY_A.output + 0.4 * (self.AMY_U.output - self.AMY_A.output) if self.AMY_U.output > self.AMY_A.output else self.AMY_A.output - 0.05 * (self.AMY_A.output - self.AMY_U.output)