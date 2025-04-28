from enum import Enum

class Phase(Enum):
    """
    Enumerates the distinct phases in the AmygdalaEngrams model:

    - PERCEPTION: Network processes sensory input and updates synaptic weights.
    - REFLECTION: (This constant exists for historical reasons -- Fiebig & Lansner (2014) postulated a replay of patterns in working memory, consolidating hippocampal engrams before they are replayed during sleep. We omit this in our model.)
    - SLEEP: Simulates memory replay and homeostatic synaptic adjustments.
    - RECALL: Evaluates network's recall capabilities without updating associative weights.
    """
    PERCEPTION = 1
    REFLECTION = 2
    SLEEP = 3
    RECALL = 4
