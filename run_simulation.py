"""
Simulation code for the Amygdala Engrams model.

At the bottom of this file, you will find the main method. Please simply uncomment the line for the figure which you would like to reproduce.
"""


from model.phase import Phase
from model.model import AmygdalaEngrams
import numpy as np
import matplotlib.pyplot as plt
from utils.util import *
import copy
import seaborn as sns


def amy_show_fear_acquisition():
    num_trials = 40
    num_runs = 1
    amy_responses = np.zeros((num_runs, num_trials))

    for run in range(num_runs):
        print("Run", run)
        model = AmygdalaEngrams()
        online_learning(model, 1000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        for i in range(num_trials):
            print("Trial", i)
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 1.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[run, i] = pattern_converged_to_amy
    amy_responses_mean = np.mean(amy_responses, axis=0)
    amy_responses_std = np.std(amy_responses, axis=0)
    plt.figure()
    num_runs_string = "(single session)" if num_runs == 1 else " (average of " + str(num_runs) + " runs)"
    plt.title(f"Fear acquisition {num_runs_string}")
    plt.plot(amy_responses_mean, label="Mean")
    # Shaded area indicating standard deviation
    plt.fill_between(x=np.arange(num_trials), y1=amy_responses_mean - amy_responses_std, y2=amy_responses_mean + amy_responses_std, alpha=0.25, label="Standard deviation")
    plt.legend(loc='lower right')
    plt.xlabel("Acquisition trial")
    plt.ylabel("C-cell activity")
    # xticks at 5, 10, 15, 20
    plt.xticks(np.arange(0, num_trials, 5))
    plt.ylim(-0.01, 1.01)
    plt.show()

def amy_show_fear_generalization():
    # Collect BA_P and BA_I responses for each similarity, on each time step
    # Train the model on acquisition in context A. 
    # Transition to Recall mode. 
    # Test the model on each similarity context, note the mean BA_P and BA_I responses for each similarity context.
    # Also test recall of context A.
    # Transition to Perception mode.
    # Continue to train the model on acquisition context A and repeat for "num_acquisition" trials.
    # Then do the same but with extinction in context A.
    # Then plot the BA_P and BA_I activities over time, for each similarity context and context A.
    similarities = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975]
    num_runs = 1

    num_accustom_trials = 0
    num_acquisition_trials = 20
    num_extinction_trials = 350
    num_test_trials = 1
    num_trials = num_acquisition_trials + num_extinction_trials

    # Initialize accumulators for responses
    total_amy_P_responses = {i: np.zeros(num_acquisition_trials + num_extinction_trials + num_test_trials) for i in similarities + [1.0]}
    total_amy_I_responses = {i: np.zeros(num_acquisition_trials + num_extinction_trials + num_test_trials) for i in similarities + [1.0]}
    total_amy_C_responses = {i: np.zeros(num_acquisition_trials + num_extinction_trials + num_test_trials) for i in similarities + [1.0]}

    for run in range(num_runs):
        model = AmygdalaEngrams()
        online_learning(model, 2000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        similarity_contexts = []
        for i, sim in enumerate(similarities):
            diff = 1 - sim
            hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * diff)
            units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
            cntx_B = cntx_A.copy()
            cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
            for j in range(units_to_change):
                cntx_B[j] = cntx_B_diff[j]
            similarity_contexts.append(cntx_B)

        amy_P_responses = {i: [] for i in similarities + [1.0]}
        amy_I_responses = {i: [] for i in similarities + [1.0]}
        amy_C_responses = {i: [] for i in similarities + [1.0]}

        for i in range(num_acquisition_trials):
            # Train the model on acquisition in context A
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 1.0)
            # Transition to Recall mode
            amy_transition_phase(model, Phase.RECALL)
            # Test the model on each similarity context
            for i, con in enumerate(similarity_contexts):
                model.update(con)
                    # Note the mean BA_P and BA_I responses for each similarity context
                amy_I_responses[similarities[i]].append(np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values())))
                amy_P_responses[similarities[i]].append(np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values())))
                amy_C_responses[similarities[i]].append(model.AMY_C.log[model.AMY_C.current_step-1])
            model.update(cntx_A)
            amy_I_responses[1.0].append(np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values())))
            amy_P_responses[1.0].append(np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values())))
            amy_C_responses[1.0].append(model.AMY_C.log[model.AMY_C.current_step-1])

        for i in range(num_extinction_trials):
            # Train the model on extinction in context A
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 0.0)
            print(np.mean(list(model.BA_I_PRE_US.log[model.BA_I.current_step-1].values())))
            # Transition to Recall mode
            amy_transition_phase(model, Phase.RECALL)
            # Test the model on each similarity context
            for i, con in enumerate(similarity_contexts):
                model.update(con)
                # Note the mean BA_P and BA_I responses for each similarity context
                amy_I_responses[similarities[i]].append(np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values())))
                amy_P_responses[similarities[i]].append(np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values())))
                amy_C_responses[similarities[i]].append(model.AMY_C.log[model.AMY_C.current_step-1])
            
            # Test recall of context A
            model.update(cntx_A)
            amy_I_responses[1.0].append(np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values())))
            amy_P_responses[1.0].append(np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values())))
            amy_C_responses[1.0].append(model.AMY_C.log[model.AMY_C.current_step-1]) 

        # Test generalization to context B
        for i, con in enumerate(similarity_contexts):
            for j in range(num_test_trials):
                model.update(con)
                amy_I_responses[similarities[i]].append(np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values())))
                amy_P_responses[similarities[i]].append(np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values())))
                amy_C_responses[similarities[i]].append(model.AMY_C.log[model.AMY_C.current_step-1])
        for j in range(num_test_trials):
            model.update(cntx_A)
            amy_I_responses[1.0].append(np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values())))
            amy_P_responses[1.0].append(np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values())))
            amy_C_responses[1.0].append(model.AMY_C.log[model.AMY_C.current_step-1])
            
        # Accumulate the responses
        for sim in similarities + [1.0]:
            total_amy_P_responses[sim] += np.array(amy_P_responses[sim])
            total_amy_I_responses[sim] += np.array(amy_I_responses[sim])
            total_amy_C_responses[sim] += np.array(amy_C_responses[sim])

    # Compute the average responses
    avg_amy_P_responses = {sim: total_amy_P_responses[sim] / num_runs for sim in similarities + [1.0]}
    avg_amy_I_responses = {sim: total_amy_I_responses[sim] / num_runs for sim in similarities + [1.0]}
    avg_amy_C_responses = {sim: total_amy_C_responses[sim] / num_runs for sim in similarities + [1.0]}

    # Plotting the average responses
    fig, axs = plt.subplots(3)
    fig.suptitle('AMY activity over time')
    for sim in similarities + [1.0]:
        axs[0].plot(avg_amy_P_responses[sim], label=f"Similarity={sim}")
        axs[1].plot(avg_amy_I_responses[sim], label=f"Similarity={sim}")
        axs[2].plot(avg_amy_C_responses[sim], label=f"Similarity={sim}")
    axs[0].set(xlabel='Trial', ylabel='AMY_P activity')
    axs[1].set(xlabel='Trial', ylabel='AMY_I activity')
    axs[2].set(xlabel='Trial', ylabel='AMY_C activity')
    axs[0].axvline(x=num_acquisition_trials, color='grey', linestyle='--', alpha=0.5)
    axs[1].axvline(x=num_acquisition_trials, color='grey', linestyle='--', alpha=0.5)
    axs[2].axvline(x=num_acquisition_trials, color='grey', linestyle='--', alpha=0.5)
    axs[0].set_ylim(0.0, max([max(avg_amy_P_responses[sim]) for sim in similarities + [1.0]]))
    axs[1].set_ylim(0.0, max([max(avg_amy_I_responses[sim]) for sim in similarities + [1.0]]))
    axs[2].set_ylim(0.0, max([max(avg_amy_C_responses[sim]) for sim in similarities + [1.0]]))
    plt.legend()
    plt.show()

    # Plot generalization gradients based on final time point
    fig, axs = plt.subplots(3, figsize=(18, 9), sharex=True)

    # Plot data with lines and markers
    for i, (ax, title, responses, ylim, yticks, yticklabels) in enumerate(
        zip(
            axs,
            ["P-cells", "I-cells", "CeM output"],
            [avg_amy_P_responses, avg_amy_I_responses, avg_amy_C_responses],
            [(0.0, 0.1), (0.0, 0.1), (0.0, 1.0)],
            [[0.0, 0.02, 0.04, 0.06, 0.08], [0.0, 0.02, 0.04, 0.06, 0.08], [0.0, 0.5, 1.0]],
            [[0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 0.5, 1]],
        )
    ):
        sims = list(similarities) + [1.0]
        activities = [responses[sim][-1] for sim in sims]
        ax.plot(sims, activities, 'k-')  # Black line for connection
        ax.plot(sims, activities, 'ro')  # Red dots
        ax.set_title(title, fontsize=22)
        ax.set_ylabel("% active" if i < 2 else "Activity", fontsize=22)
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Final adjustments for the last axis
    axs[2].set_xlabel('% Input overlap with conditioned context', fontsize=22)
    axs[2].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[2].set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=18)

    # Adjust label font sizes for all axes
    for ax in axs:
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)

    plt.tight_layout()
    plt.show()





def amy_show_fear_extinction_AA():
    num_acquisition_trials = 40
    num_extinction_trials = 100
    num_trials = num_acquisition_trials + num_extinction_trials
    num_runs = 10
    amy_responses = np.zeros((num_runs, num_trials))
    p_cell_activities = np.zeros((num_runs, num_trials))
    i_cell_activities = np.zeros((num_runs, num_trials))

    for run in range(num_runs):
        print("Run", run)
        model = AmygdalaEngrams()
        online_learning(model, 1000)
        amy_transition_phase(model, Phase.PERCEPTION)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        for i in range(num_acquisition_trials):
            print("Trial", i)
            model.update(cntx_A, 0.9)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[run, i] = pattern_converged_to_amy
            p_cell_activities[run, i] = np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values()))
            i_cell_activities[run, i] = np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values()))
        for i in range(num_extinction_trials):
            print("Trial", num_acquisition_trials + i)
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[run, num_acquisition_trials + i] = pattern_converged_to_amy
            p_cell_activities[run, num_acquisition_trials + i] = np.mean(list(model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1].values()))
            i_cell_activities[run, num_acquisition_trials + i] = np.mean(list(model.BA_I_PRE_US.log[model.BA_I_PRE_US.current_step-1].values()))
    amy_responses_mean = np.mean(amy_responses, axis=0)
    amy_responses_std = np.std(amy_responses, axis=0)
    p_cell_activities_mean = np.mean(p_cell_activities, axis=0)
    p_cell_activities_std = np.std(p_cell_activities, axis=0)
    i_cell_activities_mean = np.mean(i_cell_activities, axis=0)
    i_cell_activities_std = np.std(i_cell_activities, axis=0)
    
    #plt.figure()
    #num_runs_string = "(single session)" if num_runs == 1 else " (average of " + str(num_runs) + " runs)"
    #plt.title(f"Fear (`AA') extinction {num_runs_string}")
    #plt.plot(amy_responses_mean, label="Mean")
    # Shaded area indicating standard deviation
    #plt.fill_between(x=np.arange(num_trials), y1=amy_responses_mean - amy_responses_std, y2=amy_responses_mean + amy_responses_std, alpha=0.25, label="Standard deviation")
    #plt.legend(loc='lower right')
    #plt.xlabel("Acquisition trial")
    #plt.ylabel("C-cell activity")
    # xticks at 5, 10, 15, 20
    #plt.xticks(np.arange(0, num_trials, 5))
    #plt.ylim(-0.01, 1.01)
    #plt.show()

    fig, axes = plt.subplots(3, figsize=(9, 9))
    fig.suptitle("Fear ('AA') extinction", fontsize=24)
    axes[2].plot(amy_responses_mean, label="Mean")
    axes[2].fill_between(x=np.arange(num_trials), y1=amy_responses_mean - amy_responses_std, y2=amy_responses_mean + amy_responses_std, alpha=0.25, label="Standard deviation")
    axes[2].set_title("Fear response", fontsize=22)
    axes[2].set_xlabel("Trial", fontsize=22)
    axes[2].set_ylabel("CeM activity", fontsize=22)
    axes[2].set_ylim(-0.01, 1.01)
    axes[2].axvline(x=num_acquisition_trials, color='grey', linestyle='--', alpha=0.5)
    axes[2].set_xticks([0, 50, 100, 150])
    axes[2].set_xticklabels([0, 50, 100, 150], fontsize=18)
    axes[2].set_yticks([0, 0.5, 1])
    axes[2].set_yticklabels([0, 0.5, 1], fontsize=18)
    # axes[0].legend(loc='lower right')
    axes[0].plot(p_cell_activities_mean, label="Mean")
    axes[0].fill_between(x=np.arange(num_trials), y1=p_cell_activities_mean - p_cell_activities_std, y2=p_cell_activities_mean + p_cell_activities_std, alpha=0.25, label="Standard deviation")
    axes[0].set_title("P-cell activity", fontsize=22)
    axes[0].set_xlabel("")
    axes[0].set_ylabel(r"% cells active", fontsize=22)
    axes[0].set_ylim(-0.01, 0.071)
    axes[0].axvline(x=num_acquisition_trials, color='grey', linestyle='--', alpha=0.5)
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])
    axes[0].set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    axes[0].set_yticklabels([0, 2, 4, 6, 8], fontsize=18)
    axes[0].legend(loc='lower right', fontsize=20)
    # axes[1].legend(loc='lower right')
    axes[1].plot(i_cell_activities_mean, label="Mean")
    axes[1].fill_between(x=np.arange(num_trials), y1=i_cell_activities_mean - i_cell_activities_std, y2=i_cell_activities_mean + i_cell_activities_std, alpha=0.25, label="Standard deviation")
    axes[1].set_title("I-cell activity", fontsize=22)
    axes[1].set_xlabel("")
    axes[1].set_ylabel(r"% cells active", fontsize=22)
    axes[1].set_ylim(-0.01, 0.071)
    axes[1].axvline(x=num_acquisition_trials, color='grey', linestyle='--', alpha=0.5)
    axes[1].set_xticks([])
    axes[1].set_xticklabels([])
    axes[1].set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    axes[1].set_yticklabels([0, 2, 4, 6, 8], fontsize=18)
    plt.tight_layout()
    plt.show()



def amy_show_fear_extinction_AB():
    model = AmygdalaEngrams()
    online_learning(model, 100)

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
    cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])

    B_overlap = 0.7
    B_diff = 1 - B_overlap
    hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
    units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
    cntx_B = cntx_A.copy()
    cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    for i in range(units_to_change):
        cntx_B[model.SENSORY_CORTEX.N - units_to_change + i] = cntx_B_diff[i]
    cntx_B_vec = np.array([cntx_B[i] for i in cntx_B.keys()])

    num_accustom_trials = 4
    num_acquisition_trials = 20
    num_extinction_trials = 40
    num_trials = num_acquisition_trials + num_extinction_trials
    amy_responses = np.zeros(num_trials)

    ba_dists_A = np.zeros(num_extinction_trials)
    ba_dists_B = np.zeros(num_extinction_trials)

    for i in range(num_accustom_trials):
        model.update(cntx_A, np.random.uniform(0.5, 0.7))
    for i in range(num_accustom_trials):
        model.update(cntx_B, np.random.uniform(0.5, 0.7))
    for i in range(num_acquisition_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_A, 1.0)
        if i == 0:
            cntx_A_pattern_ba = model.BA_N.log[model.BA_N.current_step-1]
            cntx_A_pattern_ba_vec = np.array([cntx_A_pattern_ba[i] for i in cntx_A_pattern_ba.keys()])
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_A)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_B, 0.0)
        if i == 0:
            cntx_B_pattern_ba = model.BA_N.log[model.BA_N.current_step-1]
            cntx_B_pattern_ba_vec = np.array([cntx_B_pattern_ba[i] for i in cntx_B_pattern_ba.keys()])
            print("Current step:", model.BA_N.current_step)
            # assert np.array_equal(cntx_B_pattern_ba_vec, cntx_A_pattern_ba_vec) == False
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_B)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]

        pattern_converged_to_ba = model.BA_N.log[model.BA_N.current_step-1]
        pattern_converged_to_ba_vec = np.array([pattern_converged_to_ba[i] for i in pattern_converged_to_ba.keys()])
        ba_dist_A = recall_metric(cntx_A_pattern_ba_vec, pattern_converged_to_ba_vec)
        ba_dist_B = recall_metric(cntx_B_pattern_ba_vec, pattern_converged_to_ba_vec)
        print("Step", i + num_acquisition_trials)
        print("ba distance A:", ba_dist_A, "ba distance B:", ba_dist_B)
        ba_dists_A[i] = ba_dist_A
        ba_dists_B[i] = ba_dist_B

        amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    plt.figure(figsize=(10, 5))
    plt.title("AB Extinction: AMY activity over time. Overlap=" + str(B_overlap))
    plt.plot(amy_responses)
    # Plot HIP/CTX distances to contexts A and B over time
    ba_dist_comp = np.array(ba_dists_B) / (np.array(ba_dists_A) + np.array(ba_dists_B))
    print("ba dist comp:", ba_dist_comp)
    plt.plot(np.arange(num_acquisition_trials, num_trials), ba_dist_comp, label="ba distance to B / (A + B)", linestyle='--', alpha=0.5)
    # Vertical dotted line at every acquisition trial
    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    # Horizontal bar indicating context over time
    plt.fill_between(x=np.arange(num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', label='Context A', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials - 1, num_trials), y1=-0.01, y2=-0.06, color='lightcoral', label='Context B', alpha=0.8)
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("AMY activity")
    plt.legend(loc='upper right')
    plt.show()

def amy_show_fear_renewal_ABA():
    num_runs = 10
    num_acquisition_trials = 20
    num_extinction_trials =  100
    num_renewal_trials = 100
    sleep_len = 165
    num_trials = num_acquisition_trials + num_extinction_trials + num_renewal_trials
    amy_responses_dict = {i: np.zeros(num_trials) for i in range(num_runs)}

    for run in range(num_runs):

        model = AmygdalaEngrams()
        online_learning(model, 2000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])

        B_overlap = 0.8
        B_diff = 1 - B_overlap
        hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
        units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
        cntx_B = cntx_A.copy()
        cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
        for i in range(units_to_change):
            cntx_B[model.SENSORY_CORTEX.N - units_to_change + i] = cntx_B_diff[i]
        cntx_B_vec = np.array([cntx_B[i] for i in cntx_B.keys()])

        amy_responses = np.zeros(num_trials)

        ba_dists_A = np.zeros(num_extinction_trials + num_renewal_trials)
        ba_dists_B = np.zeros(num_extinction_trials + num_renewal_trials)

        for i in range(num_acquisition_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 0.9)
            print(model.AMY_A.output)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[i] = pattern_converged_to_amy
            print("AMY converged activity:", pattern_converged_to_amy)
            if i == 0:
                cntx_A_pattern_ba = model.BA_N.log[model.BA_N.current_step-1]
                cntx_A_pattern_ba_vec = np.array([cntx_A_pattern_ba[i] for i in cntx_A_pattern_ba.keys()])
        amy_transition_phase(model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()
        for i in range(num_extinction_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_B, 0.0)
            if i == 0:
                cntx_B_pattern_ba = model.BA_N.log[model.BA_N.current_step-1]
                cntx_B_pattern_ba_vec = np.array([cntx_B_pattern_ba[i] for i in cntx_B_pattern_ba.keys()])
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            print(model.AMY_A.output)

            pattern_converged_to_ba = model.BA_N.log[model.BA_N.current_step-1]
            pattern_converged_to_ba_vec = np.array([pattern_converged_to_ba[i] for i in pattern_converged_to_ba.keys()])
            ba_dist_A = recall_metric(cntx_A_pattern_ba_vec, pattern_converged_to_ba_vec)
            ba_dist_B = recall_metric(cntx_B_pattern_ba_vec, pattern_converged_to_ba_vec)
            print("Step", i + num_acquisition_trials)
            print("ba distance A:", ba_dist_A, "ba distance B:", ba_dist_B)
            ba_dists_A[i] = ba_dist_A
            ba_dists_B[i] = ba_dist_B

            amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy
            print("AMY converged activity:", pattern_converged_to_amy)
        amy_transition_phase(model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()

        for i in range(num_renewal_trials):

            ba_n_activity = model.BA_N.log[model.BA_N.current_step-1]
            active_ban_cells_A_1 = [i for i in range(len(ba_n_activity)) if ba_n_activity[i] == 1]

            ba_n_to_p_connections = model.BA_P.feedback_connections['BA_N']
            W = ba_n_to_p_connections.W # (N_BAN, N_BAP)

            # For each P-cell, compute sum of weights from active BA(N) cells
            sums_of_weights = np.sum(W[active_ban_cells_A_1, :], axis=0)

            # Plot histogram of sums of weights
            if i == 0 and run == 0:
                print(sums_of_weights)
                fig, ax = plt.subplots(figsize=(9, 9))
                plt.hist(sums_of_weights, bins=np.arange(0, 40, 2))
                ax.axvline(x=np.exp(model.BA_P.firing_threshold), color='red', linestyle='--')
                plt.title("Context. A - Before Sleep", fontsize=24)
                plt.xlabel("Summed weight", fontsize=22)
                plt.ylabel("Frequency", fontsize=22)
                ax.tick_params(labelsize=22)
                # use log scale for y-axis
                plt.yscale('log')
                plt.yticks([1, 10, 100])
                plt.ylim(0.5, 250)
                plt.show()

            ba_n_activity = model.BA_N.log[model.BA_N.current_step-1]
            active_ban_cells_A_1 = [i for i in range(len(ba_n_activity)) if ba_n_activity[i] == 1]

            ba_n_to_p_connections = model.BA_I.feedback_connections['BA_N']
            W = ba_n_to_p_connections.W # (N_BAN, N_BAP)

            # For each P-cell, compute sum of weights from active BA(N) cells
            sums_of_weights = np.sum(W[active_ban_cells_A_1, :], axis=0)

            # Plot histogram of sums of weights
            if i == 0 and run == 0:
                print(sums_of_weights)
                fig, ax = plt.subplots(figsize=(9, 9))
                plt.hist(sums_of_weights, bins=np.arange(0, 100, 2))
                ax.axvline(x=np.exp(model.BA_I.firing_threshold), color='red', linestyle='--')
                plt.title("Context. A - Before Sleep", fontsize=24)
                plt.xlabel("Summed weight", fontsize=22)
                plt.ylabel("Frequency", fontsize=22)
                ax.tick_params(labelsize=22)
                # use log scale for y-axis
                plt.yscale('log')
                plt.yticks([1, 10, 100])
                plt.ylim(0.5, 250)
                plt.show()

            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]

            pattern_converged_to_ba = model.BA_N.log[model.BA_N.current_step-1]
            pattern_converged_to_ba_vec = np.array([pattern_converged_to_ba[i] for i in pattern_converged_to_ba.keys()])
            ba_dist_A = recall_metric(cntx_A_pattern_ba_vec, pattern_converged_to_ba_vec)
            ba_dist_B = recall_metric(cntx_B_pattern_ba_vec, pattern_converged_to_ba_vec)
            print("Step", i + num_acquisition_trials + num_extinction_trials)
            print("ba distance A:", ba_dist_A, "ba distance B:", ba_dist_B)
            ba_dists_A[i + num_extinction_trials] = ba_dist_A
            ba_dists_B[i + num_extinction_trials] = ba_dist_B

            amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy
        
        amy_responses_dict[run] = amy_responses

    # Compute the average responses
    avg_amy_responses = np.zeros(num_trials)
    amy_responses_list = np.array(list(amy_responses_dict.values()))
    for run in range(num_runs):
        avg_amy_responses += amy_responses_dict[run]
    avg_amy_responses /= num_runs
    amy_response_std = np.std(amy_responses_list, axis=0)

    plt.figure(figsize=(10, 3))
    plt.title("ABA Renewal. Overlap A / B = " + str(B_overlap), fontsize=22)
    plt.plot(avg_amy_responses)
    # Show standard deviation as shaded area
    plt.fill_between(x=np.arange(num_trials), y1=avg_amy_responses - amy_response_std, y2=avg_amy_responses + amy_response_std, alpha=0.25)
    # Plot HIP/CTX distances to contexts A and B over time
    ba_dist_comp = np.array(ba_dists_B) / (np.array(ba_dists_A) + np.array(ba_dists_B))
    print("ba dist comp:", ba_dist_comp)
    # plt.plot(np.arange(num_acquisition_trials, num_trials), ba_dist_comp, label="ba distance to B / (A + B)", linestyle='--', alpha=0.5)
    # Vertical dotted line at every acquisition trial
    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    # Horizontal bar indicating context over time
    plt.fill_between(x=np.arange(num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', label='Context A', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials - 1, num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='lightcoral', label='Context B', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + num_extinction_trials - 1, num_trials), y1=-0.01, y2=-0.06, color='skyblue', alpha=0.8)
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial", fontsize=20)
    plt.ylabel("CeM activity", fontsize=20)
    plt.legend(fontsize=18, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Create some space at the bottom of the figure to prevent the xlabel from being cut off
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def amy_show_fear_renewal_ABC():
    num_runs = 5
    num_acquisition_trials = 20
    num_extinction_trials = 100
    num_renewal_trials = 100
    num_trials = num_acquisition_trials + num_extinction_trials + num_renewal_trials
    amy_responses_dict = {i: np.zeros(num_trials) for i in range(num_runs)}

    for run in range(num_runs):

        model = AmygdalaEngrams()
        online_learning(model, 1000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])

        B_overlap = 0.8
        B_diff = 1 - B_overlap
        hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
        units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
        cntx_B = cntx_A.copy()
        cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
        for i in range(units_to_change):
            cntx_B[model.SENSORY_CORTEX.N - units_to_change + i] = cntx_B_diff[i]
        cntx_B_vec = np.array([cntx_B[i] for i in cntx_B.keys()])

        cntx_C = cntx_A.copy()
        cntx_C_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
        for i in range(units_to_change):
            cntx_C[model.SENSORY_CORTEX.N - units_to_change + i] = cntx_C_diff[i]
        cntx_C_vec = np.array([cntx_C[i] for i in cntx_C.keys()])

        amy_responses = np.zeros(num_trials)

        for i in range(num_acquisition_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 1.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step - 1]
            amy_responses[i] = pattern_converged_to_amy

        for i in range(num_extinction_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_B, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step - 1]
            amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy

        for i in range(num_renewal_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_C, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step - 1]
            amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy

        amy_responses_dict[run] = amy_responses

    # Compute the average and standard deviation of responses
    amy_responses_list = np.array(list(amy_responses_dict.values()))
    avg_amy_responses = amy_responses_list.mean(axis=0)
    amy_response_std = amy_responses_list.std(axis=0)

    # Plotting
    plt.figure(figsize=(10, 3))
    plt.title("ABC Renewal. Overlap A / B, A / C = " + str(B_overlap), fontsize=22)
    plt.plot(avg_amy_responses)
    plt.fill_between(
        x=np.arange(num_trials),
        y1=avg_amy_responses - amy_response_std,
        y2=avg_amy_responses + amy_response_std,
        alpha=0.25
    )

    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    plt.fill_between(x=np.arange(num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', label='Context A', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials - 1, num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='lightcoral', label='Context B', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + num_extinction_trials - 1, num_trials), y1=-0.01, y2=-0.06, color='lightgreen', alpha=0.8, label='Context C')
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial", fontsize=20)
    plt.ylabel("CeM activity", fontsize=20)
    plt.legend(fontsize=18, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def amy_show_fear_renewal_ABCDA():
    """
    Like ABA renewal, but using 3 extinction contexts all overlap with A to the same extent
    but in different ways. 
    """
    model = AmygdalaEngrams()
    online_learning(model, 100)

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
    cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])

    B_overlap = 0.8
    B_diff = 1 - B_overlap
    hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
    units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
    cntx_B = cntx_A.copy()
    cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    # Choose a random selection of hypercolumns to change
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    print("Random hcs:", random_hcs)
    print("Length context B:", len(cntx_B))
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_B[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_B_diff[j * model.SENSORY_CORTEX.units_per_hc + i]
    cntx_B_vec = np.array([cntx_B[i] for i in cntx_B.keys()])
    # The same for context C and D
    cntx_C = cntx_A.copy()
    cntx_C_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_C[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_C_diff[j * model.SENSORY_CORTEX.units_per_hc + i]
    cntx_C_vec = np.array([cntx_C[i] for i in cntx_C.keys()])
    cntx_D = cntx_A.copy()
    cntx_D_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_D[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_D_diff[j * model.SENSORY_CORTEX.units_per_hc + i]
    cntx_D_vec = np.array([cntx_D[i] for i in cntx_D.keys()])
    
    num_acquisition_trials = 10
    num_extinction_trials = 30
    num_renewal_trials = 15
    num_trials = num_acquisition_trials + 3 * num_extinction_trials + num_renewal_trials
    amy_responses = np.zeros(num_trials)

    # Acquisition in A
    for i in range(num_acquisition_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_A, 1.0)
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Extinction in B
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_B, 0.0)
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Extinction in C
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_C, 0.0)
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Extinction in D
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_D, 0.0)
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + 2 * num_extinction_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Renewal in A
    for i in range(num_renewal_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_A, 0.0)
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + 3 * num_extinction_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)

    plt.figure(figsize=(10, 5))
    plt.title("AB(CD)A Renewal: AMY activity over time. Overlap=" + str(B_overlap))
    plt.plot(amy_responses)
    # Vertical dotted line at every acquisition trial
    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    # Horizontal bar indicating context over time
    plt.fill_between(x=np.arange(num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', label='Context A', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials - 1, num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='lightcoral', label='Context B', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + num_extinction_trials - 1, num_acquisition_trials + 2 * num_extinction_trials), y1=-0.01, y2=-0.06, color='lightgreen', label='Context C', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + 2 * num_extinction_trials - 1, num_acquisition_trials + 3 * num_extinction_trials), y1=-0.01, y2=-0.06, color='lightyellow', label='Context D', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + 3 * num_extinction_trials - 1, num_trials), y1=-0.01, y2=-0.06, color='skyblue', alpha=0.8)
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("AMY activity")
    plt.legend()
    plt.show()

def amy_show_fear_renewal_ABCDA_same_overlap():
    """
    Like ABA renewal, but using 3 extinction contexts all overlap with A to the same extent
    and in the same way.
    """
    model = AmygdalaEngrams()
    online_learning(model, 100)

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
    cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])

    B_overlap = 0.725
    B_diff = 1 - B_overlap
    hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
    units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
    cntx_B = cntx_A.copy()
    cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    # Choose a random selection of hypercolumns to change
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    print("Random hcs:", random_hcs)
    print("Length context B:", len(cntx_B))
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_B[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_B_diff[j * model.SENSORY_CORTEX.units_per_hc + i]
    cntx_B_vec = np.array([cntx_B[i] for i in cntx_B.keys()])
    # The same for context C and D
    cntx_C = cntx_A.copy()
    cntx_C_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_C[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_C_diff[j * model.SENSORY_CORTEX.units_per_hc + i]
    cntx_C_vec = np.array([cntx_C[i] for i in cntx_C.keys()])
    cntx_D = cntx_A.copy()
    cntx_D_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_D[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_D_diff[j * model.SENSORY_CORTEX.units_per_hc + i]
    cntx_D_vec = np.array([cntx_D[i] for i in cntx_D.keys()])
    
    num_accustom_trials = 300
    num_acquisition_trials = 25
    num_extinction_trials = 13
    num_renewal_trials = 15
    num_trials = num_acquisition_trials + 3 * num_extinction_trials + num_renewal_trials
    amy_responses = np.zeros(num_trials)

    # Accustom the model to all contexts
    amy_transition_phase(model, Phase.PERCEPTION)
    for i in range(num_accustom_trials):
        # Choose a random context to present to the model
        random_index = np.random.randint(0, 3)
        random_us = 0.21
        if random_index == 0:
            model.update(cntx_A, random_us)
        elif random_index == 1:
            model.update(cntx_B, random_us)
        elif random_index == 2:
            model.update(cntx_C, random_us)
        else:
            model.update(cntx_D, random_us)
    # Acquisition in A
    for i in range(num_acquisition_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_A, 1.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_A)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Extinction in B
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_B, 0.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_B)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Extinction in C
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_C, 0.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_C)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Extinction in D
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_D, 0.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_D)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + 2 * num_extinction_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
    # Renewal in A
    for i in range(num_renewal_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_A, 0.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_A)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[num_acquisition_trials + 3 * num_extinction_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)

    plt.figure(figsize=(10, 5))
    plt.title("AB(CD)A Renewal: AMY activity over time. Overlap=" + str(B_overlap))
    plt.plot(amy_responses)
    # Vertical dotted line at every acquisition trial
    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    # Horizontal bar indicating context over time
    plt.fill_between(x=np.arange(num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', label='Context A', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials - 1, num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='lightcoral', label='Context B', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + num_extinction_trials - 1, num_acquisition_trials + 2 * num_extinction_trials), y1=-0.01, y2=-0.06, color='lightgreen', label='Context C', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + 2 * num_extinction_trials - 1, num_acquisition_trials + 3 * num_extinction_trials), y1=-0.01, y2=-0.06, color='lightyellow', label='Context D', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + 3 * num_extinction_trials - 1, num_trials), y1=-0.01, y2=-0.06, color='skyblue', alpha=0.8)
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("AMY activity")
    plt.legend()
    plt.show()

def amy_show_fear_renewal_ABC_BC_sim():
    """
    Like normal ABC renwal, but with C being more similar to B than A.
    """
    model = AmygdalaEngrams()
    online_learning(model, 100)

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
    cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])

    B_overlap = 0.8
    B_diff = 1 - B_overlap
    hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
    units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
    cntx_B = cntx_A.copy()
    cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_B[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_B_diff[j * model.SENSORY_CORTEX.units_per_hc + i]

    C_overlap = 0.7
    C_diff = 1 - C_overlap
    hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * C_diff)
    units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
    cntx_C = cntx_B.copy()
    cntx_C_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_C[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_C_diff[j * model.SENSORY_CORTEX.units_per_hc + i]

    # Compute overlap between A and C
    overlap = 0
    for i in range(model.SENSORY_CORTEX.N):
        if cntx_A[i] == 1 and cntx_C[i] == 1:
            overlap += 1
    overlap /= model.SENSORY_CORTEX.num_hcs

    num_accustom_trials = 30
    num_acquisition_trials = 20
    num_extinction_trials = 25
    num_renewal_trials = 15
    num_trials = num_acquisition_trials + num_extinction_trials + num_renewal_trials
    amy_responses = np.zeros(num_trials)

    for i in range(num_accustom_trials):
        random_index = np.random.randint(0, 3)
        random_us = np.random.uniform(0.20, 0.25)
        if random_index == 0:
            model.update(cntx_A, random_us)
        elif random_index == 1:
            model.update(cntx_B, random_us)
        else:
            model.update(cntx_C, random_us)
    for i in range(num_acquisition_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_A, 1.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_A)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
        # print active I cells 
        ba_log = model.BA_I.log[model.BA_I.current_step-1]
        indices = [i for i in range(len(ba_log)) if ba_log[i] > 0.5]
        print("Active I cells:", indices)
        print("Active P cells:", [i for i in range(len(model.BA_P.log[model.BA_P.current_step-1])) if model.BA_P.log[model.BA_P.current_step-1][i] > 0.5])
    for i in range(num_extinction_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_B, 0.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_B)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]

        amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
        print("active I cells:", [i for i in range(len(model.BA_I.log[model.BA_I.current_step-1])) if model.BA_I.log[model.BA_I.current_step-1][i] > 0.5])
        print("active P cells:", [i for i in range(len(model.BA_P.log[model.BA_P.current_step-1])) if model.BA_P.log[model.BA_P.current_step-1][i] > 0.5])
    for i in range(num_renewal_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_C, 0.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_C)
        steps_before_convergence = 0
        while steps_before_convergence < 30:
            model.update()
            steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]

        amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy
        print("AMY converged activity:", pattern_converged_to_amy)
        print("active I cells:", [i for i in range(len(model.BA_I.log[model.BA_I.current_step-1])) if model.BA_I.log[model.BA_I.current_step-1][i] > 0.5])
        print("active P cells:", [i for i in range(len(model.BA_P.log[model.BA_P.current_step-1])) if model.BA_P.log[model.BA_P.current_step-1][i] > 0.5])
    plt.figure(figsize=(10, 5))
    plt.title("ABC Renewal: Overlap A/B & A/C & B/C=" + str(B_overlap) + " & " + str(overlap) + " & " + str(C_overlap))
    plt.plot(amy_responses)
    # Vertical dotted line at every acquisition trial
    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    # Horizontal bar indicating context over time
    plt.fill_between(x=np.arange(num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', label='Context A', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials - 1, num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='lightcoral', label='Context B', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + num_extinction_trials - 1, num_trials), y1=-0.01, y2=-0.06, color='lightgreen', label='Context C', alpha=0.8)
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("AMY activity")
    plt.legend(loc='lower right')
    plt.show()

def amy_show_fear_renewal_AAB():
    num_runs = 5
    num_acquisition_trials = 20
    num_extinction_trials = 100
    num_renewal_trials = 100
    num_trials = num_acquisition_trials + num_extinction_trials + num_renewal_trials
    amy_responses_dict = {i: np.zeros(num_trials) for i in range(num_runs)}

    for run in range(num_runs):

        model = AmygdalaEngrams()
        online_learning(model, 2000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])

        B_overlap = 0.8
        B_diff = 1 - B_overlap
        hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
        units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
        cntx_B = cntx_A.copy()
        cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
        for i in range(units_to_change):
            cntx_B[model.SENSORY_CORTEX.N - units_to_change + i] = cntx_B_diff[i]
        cntx_B_vec = np.array([cntx_B[i] for i in cntx_B.keys()])

        amy_responses = np.zeros(num_trials)

        for i in range(num_acquisition_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 0.9)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step - 1]
            amy_responses[i] = pattern_converged_to_amy

        for i in range(num_extinction_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step - 1]
            amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy

        for i in range(num_renewal_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_B, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step - 1]
            amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy

        amy_responses_dict[run] = amy_responses

    # Compute the average and standard deviation of responses
    amy_responses_list = np.array(list(amy_responses_dict.values()))
    avg_amy_responses = amy_responses_list.mean(axis=0)
    amy_response_std = amy_responses_list.std(axis=0)

    # Plotting
    plt.figure(figsize=(10, 3))
    plt.title("AAB Renewal. Overlap A / B = " + str(B_overlap), fontsize=22)
    plt.plot(avg_amy_responses)
    plt.fill_between(
        x=np.arange(num_trials),
        y1=avg_amy_responses - amy_response_std,
        y2=avg_amy_responses + amy_response_std,
        alpha=0.25
    )

    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    plt.fill_between(x=np.arange(num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', label='Context A', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials - 1, num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='skyblue', alpha=0.8)
    plt.fill_between(x=np.arange(num_acquisition_trials + num_extinction_trials - 1, num_trials), y1=-0.01, y2=-0.06, color='lightcoral', alpha=0.8, label='Context B')
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial", fontsize=20)
    plt.ylabel("CeM activity", fontsize=20)
    plt.legend(fontsize=18, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.subplots_adjust(bottom=0.25)
    plt.show()


def amy_show_generalization_consolidation():
    num_conditioned_trials = 25
    num_unconditioned_trials = 25
    num_delay_days = 15
    perceptions_per_day = 3
    delay_context_len = 25
    perception_phase_len = perceptions_per_day * delay_context_len

    num_runs = 10
    amy_responses_A_dict = {i: np.zeros(1 + num_delay_days) for i in range(num_runs)}
    amy_responses_B_dict = {i: np.zeros(1 + num_delay_days) for i in range(num_runs)}
    amy_responses_C_dict = {i: np.zeros(1 + num_delay_days) for i in range(num_runs)}
    validity_scores_A_dict = {i: np.zeros(1 + num_delay_days) for i in range(num_runs)}
    validity_scores_B_dict = {i: np.zeros(1 + num_delay_days) for i in range(num_runs)}
    validity_scores_C_dict = {i: np.zeros(1 + num_delay_days) for i in range(num_runs)}

    for run in range(num_runs):
        print("Run:", run)
        model = AmygdalaEngrams()
        online_learning(model, 2000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        B_overlap = 0.6
        B_diff = 1 - B_overlap
        hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
        units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
        cntx_B = cntx_A.copy()
        cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
        random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
        for j, hc in enumerate(random_hcs):
            for i in range(model.SENSORY_CORTEX.units_per_hc):
                cntx_B[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_B_diff[j * model.SENSORY_CORTEX.units_per_hc + i]

        C_overlap = 0.0
        C_diff = 1 - C_overlap
        hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * C_diff)
        units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
        cntx_C = cntx_A.copy()
        cntx_C_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
        random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
        for j, hc in enumerate(random_hcs):
            for i in range(model.SENSORY_CORTEX.units_per_hc):
                cntx_C[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_C_diff[j * model.SENSORY_CORTEX.units_per_hc + i]

        amy_responses_A = np.zeros(1 + num_delay_days)
        amy_responses_B = np.zeros(1 + num_delay_days)
        amy_responses_C = np.zeros(1 + num_delay_days)
        validity_scores_A = np.zeros(1 + num_delay_days)
        validity_scores_B = np.zeros(1 + num_delay_days)
        validity_scores_C = np.zeros(1 + num_delay_days)

        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_conditioned_trials):
            model.update(cntx_A, 0.9)
        for i in range(num_unconditioned_trials):
            model.update(cntx_B, 0.0)
        for i in range(num_unconditioned_trials):
            model.update(cntx_C, 0.0)
        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_A)
        cntx_A_vec = np.array([model.CTX.log[model.CTX.current_step-1][i] for i in model.CTX.log[model.CTX.current_step-1].keys()])
        cntx_A_vec_hip = np.array([model.HIP.log[model.HIP.current_step-1][i] for i in model.HIP.log[model.HIP.current_step-1].keys()])
        cntx_A_vec_ba = np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])
        steps_before_convergence = 0
        while steps_before_convergence < 30:
                model.update()
                steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses_A[0] = pattern_converged_to_amy
        validity_scores_A[0] = model.validity_score
        model.update(cntx_B)
        cntx_B_vec = np.array([model.CTX.log[model.CTX.current_step-1][i] for i in model.CTX.log[model.CTX.current_step-1].keys()])
        cntx_B_vec_hip = np.array([model.HIP.log[model.HIP.current_step-1][i] for i in model.HIP.log[model.HIP.current_step-1].keys()])
        cntx_B_vec_ba = np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])
        steps_before_convergence = 0
        while steps_before_convergence < 30:
                model.update()
                steps_before_convergence += 1  
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses_B[0] = pattern_converged_to_amy
        validity_scores_B[0] = model.validity_score
        model.update(cntx_C)
        cntx_C_vec = np.array([model.CTX.log[model.CTX.current_step-1][i] for i in model.CTX.log[model.CTX.current_step-1].keys()])
        cntx_C_vec_hip = np.array([model.HIP.log[model.HIP.current_step-1][i] for i in model.HIP.log[model.HIP.current_step-1].keys()])
        cntx_C_vec_ba = np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])
        steps_before_convergence = 0
        while steps_before_convergence < 30:
                model.update()
                steps_before_convergence += 1
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses_C[0] = pattern_converged_to_amy
        validity_scores_C[0] = model.validity_score

        # Sleep
        amy_transition_phase(model, Phase.SLEEP)
        for _ in range(165):
            model.update()

        for day in range(num_delay_days):
            amy_transition_phase(model, Phase.PERCEPTION)
            for _ in range(3):
                cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
                us_random = 0.9 * np.random.beta(1.25, 5)
                print("Random US:", us_random)
                for _ in range(delay_context_len):
                    model.update(cntx_random, us_random)
            amy_transition_phase(model, Phase.RECALL)
            model.update(cntx_A)
            steps_before_convergence = 0
            while steps_before_convergence < 10:
                model.update()
                steps_before_convergence += 1
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses_A[day + 1] = pattern_converged_to_amy
            validity_scores_A[day + 1] = model.validity_score
            pattern_converged_to_hip = model.HIP.log[model.HIP.current_step-1]
            pattern_converged_to_hip_vec = np.array([pattern_converged_to_hip[i] for i in pattern_converged_to_hip.keys()])
            hip_dist_A = recall_metric(cntx_A_vec_hip, pattern_converged_to_hip_vec)
            hip_dist_B = recall_metric(cntx_B_vec_hip, pattern_converged_to_hip_vec)
            hip_dist_C = recall_metric(cntx_C_vec_hip, pattern_converged_to_hip_vec)
            # Print AMY activity and which pattern CTX converged to
            pattern_converged_to_ctx = model.CTX.log[model.CTX.current_step-1]
            pattern_converged_to_ctx_vec = np.array([pattern_converged_to_ctx[i] for i in pattern_converged_to_ctx.keys()])
            ctx_dist_A = recall_metric(cntx_A_vec, pattern_converged_to_ctx_vec)
            ctx_dist_B = recall_metric(cntx_B_vec, pattern_converged_to_ctx_vec)
            ctx_dist_C = recall_metric(cntx_C_vec, pattern_converged_to_ctx_vec)
            print("Recall A on day", day)
            print(model.AMY_C.log[model.AMY_C.current_step-1])
            print("HIP distance A:", hip_dist_A, "HIP distance B:", hip_dist_B, "HIP distance C:", hip_dist_C)
            print("CTX distance A:", ctx_dist_A, "CTX distance B:", ctx_dist_B, "CTX distance C:", ctx_dist_C)
            print("BA distance A:", recall_metric(cntx_A_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])), "BA distance B:", recall_metric(cntx_B_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])), "BA distance C:", recall_metric(cntx_C_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])))
            print("P cell activity:", np.mean(list(model.BA_P.log[model.BA_P.current_step-1].values())))
            print("I cell activity:", np.mean(list(model.BA_I.log[model.BA_I.current_step-1].values())))
            print("Validity score:", model.validity_score)
            print("EC_OUT activity:", np.mean(model.EC_OUT.output))
            print("EC_IN activity:", np.mean(model.EC_IN.output))
            model.update(cntx_B)
            steps_before_convergence = 0
            while steps_before_convergence < 10:
                model.update()
                steps_before_convergence += 1
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses_B[day + 1] = pattern_converged_to_amy
            validity_scores_B[day + 1] = model.validity_score
            pattern_converged_to_hip = model.HIP.log[model.HIP.current_step-1]
            pattern_converged_to_hip_vec = np.array([pattern_converged_to_hip[i] for i in pattern_converged_to_hip.keys()])
            hip_dist_A = recall_metric(cntx_A_vec_hip, pattern_converged_to_hip_vec)
            hip_dist_B = recall_metric(cntx_B_vec_hip, pattern_converged_to_hip_vec)   
            hip_dist_C = recall_metric(cntx_C_vec_hip, pattern_converged_to_hip_vec)
            # Print AMY activity and which pattern CTX converged to
            pattern_converged_to_ctx = model.CTX.log[model.CTX.current_step-1]
            pattern_converged_to_ctx_vec = np.array([pattern_converged_to_ctx[i] for i in pattern_converged_to_ctx.keys()])
            ctx_dist_A = recall_metric(cntx_A_vec, pattern_converged_to_ctx_vec)
            ctx_dist_B = recall_metric(cntx_B_vec, pattern_converged_to_ctx_vec)
            ctx_dist_C = recall_metric(cntx_C_vec, pattern_converged_to_ctx_vec)
            print("Recall B on day", day)
            print(model.AMY_C.log[model.AMY_C.current_step-1])
            print("HIP distance A:", hip_dist_A, "HIP distance B:", hip_dist_B, "HIP distance C:", hip_dist_C)
            print("CTX distance A:", ctx_dist_A, "CTX distance B:", ctx_dist_B, "CTX distance C:", ctx_dist_C)
            print("BA distance A:", recall_metric(cntx_A_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])), "BA distance B:", recall_metric(cntx_B_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])), "BA distance C:", recall_metric(cntx_C_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])))
            print("P cell activity:", np.mean(list(model.BA_P.log[model.BA_P.current_step-1].values())))
            print("I cell activity:", np.mean(list(model.BA_I.log[model.BA_I.current_step-1].values())))
            print("Validity score:", model.validity_score)
            print("EC_OUT activity:", np.mean(model.EC_OUT.output))
            print("EC_IN activity:", np.mean(model.EC_IN.output))
            model.update(cntx_C)
            steps_before_convergence = 0
            while steps_before_convergence < 10:
                model.update()
                steps_before_convergence += 1
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses_C[day + 1] = pattern_converged_to_amy
            validity_scores_C[day + 1] = model.validity_score
            pattern_converged_to_hip = model.HIP.log[model.HIP.current_step-1]
            pattern_converged_to_hip_vec = np.array([pattern_converged_to_hip[i] for i in pattern_converged_to_hip.keys()])
            hip_dist_A = recall_metric(cntx_A_vec_hip, pattern_converged_to_hip_vec)
            hip_dist_B = recall_metric(cntx_B_vec_hip, pattern_converged_to_hip_vec)
            hip_dist_C = recall_metric(cntx_C_vec_hip, pattern_converged_to_hip_vec)
            # Print AMY activity and which pattern CTX converged to
            pattern_converged_to_ctx = model.CTX.log[model.CTX.current_step-1]
            pattern_converged_to_ctx_vec = np.array([pattern_converged_to_ctx[i] for i in pattern_converged_to_ctx.keys()])
            ctx_dist_A = recall_metric(cntx_A_vec, pattern_converged_to_ctx_vec)
            ctx_dist_B = recall_metric(cntx_B_vec, pattern_converged_to_ctx_vec)
            ctx_dist_C = recall_metric(cntx_C_vec, pattern_converged_to_ctx_vec)
            print("Recall C on day", day)
            print(model.AMY_C.log[model.AMY_C.current_step-1])
            print("HIP distance A:", hip_dist_A, "HIP distance B:", hip_dist_B, "HIP distance C:", hip_dist_C)
            print("CTX distance A:", ctx_dist_A, "CTX distance B:", ctx_dist_B, "CTX distance C:", ctx_dist_C)
            print("BA distance A:", recall_metric(cntx_A_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])), "BA distance B:", recall_metric(cntx_B_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])), "BA distance C:", recall_metric(cntx_C_vec_ba, np.array([model.BA_N.log[model.BA_N.current_step-1][i] for i in model.BA_N.log[model.BA_N.current_step-1].keys()])))
            print("P cell activity:", np.mean(list(model.BA_P.log[model.BA_P.current_step-1].values())))
            print("I cell activity:", np.mean(list(model.BA_I.log[model.BA_I.current_step-1].values())))
            print("Validity score:", model.validity_score)
            print("EC_OUT activity:", np.mean(model.EC_OUT.output))
            print("EC_IN activity:", np.mean(model.EC_IN.output))

            amy_transition_phase(model, Phase.SLEEP)
            for j in range(165):
                model.update()

        amy_responses_A_dict[run] = amy_responses_A
        amy_responses_B_dict[run] = amy_responses_B
        amy_responses_C_dict[run] = amy_responses_C
        validity_scores_A_dict[run] = validity_scores_A
        validity_scores_B_dict[run] = validity_scores_B
        validity_scores_C_dict[run] = validity_scores_C


    avg_amy_responses_A = np.zeros(1 + num_delay_days)
    avg_amy_responses_B = np.zeros(1 + num_delay_days)
    avg_amy_responses_C = np.zeros(1 + num_delay_days)
    avg_validity_scores_A = np.zeros(1 + num_delay_days)
    avg_validity_scores_B = np.zeros(1 + num_delay_days)
    avg_validity_scores_C = np.zeros(1 + num_delay_days)

    for run in range(num_runs):
        avg_amy_responses_A += amy_responses_A_dict[run]
        avg_amy_responses_B += amy_responses_B_dict[run]
        avg_amy_responses_C += amy_responses_C_dict[run]
        avg_validity_scores_A += validity_scores_A_dict[run]
        avg_validity_scores_B += validity_scores_B_dict[run]
        avg_validity_scores_C += validity_scores_C_dict[run]

    avg_amy_responses_A /= num_runs
    avg_amy_responses_B /= num_runs
    avg_amy_responses_C /= num_runs
    avg_validity_scores_A /= num_runs
    avg_validity_scores_B /= num_runs
    avg_validity_scores_C /= num_runs

    # Calculate standard deviations for shading
    std_amy_responses_A = np.zeros(1 + num_delay_days)
    std_amy_responses_B = np.zeros(1 + num_delay_days)
    std_amy_responses_C = np.zeros(1 + num_delay_days)
    std_validity_scores_B = np.zeros(1 + num_delay_days)

    for run in range(num_runs):
        std_amy_responses_A += (amy_responses_A_dict[run] - avg_amy_responses_A)**2
        std_amy_responses_B += (amy_responses_B_dict[run] - avg_amy_responses_B)**2
        std_amy_responses_C += (amy_responses_C_dict[run] - avg_amy_responses_C)**2
        std_validity_scores_B += (validity_scores_B_dict[run] - avg_validity_scores_B)**2

    std_amy_responses_A = np.sqrt(std_amy_responses_A / num_runs)
    std_amy_responses_B = np.sqrt(std_amy_responses_B / num_runs)
    std_amy_responses_C = np.sqrt(std_amy_responses_C / num_runs)
    std_validity_scores_B = np.sqrt(std_validity_scores_B / num_runs)

    # Plotting with visual upgrades
    plt.figure(figsize=(9, 9))
    plt.title("Generalization increases as fear memory ages", fontsize=22)

    # Plot mean with standard deviation shading
    plt.fill_between(range(1 + num_delay_days), 
                    avg_amy_responses_A - std_amy_responses_A, 
                    avg_amy_responses_A + std_amy_responses_A, 
                    color='blue', alpha=0.2)
    plt.fill_between(range(1 + num_delay_days), 
                    avg_amy_responses_B - std_amy_responses_B, 
                    avg_amy_responses_B + std_amy_responses_B, 
                    color='orange', alpha=0.2)
    plt.fill_between(range(1 + num_delay_days), 
                    avg_amy_responses_C - std_amy_responses_C, 
                    avg_amy_responses_C + std_amy_responses_C, 
                    color='green', alpha=0.2)
    plt.fill_between(range(1 + num_delay_days), 
                    avg_validity_scores_B - std_validity_scores_B, 
                    avg_validity_scores_B + std_validity_scores_B, 
                    color='grey', alpha=0.2)

    # Plot mean lines with increased line weights
    plt.plot(avg_amy_responses_A, label="A (conditioned)", linewidth=2.5)
    plt.plot(avg_amy_responses_B, label="B (overlap=" + str(B_overlap) + ")", linewidth=2.5)
    plt.plot(avg_amy_responses_C, label="C (overlap=" + str(C_overlap) + ")", linewidth=2.5)
    plt.plot(avg_validity_scores_B, label="HIP Recall score B", linestyle='--', linewidth=2.0)

    # Horizontal, grey dotted line at validity threshold
    plt.axhline(y=model.B_amy, color='grey', linestyle='--', alpha=0.5)

    plt.xlabel("Day", fontsize=22)
    plt.ylabel("CeM activity", fontsize=22)
    plt.ylim(-0.01, 1.01)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def amy_within_session_extinction():
    model = AmygdalaEngrams()

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

    num_acquisition_trials = 50
    num_extinction_trials = 150
    num_days = 2
    sleep_len = 165
    num_trials = num_acquisition_trials + num_days * num_extinction_trials
    amy_responses = np.zeros(num_trials)

    online_learning(model, 200)

    for i in range(num_acquisition_trials):
        amy_transition_phase(model, Phase.PERCEPTION)
        model.update(cntx_A, 1.0)
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        amy_responses[i] = pattern_converged_to_amy
    for day in range(num_days):
        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_extinction_trials):
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[num_acquisition_trials + day * num_extinction_trials + i] = pattern_converged_to_amy
        amy_transition_phase(model, Phase.SLEEP)
        if day < num_days - 1:
            for i in range(sleep_len):
                model.update()
                
    plt.figure()
    plt.title("Within-session extinction: AMY activity over time")
    plt.plot(amy_responses)
    # Vertical lines denoting day transitions
    for i in range(num_days):
        plt.axvline(x=num_acquisition_trials + i * num_extinction_trials, color='grey', linestyle='--', alpha=0.5)
    plt.xlabel("Trial")
    plt.ylabel("AMY_C output")
    plt.ylim(-0.01, 1.01)
    plt.show()

    import pandas as pd
    df = pd.DataFrame(model.IL.log)
    plt.figure()
    import seaborn as sns
    sns.heatmap(df)
    plt.title("Infralimbic cortex activity over time")
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.show()

    df = pd.DataFrame(model.IL_noise.log)
    plt.figure()
    sns.heatmap(df)
    plt.show()

def amy_show_fear_acquisition_daily_ABC():
    model = AmygdalaEngrams()
    online_learning(model, 1000)

    num_days = 10
    num_acquisition_trials = 100
    num_extinction_trials = 100
    num_renewal_trials = 100
    num_daily_trials = (num_acquisition_trials + num_extinction_trials + num_renewal_trials)
    num_trials = num_days * num_daily_trials
    amy_responses = np.zeros(num_trials)

    B_overlap = 0.0
    B_diff = 1 - B_overlap
    hcs_to_change_B = int(model.SENSORY_CORTEX.num_hcs * B_diff)
    units_to_change_B = hcs_to_change_B * model.SENSORY_CORTEX.units_per_hc

    C_overlap = 0.0
    C_diff = 1 - C_overlap
    hcs_to_change_C = int(model.SENSORY_CORTEX.num_hcs * C_diff)
    units_to_change_C = hcs_to_change_C * model.SENSORY_CORTEX.units_per_hc

    
    for day in range(num_days):
        amy_transition_phase(model, Phase.PERCEPTION)
        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_A_vec = np.array([cntx_A[i] for i in cntx_A.keys()])
        
        cntx_B = cntx_A.copy()
        cntx_B_diff = gen_random_simple_pattern(units_to_change_B, hcs_to_change_B)
        random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change_B, replace=False)
        for j, hc in enumerate(random_hcs):
            for i in range(model.SENSORY_CORTEX.units_per_hc):
                cntx_B[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_B_diff[j * model.SENSORY_CORTEX.units_per_hc + i]
        cntx_C = cntx_A.copy()
        cntx_C_diff = gen_random_simple_pattern(units_to_change_C, hcs_to_change_C)
        random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change_C, replace=False)
        for j, hc in enumerate(random_hcs):
            for i in range(model.SENSORY_CORTEX.units_per_hc):
                cntx_C[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_C_diff[j * model.SENSORY_CORTEX.units_per_hc + i]

        D = day * num_daily_trials
        for i in range(num_acquisition_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_A, 0.5)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[D + i] = pattern_converged_to_amy
            print("AMY converged activity:", pattern_converged_to_amy)
        for i in range(num_extinction_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_B, 0.5)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[D + num_acquisition_trials + i] = pattern_converged_to_amy
            print("AMY converged activity:", pattern_converged_to_amy)
        for i in range(num_renewal_trials):
            amy_transition_phase(model, Phase.PERCEPTION)
            model.update(cntx_C, 0.5)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[D + num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy
            print("AMY converged activity:", pattern_converged_to_amy)

        if day < num_days - 1:
            amy_transition_phase(model, Phase.SLEEP)
            for j in range(165):
                model.update()
    plt.figure(figsize=(10, 5))
    plt.title("Daily fear acquisition in 3 novel contexts. Maintenance rate (r) = " + str(model.maintenance_rate))
    plt.plot(amy_responses)
    for d in range(num_days):
        D = d * num_daily_trials
        plt.fill_between(x=np.arange(D, D + num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', alpha=0.8)
        plt.fill_between(x=np.arange(D + num_acquisition_trials - 1, D + num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='lightcoral', alpha=0.8)
        plt.fill_between(x=np.arange(D + num_acquisition_trials + num_extinction_trials - 1, D + num_daily_trials), y1=-0.01, y2=-0.06, color='lightgreen', alpha=0.8)
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("AMY activity")
    plt.show()

    # Now plot the mean within bins of width num_acquisition_trials
    bin_width = num_acquisition_trials
    num_bins = num_trials // bin_width
    mean_responses = np.zeros(num_bins)
    for i in range(num_bins):
        mean_responses[i] = np.mean(amy_responses[i*bin_width:(i+1)*bin_width])
    plt.figure(figsize=(10, 5))
    plt.title("Daily fear acquisition in 3 novel contexts.: Mean AMY activity per context. Maintenance rate (r) =" + str(model.maintenance_rate))
    plt.plot(mean_responses)
    plt.xlabel("Acquisition Context")
    plt.ylabel("Mean AMY activity")
    # Vertical bars at interval 3 * num_acquisition_trials
    for i in range(0, num_trials, 3 * num_acquisition_trials):
        plt.axvline(x=i // bin_width, color='grey', linestyle='--', alpha=1.0)
    # Shaded vertical bars at interval num_acquisition_trials
    for i in range(0, num_trials, num_acquisition_trials):
        plt.axvline(x=i // bin_width, color='grey', linestyle='--', alpha=0.5)
    plt.show()


def air_puff():
    model = AmygdalaEngrams()
    online_learning(model, 1000)

    num_days = 20
    num_acquisition_trials = 100
    num_extinction_trials = 100
    amy_responses = np.zeros(num_days * (num_acquisition_trials + num_extinction_trials))

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

    B_overlap = 0.6
    B_diff = 1 - B_overlap
    hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
    units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
    cntx_B = cntx_A.copy()
    cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_B[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_B_diff[j * model.SENSORY_CORTEX.units_per_hc + i]

    for day in range(num_days):
        US_A = 0.66 if day % 2 == 0 else 0.0
        US_B = 0.66 if day % 2 == 1 else 0.0

        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_acquisition_trials):
            model.update(cntx_A, US_A)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[day * (num_acquisition_trials + num_extinction_trials) + i] = pattern_converged_to_amy
        for i in range(num_extinction_trials):
            model.update(cntx_B, US_B)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[day * (num_acquisition_trials + num_extinction_trials) + num_acquisition_trials + i] = pattern_converged_to_amy

        if day < num_days - 1:
            amy_transition_phase(model, Phase.SLEEP)
            for j in range(165):
                model.update()
    plt.figure(figsize=(10, 5))
    plt.title("Air puff fear conditioning: Overlap A/B=" + str(B_overlap))
    plt.plot(amy_responses)
    for d in range(num_days):
        D = d * (num_acquisition_trials + num_extinction_trials)
        plt.fill_between(x=np.arange(D, D + num_acquisition_trials), y1=-0.01, y2=-0.06, color='skyblue', alpha=0.8)
        plt.fill_between(x=np.arange(D + num_acquisition_trials - 1, D + num_acquisition_trials + num_extinction_trials), y1=-0.01, y2=-0.06, color='lightcoral', alpha=0.8)
    plt.ylim(-0.06, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("AMY activity")
    plt.show()


def amy_show_fear_learning():
    num_days = 21 # 39
    perceptions_per_day = 4
    avg_perception_len = 30
    perception_phase_len = perceptions_per_day * avg_perception_len
    sleep_len = 165 # 165

    num_runs = 10
    agg_remembered_pattern_count_ctx = {i: [] for i in range(num_days)}
    agg_remembered_pattern_count_hip = {i: [] for i in range(num_days)}
    agg_remembered_pattern_count_ban = {i: [] for i in range(num_days)}

    agg_avg_dist_ctx = {i: [] for i in range(num_days)}
    agg_avg_dist_hip = {i: [] for i in range(num_days)}
    agg_avg_dist_ban = {i: [] for i in range(num_days)}

    for run in range(num_runs):

        model = AmygdalaEngrams()
        online_learning(model, 1500)
            
        num_training_patterns = perceptions_per_day * num_days
        input_patterns = np.zeros((num_training_patterns, model.SENSORY_CORTEX.N))
        hip_training_patterns = np.zeros((num_training_patterns, model.HIP.N))
        ctx_training_patterns = np.zeros((num_training_patterns, model.CTX.N))
        ban_training_patterns = np.zeros((num_training_patterns, model.BA_N.N))

        for day in range(num_days):
            print("Day", day)
            amy_transition_phase(model, Phase.PERCEPTION)
            daily_patterns = [None for _ in range(perceptions_per_day)]
            amy_activities = np.zeros(perceptions_per_day)
            for i in range(perceptions_per_day):
                pattern_index = perceptions_per_day*day + i
                training_pattern = gen_random_simple_pattern(model.CTX.N, model.CTX.num_hcs)
                training_pattern_vec = np.array([training_pattern[i] for i in training_pattern.keys()])
                input_patterns[perceptions_per_day*day + i, :] = training_pattern_vec
                daily_patterns[i] = training_pattern
                amy_activity = 0.35
                amy_activities[i] = amy_activity
            # Randomly partition perception phase among the patterns
            parts = partition_n_into_k(perception_phase_len, perceptions_per_day)
            for j in range(len(parts)):
                for k in range(parts[j]):
                    if k < 4:
                        model.update(daily_patterns[j], amy_activities[j])
                    else:
                        model.update(daily_patterns[j], 0.0)
                    for l in range(model.HIP.N):
                        hip_training_patterns[perceptions_per_day*day + j, l] = model.HIP.log[model.HIP.current_step-1][l]
                    for l in range(model.CTX.N):
                        ctx_training_patterns[perceptions_per_day*day + j, l] = model.CTX.log[model.CTX.current_step-1][l]
                    for l in range(model.BA_N.N):
                        ban_training_patterns[perceptions_per_day*day + j, l] = model.BA_N.log[model.BA_N.current_step-1][l]
    
            if day < num_days - 1:
                # Sleep
                amy_transition_phase(model, Phase.SLEEP)
                for j in range(sleep_len):
                    model.update()

            else:
                # Recall
                amy_transition_phase(model, Phase.RECALL)
                remembered_pattern_count_ctx = {i: 0 for i in range(num_days)}
                remembered_pattern_count_hip = {i: 0 for i in range(num_days)}
                remembered_pattern_count_ban = {i: 0 for i in range(num_days)}
                distances_ctx = {i: [] for i in range(num_days)}
                distances_hip = {i: [] for i in range(num_days)}
                distances_ban = {i: [] for i in range(num_days)}
                for j, pattern in enumerate(input_patterns):
                    day_of_pattern = int(j / (perceptions_per_day))
                    print("Pattern", j, "Day", day_of_pattern)

                    pattern_dict = {i: pattern[i] for i in range(len(pattern))}
                    # Randomly flip 10% of the pattern
                    flip_indices = np.random.choice(len(pattern), size=int(len(pattern) * 0.1), replace=False)
                    for flip_index in flip_indices:
                        pattern_dict[flip_index] = 1 - pattern_dict[flip_index]
                    model.update(pattern_dict)
                    # Wait for convergence
                    steps_before_convergence = 0
                    while steps_before_convergence < 30:
                        model.update()
                        steps_before_convergence += 1
                    pattern_converged_to_ctx = model.CTX.log[model.CTX.current_step-1]
                    pattern_converged_to_ctx_vec = np.array([pattern_converged_to_ctx[i] for i in pattern_converged_to_ctx.keys()])
                    ctx_dist = recall_metric(pattern, pattern_converged_to_ctx_vec)
                    distances_ctx[day_of_pattern].append(ctx_dist)
                    print("CTX distance:", ctx_dist)
                    if ctx_dist < model.CTX.recall_detection_threshold:
                        remembered_pattern_count_ctx[day_of_pattern] += 1
                    hip_pattern = hip_training_patterns[j, :]
                    pattern_converged_to_hip = model.HIP.log[model.HIP.current_step-1]
                    pattern_converged_to_hip_vec = np.array([pattern_converged_to_hip[i] for i in pattern_converged_to_hip.keys()])
                    hip_dist = recall_metric(hip_pattern, pattern_converged_to_hip_vec)
                    distances_hip[day_of_pattern].append(hip_dist)
                    print("HIP distance:", hip_dist)
                    if hip_dist < model.HIP.recall_detection_threshold:
                        remembered_pattern_count_hip[day_of_pattern] += 1
                    ban_pattern = ban_training_patterns[j, :]
                    pattern_converged_to_ban = model.BA_N.log[model.BA_N.current_step-1]
                    pattern_converged_to_ban_vec = np.array([pattern_converged_to_ban[i] for i in pattern_converged_to_ban.keys()])
                    ban_dist = recall_metric(ban_pattern, pattern_converged_to_ban_vec)
                    distances_ban[day_of_pattern].append(ban_dist)
                    print("BA distance:", ban_dist)
                    if ban_dist < model.BA_N.recall_detection_threshold:
                        remembered_pattern_count_ban[day_of_pattern] += 1

                print("Remembered pattern count CTX:", remembered_pattern_count_ctx)
                print("Remembered pattern count HIP:", remembered_pattern_count_hip)
                print("Remembered pattern count BAN:", remembered_pattern_count_ban)


        # After each run, append the counts to the aggregators
        for day in range(num_days):
            agg_remembered_pattern_count_ctx[day].append(remembered_pattern_count_ctx[day])
            agg_remembered_pattern_count_hip[day].append(remembered_pattern_count_hip[day])
            agg_remembered_pattern_count_ban[day].append(remembered_pattern_count_ban[day])
        for day in range(num_days):
            if distances_ctx[day]:
                agg_avg_dist_ctx[day].append(np.mean(distances_ctx[day]))
            if distances_hip[day]:
                agg_avg_dist_hip[day].append(np.mean(distances_hip[day]))
            if distances_ban[day]:
                agg_avg_dist_ban[day].append(np.mean(distances_ban[day]))

        del model

    avg_remembered_pattern_count_ctx = {day: np.mean(counts) for day, counts in agg_remembered_pattern_count_ctx.items()}
    avg_remembered_pattern_count_hip = {day: np.mean(counts) for day, counts in agg_remembered_pattern_count_hip.items()}
    avg_remembered_pattern_count_ban = {day: np.mean(counts) for day, counts in agg_remembered_pattern_count_ban.items()}
    print("avg_remembered_pattern_count_ctx", avg_remembered_pattern_count_ctx)
    print("avg_remembered_pattern_count_hip", avg_remembered_pattern_count_hip)
    print("avg_remembered_pattern_count_ban", avg_remembered_pattern_count_ban)

    # Plot remembered pattern fractions by day for CTX, HIP and BAN
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title("Recall performance by memory age", fontsize=24)
    plt.plot([num_days - t for t in list(avg_remembered_pattern_count_ctx.keys())], [avg_remembered_pattern_count_ctx[i] / perceptions_per_day for i in avg_remembered_pattern_count_ctx.keys()], label="CTX", linewidth=2.5)
    plt.plot([num_days - t for t in list(avg_remembered_pattern_count_ctx.keys())], [avg_remembered_pattern_count_hip[i] / perceptions_per_day for i in avg_remembered_pattern_count_hip.keys()], label="HIP", linewidth=2.5)
    plt.plot([num_days - t for t in list(avg_remembered_pattern_count_ctx.keys())], [avg_remembered_pattern_count_ban[i] / perceptions_per_day for i in avg_remembered_pattern_count_ban.keys()], label="BA(N)", linewidth=2.5)
    # Draw horizontal line at 0.5
    plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
    plt.xlabel("Memory age (days)", fontsize=22)
    # xticks at 0, 5, 10, 15, 20
    plt.xticks([0, 5, 10, 15, 20], fontsize=18)
    # plt.gca().invert_xaxis()
    plt.ylabel(r'% patterns recalled', fontsize=22)
    plt.ylim(0, 1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(fontsize=22)
    plt.show()

def test():
    avg_remembered_pattern_count_ctx1 = {0: 1.25, 1: 1.15, 2: 1.45, 3: 1.3, 4: 1.15, 5: 1.4, 6: 0.95, 7: 1.3, 8: 1.2, 9: 1.35, 10: 1.45, 11: 1.25, 12: 1.55, 13: 1.55, 14: 1.4, 15: 1.55, 16: 1.45, 17: 1.55, 18: 1.65, 19: 1.8, 20: 1.75, 21: 1.9, 22: 1.3, 23: 1.3, 24: 1.55}

    avg_remembered_pattern_count_hip1 = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.05, 9: 0.0, 10: 0.35, 11: 0.4, 12: 0.8, 13: 1.15, 14: 1.6, 15: 1.95, 16: 2.3, 17: 2.6, 18: 2.85, 19: 2.9, 20: 3.0, 21: 3.0, 22: 3.0, 23: 3.0, 24: 3.0}

    avg_remembered_pattern_count_ban1 = {0: 1.1, 1: 0.9, 2: 1.1, 3: 1.15, 4: 1.05, 5: 1.3, 6: 0.9, 7: 1.2, 8: 1.15, 9: 1.15, 10: 1.45, 11: 1.2, 12: 1.55, 13: 1.7, 14: 2.0, 15: 2.25, 16: 2.5, 17: 2.6, 18: 2.85, 19: 2.9, 20: 3.0, 21: 3.0, 22: 3.0, 23: 3.0, 24: 3.0}



    avg_remembered_pattern_count_ctx2 = {0: 1.1, 1: 1.05, 2: 1.35, 3: 1.1, 4: 1.45, 5: 1.25, 6: 1.15, 7: 1.2, 8: 1.2, 9: 1.4, 10: 1.55, 11: 1.35, 12: 1.6, 13: 1.45, 14: 1.65, 15: 1.65, 16: 1.55, 17: 1.9, 18: 1.6, 19: 1.85, 20: 2.2, 21: 1.7, 22: 1.4, 23: 1.1, 24: 1.5}

    avg_remembered_pattern_count_hip2 = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.05, 6: 0.0, 7: 0.0, 8: 0.1, 9: 0.05, 10: 0.1, 11: 0.35, 12: 0.85, 13: 1.1, 14: 1.6, 15: 1.95, 16: 2.7, 17: 2.9, 18: 2.85, 19: 2.95, 20: 3.0, 21: 3.0, 22: 2.95, 23: 3.0, 24: 3.0}

    avg_remembered_pattern_count_ban2 = {0: 1.0, 1: 0.9, 2: 0.85, 3: 0.95, 4: 1.25, 5: 1.2, 6: 1.05, 7: 1.05, 8: 1.1, 9: 1.25, 10: 1.35, 11: 1.4, 12: 1.65, 13: 1.7, 14: 1.9, 15: 2.45, 16: 2.75, 17: 2.95, 18: 2.85, 19: 2.95, 20: 3.0, 21: 3.0, 22: 2.95, 23: 3.0, 24: 3.0}

    avg_remembered_pattern_count_ctx3 = {0: 1.35, 1: 1.25, 2: 1.5, 3: 1.45, 4: 1.3, 5: 1.6, 6: 1.25, 7: 1.5, 8: 1.2, 9: 1.0, 10: 1.2, 11: 1.45, 12: 1.55, 13: 1.65, 14: 1.65, 15: 1.65, 16: 1.5, 17: 2.0, 18: 1.85, 19: 1.95, 20: 1.7, 21: 1.55, 22: 1.65, 23: 1.25, 24: 1.65}  
    avg_remembered_pattern_count_hip3 = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.05, 9: 0.1, 10: 0.15, 11: 0.45, 12: 0.55, 13: 1.1, 14: 1.45, 15: 1.9, 16: 2.45, 17: 2.6, 18: 2.8, 19: 2.9, 20: 2.8, 21: 3.0, 22: 3.0, 23: 3.0, 24: 3.0}
    avg_remembered_pattern_count_ban3 = {0: 1.15, 1: 1.1, 2: 1.45, 3: 1.3, 4: 0.95, 5: 1.45, 6: 1.15, 7: 1.35, 8: 1.05, 9: 0.95, 10: 1.15, 11: 1.55, 12: 1.55, 13: 1.75, 14: 2.0, 15: 2.15, 16: 2.5, 17: 2.65, 18: 2.8, 19: 2.9, 20: 2.8, 21: 3.0, 22: 3.0, 23: 3.0, 24: 3.0} 

    # Function to compute the average of two dictionaries

    def average_dicts(dict1, dict2, dict3):

        return {key: (dict1[key] + dict2[key] + dict3[key]) / 3 for key in dict1.keys()}


    avg_remembered_pattern_count_ctx = average_dicts(avg_remembered_pattern_count_ctx1, avg_remembered_pattern_count_ctx2, avg_remembered_pattern_count_ctx3)
    avg_remembered_pattern_count_hip = average_dicts(avg_remembered_pattern_count_hip1, avg_remembered_pattern_count_hip2, avg_remembered_pattern_count_hip3)
    avg_remembered_pattern_count_ban = average_dicts(avg_remembered_pattern_count_ban1, avg_remembered_pattern_count_ban2, avg_remembered_pattern_count_ban3)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title("Recall performance by memory age", fontsize=24)
    plt.plot([25 - t for t in list(avg_remembered_pattern_count_ctx.keys())], [avg_remembered_pattern_count_ctx[i] / 3 for i in avg_remembered_pattern_count_ctx.keys()], label="CTX", linewidth=2.5)
    plt.plot([25 - t for t in list(avg_remembered_pattern_count_ctx.keys())], [avg_remembered_pattern_count_hip[i] / 3 for i in avg_remembered_pattern_count_hip.keys()], label="HIP", linewidth=2.5)
    plt.plot([25 - t for t in list(avg_remembered_pattern_count_ctx.keys())], [avg_remembered_pattern_count_ban[i] / 3 for i in avg_remembered_pattern_count_ban.keys()], label="BA(N)", linewidth=2.5)
    # Draw horizontal line at 0.5
    plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
    plt.xlabel("Memory age (days)", fontsize=22)
    # plt.gca().invert_xaxis()
    plt.ylabel(r'% patterns recalled', fontsize=22)
    plt.ylim(0, 1)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(fontsize=22)
    plt.show()

def amy_sefl_protocol():
    sleep_len = 165
    trauma_len = 30
    moderate_US_len = 4
    assert trauma_len > moderate_US_len
    moderate_remainder_len = trauma_len - moderate_US_len
    harmless_len = 30

    num_runs = 5
    agg_amy_responses_trauma = np.zeros(num_runs)
    agg_amy_responses_control = np.zeros(num_runs)

    cem_history_trauma = np.zeros((num_runs, trauma_len + trauma_len + 30))
    cem_history_control = np.zeros((num_runs, trauma_len + trauma_len + 30))

    for run in range(num_runs):
        model = AmygdalaEngrams()
        online_learning(model, 1000)
        control_model = AmygdalaEngrams()
        online_learning(control_model, 1000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_B = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_C = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_D = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        # Trauma
        for i in range(trauma_len):
            model.update(cntx_A, 1.0)
            control_model.update(cntx_A, 0.0)
            cem_history_trauma[run, i] = model.AMY_C.log[model.AMY_C.current_step-1]
            cem_history_control[run, i] = control_model.AMY_C.log[control_model.AMY_C.current_step-1]
        # print feedback weights from BAN to BAP
        print(np.max(model.BA_P.feedback_connections["BA_N"].W, axis=0))
        # return 
        for i in range(harmless_len):
            model.update(cntx_C, 0.0)
            control_model.update(cntx_C, 0.0)
        for i in range(harmless_len):
            model.update(cntx_D, 0.0)
            control_model.update(cntx_D, 0.0)
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()
            control_model.update()
        # Moderate US
        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        for i in range(moderate_US_len):
            model.update(cntx_B, 0.85)
            control_model.update(cntx_B, 0.85)
            cem_history_trauma[run, trauma_len + i] = model.AMY_C.log[model.AMY_C.current_step-1]
            cem_history_control[run, trauma_len + i] = control_model.AMY_C.log[control_model.AMY_C.current_step-1]
        # return 
        for i in range(moderate_remainder_len):
            model.update(cntx_B, 0.0)
            control_model.update(cntx_B, 0.0)
            cem_history_trauma[run, trauma_len + moderate_US_len + i] = model.AMY_C.log[model.AMY_C.current_step-1]
            cem_history_control[run, trauma_len + moderate_US_len + i] = control_model.AMY_C.log[control_model.AMY_C.current_step-1]
        for i in range(harmless_len):
            model.update(cntx_C, 0.0)
            control_model.update(cntx_C, 0.0)
        for i in range(harmless_len):
            model.update(cntx_D, 0.0)
            control_model.update(cntx_D, 0.0)
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()
            control_model.update()
        # Recall test
        amy_transition_phase(model, Phase.RECALL)
        amy_transition_phase(control_model, Phase.RECALL)
        model.update(cntx_B)
        control_model.update(cntx_B)
        for i in range(30):
            model.update()
            control_model.update()
            cem_history_trauma[run, trauma_len + moderate_US_len + i] = model.AMY_C.log[model.AMY_C.current_step-1]
            cem_history_control[run, trauma_len + moderate_US_len + i] = control_model.AMY_C.log[control_model.AMY_C.current_step-1]
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        pattern_converged_to_amy_control = control_model.AMY_C.log[control_model.AMY_C.current_step-1]
        agg_amy_responses_trauma[run] = pattern_converged_to_amy
        agg_amy_responses_control[run] = pattern_converged_to_amy_control

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    axes[0].set_title("SEFL protocol")
    axes[0].plot(np.mean(cem_history_trauma, axis=0))
    axes[0].set_ylim(-0.01, 1.01)
    axes[0].set_ylabel("AMY activity")
    axes[1].set_title("SEFL protocol: AMY activity after control")
    axes[1].plot(np.mean(cem_history_control, axis=0))
    axes[1].set_ylim(-0.01, 1.01)
    axes[1].set_ylabel("AMY activity")
    plt.show()

    # print AMY_A log of model
    print("AMY_A log of model:")
    for key in[k for k in model.AMY_A.log.keys() if k > 1000 and k < 1200]:
        print(key, model.AMY_A.log[key])
    # repeat for control model
    print("AMY_A log of control model:")
    for key in[k for k in control_model.AMY_A.log.keys() if k > 1000 and k < 1200]:
        print(key, control_model.AMY_A.log[key])

    # Plot boxplots and report Wilcoxon signed-rank test
    plt.figure(figsize=(9, 9))
    plt.title("SEFL protocol", fontsize=22)
    plt.boxplot([agg_amy_responses_trauma, agg_amy_responses_control], labels=["Trauma", "Control"])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=18)
    plt.ylim(0.0, 1.0)
    plt.ylabel("CeM activity", fontsize=22)
    plt.show()

    from scipy.stats import wilcoxon
    print("Wilcoxon signed-rank test:")
    print(wilcoxon(agg_amy_responses_trauma, agg_amy_responses_control))


def amy_sefl_synapse_sum():
    sleep_len = 165
    trauma_len = 30
    moderate_US_len = 4
    assert trauma_len > moderate_US_len
    moderate_remainder_len = trauma_len - moderate_US_len
    harmless_len = 30

    num_days_before_trauma = 5
    num_days_after_trauma = 15
    num_simulations = 10 # Number of runs to average

    # Initialize arrays to store results for averaging
    all_sum_of_W_model = []
    all_sum_of_W_control = []
    all_frac_W_greater0_01_model = []
    all_frac_W_greater0_01_control = []
    all_frac_W_greater0_1_model = []
    all_frac_W_greater0_1_control = []
    all_cem_responses_model = []
    all_cem_responses_control = []

    for sim in range(num_simulations):
        model = AmygdalaEngrams()
        # model.A_thresh = 999999999 # NOTE
        online_learning(model, 2000)
        control_model = AmygdalaEngrams()
        online_learning(control_model, 2000)

        # Initialize per-run results
        sum_of_W_model = []
        sum_of_W_control = []
        frac_W_greater0_01_model = []
        frac_W_greater0_01_control = []
        frac_W_greater0_1_model = []
        frac_W_greater0_1_control = []

        # Helper function to log results
        def log_results():
            sum_of_W_model.append(np.sum(model.BA_P.feedback_connections["BA_N"].W))
            sum_of_W_control.append(np.sum(control_model.BA_P.feedback_connections["BA_N"].W))
            frac_W_greater0_01_model.append(np.sum(model.BA_P.feedback_connections["BA_N"].W > 0.01) / model.BA_P.feedback_connections["BA_N"].W.size)
            frac_W_greater0_01_control.append(np.sum(control_model.BA_P.feedback_connections["BA_N"].W > 0.01) / control_model.BA_P.feedback_connections["BA_N"].W.size)
            frac_W_greater0_1_model.append(np.sum(model.BA_P.feedback_connections["BA_N"].W > 0.1) / model.BA_P.feedback_connections["BA_N"].W.size)
            frac_W_greater0_1_control.append(np.sum(control_model.BA_P.feedback_connections["BA_N"].W > 0.1) / control_model.BA_P.feedback_connections["BA_N"].W.size)

        # Generate contexts
        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)  # Trauma context
        cntx_B = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)  # Moderate US context

        # Pre-Trauma Phase
        for day in range(num_days_before_trauma):
            amy_transition_phase(model, Phase.PERCEPTION)
            amy_transition_phase(control_model, Phase.PERCEPTION)

            # Random contexts
            for _ in range(4):
                cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
                us_random = 0.9 * np.random.beta(4, 4)
                for _ in range(harmless_len):
                    model.update(cntx_random, us_random)
                    control_model.update(cntx_random, us_random)

            log_results()
            amy_transition_phase(model, Phase.SLEEP)
            amy_transition_phase(control_model, Phase.SLEEP)
            for _ in range(sleep_len):
                model.update()
                control_model.update()
            log_results()

        # Trauma Phase
        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        for _ in range(trauma_len):
            model.update(cntx_A, 1.0)
            control_model.update(cntx_A, 0.0)
        for _ in range(4):
            cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
            us_random = 0.9 * np.random.beta(2, 4)
            for _ in range(harmless_len):
                model.update(cntx_random, us_random)
                control_model.update(cntx_random, us_random)
        log_results()
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for _ in range(sleep_len):
            model.update()
            control_model.update()
        log_results()

        # Post-Trauma Phase
        for day in range(num_days_after_trauma):
            amy_transition_phase(model, Phase.PERCEPTION)
            amy_transition_phase(control_model, Phase.PERCEPTION)

            for _ in range(4):
                cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
                us_random = 0.9 * np.random.beta(2, 4)
                for _ in range(harmless_len):
                    model.update(cntx_random, us_random)
                    control_model.update(cntx_random, us_random)

            log_results()
            amy_transition_phase(model, Phase.SLEEP)
            amy_transition_phase(control_model, Phase.SLEEP)
            for _ in range(sleep_len):
                model.update()
                control_model.update()
            log_results()

        # Moderate US Phase
        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        for i in range(moderate_US_len):
            # collect indices where BAN output is over 0.5
            if i == 1:
                indices = np.where(model.BA_N.output > 0.5)[0]
            model.update(cntx_B, 0.9)
            control_model.update(cntx_B, 0.9)
            print("Step", i, "model:", model.AMY_C.output, "control:", control_model.AMY_C.output)
        for _ in range(moderate_remainder_len):
            model.update(cntx_B, 0.0)
            control_model.update(cntx_B, 0.0)
            print("model:", model.AMY_C.output, np.mean(model.BA_P_PRE_US.output), np.mean(model.BA_I_PRE_US.output), "control:", control_model.AMY_C.output, np.mean(control_model.BA_P_PRE_US.output), np.mean(control_model.BA_I_PRE_US.output))
        for _ in range(3):
            cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
            us_random = 0.9 * np.random.beta(2, 4)
            for _ in range(harmless_len):
                model.update(cntx_random, us_random)
                control_model.update(cntx_random, us_random)
        log_results()
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for _ in range(sleep_len):
            model.update()
            control_model.update()
        log_results()

        # Recall test
        amy_transition_phase(model, Phase.RECALL)
        amy_transition_phase(control_model, Phase.RECALL)
        model.update(cntx_B)
        control_model.update(cntx_B)
        for t in range(30):
            model.update()
            control_model.update()
            print("Step", t, "model:", model.AMY_C.output, "control:", control_model.AMY_C.output)
        end_indices = np.where(model.BA_N.output > 0.5)[0]
        print("start and end indices:", indices, end_indices)
        # print overlap - what percent of indices are in end_indices
        overlap = np.sum(np.isin(indices, end_indices)) / len(indices)
        print("Overlap:", overlap)

        all_cem_responses_model.append(model.AMY_C.log[model.AMY_C.current_step-1])
        all_cem_responses_control.append(control_model.AMY_C.log[control_model.AMY_C.current_step-1])
        print("Model", model.AMY_C.log[model.AMY_C.current_step-1], "Control", control_model.AMY_C.log[control_model.AMY_C.current_step-1])

        # Collect results from this simulation
        all_sum_of_W_model.append(sum_of_W_model)
        all_sum_of_W_control.append(sum_of_W_control)
        all_frac_W_greater0_01_model.append(frac_W_greater0_01_model)
        all_frac_W_greater0_01_control.append(frac_W_greater0_01_control)
        all_frac_W_greater0_1_model.append(frac_W_greater0_1_model)
        all_frac_W_greater0_1_control.append(frac_W_greater0_1_control)

    # Average results across simulations
    avg_sum_of_W_model = np.mean(all_sum_of_W_model, axis=0)
    avg_sum_of_W_control = np.mean(all_sum_of_W_control, axis=0)
    avg_frac_W_greater0_01_model = np.mean(all_frac_W_greater0_01_model, axis=0)
    avg_frac_W_greater0_01_control = np.mean(all_frac_W_greater0_01_control, axis=0)
    avg_frac_W_greater0_1_model = np.mean(all_frac_W_greater0_1_model, axis=0)
    avg_frac_W_greater0_1_control = np.mean(all_frac_W_greater0_1_control, axis=0)

    # Plot averaged results
    plt.figure(figsize=(11, 11))
    plt.plot(avg_sum_of_W_model, label="Trauma")
    plt.plot(avg_sum_of_W_control, label="Control")
    num_days = len(avg_sum_of_W_model) // 2  # Assuming two points per day
    x_ticks = range(0, len(avg_sum_of_W_model), 2)
    x_labels = [f"{i+1}" for i in range(num_days)]
    plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=18)
    plt.xlabel("Day", fontsize=22)
    plt.ylabel("Summed strength of BA(N)-BA(P) synapses", fontsize=18)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()

    wakefulness_model = avg_sum_of_W_model[::2]
    wakefulness_control = avg_sum_of_W_control[::2]

    # Extract sleep entries (indices 1, 3, 5, ...)
    sleep_model = avg_sum_of_W_model[1::2]
    sleep_control = avg_sum_of_W_control[1::2]

    # Generate plot for wakefulness phases
    plt.figure(figsize=(11, 7))
    plt.plot(wakefulness_model, label="Trauma (Wakefulness)", marker='o')
    plt.plot(wakefulness_control, label="Control (Wakefulness)", marker='o')
    plt.xlabel("Day", fontsize=18)
    plt.ylabel("Synaptic Strength after Wakefulness", fontsize=18)
    plt.title("Net Synaptic Strength after Wakefulness", fontsize=20)
    plt.xticks(range(len(wakefulness_model)), [f"{i+1}" for i in range(len(wakefulness_model))], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()

    # Generate plot for sleep phases
    plt.figure(figsize=(11, 7))
    plt.plot(sleep_model, label="Trauma (Sleep)", marker='o')
    plt.plot(sleep_control, label="Control (Sleep)", marker='o')
    plt.xlabel("Day", fontsize=18)
    plt.ylabel("Synaptic Strength after Sleep", fontsize=18)
    plt.title("Net Synaptic Strength after Sleep", fontsize=20)
    plt.xticks(range(len(sleep_model)), [f"{i+1}" for i in range(len(sleep_model))], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()

    # Calculate wakefulness-driven increases
    wake_increase_model = wakefulness_model[1:] - sleep_model[:-1]
    wake_increase_control = wakefulness_control[1:] - sleep_control[:-1]

    # Plot wakefulness-driven increase
    plt.figure(figsize=(11, 7))
    plt.plot(wake_increase_model, label="Trauma Wake Increase", marker='o')
    plt.plot(wake_increase_control, label="Control Wake Increase", marker='o')
    plt.xlabel("Day", fontsize=18)
    plt.ylabel("Synaptic Increase during Wakefulness", fontsize=18)
    plt.title("Wakefulness-driven Synaptic Increases", fontsize=20)
    plt.xticks(range(len(wake_increase_model)), [f"{i+1}" for i in range(len(wake_increase_model))], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.grid(True)
    plt.show()

    # Calculate sleep-driven decreases
    sleep_decrease_model = sleep_model - wakefulness_model[:len(sleep_model)]
    sleep_decrease_control = sleep_control - wakefulness_control[:len(sleep_control)]

    # Plot sleep-driven decrease
    plt.figure(figsize=(11, 7))
    plt.plot(sleep_decrease_model, label="Trauma Sleep Decrease", marker='o')
    plt.plot(sleep_decrease_control, label="Control Sleep Decrease", marker='o')
    plt.xlabel("Day", fontsize=18)
    plt.ylabel("Synaptic Decrease during Sleep", fontsize=18)
    plt.title("Sleep-driven Synaptic Decreases", fontsize=20)
    plt.xticks(range(len(sleep_decrease_model)), [f"{i+1}" for i in range(len(sleep_decrease_model))], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.grid(True)
    plt.show()
    

    plt.figure(figsize=(9, 9))
    plt.title("SEFL Protocol: Avg. Fraction of W > 0.01")
    plt.plot(avg_frac_W_greater0_01_model, label="Trauma")
    plt.plot(avg_frac_W_greater0_01_control, label="Control")
    plt.legend()
    plt.show()

    """plt.figure(figsize=(9, 9))
    plt.title("SEFL Protocol: Avg. Fraction of W > 0.1")
    plt.plot(avg_frac_W_greater0_1_model, label="Trauma")
    plt.plot(avg_frac_W_greater0_1_control, label="Control")
    plt.legend()
    plt.show()

    # Do wilcoxon signed-rank test on CeM responses
    from scipy.stats import wilcoxon
    print("Wilcoxon signed-rank test:")
    print(wilcoxon(all_cem_responses_model, all_cem_responses_control))"""

    # Plot boxplots and report Wilcoxon signed-rank test
    plt.figure(figsize=(9, 9))
    plt.title("SEFL protocol", fontsize=22)
    sns.violinplot(
    data=[all_cem_responses_model, all_cem_responses_control], 
    palette="muted", 
    scale="width"
    )
    plt.xticks([0, 1], ["Trauma", "Control"], fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(0.0, 1.0)
    plt.ylabel("CeM activity", fontsize=22)
    plt.show()



def amy_sefl_reverse():
    sleep_len = 165
    trauma_len = 30
    moderate_US_len = 3
    assert trauma_len > moderate_US_len
    moderate_remainder_len = trauma_len - moderate_US_len
    harmless_len = 20

    num_runs = 10
    agg_amy_responses_trauma = np.zeros(num_runs)
    agg_amy_responses_control = np.zeros(num_runs)

    for run in range(num_runs):
        model = AmygdalaEngrams()
        online_learning(model, 1000)
        control_model = AmygdalaEngrams()
        online_learning(control_model, 1000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_B = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_C = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_D = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        # Moderate US
        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        for i in range(moderate_US_len):
            model.update(cntx_B, 0.9)
            control_model.update(cntx_B, 0.9)
        for i in range(moderate_remainder_len):
            model.update(cntx_B, 0.0)
            control_model.update(cntx_B, 0.0)
        for i in range(harmless_len):
            model.update(cntx_C, 0.0)
            control_model.update(cntx_C, 0.0)
        for i in range(harmless_len):
            model.update(cntx_D, 0.0)
            control_model.update(cntx_D, 0.0)
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()
            control_model.update()
        # Trauma
        for i in range(trauma_len):
            model.update(cntx_A, 1.0)
            control_model.update(cntx_A, 0.0)
        for i in range(harmless_len):
            model.update(cntx_C, 0.0)
            control_model.update(cntx_C, 0.0)
        for i in range(harmless_len):
            model.update(cntx_D, 0.0)
            control_model.update(cntx_D, 0.0)
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()
            control_model.update()
        # Recall test
        amy_transition_phase(model, Phase.RECALL)
        amy_transition_phase(control_model, Phase.RECALL)
        model.update(cntx_B)
        control_model.update(cntx_B)
        for i in range(30):
            model.update()
            control_model.update()
        pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
        pattern_converged_to_amy_control = control_model.AMY_C.log[control_model.AMY_C.current_step-1]
        agg_amy_responses_trauma[run] = pattern_converged_to_amy
        agg_amy_responses_control[run] = pattern_converged_to_amy_control
        print("Response Trauma:", pattern_converged_to_amy)
        print("Response Control:", pattern_converged_to_amy_control)

    # Plot boxplots and report Wilcoxon signed-rank test
    plt.figure(figsize=(9, 9))
    plt.title("SEFL protocol - order reversed.", fontsize=22)
    
    sns.violinplot(data=[agg_amy_responses_trauma, agg_amy_responses_control], palette="muted", scale="width")
    plt.xticks(plt.xticks([0, 1], ["Trauma", "Control"], fontsize=22))
    plt.yticks(fontsize=18)
    plt.ylim(0.0, 1.0)
    plt.ylabel("CeM activity", fontsize=22)
    plt.show()

    #from scipy.stats import wilcoxon
    #print("Wilcoxon signed-rank test:")
    #print(wilcoxon(agg_amy_responses_trauma, agg_amy_responses_control))
    
def amy_sleep_disruption_chronic():
    import gc
    sleep_len = 165
    sleep_omitted_fracs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    num_days = 7
    contexts_per_day = 10
    num_simulations = 1
    cem_responses = {n: [] for n in sleep_omitted_fracs}
    first_step_cem_responses = {n: [] for n in sleep_omitted_fracs}
    net_weights = {n: [] for n in sleep_omitted_fracs}
    random_cntx_len = 20
    cntx_A_len = 5
    cntx_A_strength = 0.5

    for _ in range(num_simulations):
        model = AmygdalaEngrams() # unused except for generating context A
        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        # Dict to hold US strengths
        us_dict = {}
        for day in range(num_days):
            us_dict[day] = {}
            for i in range(contexts_per_day):
                us_strength = 0.85 * np.random.beta(6,6)
                us_dict[day][i] = us_strength

        cntx_dict = {}
        for day in range(num_days):
            cntx_dict[day] = {}
            for i in range(contexts_per_day):
                cntx = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
                cntx_dict[day][i] = cntx

        for frac in sleep_omitted_fracs:
            num_sleep_steps = int(sleep_len * (1.0 - frac))
            model = AmygdalaEngrams()
            online_learning(model, 1000)
            for day in range(num_days):
                amy_transition_phase(model, Phase.PERCEPTION)
                for i in range(contexts_per_day):
                    cntx = cntx_dict[day][i]
                    us = us_dict[day][i]
                    for _ in range(random_cntx_len):
                        model.update(cntx, us)  
                amy_transition_phase(model, Phase.SLEEP)
                for _ in range(num_sleep_steps):
                    model.update()

            amy_transition_phase(model, Phase.PERCEPTION)
            for i in range(cntx_A_len):
                model.update(cntx_A, cntx_A_strength)
                print("Step", i, "model:", model.AMY_C.output)

                if i == 0:
                    first_step_cem_response = model.AMY_C.log[model.AMY_C.current_step-1]
                    first_step_cem_responses[frac].append(first_step_cem_response)

            fear_response = model.AMY_C.log[model.AMY_C.current_step-1]
            cem_responses[frac].append(fear_response)

            net_weight = np.sum(model.BA_P.feedback_connections["BA_N"].W)
            net_weights[frac].append(net_weight)
            print("For frac =", frac, "fear response =", fear_response, "num sleep steps =", num_sleep_steps)

            # Free up memory
            del model
            gc.collect()

    # Plot results, with dots as mean and shaded region as standard deviation
    # Well, first collect means and standard deviations
    cem_response_means = [np.mean(cem_responses[frac]) for frac in sleep_omitted_fracs]
    cem_response_stds = [np.std(cem_responses[frac]) for frac in sleep_omitted_fracs]
    plt.figure(figsize=(9, 9))
    plt.title("Chronic sleep deprivation")
    plt.plot(sleep_omitted_fracs, cem_response_means, marker='o', label="Mean")
    # plt.fill_between(sleep_omitted_fracs, np.array(cem_response_means) - np.array(cem_response_stds), np.array(cem_response_means) + np.array(cem_response_stds), alpha=0.3, label="Std. Dev.")
    plt.fill_between(sleep_omitted_fracs, np.array(cem_response_means) - np.array(cem_response_stds), np.array(cem_response_means) + np.array(cem_response_stds), alpha=0.3, label="Standard Error")
    plt.xlabel("Fraction of sleep omitted", fontsize=22)
    plt.ylabel("CeM activity", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(min(0.1, cem_response_means[0] - cem_response_stds[0]), max(0.6, cem_response_means[-1] + cem_response_stds[-1] + 0.05))
    # plt.gca().invert_xaxis()
    plt.show()

    # Plot cem_response minus first_step_cem_response
    cem_response_diffs = {}
    for frac in sleep_omitted_fracs:
        cem_response_diffs[frac] = np.array(cem_responses[frac]) - np.array(first_step_cem_responses[frac])
    cem_response_diff_means = [np.mean(cem_response_diffs[frac]) for frac in sleep_omitted_fracs]
    cem_response_diff_stds = [np.std(cem_response_diffs[frac]) for frac in sleep_omitted_fracs]
    plt.figure(figsize=(9, 9))
    plt.title("Chronic sleep deprivation: Saturation of learning")
    plt.plot(sleep_omitted_fracs, cem_response_diff_means, marker='o', label="Mean")
    # plt.fill_between(sleep_omitted_fracs, np.array(cem_response_diff_means) - np.array(cem_response_diff_stds), np.array(cem_response_diff_means) + np.array(cem_response_diff_stds), alpha=0.3, label="Std. Dev.")
    plt.fill_between(sleep_omitted_fracs, np.array(cem_response_diff_means) - np.array(cem_response_diff_stds), np.array(cem_response_diff_means) + np.array(cem_response_diff_stds), alpha=0.3, label="Standard Error")
    plt.xlabel("Fraction of sleep omitted", fontsize=22)
    plt.ylabel("CeM activity - first step CeM activity", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.gca().invert_xaxis()
    plt.show()

    net_weight_means = [np.mean(net_weights[frac]) for frac in sleep_omitted_fracs]
    net_weight_stds = [np.std(net_weights[frac]) for frac in sleep_omitted_fracs]
    plt.figure(figsize=(9, 9))
    plt.title("Chronic sleep deprivation")
    plt.plot(sleep_omitted_fracs, net_weight_means, marker='o', label="Mean")
    plt.fill_between(sleep_omitted_fracs, np.array(net_weight_means) - np.array(net_weight_stds), np.array(net_weight_means) + np.array(net_weight_stds), alpha=0.3, label="Std. Dev.")
    plt.xlabel("Fraction of sleep omitted", fontsize=22)
    plt.ylabel("Sum of BA(N)-BA(P) synapse weights", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.gca().invert_xaxis()
    plt.show()

    # Write all means, stds and ses to a file
    with open("chronic_sleep_deprivation.txt", "w") as f:
        f.write("Sleep fraction\tMean response\tStd. Dev.\tStd (again, nvm)\n")
        for i, frac in enumerate(sleep_omitted_fracs):
            f.write(f"{frac}\t{cem_response_means[i]}\t{cem_response_stds[i]}\t{cem_response_stds[i]}\n")

        # write individual responses and net weights
        f.write("\nIndividual responses and net weights:\n")
        f.write("Sleep fraction\tResponse\tNet weight\n")
        for frac in sleep_omitted_fracs:
            for response, net_weight in zip(cem_responses[frac], net_weights[frac]):
                f.write(f"{frac}\t{response}\t{net_weight}\n")

        from scipy.stats import ttest_ind
        for i, frac in enumerate(sleep_omitted_fracs):
            for j, frac2 in enumerate(sleep_omitted_fracs):
                if i != j:
                    t_stat, p_val = ttest_ind(cem_responses[frac], cem_responses[frac2])
                    f.write(f"{frac}\t{frac2}\t{t_stat}\t{p_val}\n")



def amy_sefl_synapse_sum_reverse():
    sleep_len = 165
    trauma_len = 40
    moderate_US_len = 4
    assert trauma_len > moderate_US_len
    moderate_remainder_len = trauma_len - moderate_US_len
    harmless_len = 20

    num_days_before_trauma = 0
    num_days_after_trauma = 0
    num_simulations = 30 # Number of runs to average

    # Initialize arrays to store results for averaging
    all_sum_of_W_model = []
    all_sum_of_W_control = []
    all_frac_W_greater0_01_model = []
    all_frac_W_greater0_01_control = []
    all_frac_W_greater0_1_model = []
    all_frac_W_greater0_1_control = []
    all_cem_responses_model = []
    all_cem_responses_control = []

    for sim in range(num_simulations):
        model = AmygdalaEngrams()
        online_learning(model, 2000)
        control_model = AmygdalaEngrams()
        online_learning(control_model, 2000)

        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)

        # Initialize per-run results
        sum_of_W_model = []
        sum_of_W_control = []
        frac_W_greater0_01_model = []
        frac_W_greater0_01_control = []
        frac_W_greater0_1_model = []
        frac_W_greater0_1_control = []

        # Helper function to log results
        def log_results():
            sum_of_W_model.append(np.sum(model.BA_P.feedback_connections["BA_N"].W))
            sum_of_W_control.append(np.sum(control_model.BA_P.feedback_connections["BA_N"].W))
            frac_W_greater0_01_model.append(np.sum(model.BA_P.feedback_connections["BA_N"].W > 0.01) / model.BA_P.feedback_connections["BA_N"].W.size)
            frac_W_greater0_01_control.append(np.sum(control_model.BA_P.feedback_connections["BA_N"].W > 0.01) / control_model.BA_P.feedback_connections["BA_N"].W.size)
            frac_W_greater0_1_model.append(np.sum(model.BA_P.feedback_connections["BA_N"].W > 0.1) / model.BA_P.feedback_connections["BA_N"].W.size)
            frac_W_greater0_1_control.append(np.sum(control_model.BA_P.feedback_connections["BA_N"].W > 0.1) / control_model.BA_P.feedback_connections["BA_N"].W.size)

        # Generate contexts
        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)  # Trauma context
        cntx_B = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)  # Moderate US context

        # Pre-Trauma Phase
        for day in range(num_days_before_trauma):
            amy_transition_phase(model, Phase.PERCEPTION)
            amy_transition_phase(control_model, Phase.PERCEPTION)

            # Random contexts
            for _ in range(3):
                cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
                us_random = 0.9 * np.random.beta(3, 4)
                for _ in range(harmless_len):
                    model.update(cntx_random, us_random)
                    control_model.update(cntx_random, us_random)

            log_results()
            amy_transition_phase(model, Phase.SLEEP)
            amy_transition_phase(control_model, Phase.SLEEP)
            for _ in range(sleep_len):
                model.update()
                control_model.update()
            log_results()

        # Moderate US Phase
        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        for _ in range(moderate_US_len):
            model.update(cntx_B, 0.9)
            control_model.update(cntx_B, 0.9)
        for _ in range(moderate_remainder_len):
            model.update(cntx_B, 0.0)
            control_model.update(cntx_B, 0.0)
        for _ in range(2):
            cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
            us_random = 0.9 * np.random.beta(3, 4)
            for _ in range(harmless_len):
                model.update(cntx_random, us_random)
                control_model.update(cntx_random, us_random)
        log_results()
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for _ in range(sleep_len):
            model.update()
            control_model.update()
        log_results()

        # Post-Trauma Phase
        for day in range(num_days_after_trauma):
            amy_transition_phase(model, Phase.PERCEPTION)
            amy_transition_phase(control_model, Phase.PERCEPTION)

            for _ in range(3):
                cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
                us_random = 0.9 * np.random.beta(3, 4)
                for _ in range(harmless_len):
                    model.update(cntx_random, us_random)
                    control_model.update(cntx_random, us_random)

            log_results()
            amy_transition_phase(model, Phase.SLEEP)
            amy_transition_phase(control_model, Phase.SLEEP)
            for _ in range(sleep_len):
                model.update()
                control_model.update()
            log_results()

        # Trauma Phase
        amy_transition_phase(model, Phase.PERCEPTION)
        amy_transition_phase(control_model, Phase.PERCEPTION)
        for _ in range(trauma_len):
            model.update(cntx_A, 1.0)
            control_model.update(cntx_A, 0.0)
        for _ in range(3):
            cntx_random = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
            us_random = 0.9 * np.random.beta(3, 4)
            for _ in range(harmless_len):
                model.update(cntx_random, us_random)
                control_model.update(cntx_random, us_random)
        log_results()
        amy_transition_phase(model, Phase.SLEEP)
        amy_transition_phase(control_model, Phase.SLEEP)
        for _ in range(sleep_len):
            model.update()
            control_model.update()
        log_results()

        # Recall test
        amy_transition_phase(model, Phase.RECALL)
        amy_transition_phase(control_model, Phase.RECALL)
        model.update(cntx_B)
        control_model.update(cntx_B)
        for _ in range(30):
            model.update()
            control_model.update()
        all_cem_responses_model.append(model.AMY_C.log[model.AMY_C.current_step-1])
        all_cem_responses_control.append(control_model.AMY_C.log[control_model.AMY_C.current_step-1])
        print("Model", model.AMY_C.log[model.AMY_C.current_step-1], "Control", control_model.AMY_C.log[control_model.AMY_C.current_step-1])

        # Collect results from this simulation
        all_sum_of_W_model.append(sum_of_W_model)
        all_sum_of_W_control.append(sum_of_W_control)
        all_frac_W_greater0_01_model.append(frac_W_greater0_01_model)
        all_frac_W_greater0_01_control.append(frac_W_greater0_01_control)
        all_frac_W_greater0_1_model.append(frac_W_greater0_1_model)
        all_frac_W_greater0_1_control.append(frac_W_greater0_1_control)

    # Average results across simulations
    avg_sum_of_W_model = np.mean(all_sum_of_W_model, axis=0)
    avg_sum_of_W_control = np.mean(all_sum_of_W_control, axis=0)
    avg_frac_W_greater0_01_model = np.mean(all_frac_W_greater0_01_model, axis=0)
    avg_frac_W_greater0_01_control = np.mean(all_frac_W_greater0_01_control, axis=0)
    avg_frac_W_greater0_1_model = np.mean(all_frac_W_greater0_1_model, axis=0)
    avg_frac_W_greater0_1_control = np.mean(all_frac_W_greater0_1_control, axis=0)

    # Plot averaged results
    plt.figure(figsize=(11, 11))
    plt.plot(avg_sum_of_W_model, label="Trauma")
    plt.plot(avg_sum_of_W_control, label="Control")
    num_days = len(avg_sum_of_W_model) // 2  # Assuming two points per day
    x_ticks = range(0, len(avg_sum_of_W_model), 2)
    x_labels = [f"{i+1}" for i in range(num_days)]
    plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=22)
    plt.xlabel("Day", fontsize=22)
    plt.ylabel("Summed strength: BA(N)-BA(P) synapses", fontsize=20)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.show()

    plt.figure(figsize=(9, 9))
    plt.title("SEFL Protocol: Avg. Fraction of W > 0.01")
    plt.plot(avg_frac_W_greater0_01_model, label="Trauma")
    plt.plot(avg_frac_W_greater0_01_control, label="Control")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9, 9))
    plt.title("SEFL Protocol: Avg. Fraction of W > 0.1")
    plt.plot(avg_frac_W_greater0_1_model, label="Trauma")
    plt.plot(avg_frac_W_greater0_1_control, label="Control")
    plt.legend()
    plt.show()

    # Do wilcoxon signed-rank test on CeM responses
    #from scipy.stats import wilcoxon
    #import seaborn as sns
    #print("Wilcoxon signed-rank test:")
    #print(wilcoxon(all_cem_responses_model, all_cem_responses_control))

    # Plot boxplots and report Wilcoxon signed-rank test
    plt.figure(figsize=(9, 9))
    # plt.title("SEFL protocol - long delay", fontsize=22)
    plt.title("SEFL protocol -- order reversed", fontsize=22)
    sns.violinplot(
    data=[all_cem_responses_model, all_cem_responses_control], 
    palette="muted", 
    scale="width"
    )
    plt.xticks([0, 1], ["Trauma", "Control"], fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(0.0, 1.0)
    plt.ylabel("CeM activity", fontsize=22)
    plt.show()
    
def amy_show_delayed_extinction():
    num_runs = 10
    num_acquisition_trials = 40
    num_extinction_trials = 250
    num_delay_trials = 10
    num_renewal_trials = 30
    sleep_len = 165
    num_trials = num_acquisition_trials + num_extinction_trials + num_renewal_trials
    amy_responses_dict = {i: np.zeros(num_trials) for i in range(num_runs)}

    for run in range(num_runs):

        model = AmygdalaEngrams()
        online_learning(model, 1000)

        amy_responses = np.zeros(num_trials)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        cntx_B = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_C = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        # Acquisition
        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_acquisition_trials):
            model.update(cntx_A, 1.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[i] = pattern_converged_to_amy
        for i in range(num_delay_trials):
            model.update(cntx_B, 0.0)
        for i in range(num_delay_trials):
            model.update(cntx_C, 0.0)
        # Sleep
        amy_transition_phase(model, Phase.SLEEP)
        for i in range(sleep_len):
            """ if i == 0:
                model.BA_I.feedback_connections['BA_N'].tau_fb = 999999999
                model.BA_P.feedback_connections['BA_N'].tau_fb = 15000
                model.maintenance_rate_P = 0.0
                model.maintenance_rate_I = 0.0
            if i == sleep_len // 4:
                model.BA_I.feedback_connections['BA_N'].tau_fb = 9999999999
                model.BA_P.feedback_connections['BA_N'].tau_fb = 9999999999
                model.maintenance_rate_P = 0.075
                model.maintenance_rate_I = 0.075 """

            model.update()
            print("I cells active:", np.mean(model.BA_I.output), "mean support of top 5 I cells:", np.mean(np.sort(model.BA_I.supports)[-5:]))
        # Extinction
        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_extinction_trials):
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy
        for i in range(num_delay_trials):
            model.update(cntx_B, 0.0)
        for i in range(num_delay_trials):
            model.update(cntx_C, 0.0)
        # Sleep
        amy_transition_phase(model, Phase.SLEEP)
        for i in range(sleep_len):
            """if i == 0:
                model.BA_I.feedback_connections['BA_N'].tau_fb = 1400
                model.BA_P.feedback_connections['BA_N'].tau_fb = 15000
                model.maintenance_rate_P = 0.0
                model.maintenance_rate_I = 0.0
            if i == sleep_len // 4:
                model.BA_I.feedback_connections['BA_N'].tau_fb = 9999999999
                model.BA_P.feedback_connections['BA_N'].tau_fb = 9999999999
                model.maintenance_rate_P = 0.075
                model.maintenance_rate_I = 0.075 """

            model.update()
            print("I cells active:", np.mean(model.BA_I.output), "mean support of top 5 I cells:", np.mean(np.sort(model.BA_I.supports)[-5:]), "A cell output:", model.AMY_A.output, "fb_tau BA N/I:", model.BA_I.feedback_connections['BA_N'].tau_fb)
        # Renewal
        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_renewal_trials):
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy
        
        amy_responses_dict[run] = amy_responses

    # Compute the average responses
    avg_amy_responses = np.zeros(num_trials)
    for run in range(num_runs):
        avg_amy_responses += amy_responses_dict[run]
    avg_amy_responses /= num_runs

    plt.figure(figsize=(10, 2))
    plt.title("Delayed extinction: AMY activity over time")
    plt.plot(avg_amy_responses)
    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    # Horizontal bar indicating context over time
    plt.ylim(-0.01, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("CeM activity")
    plt.legend()
    plt.show()

    # plot IL activity log
    plt.figure()
    plt.title("Infralimbic cortex activity over time")
    # heatmap with keys as x-axis and keys of values as y-axis
    import pandas as pd
    df = pd.DataFrame(model.IL.log)
    import seaborn as sns
    sns.heatmap(df)
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.show()


def amy_show_immediate_extinction():
    num_runs = 10
    num_acquisition_trials = 40
    num_delay_trials = 10
    num_extinction_trials = 250
    num_renewal_trials = 30
    sleep_len = 165
    num_trials = num_acquisition_trials + num_extinction_trials + num_renewal_trials
    amy_responses_dict = {i: np.zeros(num_trials) for i in range(num_runs)}

    for run in range(num_runs):

        model = AmygdalaEngrams()
        online_learning(model, 1000)

        amy_responses = np.zeros(num_trials)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        cntx_B = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_C = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        # Acquisition
        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_acquisition_trials):
            model.update(cntx_A, 1.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[i] = pattern_converged_to_amy
        # Extinction
        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_extinction_trials):
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            print("AMY_A output:", model.AMY_A.output)
            amy_responses[num_acquisition_trials + i] = pattern_converged_to_amy
        for i in range(num_delay_trials):
            model.update(cntx_B, 0.0)
        for i in range(num_delay_trials):
            model.update(cntx_C, 0.0)
        # Sleep
        amy_transition_phase(model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()
            print("I cells active:", np.mean(model.BA_I.output), "mean support of top 5 I cells:", np.mean(np.sort(model.BA_I.supports)[-5:]), "A cell output:", model.AMY_A.output, "fb_tau BA N/I:", model.BA_I.feedback_connections['BA_N'].tau_fb)
        # Delay day
        for i in range(num_delay_trials):
            model.update(cntx_B, 0.0)
        for i in range(num_delay_trials):
            model.update(cntx_C, 0.0)
        # Sleep
        amy_transition_phase(model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()
        # Renewal
        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(num_renewal_trials):
            model.update(cntx_A, 0.0)
            pattern_converged_to_amy = model.AMY_C.log[model.AMY_C.current_step-1]
            amy_responses[num_acquisition_trials + num_extinction_trials + i] = pattern_converged_to_amy
        
        amy_responses_dict[run] = amy_responses

    # Compute the average responses
    avg_amy_responses = np.zeros(num_trials)
    for run in range(num_runs):
        avg_amy_responses += amy_responses_dict[run]
    avg_amy_responses /= num_runs

    plt.figure(figsize=(10, 2))
    plt.title("Immediate extinction: AMY activity over time")
    plt.plot(avg_amy_responses)
    for i in range(num_acquisition_trials):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=0.5)
    # Horizontal bar indicating context over time
    plt.ylim(-0.01, 1.01)
    plt.xlabel("Trial")
    plt.ylabel("CeM activity")
    plt.legend()
    plt.show()

    # plot IL activity log
    plt.figure()
    plt.title("Infralimbic cortex activity over time")
    # heatmap with keys as x-axis and keys of values as y-axis
    import pandas as pd
    df = pd.DataFrame(model.IL.log)
    import seaborn as sns
    sns.heatmap(df)
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.show()


def amy_show_engram_formation():
    num_runs = 1
    presentation_lens = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    hip_syn_strength_sums_acc = {i: [] for i in presentation_lens}
    ctx_syn_strength_sums_acc = {i: [] for i in presentation_lens}
    ba_syn_strength_sums_acc = {i: [] for i in presentation_lens}

    presentation_len = np.max(presentation_lens)

    for run in range(num_runs):

        model = AmygdalaEngrams()
        online_learning(model, 1000)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(presentation_len + 1):
            strength = 0.0 if i < 20 else 1.0
            model.update(cntx_A, strength)

            if i in presentation_lens:
                # Note HIP response to input pattern
                hip_pattern = model.HIP.log[model.HIP.current_step-1]
                # Note entries which are 1 in the HIP response
                hip_engram_cells = [i for i in range(len(hip_pattern)) if hip_pattern[i] == 1]
                ctx_pattern = model.CTX.log[model.CTX.current_step-1]
                ctx_engram_cells = [i for i in range(len(ctx_pattern)) if ctx_pattern[i] > 0.5]
                ba_pattern = model.BA_N.log[model.BA_N.current_step-1]
                ba_engram_cells = [i for i in range(len(ba_pattern)) if ba_pattern[i] == 1]
                    
                # Note the sum of synaptic strengths between cells i, j that are hip engrma cells
                hip_syn_strength_sum = 0
                hip_total_sum_strength = 0
                for j in hip_engram_cells:
                    hip_total_sum_strength += np.sum(model.HIP.W[j, :])
                    for k in hip_engram_cells:
                        hip_syn_strength_sum += model.HIP.W[j, k]
                hip_syn_strength_sums_acc[i].append(hip_syn_strength_sum / hip_total_sum_strength)

                ctx_syn_strength_sum = 0
                ctx_total_sum_strength = 0
                for j in ctx_engram_cells:
                    ctx_total_sum_strength += np.sum(model.CTX.W[j, :])
                    for k in ctx_engram_cells:
                        ctx_syn_strength_sum += model.CTX.W[j, k]
                ctx_syn_strength_sums_acc[i].append(ctx_syn_strength_sum / ctx_total_sum_strength)

                ba_syn_strength_sum = 0
                ba_total_sum_strength = 0
                for j in ba_engram_cells:
                    ba_total_sum_strength += np.sum(model.BA_N.W[j, :])
                    for k in ba_engram_cells:
                        ba_syn_strength_sum += model.BA_N.W[j, k]
                ba_syn_strength_sums_acc[i].append(ba_syn_strength_sum / ba_total_sum_strength)

    # Compute the average synaptic strength sums
    avg_hip_syn_strength_sums = {i: np.mean(hip_syn_strength_sums_acc[i]) for i in presentation_lens}
    print("avg_hip_syn_strength_sums", avg_hip_syn_strength_sums)
    avg_ctx_syn_strength_sums = {i: np.mean(ctx_syn_strength_sums_acc[i]) for i in presentation_lens}
    print("avg_ctx_syn_strength_sums", avg_ctx_syn_strength_sums)
    avg_ba_syn_strength_sums = {i: np.mean(ba_syn_strength_sums_acc[i]) for i in presentation_lens}
    print("avg_ba_syn_strength_sums", avg_ba_syn_strength_sums)

    # Plot average synaptic strength sums
    plt.figure(figsize=(8, 8))
    plt.title(f"Engram formation during Perception", fontsize=24)
    plt.plot(list(avg_hip_syn_strength_sums.keys()), list(avg_hip_syn_strength_sums.values()), label="HIP", linewidth=2.5)
    plt.plot(list(avg_ctx_syn_strength_sums.keys()), list(avg_ctx_syn_strength_sums.values()), label="CTX", linewidth=2.5)
    plt.plot(list(avg_ba_syn_strength_sums.keys()), list(avg_ba_syn_strength_sums.values()), label="BA(N)", linewidth=2.5)
    # Dotted vertical line at 25
    plt.axvline(x=20, color='grey', linestyle='--', alpha=0.75, linewidth=2.5)
    # Text annotation at 25
    plt.text(16.75, 0.3, "US onset", rotation=90, verticalalignment='center', fontsize=24)
    plt.xlabel("Time step", fontsize=24)
    plt.ylabel("Encoding strength", fontsize=24)
    # set lower ylim to 0, leaving the upper limit to be what it currently is
    plt.ylim(0, plt.ylim()[1])
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(prop={'size': 24})
    plt.show()


def amy_show_sleep_replay():
    num_contexts = 10
    perception_len = 30
    sleep_len = 165
    us_strength = 0.0

    model = AmygdalaEngrams()
    online_learning(model, 1000)

    hip_perception_patterns = { i: np.zeros(model.HIP.N) for i in range(num_contexts)}
    ctx_perception_patterns = { i: np.zeros(model.CTX.N) for i in range(num_contexts)}
    ba_perception_patterns = { i: np.zeros(model.BA_N.N) for i in range(num_contexts)}

    replay_distance_hip = np.zeros((num_contexts, sleep_len))
    replay_distance_ctx = np.zeros((num_contexts, sleep_len))
    replay_distance_ba = np.zeros((num_contexts, sleep_len))

    sums_of_V = { i: np.zeros(sleep_len) for i in range(num_contexts)}
    hip_engram_cells = { i: None for i in range(num_contexts)}
    currently_replayed_context = { t: None for t in range(sleep_len) }

    amy_transition_phase(model, Phase.PERCEPTION)
    for i in range(num_contexts):
        cntx = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        if i == num_contexts // 2:
            perception_len = 30
            us_strength = 0.5
        for j in range(perception_len):
            model.update(cntx, us_strength)
        hip_perception_patterns[i] = np.array([_ for _ in model.HIP.log[model.HIP.current_step-1].values()])
        ctx_perception_patterns[i] = np.array([_ for _ in model.CTX.log[model.CTX.current_step-1].values()])
        ba_perception_patterns[i] = np.array([_ for _ in model.BA_N.log[model.BA_N.current_step-1].values()])

        hip_engram_cells[i] = [c for c in range(len(hip_perception_patterns[i])) if hip_perception_patterns[i][c] == 1]

    amy_transition_phase(model, Phase.SLEEP)
    for t in range(sleep_len):
        model.update()
        hip_current_pattern = np.array([_ for _ in model.HIP.log[model.HIP.current_step-1].values()])
        ctx_current_pattern = np.array([_ for _ in model.CTX.log[model.CTX.current_step-1].values()])
        ba_current_pattern = np.array([_ for _ in model.BA_N.log[model.BA_N.current_step-1].values()])

        for i in range(num_contexts):
            hip_dist = recall_metric(hip_perception_patterns[i], hip_current_pattern)
            ctx_dist = recall_metric(ctx_perception_patterns[i], ctx_current_pattern)
            ba_dist = recall_metric(ba_perception_patterns[i], ba_current_pattern)
            replay_distance_hip[i, t] = hip_dist
            replay_distance_ctx[i, t] = ctx_dist
            replay_distance_ba[i, t] = ba_dist
            print(hip_dist, ctx_dist, ba_dist)

            hip_total_sum_strength = 0.0

            hip_syn_strength_sum = 0.0
            for j in hip_engram_cells[i]:
                hip_total_sum_strength += np.sum(model.HIP.V[j, :])
                for k in hip_engram_cells[i]:
                    hip_syn_strength_sum += model.HIP.V[j, k]
            sums_of_V[i][t] = hip_syn_strength_sum / hip_total_sum_strength

        min_hip_dist = np.min(replay_distance_hip[:, t])    
        if min_hip_dist < 0.15:
            currently_replayed_context[t] = np.argmin(replay_distance_hip[:, t])   

    # Plot replay distances
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    # suptitle "Sleep replay"
    fig.suptitle("Replay of 10 contexts over Sleep", fontsize=24)
    # Matrix imshow for replay distances
    im = axes[0].imshow(replay_distance_hip, aspect='auto', cmap='coolwarm')
    axes[0].set_yticks([0, 5, 9])
    axes[0].set_yticklabels([1, 6, 10], fontsize=18)
    axes[0].set_ylabel("Context", fontsize=21)
    axes[0].set_xticklabels([])
    axes[0].set_xlabel("")
    axes[0].set_title("HIP", fontsize=21)
    axes[1].imshow(replay_distance_ctx, aspect='auto', cmap='coolwarm')
    axes[1].set_yticks([0, 5, 9])
    axes[1].set_yticklabels([1, 6, 10], fontsize=18)
    axes[1].set_xticklabels([])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Context", fontsize=21)
    axes[1].set_title("CTX", fontsize=21)
    axes[2].imshow(replay_distance_ba, aspect='auto', cmap='coolwarm')
    axes[2].set_yticks([0, 5, 9])
    axes[2].set_yticklabels([1, 6, 10], fontsize=18)
    axes[2].set_ylabel("Context", fontsize=21)
    axes[2].set_title("BA(N)", fontsize=21)
    axes[2].set_xlabel("Time step", fontsize=21)
    axes[2].set_xticks([0, 50, 100, 150])
    axes[2].set_xticklabels([0, 50, 100, 150], fontsize=18)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.15, 0.05, 0.7])
    cbar_ax.tick_params(labelsize=22)
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Replay distance", fontsize=21)

    plt.show()

    print(currently_replayed_context)

    plt.figure()
    contexts_to_plot = [5, 6, 7, 8, 9]
    colors = { 5: 'r', 6: 'g', 7: 'b', 8: 'c', 9: 'm'}
    for j in contexts_to_plot:
        plt.plot(sums_of_V[j], label=f"Context {j+1}", color=colors[j])
    min_y_value = np.min([np.min(sums_of_V[i]) for i in contexts_to_plot])
    # Place a horizontal rectangle below min_y_value. Fill it with color of currently replayed context
    for t in range(sleep_len):
        if currently_replayed_context[t] in contexts_to_plot:
            plt.axvspan(t-1, t, color=colors[currently_replayed_context[t]], alpha=0.5, ymin=0, ymax=0.05)
    plt.title(f'Evolution of "Cell adaptation" weights for individual context engrams')
    plt.xlabel("Sleep time step")
    plt.ylabel("Normalized sum of V")
    plt.legend()
    plt.show()

def amy_show_recruitability():

    model = AmygdalaEngrams()

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

    num_steps = 200
    cond_start = 160
    cond_end = 180
    min_rec = 0.8
    min_min_rec = 0.4
    fear_recruitability_acc = np.zeros((num_steps, model.BA_I.N))

    amy_transition_phase(model, Phase.PERCEPTION)
    for i in range(num_steps):
        model.update(cntx_A, 0.0)

        fear_recruitability_acc[i, :] = model.fear_activation

    # Sort the columns of the matrix according to model.P_cell_recruitability_phase
    fear_recruitability_acc = fear_recruitability_acc[:, np.argsort(model.P_cell_recruitability_phase)]

    # Plot fear recruitability
    plt.figure()
    plt.title("Evolution of 'P-cell recruitability' over time", fontsize=20)
    plt.imshow(fear_recruitability_acc.T, aspect='auto', cmap='coolwarm')
    plt.xlabel("Time step", fontsize=16)
    plt.ylabel("P-cell index (sorted by phase)", fontsize=16)
    plt.colorbar(label="Cell Recruitability")
    plt.show()

    # For an individual P-cell, plot its recruitability over time
    plt.figure()
    plt.title("Evolution of 'P-cell recruitability' for an individual P-cell", fontsize=20)
    plt.plot(fear_recruitability_acc[:, 0], label="P-cell 0")
    plt.xlabel("Time step", fontsize=16)
    plt.ylabel("Cell Recruitability", fontsize=16)
    plt.legend()
    plt.show()

    # For each cell, count the number of steps between cond_start and cond_end where recruitability is above min_rec
    num_steps_above_min_rec = np.sum(fear_recruitability_acc[cond_start:cond_end, :] > min_rec, axis=0)
    num_steps_above_minmin_rec = np.sum(fear_recruitability_acc[cond_start:cond_end, :] > min_min_rec, axis=0)
    print("Number of steps above min_rec for each P-cell:", num_steps_above_min_rec)
    # Cell 1 = Any cell with number of steps above min_rec = 0
    cell_1 = np.where(num_steps_above_minmin_rec == 0)[0][5]
    print("P-cells with 0 steps above min_rec:", cell_1)
    # Cell 2 = Any cell with number of steps above 0 < num_steps_above_min_rec < 5
    cell_2 = np.where((num_steps_above_min_rec > 0) & (num_steps_above_min_rec < 3))[0][0]
    # Cell 3 = Any cell with number of steps above 8 < num_steps_above_min_rec < 12
    cell_3 = np.where((num_steps_above_min_rec > 7) & (num_steps_above_min_rec < 9))[0][0]
    # Cell 4 = Any cell with number of steps above 15 < num_steps_above_min_rec < 20
    cell_4 = np.where((num_steps_above_min_rec > 15) & (num_steps_above_min_rec < 20))[0][0]

    # Plot the recruitabiity of those 4 cells
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    for i, cell in enumerate([cell_1, cell_2, cell_3, cell_4]):
        axes[i // 2, i % 2].plot(fear_recruitability_acc[:, cell])
        axes[i // 2, i % 2].set_title(f"P-cell {i + 1}", fontsize=18)
        if i == 3 or i == 2:
            axes[i // 2, i % 2].set_xlabel("Time step", fontsize=18)
        if i == 0 or i == 2:
            axes[i // 2, i % 2].set_ylabel("Recruitability", fontsize=18)
        axes[i // 2, i % 2].set_ylim(0, 1.5)
        # Shade cond time window in red
        axes[i // 2, i % 2].axvspan(cond_start, cond_end, color='red', alpha
        =0.2)
        # Separate with dotted vertical lines at cond_start and cond_end
        axes[i // 2, i % 2].axvline(x=cond_start, color='grey', linestyle='--', alpha=0.5)
        axes[i // 2, i % 2].axvline(x=cond_end, color='grey', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
        


    # Plot sum of recruitability over time
    """plt.figure()
    plt.title("Evolution of sum of 'P-cell recruitability' over time", fontsize=20)
    plt.plot(np.sum(fear_recruitability_acc, axis=1))
    plt.xlabel("Time step", fontsize=16)
    plt.ylabel("Sum of recruitability", fontsize=16)
    # set lower ylim to 0, leaving the upper limit to be what it currently is
    plt.ylim(0, plt.ylim()[1])
    plt.show()"""

def amy_show_p_cell_recruitment():
    num_trials = 40

    model = AmygdalaEngrams()
    online_learning(model, 1000)
    amy_transition_phase(model, Phase.PERCEPTION)

    cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

    for i in range(num_trials):
        print("Trial", i)
        print(model.AMY_C.output)
        model.update(cntx_A, 0.9)

    # Note currently active BA(N) engram cells
    ba_n_activity = model.BA_N.log[model.BA_N.current_step-1]
    active_ban_cells_A_1 = [i for i in range(len(ba_n_activity)) if ba_n_activity[i] == 1]

    ba_n_to_p_connections = model.BA_P.feedback_connections['BA_N']
    W = ba_n_to_p_connections.W # (N_BAN, N_BAP)

    # For each P-cell, compute sum of weights from active BA(N) cells
    sums_of_weights = np.sum(W[active_ban_cells_A_1, :], axis=0)
    print(sums_of_weights)

    # Plot histogram of sums of weights
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.hist(sums_of_weights, bins=np.arange(0, 40, 2))
    ax.axvline(x=np.exp(model.BA_P.firing_threshold), color='red', linestyle='--')
    plt.title("Context. A - Before Sleep", fontsize=24)
    plt.xlabel("Summed weight", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    ax.tick_params(labelsize=22)
    # use log scale for y-axis
    plt.yscale('log')
    plt.yticks([1, 10, 100])
    plt.ylim(0.5, 250)
    plt.show()


    ba_p_activity = model.BA_P_PRE_US.log[model.BA_P_PRE_US.current_step-1]
    active_p_cells = [i for i in range(len(ba_p_activity)) if ba_p_activity[i] > 0.5]
    # print len
    print("Number of active P-cells:", len(active_p_cells))

    fear_recruitability = model.fear_activation
    # Compute correlation between sums of weights and fear recruitability
    from scipy.stats import pearsonr
    print("Correlation between sums of weights and fear recruitability:", pearsonr(sums_of_weights, fear_recruitability))

    B_overlap = 0.5
    B_diff = 1 - B_overlap
    hcs_to_change = int(model.SENSORY_CORTEX.num_hcs * B_diff)
    units_to_change = hcs_to_change * model.SENSORY_CORTEX.units_per_hc
    cntx_B = cntx_A.copy()
    cntx_B_diff = gen_random_simple_pattern(units_to_change, hcs_to_change)
    random_hcs = np.random.choice(model.SENSORY_CORTEX.num_hcs, hcs_to_change, replace=False)
    for j, hc in enumerate(random_hcs):
        for i in range(model.SENSORY_CORTEX.units_per_hc):
            cntx_B[hc * model.SENSORY_CORTEX.units_per_hc + i] = cntx_B_diff[j * model.SENSORY_CORTEX.units_per_hc + i]

    for i in range(num_trials):
        print("Trial", i)
        model.update(cntx_B, 0.0)

    # Note currently active BA(N) engram cells
    ba_n_activity = model.BA_N.log[model.BA_N.current_step-1]
    active_ban_cells_B_1 = [i for i in range(len(ba_n_activity)) if ba_n_activity[i] == 1]

    # Note overlap between active BA(N) cells in A and B
    overlap = len(set(active_ban_cells_A_1) & set(active_ban_cells_B_1)) / len(active_ban_cells_A_1)
    print("Overlap between active BA(N) cells in A and B:", overlap)
    print("The engrams themselves have sizes:", len(active_ban_cells_A_1), len(active_ban_cells_B_1))

    ba_n_to_p_connections = model.BA_P.feedback_connections['BA_N']
    W = ba_n_to_p_connections.W # (N_BAN, N_BAP)

    # For each P-cell, compute sum of weights from active BA(N) cells
    sums_of_weights = np.sum(W[active_ban_cells_B_1, :], axis=0)

    # Plot histogram of sums of weights
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.hist(sums_of_weights, bins=np.arange(0, 40, 2))
    ax.axvline(x=np.exp(model.BA_P.firing_threshold), color='red', linestyle='--')
    plt.title("Context. B - Before Sleep", fontsize=24)
    plt.xlabel("Summed weight", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    ax.tick_params(labelsize=22)
    # use log scale for y-axis
    plt.yscale('log')
    plt.yticks([1, 10, 100])
    plt.ylim(0.5, 250)
    plt.show()

    amy_transition_phase(model, Phase.SLEEP)
    for i in range(165):
        model.update()

    amy_transition_phase(model, Phase.PERCEPTION)
    model.update(cntx_A)
    for i in range(1):
        model.update(cntx_A)

    print("C cell activity recall A:", model.AMY_C.output)
    
    # Compute and plot histogram of sums of weights again
    ba_n_activity = model.BA_N.log[model.BA_N.current_step-1]
    active_ban_cells_A_2 = [i for i in range(len(ba_n_activity)) if ba_n_activity[i] == 1]

    ba_n_to_p_connections = model.BA_P.feedback_connections['BA_N']
    W = ba_n_to_p_connections.W # (N_BAN, N_BAP)

    # For each P-cell, compute sum of weights from active BA(N) cells
    sums_of_weights = np.sum(W[active_ban_cells_A_2, :], axis=0)

    # Plot histogram of sums of weights
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.hist(sums_of_weights, bins=np.arange(0, 40, 2))
    ax.axvline(x=np.exp(model.BA_P.firing_threshold), color='red', linestyle='--')
    plt.title("Context. A - After Sleep", fontsize=24)
    plt.xlabel("Summed weight", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    ax.tick_params(labelsize=22)
    # use log scale for y-axis
    plt.yscale('log')
    plt.yticks([1, 10, 100])
    plt.ylim(0.5, 250)
    plt.show()

    model.update(cntx_B)
    for i in range(1):
        model.update(cntx_B)

    print("C cell activity recall B:", model.AMY_C.output)

    # Compute and plot histogram of sums of weights again
    ba_n_activity = model.BA_N.log[model.BA_N.current_step-1]
    active_ban_cells_B_2 = [i for i in range(len(ba_n_activity)) if ba_n_activity[i] == 1]

    # Note overlap between active BA(N) cells in A and B
    overlap = len(set(active_ban_cells_A_2) & set(active_ban_cells_B_2)) / len(active_ban_cells_A_2)
    print("Overlap between active BA(N) cells in A and B:", overlap)
    print("The engrams themselves have sizes:", len(active_ban_cells_A_2), len(active_ban_cells_B_2))

    # Overlap A_1 and A_2
    overlap = len(set(active_ban_cells_A_1) & set(active_ban_cells_A_2)) / len(active_ban_cells_A_1)
    print("Overlap between active BA(N) cells in A_1 and A_2:", overlap)
    print("The engrams themselves have sizes:", len(active_ban_cells_A_1), len(active_ban_cells_A_2))

    # Overlap B_1 and B_2
    overlap = len(set(active_ban_cells_B_1) & set(active_ban_cells_B_2)) / len(active_ban_cells_B_1)
    print("Overlap between active BA(N) cells in B_1 and B_2:", overlap)
    print("The engrams themselves have sizes:", len(active_ban_cells_B_1), len(active_ban_cells_B_2))

    ba_n_to_p_connections = model.BA_P.feedback_connections['BA_N']
    W = ba_n_to_p_connections.W # (N_BAN, N_BAP)

    # For each P-cell, compute sum of weights from active BA(N) cells
    sums_of_weights = np.sum(W[active_ban_cells_B_2, :], axis=0)

    # Plot histogram of sums of weights
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.hist(sums_of_weights, bins=np.arange(0, 40, 2))
    ax.axvline(x=np.exp(model.BA_P.firing_threshold), color='red', linestyle='--')
    plt.title("Context. B - After Sleep", fontsize=24)
    plt.xlabel("Summed weight", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    ax.tick_params(labelsize=22)
    # use log scale for y-axis
    plt.yscale('log')
    plt.yticks([1, 10, 100])
    plt.ylim(0.5, 250)
    plt.show()

def amy_assess_fear_engram_overlap():
    acquisition_len = 40
    short_home_len = 20
    home_len = 80
    sleep_len = 165

    num_runs = 40

    overlaps_AB = []
    overlaps_AC = []
    overlaps_AD = []
    fear_responses_A = []
    fear_responses_B = []
    fear_responses_C = []
    fear_responses_D = []

    for run in range(num_runs):
        print("Run", run)

        model = AmygdalaEngrams()
        online_learning(model, 1000)
        amy_transition_phase(model, Phase.PERCEPTION)

        cntx_A = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_B = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_C = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_D = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)
        cntx_HOME = gen_random_simple_pattern(model.SENSORY_CORTEX.N, model.SENSORY_CORTEX.num_hcs)

        for i in range(acquisition_len):
            model.update(cntx_A, 0.75)
        for i in range(short_home_len):
            model.update(cntx_HOME, 0.0)
        for i in range(acquisition_len):
            model.update(cntx_B, 0.75)
        for i in range(home_len):
            model.update(cntx_HOME, 0.0)
        for i in range(acquisition_len):
            model.update(cntx_C, 0.75)

        amy_transition_phase(model, Phase.SLEEP)
        for i in range(sleep_len):
            model.update()

        amy_transition_phase(model, Phase.PERCEPTION)
        for i in range(acquisition_len):
            model.update(cntx_D, 0.8)

        amy_transition_phase(model, Phase.RECALL)
        model.update(cntx_A)
        for i in range(30):
            model.update()
        P_cell_activity_A = model.BA_P.output
        P_cell_engram_A = [i for i in range(len(P_cell_activity_A)) if P_cell_activity_A[i] > 0.5]
        fear_responses_A.append(model.AMY_C.output)

        model.update(cntx_B)
        for i in range(30):
            model.update()
        P_cell_activity_B = model.BA_P.output
        P_cell_engram_B = [i for i in range(len(P_cell_activity_B)) if P_cell_activity_B[i] > 0.5]
        fear_responses_B.append(model.AMY_C.output)

        model.update(cntx_C)
        for i in range(30):
            model.update()
        P_cell_activity_C = model.BA_P.output
        P_cell_engram_C = [i for i in range(len(P_cell_activity_C)) if P_cell_activity_C[i] > 0.5]
        fear_responses_C.append(model.AMY_C.output)

        model.update(cntx_D)
        for i in range(30):
            model.update()
        P_cell_activity_D = model.BA_P.output
        P_cell_engram_D = [i for i in range(len(P_cell_activity_D)) if P_cell_activity_D[i] > 0.5]
        fear_responses_D.append(model.AMY_C.output)

        overlap_AB = len(set(P_cell_engram_A) & set(P_cell_engram_B)) / len(P_cell_engram_A)
        overlap_AC = len(set(P_cell_engram_A) & set(P_cell_engram_C)) / len(P_cell_engram_A)
        overlap_AD = len(set(P_cell_engram_A) & set(P_cell_engram_D)) / len(P_cell_engram_A)

        # Print the 4 engrams, as well as the overlap of each with the first
        print("Engram A:", P_cell_engram_A)
        print("Engram B:", P_cell_engram_B)
        print("Engram C:", P_cell_engram_C)
        print("Engram D:", P_cell_engram_D)
        print("Overlap A-B:", overlap_AB)
        print("Overlap A-C:", overlap_AC)
        print("Overlap A-D:", overlap_AD)

        overlaps_AB.append(overlap_AB)
        overlaps_AC.append(overlap_AC)
        overlaps_AD.append(overlap_AD)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Fig 1: Plot 3 vertical boxplots, showing the distribution of overlaps
    axes[0].boxplot([overlaps_AB, overlaps_AC, overlaps_AD], labels=['A-B', 'A-C', 'A-D'])
    axes[0].set_title("Overlap of recruited P-cell sets A-B, A-C, A-D")
    axes[0].set_ylabel("Overlap")
    # Fig 2: Plot 4 vertical boxplots, showing the distribution of fear responses
    axes[1].boxplot([fear_responses_A, fear_responses_B, fear_responses_C, fear_responses_D], labels=['A', 'B', 'C', 'D'])
    axes[1].set_title("Fear responses to contexts A, B, C, D")
    axes[1].set_ylabel("Fear response")
    axes[1].set_ylim(0, 1)
    plt.show()




if __name__ == "__main__":

    # Fig 4a) Engram formation
    # amy_show_engram_formation()
    # Fig 4b) Sleep replay
    # amy_show_sleep_replay()
    # Fig 4c) Recall performance over time
    # amy_show_fear_learning()
    # Fig 4d) Fear acquisition and extinction
    # amy_show_fear_extinction_AA()

    # Fig 5b) Generalization gradients
    # amy_show_fear_generalization()
    # Fig 5c) Fear renewal
    # amy_show_fear_renewal_ABA()
    # amy_show_fear_renewal_ABC()
    # amy_show_fear_renewal_AAB()
    # Fig 5d) Increases in fear generalization with memory age
    # amy_show_generalization_consolidation()

    # Fig 6 Effect of sleep homeostasis on P-cell recruitment
    # amy_show_p_cell_recruitment()

    # Fig 7) Sleep deprivation
    # amy_sleep_disruption_chronic()

    # Fig 8a, b, d) SEFL protocol
    # amy_sefl_synapse_sum()
    # Fig 8c) SEFL reverse protocol
    # amy_sefl_synapse_sum_reverse()

    # Remaining protocols (not corresponding to figures of the main text)
    # amy_show_fear_acquisition()
    # amy_show_fear_extinction_AB()
    # amy_show_fear_renewal_ABCDA()
    # amy_show_fear_renewal_ABCDA_same_overlap()
    # amy_show_fear_renewal_ABC_BC_sim()
    # amy_within_session_extinction()
    # amy_show_fear_acquisition_daily_ABC()
    # amy_sefl_protocol()
    # amy_sefl_reverse()
    # amy_show_delayed_extinction()
    # amy_show_immediate_extinction()
    # amy_show_recruitability()
    # amy_assess_fear_engram_overlap()