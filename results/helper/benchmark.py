# Helper routines for benchmarking planners
import numpy as np
from .perf_profile import welch_t_perf_prof

def distance_to_goal(data):
    return data['gap'].mean(), max(1e-4, data['gap'].std())

def solution_cost(data):
    costs = [x for x in data['sol_cost']]

    for idx, x in enumerate(data['gap']):
        if x >= 1e-2:  # not a solution gap is too big
            costs[idx] = 1e6
    costs = np.array(costs)
    return costs.mean(), max(1e-4, costs.std())

def runtime(data):
    runtimes = [x for x in data['total_time']]
    runtimes = np.array(runtimes)
    return runtimes.mean(), max(1e-4, runtimes.std())

def generated(data):
    runtimes = [x for x in data['disc_gen']]
    runtimes = np.array(runtimes)
    return runtimes.mean(), max(1e-4, runtimes.std())

def disc_planner_calls(data):
    runtimes = [x for x in data['disc_num_calls']]
    runtimes = np.array(runtimes)
    return runtimes.mean(), max(1e-4, runtimes.std())

def compute_performance_profiles(names, planners, planners_groups, baseline, baseline_groups, performance_index_fn, **params):
    data_means = []
    data_std = []
    data_n = []
    data_names = []
    S = params['S']

    for algo in planners:
        means_vec = []
        std_vec = []
        for gk in planners_groups.groups:
            S_value, algorithm, _ = gk
            if algorithm == algo and S_value == S:
                S_value, algorithm, _ = gk
                u, s = performance_index_fn(planners_groups.get_group(gk))
                means_vec += [u]
                std_vec += [s]
        if len(means_vec) == 0:
            continue
        data_names += [algo]
        data_means += [means_vec]
        data_std += [std_vec]
        data_n += [np.tile(len(means_vec), [1, 20])]
        # print(algo, means_vec, std_vec, len(means_vec))
        # two_lp_means += [two_lp_groups.get_group(gk)['gap'].mean()]
        # two_lp_std += [two_lp_groups.get_group(gk)['gap'].std()]

    if baseline is not None:
        baseline_means_vec = []
        baseline_std_vec = []

        for gk in baseline_groups.groups:
            S_value, algorithm, inst = gk
            if S_value == S:
                u, s = performance_index_fn(baseline_groups.get_group(gk))
                baseline_means_vec += [u]
                baseline_std_vec += [s]
        data_names += [baseline]
        data_means += [baseline_means_vec]
        data_std += [baseline_std_vec]
        data_n += [np.tile(len(baseline_means_vec), [1, 20])]
        # print('baseline', baseline_means_vec, baseline_std_vec, len(baseline_means_vec))

    print([len(m) for m in data_means])
    data_names = [names[n] for n in data_names]
    data_means = np.vstack(tuple(data_means)).T
    data_std = np.vstack(tuple(data_std)).T
    data_n = np.vstack(tuple(data_n)).T

    tau, rho, num_method, npts = welch_t_perf_prof(data_means, data_std, data_n, data_names, tau_min=0.5, tau_max=3,
                                                      best_fn=params['best_fn'])

    return tau, rho, num_method, npts, data_names