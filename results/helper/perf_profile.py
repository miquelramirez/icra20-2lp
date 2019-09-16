# Performance profiles
# adapted from Benjamin Recht's blog post
# http://www.argmin.net/2018/03/26/performance-profiles/

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betainc

def poor_man_welch_test(target_mean, target_std, target_n, comp_mean, comp_std, comp_n):
    # Computes a Welch test to see if the comparitor sample is larger than the target
    # sample. This is a one-sided test.
    # This works from sufficient statistics alone, and that's because this is all that
    # people report in their papers. I'm no more happy about this than you are.

    nu = ((target_std ** 2 / target_n + comp_std ** 2 / comp_n) ** 2 /
          (target_std ** 4 / target_n / (target_n - 1) + comp_std ** 4 / comp_n / (comp_n - 1)))
    t_stat = ((target_mean - comp_mean)
              / np.sqrt(target_std ** 2 / target_n + comp_std ** 2 / comp_n))

    return 0.5 * betainc(nu / 2, 1 / 2, nu / (t_stat ** 2 + nu))


def optimality_deviation(data_means, data_std, data_n, best_fn=np.argmax):
    # compute the deviation from optimality as given by the log-likehood of a Welch t-test
    num_prob, num_method = data_means.shape
    likelihood = np.zeros((num_prob, num_method))
    for prob in range(num_prob):
        best_idx = best_fn(data_means[prob, :])

        # compute the Welsh t-test to determine the p-value associated
        # with a method having mean higher than the observed highest reward
        for method in range(num_method):
            likelihood[prob, method] = -np.log10(poor_man_welch_test(
                data_means[prob, best_idx], data_std[prob, best_idx], data_n[prob, best_idx],
                data_means[prob, method], data_std[prob, method], data_n[prob, method]))

        # denote the likelihood of the best observation as being 1. This is merely counting
        # the number of times a method achieves the highest mean.
        likelihood[prob, best_idx] = 0
    return likelihood


def welch_t_perf_prof(data_means, data_std, data_n, data_names, tau_min=0.3, tau_max=3.0, npts=100, best_fn=np.argmax):
    num_prob, num_method = data_means.shape
    rho = np.zeros((npts, num_method))

    # This is the d[p,m] function discussed in the blog.
    # For this post, I'm using the log-likelihood of the Welch t-test.
    # But this is where you'd write whatever method you think would work better.
    dist_like_fun = optimality_deviation(data_means, data_std, data_n, best_fn)

    # Compute the cumulative rates of the distance being less than a fixed threshold
    tau = np.linspace(tau_min, tau_max, npts)
    for method in range(num_method):
        for k in range(npts):
            rho[k, method] = np.sum(dist_like_fun[:, method] < tau[k]) / num_prob
    return tau, rho, num_method, npts

def make_plot(tau, rho, num_method, npts, data_names):
    # make plot
    colors = ['#2D328F', '#F15C19', "#81b13c", "#ca49ac", "000000"]
    label_fontsize = 14
    tick_fontsize = 12
    linewidth = 3
    plt.figure(figsize=(20, 10))
    for method in range(num_method):
        plt.plot(tau, rho[:, method], color=colors[method], linewidth=linewidth, label=data_names[method])

    plt.xlabel(r'$-\log_{10}(\tau)$', fontsize=label_fontsize)
    plt.ylabel(r'fraction with $p_{val} \geq \tau$', fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.grid(True)
    plt.show()