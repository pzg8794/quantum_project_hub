from main import simulate_learning, simulate_spatial_learning, simulate_Waxman_constant
from quarc.blobbed_network import BlobbedGridNetwork, BlobbedQCastNetwork
import quarc.simulation
from quarc.simulation import Request
from quarc.protocol import StaticGridProtocol, QuARC
from adaptive_sim import simulate
from quarc.thresholding import get_thresholds
import quarc.plotting as plot
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import networkx as nx
import pickle
from scipy.stats import sem


learning_algs = ['QuARC', '1-by-1 Clusters', '2-by-2 Clusters', '4-by-4 Clusters', '8-by-8 Clusters', '16-by-16 Cluster']
learning_algs2 = ['QuARC', '2-by-2 Clusters', '8-by-8 Clusters']
algs = ['QuARC', 'QuARC-TS']
fairness_algs = ['QuARC', 'QuARC-1']
colors = {'QuARC': '#1f77b4', 'Q-CAST': '#ff7f0e', 'ALG-N-FUSION': '#2ca02c', 'QuARC-TS': '#6F2633', 'QuARC-1': '#d62728'}


def save_learning_fig(ps, qs, filename):
    title = ("learning-N=%d_ps=" % N) + str(ps) + "_qs=" + str(qs)
    successes_list = []
    for idx, alg in enumerate(learning_algs):
        seed = 0
        statsfile = os.path.join('results', title, alg, 'stats' + str(seed) + '.pkl')
        if os.path.exists(statsfile):
            f = open(statsfile, 'rb')
            stats = pickle.load(f)
            f.close()
            s = stats.successes
            print(alg, len(s))
        else:
            s = [-1]*len(s) # Include dummy data for this curve to keep coloring scheme consistent
        successes_list.append(s)

    bin_size = 500
    save_path = os.path.join('figures', filename)
    plot.binned_throughput(successes_list, bin_size, labels=learning_algs, title='', save_path=save_path, lw=True, legend=False)
    print('Saved figure as %s\n' % save_path)

def save_spatial_learning_fig(path_to_fig, filename):
    save_path = os.path.join('figures', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.copyfile(path_to_fig, save_path)
    print('Saved figure as %s\n' % save_path)

def save_throughput_fig(E_p, ns, qs, xlabel, filename, E_d=6, nsd=10):
    assert(isinstance(ns, list) and isinstance(qs, list))
    if xlabel == 'n':
        # Plot n on x-axis
        xs = ns
        assert(len(qs) == 1)
    elif xlabel == 'q':
        # Plot q on x-axis
        xs = qs
        assert(len(ns) == 1)
    else:
        raise RuntimeError('Invalid args.')
    
    
    plt.rcParams.update({'font.size': 12})
    throughput_type = 'regular'
    start, end = (4000, 5000)

    ps = [E_p]
    width = None
    nqubits = None

    results = {alg : {x : (0,0) for x in xs} for alg in algs}
    for i, E_p in enumerate(ps):
        for n in ns:
            for q in qs:
                title = ("n=%d_E_p=%.2f_q=%.2f_E_d=%.1f_nsd=%d" % (n, E_p, q, E_d, nsd) 
                         + ('_w=%d' % width if width else '')
                         + ('_qb=%d' % nqubits if isinstance(nqubits, int) else ('_qb=max' if nqubits == 'max' else ''))
                        )

                print(n, q, E_p)
                for idx, alg in enumerate(algs):
                    folder = os.path.join('results', title, alg)
                    tps = []
                    seed = 0
                    while os.path.exists(os.path.join(folder, 'stats' + str(seed) + '.pkl')):
                        f = open(os.path.join(folder, 'stats' + str(seed) + '.pkl'), 'rb')
                        stats = pickle.load(f)
                        f.close()
                        if n > 400 and alg in ['Q-CAST', 'ALG-N-FUSION']:
                            tp = stats.avg_throughput(0, 1000, throughput_type)
                        else:
                            tp = stats.avg_throughput(start, end, throughput_type)

                        tps.append(tp)
                        #print(seed, tp)
                        seed += 1
                    print('  %d' % (len(tps)))

                    mean = np.mean(tps)
                    std_err = sem(tps)
                    x = n if xlabel == 'n' else q
                    results[alg][x] = (mean, std_err)

    styles = ['-', ':']
    for i, alg in enumerate(algs):
        for j, E_p in enumerate(ps):
            alg_label = 'QuARC-2D' if alg == 'QuARC' else alg
            label = "%s ($E_p=%.1f$)" % (alg_label, E_p)
            color = colors[alg] 
            style = styles[j]
            ys = [results[alg][x][0] for x in xs]
            print(alg, ys)
            errs = [results[alg][x][1] for x in xs]
            errs2 = np.array(errs)
            errs2[errs>=ys] = ys[errs>=ys]*.999999
            yerr = errs if xlabel == 'n' else [errs2, errs]
            #plt.plot(ns, ys, style, color=color, label=label)
            plt.errorbar(xs, ys, yerr=yerr, color=color, linestyle=style, marker='o', label=label)
            

    plt.xlabel(xlabel)
    plt.ylabel('Mean Aggregate Throughput' if throughput_type == 'aggregate' else 'Mean Throughput')
    plt.ylim(bottom=0)
    
    if xlabel == 'q':
        plt.yscale('log')
        plt.xticks(qs)
        plt.ylim(1E-2,6)
        if len(ps) == 1:
            if ps[0] == 0.6:
                plt.ylim(2E-1,10)
            if ps[0] == 0.3:
                plt.ylim(1E-2,1.5)

    #plt.legend()
    save_path = os.path.join('figures', filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print('Saved figure as %s\n' % save_path)
    plt.show()

def gen_data(E_p, ns, qs, trial_numbers, T=5000, algs=['QuARC'], nsd=10, cutoff=10**9):
    for seed in trial_numbers:
        print("Trial", seed)
        for n in ns:
            for q in qs:
                print("n, q =", n, q)
                simulate_Waxman_constant(seed=seed, T=T, n=n, E_p=E_p, q=q, algs=algs, nsd=nsd, cutoff=cutoff)
                print()

def aggregate_fairness_data(n, E_p, algs, nsds=[1,10], q=0.9, E_d=6):
    cutoffs = [1, 10**9]
    width = None
    nqubits = None

    discards_by_distance = {alg : {d : [0,0] for d in range(n)} for alg in algs}
    attempts_by_distance = {alg : {d : [0,0] for d in range(n)} for alg in algs}
    discards_by_cutoff = {alg : {c : [0,0] for c in cutoffs if c < 10**9} for alg in algs}
    latency_by_distance = {alg : {nsd : {d : [] for d in range(n)} for nsd in nsds} for alg in algs}
    for nsd in nsds:
        for i, cutoff in enumerate(cutoffs):
            title = ("n=%d_E_p=%.2f_q=%.2f_E_d=%.1f_nsd=%d" % (n, E_p, q, E_d, nsd) 
                     + ('_w=%d' % width if width else '')
                     + ('_qb=%d' % nqubits if isinstance(nqubits, int) else ('_qb=max' if nqubits == 'max' else ''))
                     + ('_c=%d' % cutoff if cutoff < 10**9 else '')
                    )

            print(title)
            for idx, alg in enumerate(algs):
                folder = os.path.join('results', title, alg)
                latencies = []
                discards = []
                seed = 0
                while os.path.exists(os.path.join(folder, 'stats' + str(seed) + '.pkl')):
                    f = open(os.path.join(folder, 'stats' + str(seed) + '.pkl'), 'rb')
                    stats = pickle.load(f)
                    f.close()

                    # Ignore early requests, so we only look at results for after quarc converged
                    if alg == 'QuARC' and nsd==1:
                        stats.latencies = stats.latencies[:len(stats.latencies)//2]
                        stats.discards = stats.discards[:len(stats.discards)//2]

                    latencies += stats.latencies
                    discards += stats.discards
                    if cutoff == 1 and nsd == 10:
                        net = BlobbedQCastNetwork(n=n, E_p=E_p, q=q, seed=seed)
                        dists = {s : ds for (s, ds) in nx.shortest_path_length(net.G)}
                        for r in stats.discards:
                            # Discards
                            src, dst = (r.src, r.dst) if isinstance(r, Request) else r
                            dist = dists[src][dst]
                            discards_by_distance[alg][dist][0] += 1
                            discards_by_distance[alg][dist][1] += 1
                        for src, dst, _ in stats.latencies:
                            # Successes
                            dist = dists[src][dst]
                            discards_by_distance[alg][dist][1] += 1
                        for src, dst in stats.tried_pairs:
                            dist = dists[src][dst]
                            attempts_by_distance[alg][dist][0] += 1
                        for src, dst in stats.untried_pairs:
                            dist = dists[src][dst]
                            attempts_by_distance[alg][dist][1] += 1
                    if cutoff >= 10**9:
                        net = BlobbedQCastNetwork(n=n, E_p=E_p, q=q, seed=seed)
                        dists = {s : ds for (s, ds) in nx.shortest_path_length(net.G)}
                        for src, dst, l in stats.latencies:
                            # Successes
                            dist = dists[src][dst]
                            latency_by_distance[alg][nsd][dist].append(l+1)

                    seed += 1

                print('  %d' % seed)

                if cutoff < 10**9:
                    discards_by_cutoff[alg][cutoff][0] = len(discards)
                    discards_by_cutoff[alg][cutoff][1] = len(discards) + len(latencies) # Total number of requests

    ideal_rates = {}
    if 1 in nsds:
        nsd = 1
        for i, alg in enumerate(algs):
            d = 1
            xx = []
            yy = []
            errs = []
            print()
            print(alg)
            while len(latency_by_distance[alg][nsd][d]) > 0:
                xx.append(d)
                avg_lat = np.mean(latency_by_distance[alg][nsd][d])
                err = sem(latency_by_distance[alg][nsd][d])
                print(d, len(latency_by_distance[alg][nsd][d]))
                yy.append(1/avg_lat)
                errs.append(err)
                d += 1
            ideal_rates[alg] = yy

    actual_rates = {}
    for i, alg in enumerate(algs):
        d = 1
        xx = []
        yy = []
        print()
        print(alg)
        while discards_by_distance[alg][d][1] >= 200: # Want sample size >= 200
            xx.append(d)
            failures = discards_by_distance[alg][d][0]
            total = discards_by_distance[alg][d][1]
            successes = total-failures
            prop_succeeded = successes/total
            tried = attempts_by_distance[alg][d][0]
            untried = attempts_by_distance[alg][d][1]
            prop_attempted = max(tried/(tried+untried),(total-failures)/total)
            success_rate = prop_succeeded/prop_attempted
            yy.append(success_rate)
            print(d, total)
            d += 1
        if alg == 'QuARC-1':
            yy = ideal_rates[alg][:15] # No difference bw ideal and actual rates, since quarc-1 only picks one request at a time
        actual_rates[alg] = yy
    
    return actual_rates, ideal_rates, range(1,16)

def save_starvation_fig(algs, actual_rates, xx, filename='fig_7a.pdf'):
    for alg in algs:
        plt.plot(xx, actual_rates[alg], label=alg, color=colors[alg], marker='o')

    plt.xlabel("Hop Count")
    plt.ylabel("Proportion of Requests")
    plt.ylim((0,1))
    #plt.legend()    
    save_path = os.path.join('figures', filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print('Saved figure as %s\n' % save_path)
    plt.show()

def save_alloc_bias_fig(algs, actual_rates, ideal_rates, xx, filename='fig_7b.pdf'):
    for i, alg in enumerate(algs):
        if alg == 'QuARC-1':
            continue
        actual = np.array(actual_rates[alg])
        ideal = np.array(ideal_rates[alg])
        dmax = min(len(actual), len(ideal))
        yy = np.minimum(actual[:dmax]/ideal[:dmax], 1)
        plt.plot(xx, yy, label=alg, marker='o')

    plt.xlabel("Hop Count")
    plt.ylabel("Proportion of Maximum Rate")
    plt.ylim((0,1))
    # plt.legend()
    save_path = os.path.join('figures', filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print('Saved figure as %s\n' % save_path)
    plt.show()

    
    
    
    
    
    
if __name__ == "__main__":

    ####################################################
    # Generate all data needed for Fig. 5 (a-d) and plot
    ####################################################

    N = 16
    T = 25000
    seed = 0

    # Fig. 5(a)
    print("Fig. 5(a)...")
    ps = [.7, .6, .8, .5, .9]
    qs = [.8, 1, .7, .9, .7]
    simulate_learning(seed, T, ps, qs, N=N, algs=learning_algs)
    save_learning_fig(ps, qs, 'fig_5a.pdf')

    # Fig. 5(b)
    ps = np.arange(.9, .5, -0.01)
    qs = [1.0]
    simulate_learning(seed, T, ps, qs, N=N, algs=learning_algs)
    save_learning_fig(ps, qs, 'fig_5b.pdf')

    # Fig. 5(c)
    ps = [0.6, 0.9] * 12 + [0.6]
    qs = [1.0]
    simulate_learning(seed, T, ps, qs, N=N, algs=learning_algs2)
    save_learning_fig(ps, qs, 'fig_5c.pdf')

    # Fig. 5(d)
    T = 1500
    path5d = simulate_spatial_learning(seed, T)
    save_spatial_learning_fig(path5d, 'fig_5d.html')


    #########################################################
    # Generate all data needed for Fig. 6 (a-d) and then plot
    #########################################################

    n_trials = 10
    ns = [50, 100, 200, 400, 800]
    qs = [0.6, 0.7, 0.8, 0.9, 1.0]

    # Fig. 6(a)
    print("Fig. 6(a)...")
    gen_data(0.3, ns, [0.9], range(n_trials), algs=algs)
    save_throughput_fig(0.3, ns, [0.9], 'n', 'fig_6a.pdf')

    # Fig. 6(b)
    gen_data(0.6, ns, [0.9], range(n_trials), algs=algs)
    save_throughput_fig(0.6, ns, [0.9], 'n', 'fig_6b.pdf')

    # Fig. 6(c)
    gen_data(0.3, [100], qs, range(n_trials), algs=algs)
    save_throughput_fig(0.3, [100], qs, 'q', 'fig_6c.pdf')

    # Fig. 6(d)
    gen_data(0.6, [100], qs, range(n_trials), algs=algs)
    save_throughput_fig(0.6, [100], qs, 'q', 'fig_6d.pdf')


    #########################################################
    # Generate all data needed for Fig. 7 (a-b) and then plot
    #########################################################

    # Generate data for figure 7
    n_trials = 40
    gen_data(0.5, [200], [0.9], range(n_trials), algs=fairness_algs, nsd=10)
    gen_data(0.5, [200], [0.9], range(n_trials), algs=fairness_algs, nsd=1)
    gen_data(0.5, [200], [0.9], range(n_trials), algs=fairness_algs, nsd=10, cutoff=1)
    actual_rates, ideal_rates, xx = aggregate_fairness_data(200, 0.5, fairness_algs)

    # Fig. 7(a)
    save_starvation_fig(fairness_algs, actual_rates, xx, 'fig_7a.pdf')

    # Fig. 7(b)
    save_alloc_bias_fig(fairness_algs, actual_rates, ideal_rates, xx, 'fig_7b.pdf')