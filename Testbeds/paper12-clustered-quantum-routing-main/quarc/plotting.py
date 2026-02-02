import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import savgol_filter
import math
from pathlib import Path
import os


def simple_plot(rates, xx=None):
    if xx != None:
        plt.plot(xx, rates, '-')
        plt.xlim((0, xx[-1]+xx[0]-1))
    else:
        plt.plot(range(1,len(rates)+1), rates, 'o')
        
    plt.xlabel('Timestep')
    plt.ylabel('Rate (no. of GHZ states/cycle)')
    plt.ylim(bottom=0)
    
    plt.title('Simple Simulation')
    plt.show()

def plot(rates_list, xx, style='-', labels=None, title='', ymax=None, save_path=None, lw=True, legend=True):
    for i, rates in enumerate(rates_list):
        label = labels[i] if labels else None
        if lw:
            _lw = 3 if i == 0 else 1.5
            plt.plot(xx, rates, style, label=label, lw=_lw)
        else:
            plt.plot(xx, rates, style, label=label)
        
    plt.xlim((0, xx[-1]+xx[0]-1))
    plt.xlabel('Time Slot')
    plt.ylabel('Throughput (successful requests/cycle)')
    plt.ylim(bottom=0)
    if ymax:
        plt.ylim(top=ymax)

    if legend: plt.legend()
    plt.title(title)

    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    
def moving_average_success_rate(successes, attempts, window_width):
    cumsum_succ = np.cumsum(np.insert(successes, 0, 0))
    cumsum_attm = np.cumsum(np.insert(attempts, 0, 0))
    ma_succ = (cumsum_succ[window_width:] - cumsum_succ[:-window_width]) / window_width
    ma_attm = (cumsum_attm[window_width:] - cumsum_attm[:-window_width]) / window_width
    ma_rates = ma_succ/ma_attm
    offset_l = window_width//2
    offset_r = (window_width-1)//2
    xx = range(1+offset_l, len(attempts)+1-offset_r)
    simple_plot(ma_rates, xx=xx)

def moving_average_throughput(successes, window_width):
    a = np.ones_like(successes)
    moving_average_success_rate(successes, a, window_width)

def _binned_success_rate(successes, attempts, bin_size):
    successes, attempts = np.array(successes), np.array(attempts)
    N = len(attempts)
    n = N // bin_size
    binned_attm = np.zeros(n)
    binned_succ = np.zeros(n)

    for i in range(n):
        l = i * bin_size
        r = l + bin_size
        if i == n-1:
            r = N
        binned_attm[i] = sum(attempts[l:r]) / (r-l)
        binned_succ[i] = sum(successes[l:r]) / (r-l)
    binned_rates = binned_succ / binned_attm
    return binned_rates
    
def binned_throughput(successes, bin_size, style='-', labels=None, title='', ymax=None, save_path=None, lw=True, legend=True):
    # Successes should be a list of np.arrays, one array per trial
    
    rates_list = []
    for s in successes:
        a = np.ones_like(s)
        rates = _binned_success_rate(s, a, bin_size)
        rates_list.append(rates)
    
    xx = list(np.arange((bin_size+1)/2, len(successes[0]), bin_size))
    plot(rates_list, xx, style=style, labels=labels, title=title, ymax=ymax, save_path=save_path, lw=lw, legend=legend)
        
        
def latency(net, latencies_list, method, labels, bins=None, log=False, **kwargs):
    if method == 'histogram':
        lo = 100
        hi = 0
        all_data = []
        for i, latencies in enumerate(latencies_list):
            data = [l for (_,_,l) in latencies]
            all_data.append(data)
            lo = min(lo, min(data))
            hi = max(hi, max(data))
        if not bins: bins = hi - lo
        plt.hist(all_data, bins=bins, rwidth=0.9, label=labels, log=log, **kwargs)
        plt.title('Latency Distribution')
        plt.xlabel('Latency (Time Slots)')
        plt.ylabel('Proportion of Requests')
        plt.legend()
        plt.show()
    elif method == 'scatter':
        for i, latencies in enumerate(latencies_list):
            dists = {s : ds for (s, ds) in nx.shortest_path_length(net.G, weight=lambda n1, n2, _ : net.edgeLen((n1,n2)))}
            xx = [dists[src][dst] for (src, dst, _) in latencies]
            yy = [l for (_, _, l) in latencies]
            plt.plot(xx, yy, 'o', label=labels[i], **kwargs)
        plt.title('Latency vs Distance') #plt.title('Age of Unfilled Requests')
        plt.xlabel('Shortest Path Distance')
        plt.ylabel('Latency (Time Slots)') # plt.ylabel('Age at End of Simulation')
        if log:
            plt.yscale('log')
        plt.legend()
        plt.show()
    elif method == 'cutoff':
        if 'cutoff' in kwargs:
            cutoff = kwargs['cutoff']
            del kwargs['cutoff']
        else:
            cutoff = 100
        width = 0.25
        for i, latencies in enumerate(latencies_list):
            dists = {s : ds for (s, ds) in nx.shortest_path_length(net.G)}
            data = [dists[src][dst] for (src, dst, l) in latencies if l > cutoff]
            xx = np.array(range(nx.diameter(net.G)+1))
            heights = [data.count(h) for h in xx]
            offset = width*i
            rects = plt.bar(xx+offset, heights, width, label=labels[i])
            #plt.bar_label(rects, padding=3)
        plt.xticks(xx + width, xx)
        plt.title('Starved Requests')
        plt.xlabel('Hop Distance')
        plt.ylabel('Number of Requests Unfilled After %d Time Slots' % cutoff)
        plt.legend()
        plt.show()
    else:
        raise NotImplementedError("Supported methods: 'histogram' and 'scatter'")







