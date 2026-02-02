from quarc.blobbed_network import BlobbedGridNetwork, BlobbedQCastNetwork
import quarc.simulation
from quarc.thresholding import get_thresholds
import quarc.protocol
from quarc.protocol import QuARC, QuARC_TS_Waxman, QCastProtocol, AlgNFusionProtocol, StaticGridProtocol
from adaptive_sim import simulate

import os
import os.path
import random
import pickle
import numpy as np
import networkx as nx

def gen_requests_file(path, net, seed, request_type, N=70_000):
    random.seed(seed)
    
    if request_type == 'random':
        reqs = quarc.simulation._generate_requests(net, N, 0)
        L = [(r.src,r.dst) for r in reqs]
    elif request_type == 'bimodal':
        diam = nx.diameter(net.G)
        dists = {s : ds for (s, ds) in nx.shortest_path_length(net.G)}
        low_cutoff = round(diam * 0.25)
        high_cutoff = round(diam * 0.75)
        L_low = []
        L_high = []
        while len(L_low) < N//2:
            reqs = quarc.simulation._generate_requests(net, N//2, 0)
            L_low += [(r.src,r.dst) for r in reqs if dists[r.src][r.dst] == low_cutoff]
        while len(L_high) < N//2:
            reqs = quarc.simulation._generate_requests(net, N//2, 0)
            L_high += [(r.src,r.dst) for r in reqs if dists[r.src][r.dst] == high_cutoff]
        L = L_low[:N//2] + L_high[:N//2]
        random.shuffle(L)
    else:
        assert False, "invalid request type: %s" % request_type

    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w')
    f.write(str(L))
    f.close()
    

def simulate_Waxman_constant(seed, T, algs=['QuARC', 'Q-CAST', 'ALG-N-FUSION'], request_type='random',
                             n=100, E_p=0.6, q=0.9, E_d=6, width=None, nqubits=None, nsd=10, cutoff=10**9, 
                             custom_sim=None):
    # Run simulation on Waxman topology with constant p/q
    random.seed(seed)
    

    if nqubits == None:
        qubits = None
    elif isinstance(nqubits, int):
        qubits = lambda x : nqubits
    elif nqubits == 'max':
        qubits = lambda x : x
    
    net = BlobbedQCastNetwork(n=n, E_p=E_p, q=q, cap=width, E_d=E_d, qubits=qubits, seed=seed)

    requests_file = os.path.join('requests', '%s_requests_n%d_%d.txt' % (request_type, len(net.G.nodes), seed))
    if not os.path.exists(requests_file):
        # print("Generating requests file: %s" % requests_file)
        state = random.getstate()
        gen_requests_file(requests_file, net, seed, request_type)
        random.setstate(state)

    if custom_sim:
        sim = custom_sim
        sim.add_requests_file(os.path.abspath(requests_file))
    else:
        sim = quarc.simulation.constant(net, T, p=E_p, nsd=nsd, q=q, E_p=True, requests_file=os.path.abspath(requests_file))
    
    N = len(net.G.nodes)
    merge_th, split_th = get_thresholds(N)
    protocols = {
        'QuARC' : QuARC(500, merge_th, split_th),
        'QuARC-TS': QuARC_TS_Waxman(500, N),
        'QuARC-1' : QuARC(10**6, merge_th, split_th, b0=1),
        'Q-CAST' : QCastProtocol(memoize=False),
        'ALG-N-FUSION' : AlgNFusionProtocol(memoize=False)
    }
    delete = []
    for p in protocols:
        if p not in algs:
            delete.append(p)
    for p in delete:
        del protocols[p]

    title = (((request_type+'-') if request_type != 'random' else '') + "n=%d_E_p=%.2f_q=%.2f_E_d=%.1f_nsd=%d" % (n, E_p, q, E_d, nsd) 
             + ('_w=%d' % width if width else '')
             + ('_qb=%d' % nqubits if isinstance(nqubits, int) else ('_qb=max' if nqubits == 'max' else ''))
             + ('_c=%d' % cutoff if cutoff < 10**9 else '')
            )
    print(sim.name, "(T=%d, n=%d, E_p=%.2f, q=%.2f, E_d=%.1f, nsd=%d)" % (T, n, E_p, q, E_d, nsd))
    if cutoff < 10**9: print("cutoff =", cutoff)
    print(net.label())
    
    topofile = os.path.join('results', title, 'topo%d.net' % seed)
    os.makedirs(os.path.dirname(topofile), exist_ok=True)
    net.to_file(topofile, 0, q)
    quarc.protocol.StaticProtocolGN(1).init_blobs(net)
    net.show_blobs(filename='topo%d.html' % seed, path=os.path.join('results', title))
    
    
    
    for i, alg_name in enumerate(protocols):
        protocol = protocols[alg_name]
        print('%d/%d:' % (i+1,len(protocols)), protocol.label(net=net))
        _, _, stats = simulate(sim, protocol, seed=seed, cutoff=cutoff)
        resultsfile = os.path.join('results', title, alg_name, 'stats%d.pkl' % seed)
        
        os.makedirs(os.path.dirname(resultsfile), exist_ok=True)
        with open(resultsfile, 'wb') as stats_f:
            pickle.dump(stats, stats_f)
    
    
def simulate_learning(seed, T, ps, qs, N=16, nsd=10, algs='all'):
    random.seed(seed)
    net = BlobbedGridNetwork(N=N, cap=1)
    requests_file = os.path.join('requests', 'random_requests_n%d.txt' % len(net.G.nodes))
    if not os.path.exists(requests_file):
        # print("Generating requests file: %s" % requests_file)
        state = random.getstate()
        gen_requests_file(requests_file, net, seed, 'random')
        random.setstate(state)
    
    sim = quarc.simulation.custom(net, T, ps=ps, qs=qs, requests_file=requests_file)

    merge_th, split_th = get_thresholds(N*N)
    if algs == 'all':
        protocols = [QuARC(500, merge_th, split_th)] + [StaticGridProtocol(2**(2*b)) for b in range(round(np.log2(N))+1)]
    else:
        protocols = []
        if 'QuARC' in algs:
            protocols.append(QuARC(500, merge_th, split_th))
        for x in algs:
            if isinstance(x, int):
                protocols.append(StaticGridProtocol(x))
            if x in ['1-by-1 Clusters', '2-by-2 Clusters', '4-by-4 Clusters', '8-by-8 Clusters', '16-by-16 Cluster']:
                l = int(x[:x.find('-')])
                protocols.append(StaticGridProtocol(N**2/l**2))

    title = ("learning-N=%d_ps=" % N) + str(ps) + "_qs=" + str(qs)
    
    for i, protocol in enumerate(protocols):
        print('%d/%d:' % (i+1,len(protocols)), protocol.label(net=net))
        _, _, stats = simulate(sim, protocol, seed=seed)
        resultsfile = os.path.join('results', title, protocol.label(net=net), 'stats%d.pkl' % seed)
        
        os.makedirs(os.path.dirname(resultsfile), exist_ok=True)
        with open(resultsfile, 'wb') as stats_f:
            pickle.dump(stats, stats_f)


def simulate_spatial_learning(seed, T, n=200, p_high=0.6, p_low=0.3, nsd=10, E_d=6, width=None):
    net = BlobbedQCastNetwork(n=n, E_p=0.6, q=0.9, cap=width, E_d=E_d, seed=seed)
    sim = quarc.simulation.half_and_half(net, T, p_high=p_high, p_low=p_low)
    simulate_Waxman_constant(seed, T, algs=['QuARC'], n=n, nsd=nsd, custom_sim=sim)
    merge_th, split_th = get_thresholds(n)
    protocol = QuARC(500, merge_th, split_th)
    save_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                             'graphs', net.label(), sim.name, protocol.label(net=net), 't%d.html' % T)
    return save_file
            
            
            
def simulate_perturbation(seed, T, N, p_pert, q_pert, algs=['QuARC', 'Q-CAST', 'ALG-N-FUSION'],
                          E_p=0.6, q=0.9, width=3, nqubits='max2', nsd=10):
    random.seed(seed)
    
    if isinstance(nqubits, int):
        qubits = lambda x : nqubits
    elif nqubits == 'max':
        qubits = lambda x : x
    elif nqubits == 'max2':
        qubits = lambda x : x // 2
    
    net = BlobbedGridNetwork(N=N, cap=width, qubits=qubits)
    
    requests_file = os.path.join('requests', 'random_requests_n%d_%d.txt' % (len(net.G.nodes), seed))
    if not os.path.exists(requests_file):
        state = random.getstate()
        gen_requests_file(requests_file, net, seed)
        random.setstate(state)
        
    sim = quarc.simulation.constant_perturbation(net, T, E_p, q, p_pert, q_pert, requests_file=os.path.abspath(requests_file))

    merge_th, split_th = get_thresholds(len(net.G.nodes))
    protocols = {
        'QuARC' : QuARC(500, merge_th, split_th),
        'Q-CAST' : QCastProtocol(memoize=False),
        'ALG-N-FUSION' : AlgNFusionProtocol(memoize=False)
    }
    delete = []
    for p in protocols:
        if p not in algs:
            delete.append(p)
    for p in delete:
        del protocols[p]

    title = ("perturbation-N=%d_E_p=%.2f_q=%.2f_pp=%.2f_qp=%.2f_nsd=%d_w=%d" % (N, E_p, q, p_pert, q_pert, nsd, width)
             + ('_qb=%d' % nqubits if isinstance(nqubits, int) else ('_qb=%s' % nqubits))
            )
    
    for i, alg_name in enumerate(protocols):
        protocol = protocols[alg_name]
        print('%d/%d:' % (i+1,len(protocols)), protocol.label(net=net))
        _, _, stats = simulate(sim, protocol, seed=seed)
        resultsfile = os.path.join('results', title, protocol.label(net=net), 'stats%d.pkl' % seed)
        
        os.makedirs(os.path.dirname(resultsfile), exist_ok=True)
        with open(resultsfile, 'wb') as stats_f:
            pickle.dump(stats, stats_f)
