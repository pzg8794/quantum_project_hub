from quarc.blobbed_network import BlobbedGridNetwork
import quarc.protocol as qpc
from quarc.progress_bar import progressBar
# from qcast.qcast_interface import runQCast
import networkx as nx
import random
import numpy as np
import time
import os.path


def simulate_request(net, src, dst, path, stats, t, blob_utilizations, dynamic_fusions=True):
    # Simulates the protcol, servicing request (src, dst) using the blobs in 'path'
    # path: list of blob identifiers
    
    # Assign qubits to channels; i.e., select channels to attempt entanglement generation in order of priority
    start = time.time_ns()
    all_nodes = [n for b in path for n in net.nodes_in_blob(b)]
    singleton_nodes = [net.nodes_in_blob(b)[0] for b in path if net.is_singleton(b)]
    GG = nx.induced_subgraph(net.G, all_nodes)
    all_channels = [ch for e in GG.edges for ch in net.get_channels(e)]
    width = {(min(u,v), max(u,v)) : 0 for u,v in GG.edges}
    def qubits_available(n):
        nq = net.get_nqubits(n)
        b = net.get_blob_id(n)
        if net.is_singleton(b):
            c = net.subsingleton_clusters[b]; assert(c == 1)
            i = blob_utilizations[b]
            return nq//c + (1 if i<(nq%c) else 0)
        else:
            return nq
    qubits_remaining = {n : qubits_available(n) for n in GG.nodes}
    for ch in sorted(all_channels, key=lambda c : c.priority):
        u, v = ch.edge
        if qubits_remaining[u] > 0 and qubits_remaining[v] > 0: 
            # Enough resources to attempt entanglement generation on e
            width[(min(u,v), max(u,v))] += 1
            qubits_remaining[u] -= 1
            qubits_remaining[v] -= 1
    stats.record_timing('route', time.time_ns()-start, t)
    for b in path:
        blob_utilizations[b] += 1
    #print("Qubit assignment: %.2f" % ((time.time_ns()-start)/10**6))
    shortest_path = nx.shortest_path(GG, src, dst)
    
    agg_tp = 0
    direct_success = False
    if dynamic_fusions:
        # Follow the local dynamic fusion protocol
        
        assert(src in GG.nodes and dst in GG.nodes)
        
        # Simulate success/failure of entanglement generation on edge e
        successful_links = {n : [] for n in GG.nodes}
        for e in GG.edges:
            w = width[(min(e[0],e[1]), max(e[0],e[1]))]
            channels = net.get_channels(e)[:w]
            for ch in channels:
                p = net.get_p(ch.edge)
                if random.random() <= p: # Successful entanglement generation
                    u, v = ch.edge
                    if (u, v) in [(src, dst), (dst, src)]:
                        # trivial success (edge case)
                        direct_success = True
                        agg_tp += 1
                        continue
                    successful_links[u].append(ch)
                    successful_links[v].append(ch)

        # Compute fusions (locally on each node)
        fusions = {n : [] for n in GG.nodes}
        for n in GG.nodes:
            if n not in [src, dst]:
                links = successful_links[n].copy()
                fs = []
                while len(links) > 0:
                    curr = []
                    for v in GG.neighbors(n):
                        outgoing_links = sorted([link for link in links if link.other_end_of(n) == v], key=lambda c : c.priority)
                        if len(outgoing_links) > 0:
                            l = outgoing_links[0]
                            curr.append(l)
                            links.remove(l)
                    if len(curr) > 1:
                        fs.append(curr)
                    if len(curr) == 1 and len(fs) > 0:
                        fs[-1] += curr
                fusions[n] = fs

        # Compute the routing graph (different fusions are given their own nodes)
        edgelist = []
        max_n_fusions = 0
        for n in GG.nodes:
            for i, fusion_set in enumerate(fusions[n]):
                u = (n, i)
                max_n_fusions = max(max_n_fusions, i)
                for ch in fusion_set:
                    other_end = ch.other_end_of(n)
                    v = other_end if other_end in [src, dst] else (other_end, i)
                    e = (u, v) # append to edgelist if unique
                    edgelist.append(e)
        H1 = nx.from_edgelist(edgelist)

        # Perform fusions
        failed_nodes = [] # fusions that failed
        for n in H1.nodes:
            if n in [src, dst] or random.random() <= net.get_q(n[0]):
                pass # Successful fusion
            else:
                failed_nodes.append(n)
                #print("Failed fusion:", n)
        H1.remove_nodes_from(failed_nodes)
            
        success = src in H1 and dst in H1 and nx.has_path(H1, src, dst)
        
        if success:
            # Compute aggregate throughput
            H2 = H1.copy()
            #nx.draw(H2, with_labels=True)
            src_links = {"src%d"%i for i in range(len(list(H2.neighbors(src))))}
            dst_links = {"dst%d"%i for i in range(len(list(H2.neighbors(dst))))}
            H2.add_edges_from([(u,"src%d"%i) for i, u in enumerate(H2.neighbors(src))])
            H2.remove_node(src)
            H2.add_edges_from([(u,"dst%d"%i) for i, u in enumerate(H2.neighbors(dst))])
            H2.remove_node(dst)

            components = nx.connected_components(H2)
            for c in components:
                # Aggregate throughput is the number of connected components reaching both the source and destination
                if len(c.intersection(src_links)) > 0 and len(c.intersection(dst_links)) > 0:
                    agg_tp += 1
        
        success = success or direct_success
        
    
    else:
        # Fuse all successful links at every node
        
        # Simulate success/failure of n-fusion at node n
        nodes = []
        for b in path:
            ns = net.nodes_in_blob(b)
            for n in ns:
                if n in [src, dst] or random.random() <= net.get_q(n):
                    # src/dst nodes do not fuse
                    nodes.append(n)
        H0 = nx.induced_subgraph(net.G, nodes)
        assert(src in H0 and dst in H0)

        # Simulate success/failure of entanglement generation on edge e
        edges = []
        for e in H0.edges:
            p = net.get_p(e)
            n_channels = width[(min(e[0],e[1]), max(e[0],e[1]))]
            if random.random() <= 1 - ((1 - p)**n_channels):
                edges.append(e)
        H1 = nx.edge_subgraph(H0, edges)


        success = src in H1 and dst in H1 and nx.has_path(H1, src, dst)
    
    
    # Record whether entanglement was passed between each blob
    SP_multi = {(x, i) for x in shortest_path[1:-1] for i in range(max_n_fusions)}.union({src, dst})
    for i, blob in enumerate(path):
        prev = set([src]) if i == 0 else set(net.nodes_in_blob(path[i-1]))
        curr = set(net.nodes_in_blob(blob))
        next = set([dst]) if i == len(path)-1 else set(net.nodes_in_blob(path[i+1]))
        passed_entanglement = False
        passed_entanglement_sp = False
        dont_record_sp = False
        
        # For shortest path entanglement passing rate calculation
        try:
            prev_sp0 = sorted(list(prev.intersection(set(shortest_path))), key=shortest_path.index)[-1]
            next_sp0 = sorted(list(next.intersection(set(shortest_path))), key=shortest_path.index)[0]
        except:
            # Edge case where the shortest path doesn't go through a blob at all; don't record anythin
            dont_record_sp = True
            prev_sp0 = -1
            next_sp0 = -1
        
        # Node names change when we do dynamic fusions
        if dynamic_fusions:
            if i != 0:
                prev = set([node for node in H1.nodes if (node in prev or (node not in [src, dst] and node[0] in prev))])
            curr = set([node for node in H1.nodes if (node in [src, dst] or node[0] in curr)])
            if i != len(path)-1:
                next = set([node for node in H1.nodes if (node in next or (node not in [src, dst] and node[0] in next))])
        
        
        prev_sp = {x for x in prev if (x == prev_sp0 or (isinstance(x, tuple) and x[0] == prev_sp0))}
        next_sp = {x for x in next if (x == next_sp0 or (isinstance(x, tuple) and x[0] == next_sp0))}
        
        for x in prev_sp:
            for y in next_sp:
                if passed_entanglement_sp:
                    continue
                HSP = nx.induced_subgraph(H1, SP_multi)
                if x in HSP and y in HSP and nx.has_path(HSP, x, y):
                    passed_entanglement_sp = True
        
        if not dont_record_sp:
            path_len_in_blob = shortest_path.index(next_sp0) - shortest_path.index(prev_sp0)
            stats.record_SPEP(blob, passed_entanglement_sp, path_len_in_blob)
        
                
        for x in prev:
            for y in next:
                if passed_entanglement:
                    continue
                H2 = nx.induced_subgraph(H1, prev.union(curr).union(next))

                if x in H2 and y in H2 and nx.has_path(H2, x, y):
                    passed_entanglement = True

        stats.record_entanglement_passing(blob, passed_entanglement)
        
    return success, agg_tp

def simulate_adaptive(sim, protocol, show_progress=True, seed=0, cutoff=10**9):
    # sim : simulation setup (defines the network and controls how it will change over time)
    # protocol : blobbing protocol to run (defines how the network will process requests and blob)
    
    random.seed(seed)
        
    successes = np.zeros(sim.T)
    attempts = np.zeros(sim.T)
    
    requests = []
    net = sim.net
    dynamic_fusions = protocol.dynamic_fusions if hasattr(protocol, 'dynamic_fusions') else True
    
    start = time.time_ns()
    protocol.init_blobs(net)
    stats = protocol.get_new_stats_collector(net)
    #protocol.prioritize_qubits(net, 0)
    stats.record_timing('init', time.time_ns()-start, 1)
    
    sim.process_events(0)
        
    prefix = os.path.join('graphs', net.label(), sim.name)
    if protocol.static:
        net.show_blobs('%s.html' % protocol.label(net=net), path=os.path.join(prefix, 'static'))
    else:
        net.show_blobs('t0.html', path=os.path.join(prefix, protocol.label(net=net)))
            
    for t in range(1, sim.T+1):
        if show_progress and t%50==0: progressBar(t-1, sim.T)
        sim.process_events(t) # Update the network
        
        start = time.time_ns()
        protocol.reblob(net, t, stats) # Update (merge/split) blobs according to protocol 
        stats.record_timing('reconfig', time.time_ns()-start, t)
        
        if not protocol.static and t % protocol.epoch_len == 0:
            net.show_blobs('t%d.html' % t, path=os.path.join(prefix, protocol.label(net=net))) # Display intermediate blobbings
        
        discards = [r for r in requests if t - r.gen_time >= cutoff] # Remove old requests
        stats.discards += discards
        requests = [r for r in requests if r not in discards]
        requests = sim.generate_requests(t, requests) # Generate requests for this timestep
        
        start = time.time_ns()
        routes = protocol.route(net, requests) # Select paths
        protocol.prioritize_qubits(net, t, routes=routes) # Give each link a priority
        stats.record_timing('route', time.time_ns()-start, t)
        
        # Simulate
        untried_pairs = [(r.src, r.dst) for r in requests]
        blob_utilizations = {b : 0 for b in net.blobs} # For tracking usage of sub-singleton clusters
        for r in routes:
            src, dst = r.src, r.dst
            stats.tried_pairs.append((src, dst))
            untried_pairs.remove((src, dst))
            path = routes[r]
            success, agg_successes = simulate_request(net, src, dst, path, stats, t, blob_utilizations, dynamic_fusions)
            if success:
                # Successful entanglement generation!
                requests.remove(r)
                successes[t-1] += 1
                
            start = time.time_ns()
            protocol.record(stats, request=r, outcome=success, path=path, t=t, agg=agg_successes) # Record stats
            stats.record_timing('record', time.time_ns()-start, t)
            
            attempts[t-1] += 1
        stats.untried_pairs += untried_pairs
    
    stats.leftovers = [(r.src, r.dst, r.gen_time) for r in requests]
    while len(stats.successes) < sim.T: stats.successes.append(0)
    while len(stats.aggregate_successes) < sim.T: stats.aggregate_successes.append(0)
    assert(list(successes) == stats.successes)
    if show_progress: progressBar(sim.T, sim.T, done=True)
    return successes, attempts, stats


def simulate_QCast(sim, protocol, show_progress=True, seed=19900111, cutoff=10**9):
    successes = np.zeros(sim.T)
    aggregate_successes = np.zeros(sim.T)
    stats = protocol.get_new_stats_collector(sim.net)
    stats.cutoff = cutoff
    alpha = None
    q = None
    
    events = sim.events
    event_times = sorted(events) + [sim.T]
    # Run Q-CAST separately for each segment where the network stays the same
    for i in range(len(event_times) - 1):
        t = event_times[i]
        end = event_times[i+1]
        if show_progress: progressBar(t, sim.T)
            
        for E in events[t]:
            if E.event_type == 'p':
                assert(isinstance(sim.net, BlobbedGridNetwork) and 
                       sim.net.G.nodes[(1,0)]['x'] - sim.net.G.nodes[(0,0)]['x'] == 1.0) # i.e., assume all links have length 1
            alpha, q = sim.process_QCast_event(E, alpha, q)
        
        duration = end - t
        ##########
        # Optionally reduce run time by reusing results for runs over 'max_trials' time steps
        max_trials = 10**10
        while duration > 0:
            n_trials = min(duration, max_trials)
            # results, (latencies, leftovers, discards, tried_pairs, untried_pairs), timing_data = (
            #     runQCast(sim.net, n_trials, alpha, q, sim.nsd(t), protocol.k, req_gen_method=sim.req_gen_method, 
            #              req_file=sim.requests_file, algorithm=protocol.internal_name, memoize=protocol.memoize, seed=seed, 
            #              cutoff=cutoff, p_pert=sim.p_pert, q_pert=sim.q_pert)
            # )
            raise RuntimeError('Q-CAST code is not included with the QuARC source code for copyright reasons. Please refer to the Q-CAST source code or reach out to Connor Clayton (QuARC author) with any questions.')
            successes[t:t+n_trials] = results[0]
            aggregate_successes[t:t+n_trials] = results[1]
            stats.latencies += latencies
            stats.leftovers = leftovers
            stats.discards += discards
            stats.tried_pairs += tried_pairs
            stats.untried_pairs += untried_pairs
            stats.timing += timing_data
            t += n_trials
            duration = end - t
        assert(duration == 0)
        ##########

    stats.successes += list(successes)
    stats.aggregate_successes += list(aggregate_successes)
    if show_progress: progressBar(sim.T, sim.T, done=True)
    return successes, [], stats



def simulate(sim, protocol, show_progress=True, seed=2224, cutoff=10**9):
    if isinstance(protocol, qpc.BlobbingProtocol):
        return simulate_adaptive(sim, protocol, show_progress, seed=seed, cutoff=cutoff)
    elif isinstance(protocol, qpc.QCastProtocol) or isinstance(protocol, qpc.AlgNFusionProtocol):
        return simulate_QCast(sim, protocol, show_progress, seed=seed, cutoff=cutoff)
    else:
        raise NotImplementedError()


