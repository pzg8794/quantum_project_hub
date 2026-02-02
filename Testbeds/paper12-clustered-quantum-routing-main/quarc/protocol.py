from quarc.stats import BlobStatsCollector, QCastStatsCollector
from quarc.thresholding import get_thresholds_TS
import numpy as np

# Class to support a generic blobbing protocol
class BlobbingProtocol():
    def __init__(self, route, reblob, record):
        # route : function ( (network, list requests) -> list list blobs) that decides which blobs should fuse
        # reblob : function that decides how to blob the network
        # record : function that records stats on entanglement generation success/failures
        
        assert(callable(route) and callable(reblob))
    
    def init_blobs(self, net):
        pass
    
    def get_new_stats_collector(self, net):
        return BlobStatsCollector(net) # Replace this with other StatsCollector sub-classes for different protocols as needed
    
    def route(self, net, requests):
        pass
    
    def reblob(self, net, t, stats):
        pass
    
    def prioritize_qubits(self, net, t):
        pass
    
    def record(self, stats, **kwargs):
        outcome = kwargs['outcome']
        path = kwargs['path']
        r = kwargs['request']
        t = kwargs['t']
        agg = kwargs['agg']
        assert(agg >= 1 if outcome else agg == 0)
        stats.record_attempt(r, path, t)
        if outcome:
            stats.record_success(r, path, t, agg)

    
####################################################
################ Routing Functions #################
####################################################

import networkx as nx

# Greedy routing protocol that chooses a shortest path for each request and tries to fill it
def greedy_route(net, requests, f=None):
    routes = {}
    remaining_assignments = net.subsingleton_clusters.copy()
    for r in requests:
        reserved_blobs = [b for b in net.blobs if remaining_assignments[b] <= 0]
        H0 = nx.restricted_view(net.H, reserved_blobs, [])
        bsrc = net.get_blob_id(r.src)
        bdst = net.get_blob_id(r.dst)
        try:
            p = nx.shortest_path(H0, source=bsrc, target=bdst, weight=f)
            routes[r] = p
            for b in p:
                remaining_assignments[b] -= 1
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            pass

    return routes

def wide_greedy_route(net, requests):
    def f(u, v, attr):
        n_links = attr['weight']
        nnodes_u = net.H.nodes[u]['nnodes']
        nnodes_v = net.H.nodes[v]['nnodes']
        return nnodes_v/n_links

    return greedy_route(net, requests, f)

def sorted_route(net, requests):
    def dist2(u, v):
        return (net.x(u) - net.x(v))**2 + (net.y(u) - net.y(v))**2
    rs = sorted(requests, key=lambda r : dist2(r[0],r[1]))
    return wide_greedy_route(net, rs)


####################################################
########### Qubit Assignment Protocols #############
####################################################

# Qubit assignment decides which links will attempt
# entanglement generation. Effectively, we must
# prioritize links.

import random

# Randomly select edges until saturated
def qubit_assignment_random(net, t, **kwargs):
    if t == 0:
        for e in net.G.edges:
            for c in net.get_channels(e):
                c.priority = random.random()

def qubit_assignment_random_disbursed(net, t, **kwargs):
    #if t == 0:
    for e in net.G.edges:
        for i, c in enumerate(net.get_channels(e)):
            c.priority = random.random() + i

def qubit_assignment_distance_heuristic(net, t, **kwargs):
    routes = kwargs['routes']
    multiplier = kwargs['multiplier']
    
    
    # Initialize all priorities to 0
    for e in net.G.edges:
        for i, c in enumerate(net.get_channels(e)):
            c.priority = i + random.random()/100
    
    for r in routes:
        src, dst = r.src, r.dst
        path = routes[r]
        all_nodes = set(net.nodes_in_blobs(path))
        for i, blob in enumerate(path):
            # Edge case for src and dst blobs
            first_blob = (i == 0)
            last_blob = (i == len(path)-1)
            prev = path[i-1] if not first_blob else src
            next = path[i+1] if not last_blob else dst
            for n in net.nodes_in_blob(blob):
                dp = net.distance_to_blob(n, prev, single_node=first_blob)
                dn = net.distance_to_blob(n, next, single_node=last_blob)
                for x in net.G.neighbors(n):
                    if x in all_nodes: # don't bother with edges outside the path
                        dp2 = net.distance_to_blob(x, prev, single_node=first_blob)
                        dn2 = net.distance_to_blob(x, next, single_node=last_blob)
                        # How much closer would x bring us to the previous or next blob?
                        delta_p = dp2-dp
                        delta_n = dn2-dn
                        score = min(delta_p, delta_n) # {-1, 0, 1}
                        for c in net.get_channels((n,x)):
                            # prioritize edges which reduce distance to the next blob or to the previous blob
                            c.priority += multiplier*score
                            #c.priority += random.random() + multiplier*score
                        
                        
    




####################################################
########### Merging/Splitting Protocols ############
####################################################


import itertools

# Split 'blob' into 's' new blobs
def split_Girvan_Newman(blob, net, s=4, nodes=[]):
    if not nodes:
        nodes = net.nodes_in_blob(blob)
    subG = nx.subgraph(net.G, nodes)
    new_blobs = []

    if len(nodes) < s:
        # Make each node its own blob
        for n in nodes:
            new_blobs.append([n])
        return new_blobs

    else:
        # Partition nodes using the Girvan-Newman algorithm
        for partition in nx.community.girvan_newman(subG):
            if len(partition) >= s:
                for S in partition:
                    new_blobs.append(list(S))
                return new_blobs



def merge_blobs_no_split(B, *args):
    b0 = min(args)
    new_blob = []
    for b in args:
        new_blob += B[b]
        if b != b0:
            del B[b]
    B[b0] = new_blob

# Update B to reflect the following process:
# 1) merge the blobs in 'args' into one blob
# 2) split this new blob into 'n' blobs
def merge_blobs(B, net, *args, n=1):
    assert n >= 1 and n < len(args) # Second part here required for naming purposes
    
    if n == 1:
        return merge_blobs_no_split(B, *args)
    else:        
        nodes = []
        blob_names = args
        for b in blob_names:
            ns = B[b]
            nodes += ns
        new_blobs = split_Girvan_Newman(None, net, s=n, nodes=nodes)
        assert(len(new_blobs) == n)
                

        # Select names for new blobs (try to make them conisitent with the old blobs)
        labels = {}
        it = 0
        while len(labels) < n:
            it += 1
            labels = {}
            for i, new_nodes in enumerate(new_blobs):
                rep = random.choice(new_nodes) # Random node in new blob
                b = net.get_blob_id(rep)
                labels[b] = i
            if it > 0 and it % 10 == 0:
                print("Inefficient name selection when merging blobs", args, "-- choosing names arbitrarily")
                labels = {}
                for i, name in enumerate(blob_names[:n]):
                    labels[name] = i
                
        assert(len(labels) == n)
        
        # Update B with new blobs
        for b in blob_names:
            if b in labels:
                # Blob id is being reused
                i = labels[b]
                B[b] = new_blobs[i]
            else:
                # Blob id should be discarded
                del B[b]





def _valid_neighbors(blob, net, exclude):
    neighbors = list(net.H.neighbors(blob))
    for n in exclude:
        if n in neighbors:
            neighbors.remove(n)
    return neighbors

def _all_neighbors(blob, net):
    return list(net.H.neighbors(blob))

def _min_neighbors(blob, neighbors, f, m):
    # f : list blob -> real -- function on neighboring blobs to minimize
    # Returns the 'm' neighbors of 'blob' that minimize f
    
    
    #print("Neighbors of %d:" % blob, neighbors)
    
    lo = 10**10
    bs = None
    subsets = list(itertools.combinations(neighbors, m))
    for S in subsets:
        S = list(S)
        val = f(S)
        #print("\t", S, val)
        if val < lo:
            lo = val
            bs = S

    #print("Top %d neighbors of %d:" % (m, blob), bs)
    return bs

def _max_neighbors(blob, neighbors, f, m):
    # f : blob -> real -- function on neighboring blobs to maximize
    # Returns the 'm' neighbors of 'blob' that maximize f
    
    g = lambda S : -f(S)
    return _min_neighbor(blob, neighbors, g, m)

# Randomly select a neighbor to merge with
def select_random(blob, net, neighbors, m):
    neighbors = neighbors.copy()
    L = []
    while len(L) < m:
        L.append(random.choice(neighbors))
        neighbors.remove(L[-1])

    return L

# Select the largest neighbor to merge with
def select_largest(blob, net, neighbors, m):
    def f(S):
        s = 0
        for n in S:
            s += net.H.nodes[n]['nnodes']
        return s
    return _max_neighbors(blob, neighbors, f, m)

# Select the smallest neighbor to merge with
def select_smallest(blob, net, neighbors, m):
    def f(S):
        s = 0
        for n in S:
            s += net.H.nodes[n]['nnodes']
        return s
    return _min_neighbors(blob, neighbors, f, m)

# Select the neighbor that induces the smallest diameter
def select_min_diam(blob, net, neighbors, m):
    def f(S):
        nbunch = net.nodes_in_blobs([blob] + S)
        new_blob = nx.induced_subgraph(net.G, nbunch)
        diam = nx.diameter(new_blob)
        return diam
    
    return _min_neighbors(blob, neighbors, f, m)

# Select the neighbor that shares the most links
def select_max_links(blob, net, neighbors, m):
    def f(S):
        w = 0
        for n in S:
            w += net.H.edges[(blob, n)]['weight']
        return w
    return _max_neighbors(blob, neighbors, f, m)

# Select the neighbor that induces the lowest kemeny constant (a measure of closeness)
def select_min_kemeny(blob, net, neighbors, m):
    def f(S):
        nbunch = net.nodes_in_blobs([blob] + S)
        new_blob = nx.induced_subgraph(net.G, nbunch)
        kc = nx.kemeny_constant(new_blob)
        return kc
    return _min_neighbors(blob, neighbors, f, m)




# Select 'merge_number' of blob's neighbors to merge with, but not any of the blobs in 'exclude'
def merge_select(merge_number, merge_criterion, blob, net, exclude=[]):
    if merge_criterion == 'random':
        sel = select_random
    elif merge_criterion == 'largest':
        sel = select_largest
    elif merge_criterion == 'smallest':
        sel = select_smallest
    elif merge_criterion == 'min_diam':
        sel = select_min_diam
    elif merge_criterion == 'max_links':
        sel = select_max_links
    elif merge_criterion == 'kemeny_constant':
        sel = select_min_kemeny
    else:
        raise NotImplementedError("Merge criterion '%s' is not supported" % str(merge_criterion))

    n_neighbors = len([x for x in net.H.neighbors(blob)])
    if n_neighbors < merge_number:
        # Edge case where there are too few blobs to merge using merge_number
        merge_number = n_neighbors
        
    #neighbors = _valid_neighbors(blob, net, exclude)
    neighbors = _all_neighbors(blob, net)
    best_neighbors = sel(blob, net, neighbors, merge_number)
    
    if best_neighbors == None:
        return None
    
    for n in best_neighbors:
        if n in exclude:
            return None
    
    return best_neighbors


# Possible criteria: 
#   X highest centrality after merging
#   X highest connectivity after merging
#   - lowest radius after merging
#   - highest kemeny_constant after merging
#   - highest success score
#   - lowest success score
#   - communicability?



def p_merge(r, m, s, exp):
    # r : success prob
    # m : merge threshold
    # s : split threshold
    # exp : exponent of probability decay (exp < 0 indicates 0 probability between m and s)
    
    mid = s
    #mid = (s+m)/2
    if r > mid:
        return 0
    if r <= m:
        return 1
    
    if exp < 0:
        return 0
    if exp == 0:
        raise NotImplementedError("exp = 0 not implmented")
    else:
        return (1/(mid-m))**exp * (abs(r - mid))**exp

def p_split(r, m, s, exp):
    mid = (s+m)/2
    if r < mid:
        return 0
    if r > s:
        return 1
    
    if exp < 0:
        return 0
    if exp == 0:
        raise NotImplementedError("exp = 0 not implmented")
    else:
        return (2/(s-m))**exp * (abs(r - mid))**exp


####################################################
#################### Protocols #####################
####################################################

class StaticGridProtocol(BlobbingProtocol):
    def __init__(self, b0, dynamic_fusions=True):
        self.static = True
        self.dynamic_fusions = dynamic_fusions
        self.b0 = b0
    
    def init_blobs(self, net):
        # Reset the network to have b0 square blobs
        N = net.N
        d = net.d
        b0 = self.b0
        
        assert round(b0**(1/d))**d == b0
        assert (N**d) % b0 == 0, 'Grid size (%d) must be multiple of number of blobs (%d).' % (N**d, b0)
        
        B = self._grid_blobbing(net.G, N, d)
        net.reset(B)

    def _grid_blobbing(self, G, N, d):
        B = {}
        b = N//round(self.b0**0.5) # side length of each square blob
        for n in G.nodes:
            blob_id = self._get_blob_id(n, N, d, b)
            if blob_id not in B:
                B[blob_id] = []
            B[blob_id].append(n)
        self._shuffle_blob_labels(B)
        return B
    
    def _shuffle_blob_labels(self, B):
        # Randomly reorder blob labels in B; fixes a bug where deterministic blob labels bias the routing algorithm
        labels = list(B.keys())
        random.shuffle(labels)
        for i in range(len(labels)//2):
            l1 = labels[2*i]
            l2 = labels[2*i+1]
            tmp = B[l1].copy()
            B[l1] = B[l2].copy()
            B[l2] = tmp
        
    def _get_blob_id(self, n, N, d, b):
        # Create hypercubic blobs with side length b
        blob_id = 0
        for i in range(d):
            blob_id += (N//b)**i * (n[i]//b)
        return blob_id
    
    def route(self, net, requests):
        return wide_greedy_route(net, requests)
    
    def prioritize_qubits(self, net, t, **kwargs):
        qubit_assignment_random_disbursed(net, t, multiplier=0.5, **kwargs)
    
    def label(self, **kwargs):
        N = kwargs['net'].N
        b = round(N/self.b0**0.5)
        if self.b0 == 1:
            return '%d-by-%d Cluster' % (N, N)
        return '%d-by-%d Clusters' % (b, b)


# Static protocol which maintains a constant b blobs partitioned by the Girvan-Newman algorithm
class StaticProtocolGN(BlobbingProtocol):
    def __init__(self, b0, dynamic_fusions=True):
        # b0 : number of blobs
        self.static = True
        self.dynamic_fusions = dynamic_fusions
        b0 = max(1, b0)
        self.b0 = b0

    # Initialize blobbing at t = 0 with b0 blobs
    def init_blobs(self, net):
        nodes = list(net.G.nodes)
        random.shuffle(nodes)
        new_blobs = []
        if len(nodes) <= self.b0:
            # Make each node its own blob
            for n in nodes:
                new_blobs.append([n])
        elif self.b0 == 1:
            new_blobs.append(nodes)
        else:
            # Partition nodes using the Girvan-Newman algorithm
            for partition in nx.community.girvan_newman(net.G):
                if len(partition) >= self.b0:
                    for S in partition:
                        new_blobs.append(list(S))
                    break

        B = {}
        for i, S in enumerate(new_blobs):
            B[i] = S

        net.reset(B)

    def route(self, net, requests):
        return wide_greedy_route(net, requests)
    
    def prioritize_qubits(self, net, t, **kwargs):
        qubit_assignment_random_disbursed(net, t, multiplier=0.5, **kwargs)
    
    def label(self, **kwargs):
        if self.b0 == 1:
            return '1 Cluster'
        return '%d Clusters' % self.b0
    
    

class AdaptiveGridProtocol(BlobbingProtocol):
    def __init__(self, epoch_len, merge_th, split_th, merge_criterion='kemeny_constant', 
                 b0=1, merge_number=2, exp_merge=-1, exp_split=-1, dynamic_fusions=True, ent_pass_th=True):
        # epoch_len : number of timesteps in between each reblob
        # merge_th : success probability below which a blob will merge
        # split_th : success probability above which a blob will split 
        # b0 : initial number of blobs
        
        self.static = False
        self.dynamic_fusions = dynamic_fusions
        self.ent_pass_th = ent_pass_th
        
        # Hyperparameters
        self.split_count = 4
        self.super_merge_factor = 0.75
        
        self.b0 = b0
        self.epoch_len = epoch_len
        if isinstance(merge_th, float):
            self.merge_th = lambda n : merge_th
        else:
            self.merge_th = merge_th
        if isinstance(split_th, float):
            self.split_th = lambda n : split_th
        else:
            self.split_th = split_th
        
        self.supported_merge_criteria = ['random', 'largest', 'smallest', 'min_diam', 'max_links', 
                                        'kemeny_constant', 'kemeny_constant2', 'kemeny_constant3', 'kemeny_constant4']
        self.merge_number = merge_number
        self.merge_criterion = merge_criterion
        self.exp_split = exp_split
        self.exp_merge = exp_merge
        
        #assert merge_th < split_th, "Merging threshold (%f) must be less than splitting threshold (%f)." % (merge_th, split_th)
        assert merge_criterion in self.supported_merge_criteria, "Unsupported merge criterion '%s'" % merge_criterion
    
    def init_blobs(self, net):
        if self.b0 == len(net.G.nodes):
            B = {i : [n] for i, n in enumerate(list(net.G.nodes))}
            net.reset(B)
        elif self.b0 == 1:
            B = {0 : list(net.G.nodes)}
            net.reset(B)
        else:
            raise NotImplementedError("Adaptive protocol only supports b0=1 or b0=n for now")

    def route(self, net, requests):
        return wide_greedy_route(net, requests)
    
    def prioritize_qubits(self, net, t, **kwargs):
        qubit_assignment_random(net, t)
    
    def reblob(self, net, t, stats):
        if t > 0 and t % self.epoch_len == 0:
            # Reblob every self.epoch_len timesteps
            to_split = []
            to_merge = []
            to_super_merge = [] # EXPERIMENTAL
            #singletons = {x for x in net.G.nodes if net.get_avg_effective_p(x) > 0.97}
            ps = [] # DELETE
            stats.blob_counts.append(len(net.blobs))
            for blob in net.blobs:
                p = stats.get_entanglement_passing_rate(blob) # stats.get_success_rate(blob)
                n = net.blob_size(blob)
                ps.append((blob, p)) # DELETE
                #if p > self.split_th(n):
                #    to_split.append(blob)
                #elif p < self.merge_th(n):
                #    to_merge.append(blob)
                
                # Probabilitically split and merge
                #avg_channel_width = net.blob_average_capacity(blob)
                #split_merge_arg = n * avg_channel_width if self.experimental_th else n
                mth = self.merge_th(n)
                sth = self.split_th(n)
                #p_s = p_split(p, mth, sth, self.exp_split)
                #p_m = p_merge(p, mth, sth, self.exp_merge)
                
                #split_cutoff = 0.9
                #merge_cutoff = 0.95
                #singleton_cutoff = 0.6
                #qubit_deficit = min(1, net.get_nqubits_blob(blob)/net.blob_total_channels(blob))
                #deficit_factor = min(qubit_deficit, 1 - (1-qubit_deficit)/n)
                #deficit_factor = 1 #qubit_deficit # if n <= 5 else 1
                
                #if len(set(net.nodes_in_blob(blob)).intersection(singletons)) > 0:
                #    continue
                
                rate = stats.get_entanglement_passing_rate(blob) if self.ent_pass_th else stats.get_success_rate(blob)

                #net_diam = nx.diameter(net.G)
                #blob_diam = max(nx.diameter(nx.induced_subgraph(net.G, net.nodes_in_blob(blob))), 1)
                #stats.get_SPEP_rate(blob)**((net_diam/blob_diam)/3) > .5
                neighbor_blobs = list(net.H.neighbors(blob))
                my_size = len(net.nodes_in_blob(blob))
                k = self.split_count
                sizes = [len(net.nodes_in_blob(b)) for b in neighbor_blobs if (len(net.nodes_in_blob(b)) <= max(my_size/k,1))]
                neighbor_ratio = len(sizes)/len(neighbor_blobs) if len(neighbor_blobs) > 0 else 0 # Love thy neighbor
                
                if self.split_th(n) <= rate or neighbor_ratio >= .5:
                    to_split.append(blob)
                elif rate <= self.merge_th(n):
                    to_merge.append(blob)
                
                '''
                #if (self.split_th(n) <= (stats.get_success_rate(blob)+stats.get_entanglement_passing_rate(blob))/2):
                if (self.split_th(n) <= stats.get_entanglement_passing_rate(blob)):
                    to_split.append(blob)
                #elif stats.get_success_rate(blob) <= self.merge_th(n) and stats.get_entanglement_passing_rate(blob) < merge_cutoff and (n > 1 or stats.get_entanglement_passing_rate(blob) < singleton_cutoff):
                elif stats.get_entanglement_passing_rate(blob) <= self.merge_th(n):
                    to_merge.append(blob)
                #elif (self.split_th(n)-stats.get_entanglement_passing_rate(blob) <= 0.1 and random.random() <= 0.5):
                #    to_split.append(blob)
                '''
                
                    # EXPERIMENTAL
                    # When many blobs are all performing very poorly, merge them all together
                    #if p < self.super_merge_factor * mth:
                    #    to_super_merge.append(blob)
            
            ####################
            # for debugging
            '''
            print('t =', t, '(%d blobs)' % len(net.blobs))
            for (b, p) in sorted(ps, key=lambda t : t[1]):
                n = net.blob_size(b)
                m = net.blob_edge_count(b)
                qubit_deficit = min(1, net.get_nqubits_blob(b)/net.blob_total_channels(b))
                deficit_factor = 1 - (1-qubit_deficit)/n
                avg_eff_p = np.mean([net.get_avg_effective_p(x) for x in net.nodes_in_blob(b)])
                mth = self.merge_th(n)
                sth = self.split_th(n)
                net_diam = nx.diameter(net.G)
                blob_diam = max(nx.diameter(nx.induced_subgraph(net.G, net.nodes_in_blob(b))), 1)
                print(b, "%.2f, %.2f, %.2f (%.2f, %.2f) %s (n=%d, succ=%d/%d, spl=%.2f)" % (stats.get_success_rate(b),
                                                                                       stats.get_entanglement_passing_rate(b),
                                                              stats.get_SPEP_rate(b),
                                                                          mth, sth,
                                                             "split" if b in to_split else "merge" if b in to_merge else "     ",
                                                                  n, stats.SPEP_stats[b]['successes'], stats.SPEP_stats[b]['attempts'],
                                                                                       1/stats.get_avg_shortest_path_len(b)))
                                                                   #p_merge(p, mth, sth, self.exp_merge), 
                                                                   #p_split(p, mth, sth, self.exp_split),
                                                                   #"SUPER_MERGE" if b in to_super_merge else "           "))
            print()
            '''
            ####################
                    
            # Perform all splits first
            #B = net._blob_to_nodes.copy()
            #while len(to_split) > 0:
            B = net._blob_to_nodes.copy()
            new_ids = net.available_blob_ids(len(to_split) * (self.split_count-1))
            for blob in to_split:
                new_blobs = split_Girvan_Newman(blob, net, s=self.split_count)
                # Give new blobs new labels, update B
                for i, S in enumerate(new_blobs):
                    if i == 0:
                        # Reuse old blob id
                        B[blob] = S
                    else:
                        B[new_ids.pop(0)] = S
            net._update_blobs(B) # Update the network with split blobs
                #to_split = [b for b in net.blobs if net.blob_size(b) in (2,3)]
            
            # Perform any super-merges (name still in beta)
            already_merged = []
            H0 = nx.induced_subgraph(net.H, to_super_merge)
            components = nx.connected_components(H0)
            for c in components:
                if len(c) > self.merge_number:
                    merge_blobs(B, net, *c, n=1)
                    already_merged += c
            
            # Perform merges
            to_merge.sort(key=stats.get_success_rate) # stats.get_entanglement_passing_rate
            for b1 in to_merge:
                if b1 not in already_merged: # Don't merge any blobs more than once in an epoch 
                    bs = merge_select(self.merge_number, self.merge_criterion, b1, net, exclude=already_merged)

                    if bs:
                        #print("Merging", [b1] + bs)
                        merge_blobs(B, net, b1, *bs, n=len(bs))
                        already_merged += [b1] + bs 
                    #print()
            net._update_blobs(B) # Update the network with merged blobs
            
            # Reset stats for next epoch
            stats.reset_stats(net)
            
    
    def label(self, **kwargs):
        return 'Adaptive'
    
    
class AdaptiveGridSorted(AdaptiveGridProtocol):
    def route(self, net, requests):
        return sorted_route(net, requests)

    def label(self, **kwargs):
        return 'Adaptive (Sorted)'
    
    
class simpleQuARC(AdaptiveGridProtocol):
    def label(self, **kwargs):
        df = ' (static fusions)' if not self.dynamic_fusions else ''
        return 'simpleQuARC' + df

    def prioritize_qubits(self, net, t, **kwargs):
        qubit_assignment_random_disbursed(net, t)

class QuARC(AdaptiveGridProtocol):
    def __init__(self, epoch_len, merge_th, split_th, merge_criterion='kemeny_constant', 
                 b0=1, merge_number=2, exp_merge=-1, exp_split=-1, dynamic_fusions=True, multiplier=0.5):
        self.multiplier = multiplier
        super().__init__(epoch_len, merge_th, split_th, merge_criterion, 
                       b0, merge_number, exp_merge, exp_split, dynamic_fusions, ent_pass_th=True)
        
    def label(self, **kwargs):
        df = ' (static fusions)' if not self.dynamic_fusions else ''
        return 'QuARC' + df 

    def prioritize_qubits(self, net, t, **kwargs):
        qubit_assignment_random_disbursed(net, t, multiplier=self.multiplier, **kwargs)
        
class QuARC_TS_Waxman(QuARC):
    def __init__(self, epoch_len, N):
        merge_th_TS, split_th_TS = get_thresholds_TS(N)
        super().__init__(epoch_len, merge_th_TS, split_th_TS)
    
    def label(self, **kwargs):
        return 'QuARC-TS'

        
class QuARC_success_rate(AdaptiveGridProtocol):
    def __init__(self, epoch_len, merge_th, split_th, merge_criterion='kemeny_constant', 
                 b0=1, merge_number=2, exp_merge=-1, exp_split=-1, dynamic_fusions=True, multiplier=0.5):
        self.multiplier = multiplier
        super().__init__(epoch_len, merge_th, split_th, merge_criterion, 
                       b0, merge_number, exp_merge, exp_split, dynamic_fusions, ent_pass_th=False)
        
    def prioritize_qubits(self, net, t, **kwargs):
        qubit_assignment_distance_heuristic(net, t, multiplier=self.multiplier, **kwargs)
    
    def label(self, **kwargs):
        df = ' (static fusions)' if not self.dynamic_fusions else ''
        return 'QuARC (success rate)' + df 
    

    
    
class QCastProtocol():
    def __init__(self, k=3, succ_pairs_only=True, reuse_requests=True, allow_recovery_paths=True, memoize=True):
        # k : k-hop distance for local link-state sharing
        # succ_pairs_only : if true, limit throughput calculation to only the number of successful pairs
        # reuse_requests : if false, use the original Q-CAST request generation method 
        #                  (throw away unfilled requests after each timestep)

        self.k = k
        self.succ_pairs_only = succ_pairs_only
        self.reuse_requests = reuse_requests
        self.memoize = memoize
        self.internal_name = "Q-CAST" if allow_recovery_paths else "Q-CAST-R"
    
    def get_new_stats_collector(self, net):
        return QCastStatsCollector(net)
        
    def label(self, **kwargs):
        limiting = '' if self.succ_pairs_only else '(# EPR Pairs)'
        k = ' (k = %d)' % self.k if self.k != 3 else ''
        req_gen = ' (Discarding Unfilled Requests)' if not self.reuse_requests else ''
        return self.internal_name + limiting + k + req_gen
    
class QPassProtocol(QCastProtocol):
    def __init__(self, succ_pairs_only=True, reuse_requests=True, allow_recovery_paths=True, memoize=True):
        # k : k-hop distance for local link-state sharing
        # succ_pairs_only : if true, limit throughput calculation to only the number of successful pairs
        # reuse_requests : if false, use the original Q-CAST request generation method 
        #                  (throw away unfilled requests after each timestep)

        self.k = -1 # For compatibility
        self.succ_pairs_only = succ_pairs_only
        self.reuse_requests = reuse_requests
        self.memoize = memoize
        self.internal_name = "Q-PASS" if allow_recovery_paths else "Q-PASS-R"
        
    def label(self, **kwargs):
        limiting = '' if self.succ_pairs_only else '(# EPR Pairs)'
        req_gen = ' (Discarding Unfilled Requests)' if not self.reuse_requests else ''
        return self.internal_name + limiting + req_gen

class AlgNFusionProtocol():
    def __init__(self, succ_pairs_only=True, reuse_requests=True, allow_recovery_paths=True, memoize=True):
        # succ_pairs_only : if true, limit throughput calculation to only the number of successful pairs
        # reuse_requests : if false, use the original Q-CAST request generation method 
        #                  (throw away unfilled requests after each timestep)

        if succ_pairs_only == False:
            raise NotImplementedError()
            
        self.k = -1 # For compatibility
        self.succ_pairs_only = succ_pairs_only
        self.reuse_requests = reuse_requests
        self.memoize = memoize
        self.internal_name = "AlgNFusion" if allow_recovery_paths else "AlgNFusion-R"
    
    def get_new_stats_collector(self, net):
        return QCastStatsCollector(net)
        
    def label(self, **kwargs):
        limiting = '' if self.succ_pairs_only else '(# EPR Pairs)'
        req_gen = ' (Discarding Unfilled Requests)' if not self.reuse_requests else ''
        return 'ALG-N-Fusion' + limiting + req_gen











