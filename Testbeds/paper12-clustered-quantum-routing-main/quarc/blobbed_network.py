import networkx as nx
import random
import pyvis
from ast import literal_eval as make_tuple
from scipy.spatial import ConvexHull
import numpy as np
import os, os.path
# from qcast.qcast_interface import genQCastTopology

class QuantumChannel:
    def __init__(self, net, edge, priority=0, color=None):
        self.priority = priority
        #self.color = color
        self.edge = edge
        self.length = net.edgeLen(edge)

    def __repr__(self):
        return f"Channel({self.edge[0]} <--> {self.edge[1]}, priority={self.priority:0,.3f})"
    
    def other_end_of(self, n):
        assert(n in self.edge)
        u, v = self.edge
        return u if n == v else v
        
class BlobbedNetwork:
    def __init__(self, G, cap=None, qubits=None):
        # G : networkX graph
        # B : maps a blob id to the nodes in that blob
        # cap : capacity of each edge
        # qubits : defines number of qubits on each node, relative to the total outgoing capacity of that node
                
        self.G = G
        self._colors = {}
        self._blob_to_nodes = None
        self.apsp_lengths = dict(nx.all_pairs_shortest_path_length(G))
        #self._update_blobs(B)
        
        self.add_attributes('length', 'edge', self.edgeLen)
        
        if cap:
            self.add_attributes('capacity', 'edge', cap)
            self.add_attributes('channels', 'edge', lambda e : [QuantumChannel(self, e) for _ in range(self.get_cap(e))])
            if isinstance(cap, int):
                self.cap = cap
        if qubits:
            if callable(qubits):
                # Special case where # of qubits on a node is the sum of all incident capacities
                _qubits = lambda n : qubits(sum([self.G.edges[e]['capacity'] for e in self.G.edges(n)]))
                self.add_attributes('qubits', 'node', _qubits)
            else:
                self.add_attributes('qubits', 'node', qubits)
    
    def reset(self, B):
        # Reset the network with blobbing B
        # Useful for reusing the same network in multiple simulations
        for n in self.G.nodes:
            self.set_q(n, None)
        for e in self.G.edges:
            self.set_p(e, None)
        
        self._update_blobs(B)
    
    def _update_blobs(self, B):
        # B : new blobbing of network (B : blob_id -> list node)
        B = B.copy()
                
        self._blob_to_nodes = B
        self.blobs = B.keys()
        self._node_to_blob = {}
        self.n_blobs = 0
        for blob_id in B:
            nodes = B[blob_id]
            if len(nodes) > 0: self.n_blobs += 1
            for n in nodes:
                self._node_to_blob[str(n)] = blob_id

        #self.perimeters = {}
        #self.areas = {}
        
        # Construct minor induced by blobs (H)
        self.H = nx.quotient_graph(self.G, self._blob_to_nodes)
        labels = {fs : self.get_blob_id(list(fs)[0]) for fs in self.H.nodes}
        nx.relabel_nodes(self.H, labels, copy=False)
        
        # Update sub-singleton clusters
        self.subsingleton_clusters = {b : 1 for b in self.blobs} # hard-coded, not used in this implementation of quarc
        for b in self.blobs:
            assert (self.is_singleton(b)) or self.subsingleton_clusters[b] == 1
        
    # Get blob id from node
    def get_blob_id(self, n): 
        if not isinstance(n, str):
            n = str(n)
        return self._node_to_blob[n]
    
    # Get a list of nodes in blob b
    def nodes_in_blob(self, b):
        return self._blob_to_nodes[b]
    
    # Get a list of all nodes in more than one blob
    def nodes_in_blobs(self, blobs):
        nodes = []
        for b in blobs:
            nodes += self.nodes_in_blob(b)
        return nodes
    
    # Get the number of nodes in a blob
    def blob_size(self, blob):
        return len(self.nodes_in_blob(blob))
    
    # Get the number of edges within a blob
    def blob_edge_count(self, blob):
        H0 = nx.induced_subgraph(self.G, self.nodes_in_blob(blob))
        return len(H0.edges)
    
    # Get the number of edges within a blob
    def blob_average_capacity(self, blob):
        H0 = nx.induced_subgraph(self.G, self.nodes_in_blob(blob))
        return sum([len(self.get_channels(e)) for e in H0.edges]) / self.blob_size(blob)
    
    def blob_total_channels(self, blob):
        # Includes internal and external edges
        count = 0
        nodes = self.nodes_in_blob(blob)
        for x in nodes:
            for y in self.G.neighbors(x):
                e = (x, y)
                nc = len(self.get_channels(e))
                if y in nodes:
                    count += nc/2
                else:
                    count += nc
        return count
    
    # Return the n smallest integers that are not currently used as blob labels
    def available_blob_ids(self, n):
        ids = []
        i = 0
        while len(ids) < n:
            if i not in self.blobs:
                ids.append(i)
            i += 1
        return ids
    
    # Get/set link success probability of edge e (not robust)
    def get_p(self, e):
        return self.G.edges[e]['p']
    def set_p(self, e, p):
        self.G.edges[e]['p'] = p
    def get_avg_effective_p(self, x):
        total = 0
        for y in self.G.neighbors(x):
            nc = len(self.get_channels((x,y)))
            p = self.get_p((x,y))
            p_eff = 1 - ((1 - p)**nc)
            total += p_eff
        avg_p_eff = total / len(list(self.G.neighbors(x)))
        return avg_p_eff
    
    # Get/set fusion success probability of node n (not robust)
    def get_q(self, n):
        return self.G.nodes[n]['q']
    def set_q(self, n, q):
        self.G.nodes[n]['q'] = q
    
    # Compute the value of alpha needed to make the average value of p be 'E_p'
    def getAlphaFromE_p(self, E_p):
        alpha = 0.1
        step = 0.1
        lastAdd = True

        while True:
            avgP = np.mean([np.exp(-alpha * self.edgeLen(e)) for e in self.G.edges])

            if (abs(avgP - E_p) / E_p < 0.001):
                break
            elif (avgP > E_p):
                if not lastAdd: step /= 2
                alpha += step
                lastAdd = True
            else:
                if lastAdd: step /= 2
                alpha -= step
                lastAdd = False

        return alpha
    
    # Get capacity of edge e
    def get_cap(self, e):
        return self.G.edges[e]['capacity']
    
    # Get channels within edge e
    def get_channels(self, e):
        return self.G.edges[e]['channels']
    
    def n_adjacent_channels(self, n):
        return sum([len(self.get_channels((n,y))) for y in self.G.neighbors(n)])
    
    # Get the number of qubits on node n
    def get_nqubits(self, n):
        q = self.G.nodes[n]['qubits']
        channels = self.n_adjacent_channels(n)
        return min(q, channels)
    
    def get_nqubits_blob(self, blob):
        return sum([self.get_nqubits(n) for n in self.nodes_in_blob(blob)])
    
    def x(self, n):
        return self.G.nodes[n]['x']
    def y(self, n):
        return self.G.nodes[n]['y']
    
    def edgeLen(self, e):
        (u,v) = e
        x1, y1 = self.x(u), self.y(u)
        x2, y2 = self.x(v), self.y(v)
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def distance_to_blob(self, n, b, single_node=False):
        # Should memoize this
        if single_node:
            # b is actually just a node, not a blob
            return self.apsp_lengths[n][b]
        
        targets = self.nodes_in_blob(b)
        return min([self.apsp_lengths[n][t] for t in targets])
    
    def is_singleton(self, b):
        return self.blob_size(b) == 1
    
    
    # Add an edge or node attribute to the graph
    def add_attributes(self, name, attr_type, attr):
        assert(attr_type in ['edge', 'node'] and type(name) == str)
        if attr_type == 'edge':
            if isinstance(attr, (float, int)):
                for e in self.G.edges:
                    self.G.edges[e][name] = attr
            elif isinstance(attr, dict):
                for e in attr:
                    self.G.edges[e][name] = attr[e]
            elif callable(attr): # attr is a function (edge -> int/float)
                for e in self.G.edges:
                    self.G.edges[e][name] = attr(e)
            else:
                raise TypeError('attr must be instance of int, float, dict, or fuction, not %s.' % type(attr))
        if attr_type == 'node':
            if isinstance(attr, (float, int)):
                for n in self.G.nodes:
                    self.G.nodes[n][name] = attr
            elif isinstance(attr, dict):
                for n in attr:
                    self.G.nodes[n][name] = attr[n]
            elif callable(attr): # attr is a function (node -> int/float)
                for n in self.G.nodes:
                    self.G.nodes[n][name] = attr(n)
            else:
                raise TypeError('attr must be instance of int, float, dict, or fuction, not %s.' % type(attr))
    
    def label(self):
        return ''
    
    def to_string(self, alpha, q, k=3):
        # Format:
        # n
        # alpha
        # q
        # k
        # x y qubits
        # ...
        # u v p capacity
        # ...
        
        s = ''
        
        s += (str(self.G.number_of_nodes()) + '\n')
        s += ("{:.8g}".format(alpha) + '\n')
        s += (str(float(q)) + '\n')
        s += (str(int(k)) + '\n')
        
        nodes = sorted(self.G.nodes)
        for n in nodes:
            x = str(self.G.nodes[n]['x'])
            y = str(self.G.nodes[n]['y'])
            qubits = str(self.G.nodes[n]['qubits'])
            s += (qubits + ' ' + x + ' ' + y + '\n')
        
        edges = self.G.edges
        for e in edges:
            u, v = e
            if not (isinstance(u, int) and isinstance(v, int)):
                u = nodes.index(u)
                v = nodes.index(v)
            cap = str(self.G.edges[e]['capacity'])
            s += (str(u) + ' ' + str(v) + ' ' + cap + '\n')
        
        return s
    
    def to_file(self, filename, alpha, q, k=3):
        f = open(filename, "w")
        f.write(self.to_string(alpha, q, k))
        f.close()
    
    # Display the network and blobs
    def show_blobs(self, filename='nx.html', path='graphs', size=(700,700)):
        if not self._blob_to_nodes:
            raise RuntimeError("No blobs to show! Call protocol.init_blobs() first.")
        if isinstance(self, BlobbedIslandNetwork):
            size = (700, 1400)
            
        def get_color(b):
            if b in self._colors:
                return self._colors[b]
            basic_colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'fuchsia', 'lime', 'teal', 'olive']
            
            if b < len(basic_colors):
                c = basic_colors[b]
            else:
                c = 'rgb' + str((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            self._colors[b] = c
            return c

        nt = pyvis.network.Network('%dpx'%size[0], '%dpx'%size[1], notebook=True, cdn_resources='in_line')
        # populates the nodes and edges data structures
        G2 = nx.relabel_nodes(self.G, {n : str(n) for n in self.G.nodes})
        for n1, n2, d in G2.edges(data=True): #remove channel attributes to avoid errors when plotting
            d.pop('channels', None)
        
        nt.from_nx(G2)
        nt.toggle_physics(False)
        
        for n in nt.nodes:
            b = self.get_blob_id(n['id'])
            n['color'] = get_color(b)
            n['font'] = '30px arial black'
            n['title'] = 'Node ' + str(n['id']) + '\nCluster ' + str(b) + '\n%d Qubits' % n['qubits']
            n['label'] = ''
            m = 100 if isinstance(self, BlobbedGridNetwork) else 15
            n['x'] = n['x']*m
            n['y'] = n['y']*m

        for e in nt.edges:
            e['title'] =  '(%s, %s)\nCapacity %s' % (e['from'], e['to'], e['capacity'])
            if self.get_blob_id(e['from']) == self.get_blob_id(e['to']):
                # e is not cut
                e['color'] = get_color(self.get_blob_id(e['from']))
                e['width'] = 5
            else:
                # e is cut
                e['color'] = 'black'
                e['width'] = 2.5

        os.makedirs(path, exist_ok=True)
        nt.show_buttons()
        nt.show(os.path.join(path, filename))


        
        
class BlobbedGridNetwork(BlobbedNetwork):
    def __init__(self, N, d=2, cap=1, qubits=lambda x : x):
        # N: Size of each dimension
        # d: Grid dimensionality        
        # cap : Constant, dict, or function that defines capacity of each edge
        
        # Create an N-by-N grid graph
        G = nx.grid_graph([N]*d)
        
        if d == 2:
            for (x,y) in G.nodes:
                G.nodes[(x,y)]['x'] = x
                G.nodes[(x,y)]['y'] = y
        
        self.N = N
        self.d = d
        #G = nx.convert_node_labels_to_integers(self.G, first_label=1, label_attribute='pos')
            
        # Populate blobbing map
        #B = self._grid_blobbing(G, b0)
        
        super().__init__(G, cap, qubits)
        
    def label(self):
        try:
            cap = '(cap %d)' % self.cap if self.cap > 1 else ''
        except:
            cap = ''
        return ('%dD-%dx%d-grid'+cap) % (self.d, self.N, self.N)



class ReducedBlobbedGridNetwork(BlobbedGridNetwork):
    def __init__(self, N, d, r=0.1):
        # N: Size of each dimension
        # d: Grid dimensionality
        # b0: Initial blob size (per dimension)
        # r: Probability each grid point is absent
        
        self.r = r
        
        connected = False
        i = 0
        while not connected:
            if i >= 10:
                raise RuntimeError("Could not generate connected graph after 10 attempts. Try removing fewer nodes.")
                
            # Create an N-by-N grid graph
            G = nx.grid_graph([N]*d)

            if d == 2:
                for (x,y) in G.nodes:
                    G.nodes[(x,y)]['x'] = x
                    G.nodes[(x,y)]['y'] = y

            self.N = N
            self.d = d

            to_remove = []
            for n in G.nodes:
                if random.random() < r:
                    to_remove.append(n)
            for n in to_remove:
                G.remove_node(n)

            connected = nx.is_connected(G)
            i += 1
            
        # Populate blobbing map
        #B = self._grid_blobbing(G, b0)
        
        super(BlobbedGridNetwork, self).__init__(G)
    
    def label(self):
        return super().label() + 'reduced-%.2f' % self.r
        
class BlobbedHexNetwork(BlobbedNetwork):
    def __init__(self, N):
        # N : number of hexagons in each row/column
        G = nx.hexagonal_lattice_graph(N, N)
        self.N = N
        
        for n in G.nodes:
            x, y = G.nodes[n]['pos']
            G.nodes[n]['x'] = x
            G.nodes[n]['y'] = y
        
        super().__init__(G)
    
    def label(self):
        return '%dx%d-hexagonal-lattice' % (self.N, self.N)
    
class BlobbedRandomNetwork(BlobbedNetwork):
    def __init__(self, n, p=None):
        # n : number of nodes
        
        if p == None:
            p = 2*np.log(n)/n
        
        self.N = n
        self.p = p
        
        connected = False
        i = 0
        while not connected:
            if i >= 10:
                raise RuntimeError("Could not generate connected random graph after 10 attempts. Try increasing p.")
                
            # Create a random graph
            G = nx.gnp_random_graph(n, p)

            for node in G.nodes:
                angle = 2*np.pi * node / n 
                G.nodes[node]['x'] = np.cos(angle) *10
                G.nodes[node]['y'] = np.sin(angle) *10

            connected = nx.is_connected(G)
            i += 1
        
        super().__init__(G)
    
    def label(self):
        return 'random-graph-(%d,%.2f)' % (self.N, self.p)

    
class BlobbedIslandNetwork(BlobbedNetwork):
    def __init__(self, N, n, alpha=0.1, beta=0.4, p=0.3, cap=1, qubits=lambda x : x, seed=None):
        # N : width of central island
        # n : number of nodes in side islands
        # p : fraction of side island nodes connected to central island
                
        self.N = 2*n + N**2
        
        # Create a Waxman island graphs
        box_size = 40
        if seed: random.seed(seed)
        L = nx.waxman_graph(n, domain=(0, 0, box_size, box_size), alpha=alpha, beta=beta)
        R = nx.waxman_graph(n, domain=(4*box_size, 0, 5*box_size, box_size), alpha=alpha, beta=beta)
        nx.relabel_nodes(R, lambda x : x+len(L), copy=False)
        
        for G0 in [L, R]:
            for n in G0.nodes:
                x, y = G0.nodes[n]['pos']
                G0.nodes[n]['x'] = x
                G0.nodes[n]['y'] = y
        
        # Create grid central graph
        C = nx.grid_graph([N, N])
        l = 2*box_size
        r = 3*box_size
        for n in C.nodes:
            x, y = n
            C.nodes[n]['x'] = x/(N-1) * (r-l) + l
            C.nodes[n]['y'] = y/(N-1) * (r-l)
        nx.relabel_nodes(C, lambda n : len(L)+len(R)+n[1]*N+n[0], copy=False)
        
        # Compute edges between islands
        edges = []
        for n in sorted(L.nodes, key=lambda n : L.nodes[n]['x'])[round((1-p)*len(L)):]:
            y = L.nodes[n]['y']
            i = round(y/box_size * (N-1))
            v = len(L)+len(R)+i*N+0
            edges.append((n,v))
        for n in sorted(R.nodes, key=lambda n : R.nodes[n]['x'])[:round(p*len(L))]:
            y = R.nodes[n]['y']
            i = round(y/box_size * (N-1))
            v = len(L)+len(R)+i*N+N-1
            edges.append((n,v))


        G = nx.union(nx.union(L, R), C)
        G.add_edges_from(edges)
        
        
        

        
        super().__init__(G, cap, qubits)
    
    def label(self):
        return '3-island-network'
    

    
class BlobbedQCastNetwork(BlobbedNetwork):
    def __init__(self, n, E_p, q, E_d=6, k=3, cap=None, qubits=None, seed=19900111):
        # Note: QCast network == Waxman network
        # n : number of nodes
        # E_p : the average success rate of all channels
        # E_d : average number of neighbors (doesn't actually work because of a Q-CAST bug)
        # cap : capacity of each edge; cap=None lets Q-CAST decide the capacities
        # qubits : # of qubits on each node; qubits=None lets Q-CAST decide
        
        self.n = n
        self.avg_degree = E_d
        self.cap = cap

        def read_QCAST_graph_from_file(f):
            n = int(f.readline().strip())
            alpha = float(f.readline().strip())
            q = float(f.readline().strip())
            k = int(f.readline().strip())

            nodes = []
            for i in range(n):
                _qubits, _x, _y = f.readline().strip().split()
                qubits, x, y = int(_qubits), float(_x), float(_y)
                nodes.append((x, y, qubits))

            edges = []
            while line := f.readline().strip():
                _u, _v, _cap = line.split()
                u, v, cap = int(_u), int(_v), int(_cap)
                edges.append((u, v, cap))
            return nodes, edges
        
        
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'waxman_topologies')
        file = "waxman_%d_%d_%d_%d_%d_%d.net" % (n, E_p*100, q*100, E_d, k, seed)
        filename = os.path.join(folder, file)
        
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            # genQCastTopology(filename, n, E_p, q, E_d, k, seed)
            raise RuntimeError('''Waxman topologies are generated using the Q-CAST source code for consistency in evaluation. '''
                               '''This code is not included with the QuARC source code for copyright reasons. '''
                               '''Some reference topologies are provided in the directory 'Waxman_topolgies'. '''
                               '''Please refer to the function 'generate' in Topo.kt of the Q-CAST source code for more details.''')
        
        f = open(filename)
        
        nodes, edges = read_QCAST_graph_from_file(f)
        qubits = {i : qc for i, (_, _, qc) in enumerate(nodes)} if not qubits else qubits
        locs = {i : (x, y) for i, (x, y, _) in enumerate(nodes)}
        capacities = {(u, v) : c for (u, v, c) in edges} if not cap else cap
        G = nx.from_edgelist([(u,v) for (u, v, _) in edges])

        for n in G.nodes:
            x, y = locs[n]
            G.nodes[n]['x'] = x
            G.nodes[n]['y'] = y
            
        # Make capacities inversely proportional to edge length
        self.invlen = False
        if cap == 'invlen':
            self.invlen = True
            self.G = G
            caps = [3,4,5,6,7]
            sorted_edges = sorted(G.edges, key=lambda x : -1*self.edgeLen(x))
            m = len(sorted_edges)
            
            capacities = {e : caps[int(sorted_edges.index(e)/m*len(caps))] for e in G.edges}
        
        
        super().__init__(G, cap=capacities, qubits=qubits)
        
        

    
    def label(self):
        invlen = '(invlen) ' if self.invlen else ''
        cap = ', cap=%d' % self.cap if isinstance(self.cap, int) else ''
        if (not cap) and self.cap:
            print("Non standard or integer edge capacity")
        return 'Q-CAST Topology %s(n=%d, k=%d%s)' % (invlen, self.n, self.avg_degree, cap)
    
    
    
        
        