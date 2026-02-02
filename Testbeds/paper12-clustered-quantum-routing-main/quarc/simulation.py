# The Simulation class contains all the information needed to run a simulation
# i.e., it describes and controls how the network will change over time

import random as rd
import numpy as np
from quarc.blobbed_network import BlobbedGridNetwork

class Event():
    # Change in p, q, (topology?)
    def __init__(self, event_type, time, action):
        valid_event_types = ['p', 'q', 'E_p']
        if event_type not in valid_event_types:
            raise NotImplementedError(event_type)
        self.event_type = event_type
        self.time = time
        self.action = action


class Request():
    def __init__(self, src, dst, t):
        self.src = src
        self.dst = dst
        self.gen_time = t
    
    def __repr__(self):
        return f"Request(src={self.src}, dst={self.dst}, t={self.gen_time})"

        
def _generate_requests(net, r, t):
    # Generate r random S-D pairs
    requests = []
    ns = list(net.G.nodes)
    if isinstance(net, BlobbedGridNetwork): ns = list(range(len(net.G.nodes)))
    for i in range(r):
        src, dst = rd.choice(ns), rd.choice(ns)
        while dst == src:
            dst = rd.choice(ns) # require src != dst
        request = Request(src, dst, t)
        requests.append(request)
    return requests

def _get_requests(net, r, t, requests_queue=None):
    if requests_queue:
        SDs = [requests_queue.pop(0) for _ in range(r)]
        
        if isinstance(net, BlobbedGridNetwork):
            SDs = [((n1%net.N, n1//net.N), (n2%net.N, n2//net.N)) for (n1, n2) in SDs]
        return [Request(sd[0], sd[1], t) for sd in SDs]
    else:
        return _generate_requests(net, r, t)

def _read_requests_file(requests_file):
    f = open(requests_file, 'r')
    s = f.readline().strip()
    requests_queue = list(eval(s))
    f.close()
    return requests_queue


class Simulation():
    def __init__(self, net, T, nsd=10, req_gen_method='constant', name='', requests_file=None,
                 p_pert=0, q_pert=0):
        # net : network associated with this simulation
        # T : simulation time
        # generate_requests : request generation method; only supports 'constant' for now
        self.events = {}
        self.net = net
        self.T = T
        self._nsd = nsd
        self.req_gen_method = req_gen_method
        self.requests_file = requests_file
        self.p_pert = p_pert
        self.q_pert = q_pert
        self.add_requests_file(requests_file)
        self.name = name
    
    def add_requests_file(self, requests_file=None):
        if requests_file:
            self.requests_queue = _read_requests_file(requests_file)
            self.requests_file = requests_file
        else:
            self.requests_queue = None
            self.requests_file = None
    
    def nsd(self, t):
        if isinstance(self._nsd, int):
            return self._nsd
        else:
            raise NotImplementedError("nsd cannot be of type %s" % str(type(self._nsd)))
    
    def generate_requests(self, t, remaining_requests):
        if t == 1 and self.requests_queue:
            # Reset the request queue when running multiple simulations in a row
            self.requests_queue = _read_requests_file(self.requests_file)

        if self.req_gen_method == 'constant':
            # Add new requests so there are 'nsd' requests in total
            return remaining_requests + _get_requests(self.net, self.nsd(t)-len(remaining_requests), t, self.requests_queue)
        if self.req_gen_method == 'constant_replace':
            # Ignore the old requests, just generate 'nsd' fresh ones
            return _get_requests(self.net, self.nsd(t), t, self.requests_queue)
        if self.req_gen_method == 'continuous':
            # Always generate 'nsd' new requests each time step
            return remaining_requests + _get_requests(self.net, self.nsd(t), t, self.requests_queue)
        raise NotImplementedError()
    
    def _add_event(self, E):
        t = E.time
        if not (isinstance(t, int) and 0 <= t <= self.T):
            raise RunTimeError("t must be an integer with 0 <= t <= T")
        if t in self.events.keys():
            self.events[t].append(E)
        else:
            self.events[t] = [E]
    
    def add_p_event(self, time, p):
        #assert 0 <= p <= 1+10**-6, 'invalid p value: %f' % p
        E = Event('p', time, p)
        self._add_event(E)
    
    def add_Ep_event(self, time, E_p):
        # E_p : expected value of p
        #assert 0 <= E_p <= 1+10**-6, 'invalid E_p value: %f' % E_p
        E = Event('E_p', time, E_p)
        self._add_event(E)
    
    def add_q_event(self, time, q):
        #assert(0 <= q <= 1)
        E = Event('q', time, q)
        self._add_event(E)
    
    def get_events(self, t):
        if t in self.events.keys():
            return self.events[t]
        else:
            return []

    def _process_event(self, E):
        if E.event_type == 'p':
            # reset 'p' parameter of each edge according to E.action
            p = E.action
            if isinstance(p, (float, int)):
                for e in self.net.G.edges:
                    self.net.set_p(e, p)
            elif isinstance(p, dict):
                for e in p:
                    self.net.set_p(e, p[e])
            elif callable(p): # p is a function (edge -> float)
                for e in self.net.G.edges:
                    self.net.set_p(e, p(e))
            else:
                raise TypeError('p must be instance of int, float, dict, or fuction, not %s.' % type(p))
        
        elif E.event_type == 'q':
            # reset 'q' parameter of each node according to E.action
            q = E.action
            if isinstance(q, (float, int)):
                if self.q_pert != 0: print("WARNING: Adding %d%% error to q!" % (self.q_pert * 100))
                for n in self.net.G.nodes:
                    q_actual = q * (1 + self.q_pert * 2*(rd.randint(0,1) - 0.5))
                    ##########
                    #maxQError = 0.1
                    #q_actual = q * (maxQError * 2*(rd.random() - 0.5))
                    #q_actual = q + 0.1 * 2*(rd.randint(0,1) - 0.5) # +/- 0.1
                    ##########
                    self.net.set_q(n, q_actual)
            elif isinstance(q, dict):
                for n in q:
                    self.net.set_q(n, q[n])
            elif callable(q): # q is a function (node -> float)
                for q in self.net.G.nodes:
                    self.net.set_q(n, q(n))
            else:
                raise TypeError('q must be instance of int, float, dict, or fuction, not %s.' % type(q))
        
        elif E.event_type == 'E_p':
            E_p = E.action
            if isinstance(E_p, (float, int)):
                alpha = self.net.getAlphaFromE_p(E_p)
                if self.p_pert != 0: print("WARNING: Adding %d%% error to p!" % (self.p_pert * 100))
                for e in self.net.G.edges:
                    p = np.e**(-alpha * self.net.edgeLen(e))
                    p *= (1 + self.p_pert * 2*(rd.randint(0,1) - 0.5))
                    #print(e, "p", np.e**(-alpha * self.net.edgeLen(e)), "-->", p)
                    ##########
                    #maxPError = 0.2
                    #p += p * (maxPError * 2*(rd.random() - 0.5))
                    #p += 0.1 * 2*(rd.randint(0,1) - 0.5) # +/- 0.1
                    ##########
                    self.net.set_p(e, p)
            else:
                raise TypeError('E_p must be instance of int, float, dict, or fuction, not %s.' % type(E_p))
        
        else:
            raise NotImplementedError(event_type)
        
    # Find all events which occur at time t and apply them to the network self.net
    def process_events(self, t):
        events = self.get_events(t)
        for E in events:
            self._process_event(E)
    
    # Process an event for use in a Q-CAST simulation
    def process_QCast_event(self, E, alpha, q):
        if E.event_type == 'p':
            # set alpha according to p; Assumes each link has length 1!
            _p = E.action
            if isinstance(_p, (float, int)):
                alpha = -np.log(_p)
            else:
                raise NotImplementedError('p must be instance of int or float, not %s.' % type(_p))

        elif E.event_type == 'q':
            # set fusion success probability of all nodes
            _q = E.action
            if isinstance(_q, (float, int)):
                q = float(_q)
            else:
                raise NotImplementedError('q must be instance of int or float, not %s.' % type(_q))

        elif E.event_type == 'E_p':
            # set alpha such that the average p is E_p
            E_p = E.action
            if isinstance(E_p, (float, int)):
                alpha = self.net.getAlphaFromE_p(E_p)
            else:
                raise NotImplementedError('p must be instance of int or float, not %s.' % type(_p))

        else:
            raise NotImplementedError(E.event_type)

        return alpha, q
        

        
        
####################################################
################### Simulations ####################
####################################################

def constant(net, T, p=.8, q=1.0, E_p=False, name=None, **kwargs):
    if name == None:
        name = 'Constant p Simulation (p=%.2f)' % p

    sim = Simulation(net=net, T=T, name=name, **kwargs)
    sim.add_q_event(0, q)
    
    pevent = sim.add_Ep_event if E_p else sim.add_p_event
    pevent(0, p)
    return sim

def dip(net, T, q=1.0, E_p=False, name='Dip Simulation', **kwargs):
    sim = Simulation(net=net, T=T, name=name, **kwargs)
    sim.add_q_event(0, q)
    
    pevent = sim.add_Ep_event if E_p else sim.add_p_event
    pevent(0, .8)
    pevent(sim.T//3, .6)
    pevent(2*sim.T//3, .8)
    return sim

# p increases throughout the simulation
def increasing(net, T, nsd=10, name='Increasing p Simulation'):
    sim = Simulation(net=net, T=T, name=name, nsd=nsd)
    sim.add_p_event(0, .5)
    sim.add_q_event(0, 1)
    sim.add_p_event(sim.T//6, .6)
    sim.add_p_event(2*sim.T//6, .7)
    sim.add_p_event(3*sim.T//6, .8)
    sim.add_p_event(4*sim.T//6, .9)
    sim.add_p_event(5*sim.T//6, 1.0)
    return sim

# p decreases throughout the simulation
def decreasing(net, T, q=1.0, E_p=False, name='Decreasing p Simulation', **kwargs):
    # if E_p, use the same p values as the _average_ link success probabilities
    sim = Simulation(net=net, T=T, name=name, **kwargs)
    sim.add_q_event(0, 1)
    
    pevent = sim.add_Ep_event if E_p else sim.add_p_event
    pevent(0, 1.0)
    pevent(sim.T//6, .9)
    pevent(2*sim.T//6, .8)
    pevent(3*sim.T//6, .7)
    pevent(4*sim.T//6, .6)
    pevent(5*sim.T//6, .5)
    return sim

def decreasing2(net, T, start=.8, stop=.1, step=.1, ps=None, q=1.0, E_p=False, name='Decreasing p Simulation (2)', **kwargs):
    # p decreases throughout the simulation
    sim = Simulation(net=net, T=T, name=name, **kwargs)
    sim.add_q_event(0, q)
    
    pevent = sim.add_Ep_event if E_p else sim.add_p_event
    if not ps:
        ps = np.arange(start, stop-step, -step)
    n = len(ps)
    for i, p in enumerate(ps):
        pevent(i*sim.T//n, p)
    return sim

def custom(net, T, ps, qs, E_p=False, name='Custom Simulation', **kwargs):
    sim = Simulation(net=net, T=T, name=name, **kwargs)
    
    assert(len(ps) > 0 and len(qs) > 0)
    
    pevent = sim.add_Ep_event if E_p else sim.add_p_event

    n = len(ps)
    for i, p in enumerate(ps):
        pevent(i*sim.T//n, p)
        
    m = len(qs)
    for j, q in enumerate(qs):
        sim.add_q_event(j*sim.T//m, q)
        
    return sim

def drop(net, T, t_drop=2500, p2=0.9,  nsd=10, name='Drop Simulation'):
    assert(T>t_drop)
    sim = Simulation(net=net, T=T, name=name, nsd=nsd)
    sim.add_p_event(0, 1.0)
    sim.add_q_event(0, 1)
    sim.add_p_event(t_drop, p2)
    return sim

# p ∈ [0.5, 1] changes randomly every r timesteps
def random(net, T, r=4000,  nsd=10, name='Random Simulation'):
    sim = Simulation(net=net, T=T, name=name, nsd=nsd)
    sim.add_q_event(0, 1)
    for t in range(0, T, r):
        p = rd.random()/2 + .5
        sim.add_p_event(t, p)
        sim.name = sim.name + ' %.2f' % p
    return sim

def varied(net, T, nsd=10, name='Varied p Simulation'):
    ps = [.8, .9, .7, .9, .6, .85, .65, .95]
    sim = Simulation(net=net, T=T, name=name, nsd=nsd)
    sim.add_q_event(0, 1)
    for i, p in enumerate(ps):
        sim.add_p_event(i*sim.T//len(ps), p)
    return sim

def sweep(net, T=13000, q=1.0, nsd=10, name='Sweep'):
    ps = np.arange(.4, 1.001, .05)
    sim = Simulation(net=net, T=T, name=name, nsd=nsd)
    sim.add_q_event(0, q)
    for i, p in enumerate(ps):
        sim.add_p_event(i*sim.T//len(ps), p)
    return sim

# p decreases linearly from 1 in the top-left to .6 in the lower-right
def spatial_2D_grid(net, T, lo=0.6, exp=1, name='Spatial Varying'):
    sim = Simulation(net=net, T=T, name=name)
    sim.add_q_event(0, 1)
    def p(e):
        N = net.N - 1
        c = (1 - lo)/(-2*N)**exp
        
        ((x1, y1), (x2, y2)) = e
        x = min(x1, x2)
        y = min(y1, y2)
        return (c*(x+y - 2*N)**exp + lo).real
    
    sim.add_p_event(0, p)
    return sim


def constant_perturbation(net, T, E_p, q, p_pert, q_pert, **kwargs):
    # p_pert: perturbation of p, in percent
    # q_pert: perturbation of q, in percent
    
    name = "Constant Perturbation (%.2f, %.2f)" % (p_pert, q_pert)
    sim = Simulation(net=net, T=T, name=name, p_pert=p_pert, q_pert=q_pert, **kwargs)
    
    #alpha = net.getAlphaFromE_p(E_p)
    #ps = lambda e : np.e**(-alpha * net.edgeLen(e)) * (1 + p_pert * 2*(rd.randint(0,1) - 0.5))
    #qs = lambda v :q * (1 + q_pert * 2*(rd.randint(0,1) - 0.5))
    
    sim.add_Ep_event(0, E_p)
    sim.add_q_event(0, q)

    return sim

def half_and_half(net, T, p_high=0.6, p_low=0.4, q=0.9, E_p=False, **kwargs):
    name = 'Half-and-Half (%.2f, %.2f)' % (p_low, p_high)
    sim = Simulation(net=net, T=T, name=name, **kwargs)
    sim.add_q_event(0, q)
    
    pevent = sim.add_Ep_event if E_p else sim.add_p_event
    n = len(net.G.nodes)
    def p(e):
        u, v = e
        if u + v < n:
            return p_high
        else:
            return p_low
    
    pevent(0, p)
    return sim


