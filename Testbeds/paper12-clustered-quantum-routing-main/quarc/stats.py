import numpy as np

# Class to keep track of statistics in a network. Allows protocols to be independent of networks
class BlobStatsCollector():
    def __init__(self, net):
        self.stats = {}
        self.entanglement_passing_stats = {}
        self.SPEP_stats = {}
        self.successes = []
        self.aggregate_successes = []
        self.attempts = []
        self.latencies = []
        self.leftovers = []
        self.discards = []
        self.tried_pairs = []
        self.untried_pairs = []
        self.blob_counts = []
        self.cutoff = -1
        self.timing = {'init': [],
                       'reconfig': [],
                       'route' : [],
                       'record' : []
                      }
        self.reset_stats(net)
        
    def reset_stats(self, net):
        self.stats.clear()
        self.entanglement_passing_stats.clear()
        self.total_successes = 0
        self.total_attempts = 0
        for b in net.blobs:
            self.stats[b] = {}
            self.stats[b]['attempts'] = 0
            self.stats[b]['successes'] = 0
            self.stats[b]['w_attempts'] = 0
            self.stats[b]['w_successes'] = 0
            self.entanglement_passing_stats[b] = {}
            self.entanglement_passing_stats[b]['attempts'] = 0
            self.entanglement_passing_stats[b]['successes'] = 0
            self.SPEP_stats[b] = {}
            self.SPEP_stats[b]['attempts'] = 0
            self.SPEP_stats[b]['successes'] = 0
            self.SPEP_stats[b]['path_lens'] = []
    
    def record_attempt(self, request, path, t):
        for blob in path:
            self.stats[blob]['attempts'] += 1
            self.stats[blob]['w_attempts'] += 1/len(path)
        self.total_attempts += 1
        
        while len(self.attempts) < t:
            self.attempts.append(0)
        self.attempts[t-1] += 1
        
    def record_success(self, request, path, t, agg):
        for blob in path:
            self.stats[blob]['successes'] += 1
            self.stats[blob]['w_successes'] += 1/len(path)
        self.total_successes += 1
        
        latency = t - request.gen_time
        self.latencies.append((request.src, request.dst, latency))
        
        while len(self.successes) < t:
            self.successes.append(0)
        self.successes[t-1] += 1
        
        while len(self.aggregate_successes) < t:
            self.aggregate_successes.append(0)
        self.aggregate_successes[t-1] += agg
    
    def avg_success_rate(self):
        return self.total_successes / self.total_attempts
    
    def avg_blob_success_rate(self):
        s = 0
        a = 0
        for blob in self.stats:
            s += self.stats[blob]['successes']
            a += self.stats[blob]['attempts']
        return s / a
    
    def get_success_rate(self, blob):
        if self.stats[blob]['attempts'] > 0:
            return self.stats[blob]['successes'] / self.stats[blob]['attempts']
        else:
            return 0.75 # Arbitrary value; avoids division by 0
    
    def get_weighted_success_rate(self, blob):
        if self.stats[blob]['w_attempts'] > 0:
            return self.stats[blob]['w_successes'] / self.stats[blob]['w_attempts']
        else:
            return 0.75 # Arbitrary value; avoids division by 0
    
    def record_entanglement_passing(self, blob, success:bool):
        self.entanglement_passing_stats[blob]['attempts'] += 1
        if success:
            self.entanglement_passing_stats[blob]['successes'] += 1
            
    def get_entanglement_passing_rate(self, blob):
        if self.entanglement_passing_stats[blob]['attempts'] > 0:
            return self.entanglement_passing_stats[blob]['successes'] / self.entanglement_passing_stats[blob]['attempts']
        else:
            return -1
            #raise RuntimeError()
    
    def avg_entanglement_passing_rate(self):
        s = 0
        a = 0
        for blob in self.entanglement_passing_stats:
            s += self.entanglement_passing_stats[blob]['successes']
            a += self.entanglement_passing_stats[blob]['attempts']
        return s / a
    
    # Shortest path entanglement passing
    def record_SPEP(self, blob, success:bool, path_len_in_blob):
        self.SPEP_stats[blob]['attempts'] += 1
        self.SPEP_stats[blob]['path_lens'].append(path_len_in_blob)
        if success:
            self.SPEP_stats[blob]['successes'] += 1
            
    def get_SPEP_rate(self, blob):
        if self.SPEP_stats[blob]['attempts'] > 0:
            return self.SPEP_stats[blob]['successes'] / self.SPEP_stats[blob]['attempts']
        else:
            return -1
            #raise RuntimeError()
    
    def get_avg_shortest_path_len(self, blob):
        if self.SPEP_stats[blob]['attempts'] > 0:
            return np.mean(self.SPEP_stats[blob]['path_lens'])
        else:
            return -1
            
    
    def avg_throughput(self, start, end, throughput_type='regular'):
        if throughput_type == 'regular':
            return sum(self.successes[start:end]) / (end-start)
        elif throughput_type == 'aggregate':
            return sum(self.aggregate_successes[start:end]) / (end-start)
    
    def record_timing(self, task, elapsed, t):
        assert(task in self.timing)
        L = self.timing[task]
        if len(L) < t:
            L.append(elapsed)
        else:
            L[t-1] += elapsed

    def total_time_ms(self):
        return sum([sum(ts) for ts in self.timing.values()]) / 10**6
    
    def total_routing_time_ms(self):
        return self.total_time_ms() - self.total_reconfig_time_ms()
    
    def total_reconfig_time_ms(self):
        return sum(self.timing['reconfig']) / 10**6
        

class QCastStatsCollector():
    def __init__(self, net):
        self.successes = []
        self.aggregate_successes = []
        self.latencies = []
        self.leftovers = []
        self.discards = []
        self.tried_pairs = []
        self.untried_pairs = []
        self.cutoff = -1
        self.timing = []
        
    def avg_throughput(self, start, end, throughput_type='regular'):
        if throughput_type == 'regular':
            return sum(self.successes[start:end]) / (end-start)
        elif throughput_type == 'aggregate':
            return sum(self.aggregate_successes[start:end]) / (end-start)

    def total_time_ms(self):
        return sum(self.timing) / 10**6