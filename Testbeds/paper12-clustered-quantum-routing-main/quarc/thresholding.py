from scipy.interpolate import interp1d
import numpy as np
import os.path



merge_thresholds = {
    1 : 0.618376,
    2 : 0.551486,
    4 : 0.358346,
    8 : 0.132183,
    16 : 0.000000
}

split_thresholds = {
    1 : 1.000000,
    2 : 0.985486,
    4 : 0.955994,
    8 : 0.716107,
    16 : 0.226608
}


merge_thresholds_16x16_10nsd = {
    1 : 0.678959,
    4 : 0.580743,
    16 : 0.404227,
    64 : -0.069254,
    256 : 0.000000
}

split_thresholds_16x16_10nsd = {
    1 : 1.000000,
    4 : 0.990849,
    16 : 0.953894,
    64 : 0.723375,
    256 : -0.162634
}

merge_thresholds_8x8_5nsd_q100 = {
    1 : 0.717220,
    4 : 0.623051,
    16 : 0.435922,
    64 : 0.000000
}

split_thresholds_8x8_5nsd_q100 = {
    1 : 1.000000,
    4 : 0.986561,
    16 : 0.919657,
    64 : 0.629271
}

merge_thresholds_8x8_6nsd_q100 = {
    1 : 0.632623,
    4 : 0.505667,
    16 : 0.000000,
    64 : 0.000000
}

split_thresholds_8x8_6nsd_q100 = {
    1 : 1.000000,
    4 : 0.972346,
    16 : 0.827180,
    64 : 0.000000
}



merge_thresholds_8x8_6nsd_q90 = {
    1 : 1.000000,
    4 : 0.487490,
    16 : 0.000000,
    64 : 0.000000
}

split_thresholds_8x8_6nsd_q90 = {
    1 : 1.000000,
    4 : 1.000000,
    16 : 0.794680,
    64 : 0.000000
}

merge_thresholds_8x8_6nsd_q90_reusing = {
    1 : 1.000000,
    4 : 0.588982,
    16 : 0.000000,
    64 : 0.000000
}

split_thresholds_8x8_6nsd_q90_reusing = {
    1 : 1.000000,
    4 : 1.000000,
    16 : 0.892183,
    64 : 0.000000
}


merge_thresholds_8x8_10nsd = {
    1 : 0.603335,
    4 : 0.434984,
    16 : 0.007159,
    64 : 0.000000
}

split_thresholds_8x8_10nsd = {
    1 : 1.000000,
    4 : 0.964151,
    16 : 0.765392,
    64 : 0.031877
}


merge_thresholds_8x8_25nsd = {
    1 : 0.552892,
    4 : 0.076956,
    16 : -0.048261,
    64 : 0.000000
}

split_thresholds_8x8_25nsd = {
    1 : 1.000000,
    4 : 0.948427,
    16 : 0.131295,
    64 : -0.127997
}


merge_thresholds_16x16_10nsd_cap4 = {
    1 : 0.664902,
    4 : 0.582881,
    16 : 0.405404,
    64 : 0.000000,
    256 : 0.000000
}

split_thresholds_16x16_10nsd_cap4 = {
    1 : 1.000000,
    4 : 0.991308,
    16 : 0.949442,
    64 : 0.713680,
    256 : 0.000000
}


merge_thresholds_16x16_10nsd_q90 = {
    1 : 1.000000,
    4 : 0.648468,
    16 : 0.194847,
    64 : 0.000000,
    256 : 0.000000
}

split_thresholds_16x16_10nsd_q90 = {
    1 : 1.000000,
    4 : 1.000000,
    16 : 0.975183,
    64 : 0.395778,
    256 : 0.000000
}

merge_thresholds_16x16_10nsd_q100 = {
    1 : 0.707435,
    4 : 0.648244,
    16 : 0.548330,
    64 : 0.343927,
    256 : 0.000000
}

split_thresholds_16x16_10nsd_q100 = {
    1 : 1.000000,
    4 : 0.994034,
    16 : 0.969967,
    64 : 0.867061,
    256 : 0.525234
}




merge_thresholds_8x8_ent_passing = {
    1 : 0.889601,
    4 : 0.844869,
    16 : 0.561084,
    64 : 0.000000
}

split_thresholds_8x8_ent_passing = {
    1 : 1.000000,
    4 : 0.992130,
    16 : 0.953083,
    64 : 0.379098
}

merge_thresholds_16x16_ent_passing = {
    1 : 0.960370,
    4 : 0.929697,
    16 : 0.873250,
    64 : 0.626505,
    256 : 0.000000
}

split_thresholds_16x16_ent_passing = {
    1 : 1.000000,
    4 : 0.999104,
    16 : 0.992260,
    64 : 0.927781,
    256 : 0.455765
}


merge_thresholds_32x32_ent_passing = {
    1 : 0.985532,
    4 : 0.979573,
    16 : 0.955073,
    64 : 0.856107,
    256 : 0.398067,
    1024 : 0.000000
}

split_thresholds_32x32_ent_passing = {
    1 : 1.000000,
    4 : 1.000000,
    16 : 0.998997,
    64 : 0.984456,
    256 : 0.855926,
    1024 : 0.110000
}






def threshold(blob_sizes, merge_thresholds, split_thresholds, xx='nodes'):
    split_vals = []
    merge_vals = []
    
    blob_sizes = sorted(blob_sizes)
    for b in blob_sizes:
        split_vals.append(split_thresholds[b])
        merge_vals.append(merge_thresholds[b])

    if xx == 'nodes':
        # By node count
        split_blob_sizes = blob_sizes.copy()
        while min(split_vals) <= 0:
            idx = split_vals.index(min(split_vals))
            split_vals.pop(idx)
            split_blob_sizes.pop(idx)
        split_th = interp1d(split_blob_sizes, split_vals, fill_value='extrapolate', kind='linear')
        while min(merge_vals) <= 0:
            idx = merge_vals.index(min(merge_vals))
            merge_vals.pop(idx)
            blob_sizes.pop(idx)
        
        merge_th = lambda x : 1/interp1d(blob_sizes, 1/np.array(merge_vals), fill_value='extrapolate', kind='linear')(x)
    else:
        raise NotImplementedError()
    
    return merge_th, split_th


b0s = [1, 4, 16, 64, 256]


#merge_th, split_th = threshold(b0s, merge_thresholds, split_thresholds)
merge_th_8x8_6nsd_q100, split_th_8x8_6nsd_q100 = threshold(b0s[:4], merge_thresholds_8x8_6nsd_q100, split_thresholds_8x8_6nsd_q100)
merge_th_8x8_6nsd_q90, split_th_8x8_6nsd_q90 = threshold(b0s[:4], merge_thresholds_8x8_6nsd_q90, split_thresholds_8x8_6nsd_q90)
merge_th_8x8_6nsd_q90_reusing, split_th_8x8_6nsd_q90_reusing = threshold(b0s[:4], merge_thresholds_8x8_6nsd_q90_reusing, split_thresholds_8x8_6nsd_q90_reusing)
merge_th_8x8_10nsd, split_th_8x8_10nsd = threshold(b0s[:4], merge_thresholds_8x8_10nsd, split_thresholds_8x8_10nsd)
merge_th_8x8_25nsd, split_th_8x8_25nsd = threshold(b0s[:4], merge_thresholds_8x8_25nsd, split_thresholds_8x8_25nsd)
merge_th_16x16_10nsd, split_th_16x16_10nsd = threshold(b0s[:5], merge_thresholds_16x16_10nsd, split_thresholds_16x16_10nsd)
merge_th_16x16_10nsd_cap4, split_th_16x16_10nsd_cap4 = threshold(b0s[:5], merge_thresholds_16x16_10nsd_cap4, split_thresholds_16x16_10nsd_cap4)
merge_th_16x16_10nsd_q90, split_th_16x16_10nsd_q90 = threshold(b0s[:5], merge_thresholds_16x16_10nsd_q90, split_thresholds_16x16_10nsd_q90)
merge_th_16x16_10nsd_q100, split_th_16x16_10nsd_q100 = threshold(b0s[:5], merge_thresholds_16x16_10nsd_q100, split_thresholds_16x16_10nsd_q100)
merge_th_8x8_5nsd_q100, split_th_8x8_5nsd_q100 = threshold(b0s[:4], merge_thresholds_8x8_5nsd_q100, split_thresholds_8x8_5nsd_q100)
merge_th_16x16_ent_passing, split_th_16x16_ent_passing = threshold(b0s[:5], merge_thresholds_16x16_ent_passing, split_thresholds_16x16_ent_passing)
merge_th_32x32_ent_passing, split_th_32x32_ent_passing = threshold(b0s[:6], merge_thresholds_32x32_ent_passing, split_thresholds_32x32_ent_passing)
merge_th_8x8_ent_passing, split_th_8x8_ent_passing = threshold(b0s[:4], merge_thresholds_8x8_ent_passing, split_thresholds_8x8_ent_passing)



def scale_thresholds(n, orig_n, m, s):
    # n : number of nodes in network
    factor = orig_n/n
    merge_th = lambda x : m(factor * x)
    split_th = lambda x : s(factor * x)
    return merge_th, split_th
    
def get_success_rate_thresholds(n):
    scale = n/8
    merge_th = lambda x : scale/(x+scale-1)
    split_th = lambda x : -(x-1)/(n-1)+1
    return merge_th, split_th
    

def get_thresholds2(n, c=0.9):
    _merge_th, _ = get_thresholds(n)
    merge_th = lambda n : (.5 if (n == 1) else _merge_th(n)) - _merge_th(2) + c
    split_th = lambda x : -c*(x-1)/(n-1)+c
    return merge_th, split_th
    

def get_thresholds(n):
    return scale_thresholds(n, 16**2, merge_th_16x16_ent_passing, split_th_16x16_ent_passing)

def get_thresholds_TS(n, topo='Q-CAST-ref'):
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TS-thresh', topo, '%s.th' % n)
    f = open(file)
    split_cap = float(f.readline().strip())
    merge_cap = float(f.readline().strip())
    f.close()

    m1, s1 = get_thresholds(n)
    split_th = lambda n : min(s1(n), split_cap)
    merge_th = lambda n : min(m1(n), merge_cap)
    return merge_th, split_th



