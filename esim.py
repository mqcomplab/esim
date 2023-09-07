import numpy as np
import math
from math import log

"""                     ESIM_MODULES: Extended similarity
    ----------------------------------------------------------------------
    
    Miranda-Quintana Group, Department of Chemistry, University of Florida 
    
    ----------------------------------------------------------------------
    
    Please, cite the original papers on the n-ary indices:

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4 

    ----------------------------------------------------------------------

    Github: @mqcomplab
    Latest update: 9/7/2023 by KLP

    ----------------------------------------------------------------------"""



def calculate_counters(data, n_objects = None, c_threshold = None, w_factor = "fraction"):
    """Calculates 1-similarity, 0-similarity, and dissimilarity counters from a array of binary vectors
       or the column-wise sum of those vectors. If column-wise is the inpute, indicate number of objects

    Arguments
    ---------
    data : np.ndarray
        Array of arrays, each sub-array contains the binary object
        OR Array with the columnwise sum, if so specify n_objects

    n_objects: int
        Number of objects, only necessary if the column-wise sum is the input data.

    c_threshold : {None, 'dissimilar', 'min', int, flot}
        Coincidence threshold.
        None : Default, c_threshold = n_objects % 2
        'min' : c_threshold = n_objects % 2
        'dissimilar' : c_threshold = ceil(n_objects / 2)
        int : Integer number < n_objects
        float: number between (0, 1)

    w_factor : {"fraction", "power_n"}
        Type of weight function that will be used.
        'fraction' : similarity = d[k]/n
                     dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
        'power_n' : similarity = n**-(n_objects - d[k])
                    dissimilarity = n**-(d[k] - n_objects % 2)
        other values : similarity = dissimilarity = 1

    Returns
    -------
    counters : dict
        Dictionary with the weighted and non-weighted counters.

    """

    # Check if the data is a np.ndarray of a list
    if not isinstance(data, np.ndarray):
        raise TypeError("Warning: Input data is not a np.ndarray, to secure the right results please input the right data type")

    # Check if the input np.array corresponds to the array of arrays OR the column wise sum array
    if data.ndim == 1:
        c_total = data 

        if not n_objects:
            raise ValueError("Input data is the column-wise sum, please specify number of objects")
    else:
        c_total = np.sum(data, axis = 0)

        if not n_objects:
            n_objects = len(data)

        elif n_objects and n_objects != len(data):
            print("Warning, specified number of objects is different from the number of objects in data")
            n_objects = len(data)
            print("Performing calculations with", n_objects, "objects.")

    # Assign coincidence threshold, this value will be the threshold to classify a counter as 1-similarity, 0-similarity or dissimilarity
    if not c_threshold:
        c_threshold = n_objects % 2

    if c_threshold == 'dissimilar':
        c_threshold = math.ceil(n_objects / 2)

    if c_threshold == 'min':
        c_threshold = n_objects % 2

    if isinstance(c_threshold, int):
        if c_threshold >= n_objects:
            raise ValueError("c_threshold cannot be equal or greater than n_objects.")
        c_threshold = c_threshold

    if 0 < c_threshold < 1:
        c_threshold *= n_objects


    # Setting the weighting function to weigh the partial coincidences 
    if w_factor:
        if "power" in w_factor:
            power = int(w_factor.split("_")[-1])

            def f_s(d):
                return power**-float(n_objects - d)

            def f_d(d):
                return power**-float(d - n_objects % 2)
            
        elif w_factor == "fraction":
            def f_s(d):
                return d/n_objects

            def f_d(d):
                return 1 - (d - n_objects % 2)/n_objects
            
        else:
            def f_s(d):
                return 1

            def f_d(d):
                return 1
    else:
        def f_s(d):
            return 1

        def f_d(d):
            return 1

    # Calculate a (1-similarity), d (0-similarity), b + c (dissimilarity)

    a_indices = 2 * c_total - n_objects > c_threshold
    d_indices = n_objects - 2 * c_total > c_threshold
    dis_indices = np.abs(2 * c_total - n_objects) <= c_threshold

    a = np.sum(a_indices)
    d = np.sum(d_indices)
    total_dis = np.sum(dis_indices)

    a_w_array = f_s(2 * c_total[a_indices] - n_objects)
    d_w_array = f_s(abs(2 * c_total[d_indices] - n_objects))
    total_w_dis_array = f_d(abs(2 * c_total[dis_indices] - n_objects))

    w_a = np.sum(a_w_array)
    w_d = np.sum(d_w_array)
    total_w_dis = np.sum(total_w_dis_array)

    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis

    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}
    return counters

def gen_sim_dict(data, n_objects = None, c_threshold = None, w_factor = "fraction"):
    """Calculate a dictionary containing all the available similarity indexes

    Arguments
    ---------
    See calculate_counters.

    Returns
    -------
    sim_dict : dict
        Dictionary with the weighted and non-weighted similarity indexes."""

    # Indices
    # AC: Austin-Colwell
    # BUB: Baroni-Urbani-Buser
    # CTn: Consoni-Todschini n
    # Fai: Faith
    # Gle: Gleason
    # Ja: Jaccard
    # Ja0: Jaccard 0-variant
    # JT: Jaccard-Tanimoto
    # RT: Rogers-Tanimoto
    # RR: Russel-Rao
    # SM: Sokal-Michener
    # SSn: Sokal-Sneath n

    # Calculate the similarity and dissimilarity counters
    counters = calculate_counters(data, n_objects, c_threshold = c_threshold, w_factor = w_factor)

    # Weighted Indices
    ac_w = (2/np.pi) * np.arcsin(np.sqrt(counters['total_w_sim']/
                                         counters['w_p']))
    bub_w = ((counters['w_a'] * counters['w_d'])**0.5 + counters['w_a'])/\
            ((counters['w_a'] * counters['w_d'])**0.5 + counters['w_a'] + counters['total_w_dis'])
    ct1_w = (log(1 + counters['w_a'] + counters['w_d']))/\
            (log(1 + counters['w_p']))
    ct2_w = (log(1 + counters['w_p']) - log(1 + counters['total_w_dis']))/\
            (log(1 + counters['w_p']))
    ct3_w = (log(1 + counters['w_a']))/\
            (log(1 + counters['w_p']))
    ct4_w = (log(1 + counters['w_a']))/\
            (log(1 + counters['w_a'] + counters['total_w_dis']))
    fai_w = (counters['w_a'] + 0.5 * counters['w_d'])/\
            (counters['w_p'])
    gle_w = (2 * counters['w_a'])/\
            (2 * counters['w_a'] + counters['total_w_dis'])
    ja_w = (3 * counters['w_a'])/\
           (3 * counters['w_a'] + counters['total_w_dis'])
    ja0_w = (3 * counters['total_w_sim'])/\
            (3 * counters['total_w_sim'] + counters['total_w_dis'])
    jt_w = (counters['w_a'])/\
           (counters['w_a'] + counters['total_w_dis'])
    rt_w = (counters['total_w_sim'])/\
           (counters['w_p'] + counters['total_w_dis'])
    rr_w = (counters['w_a'])/\
           (counters['w_p'])
    sm_w =(counters['total_w_sim'])/\
          (counters['w_p'])
    ss1_w = (counters['w_a'])/\
            (counters['w_a'] + 2 * counters['total_w_dis'])
    ss2_w = (2 * counters['total_w_sim'])/\
            (counters['w_p'] + counters['total_w_sim'])
    

    # Non-Weighted Indices
    ac_nw = (2/np.pi) * np.arcsin(np.sqrt(counters['total_w_sim']/
                                          counters['p']))
    bub_nw = ((counters['w_a'] * counters['w_d'])**0.5 + counters['w_a'])/\
             ((counters['a'] * counters['d'])**0.5 + counters['a'] + counters['total_dis'])
    ct1_nw = (log(1 + counters['w_a'] + counters['w_d']))/\
             (log(1 + counters['p']))
    ct2_nw = (log(1 + counters['w_p']) - log(1 + counters['total_w_dis']))/\
             (log(1 + counters['p']))
    ct3_nw = (log(1 + counters['w_a']))/\
             (log(1 + counters['p']))
    ct4_nw = (log(1 + counters['w_a']))/\
             (log(1 + counters['a'] + counters['total_dis']))
    fai_nw = (counters['w_a'] + 0.5 * counters['w_d'])/\
             (counters['p'])
    gle_nw = (2 * counters['w_a'])/\
             (2 * counters['a'] + counters['total_dis'])
    ja_nw = (3 * counters['w_a'])/\
            (3 * counters['a'] + counters['total_dis'])
    ja0_nw = (3 * counters['total_w_sim'])/\
             (3 * counters['total_sim'] + counters['total_dis'])
    jt_nw = (counters['w_a'])/\
            (counters['a'] + counters['total_dis'])
    rt_nw = (counters['total_w_sim'])/\
            (counters['p'] + counters['total_dis'])
    rr_nw = (counters['w_a'])/\
            (counters['p'])
    sm_nw =(counters['total_w_sim'])/\
           (counters['p'])
    ss1_nw = (counters['w_a'])/\
             (counters['a'] + 2 * counters['total_dis'])
    ss2_nw = (2 * counters['total_w_sim'])/\
             (counters['p'] + counters['total_sim'])
    

    # Dictionary with all the results
    Indices = {'nw': {'AC': ac_nw, 'BUB':bub_nw, 'CT1':ct1_nw, 'CT2':ct2_nw, 'CT3':ct3_nw,
                      'CT4':ct4_nw, 'Fai':fai_nw, 'Gle':gle_nw, 'Ja':ja_nw,
                      'Ja0':ja0_nw, 'JT':jt_nw, 'RT':rt_nw, 'RR':rr_nw,
                      'SM':sm_nw, 'SS1':ss1_nw, 'SS2':ss2_nw},
                'w': {'AC': ac_w, 'BUB':bub_w, 'CT1':ct1_w, 'CT2':ct2_w, 'CT3':ct3_w,
                      'CT4':ct4_w, 'Fai':fai_w, 'Gle':gle_w, 'Ja':ja_w,
                      'Ja0':ja0_w, 'JT':jt_w, 'RT':rt_w, 'RR':rr_w,
                      'SM':sm_w, 'SS1':ss1_w, 'SS2':ss2_w}}
    
    return Indices

def calculate_medoid(data, n_ary = 'RR', c_threshold = None, w_factor = 'fraction', weight = 'nw', c_total = None):
    """Calculate the medoid of a set"""
    """ Arguments 
        --------
        
        data: np.array
            np.array of all the binary objects

        n_ary: string
            Default: 'RR'
            string with the initials of the desired similarity index to calculate the medoid from. 
            See gen_sim_dict description for keys.
       
        c_threshold: {int, float, 'min', 'dissimilar', None}
            Default: None
            threshold for the counters. If not provided, it will be calculated with n_objects % 2.

        w_factor: 
            Default: 'fraction'
            desired weighing factors for the counters. 
        
        weight: string
            Default: 'nw'
            {'nw', 'w'} desired weighing method for the similarity index.

        c_total: np.array
            Default: None
            Columnwise sum, not necessary to provide
            
        -----------------
        Returns
        -----------------
        index: int
            index of the medoid in the data array"""

    # Check for input errors
    if n_ary not in ['AC', 'BUB', 'CT1', 'CT2', 'CT3', 'CT4', 'Fai', 'Gle', 'Ja', 'Ja0', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2']: 
        raise ValueError("Desired similarity index not available")

    if weight not in ['nw', 'w']: raise ValueError("weight must be 'nw' or 'w'")

    # Calculate and check columnwise sum
    # If not provided, calculate it. If provided, check if it is correct    
    if c_total is None: c_total = np.sum(data, axis = 0)
    elif c_total is not None and len(data[0]) != len(c_total): 
        raise ValueError("Dimensions of objects and columnwise sum differ")

    # Calculate necessary counters to find medoid
    n_objects = len(data)
    index = n_objects + 1
    min_sim = 1.01

    # Calculate complementary sums
    comp_sums = c_total - data

    # Calculate complementary similarity and find object with the lowest (medoid)
    for i, obj in enumerate(comp_sums):
        sim_dict = gen_sim_dict(obj, n_objects = n_objects - 1, c_threshold = c_threshold, w_factor = w_factor)
        sim_index = sim_dict[weight][n_ary]
        if sim_index < min_sim:
            min_sim = sim_index
            index = i
        else:
            pass

    return index

def calculate_outlier(data, n_ary = 'RR', c_threshold = None, w_factor = 'fraction', weight = 'nw', c_total = None):
    """Calculate the outlier of a set"""
    """ Arguments 
        --------
        
        data: np.array
            np.array of all the binary objects

        n_ary: string
            Default: 'RR'
            string with the initials of the desired similarity index to calculate the medoid from. 
            See gen_sim_dict description for keys.
       
        c_threshold: {int, float, 'min', 'dissimilar', None}
            Default: None
            threshold for the counters. If not provided, it will be calculated with n_objects % 2.

        w_factor: 
            Default: 'fraction'
            desired weighing factors for the counters. 
        
        weight: string
            Default: 'nw'
            {'nw', 'w'} desired weighing method for the similarity index.

        c_total: np.array
            Default: None
            Columnwise sum, not necessary to provide
            
        -------------
        Returns
        -------------
        index: int
            index of the outlier in the data array"""

    # Check for input errors
    if n_ary not in ['AC', 'BUB', 'CT1', 'CT2', 'CT3', 'CT4', 'Fai', 'Gle', 'Ja', 'Ja0', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2']: 
        raise ValueError("Desired similarity index not available")

    if weight not in ['nw', 'w']: raise ValueError("weight must be 'nw' or 'w'")

    # Calculate and check columnwise sum
    # If not provided, calculate it. If provided, check if it is correct    
    if c_total is None: c_total = np.sum(data, axis = 0)
    elif c_total is not None and len(data[0]) != len(c_total): 
        raise ValueError("Dimensions of objects and columnwise sum differ")

    # Calculate necessary counters to find medoid
    n_objects = len(data)
    index = n_objects + 1
    min_sim = -0.01

    # Calculate complementary sums
    comp_sums = c_total - data

    # Calculate complementary similarity and find object with the lowest (medoid)
    for i, obj in enumerate(comp_sums):
        sim_dict = gen_sim_dict(obj, n_objects = n_objects - 1, c_threshold = c_threshold, w_factor = w_factor)
        sim_index = sim_dict[weight][n_ary]
        if sim_index > min_sim:
            min_sim = sim_index
            index = i
        else:
            pass

    return index

def calculate_comp_sim(data, c_threshold = None, n_ary = 'RR', w_factor = 'fraction', weight = 'nw', c_total = None):
    """Calculate the complementary similarity for each element of a set"""
    """ Arguments 
        --------
        
        data: np.array
            np.array of all the binary objects

        n_ary: string
            Default: 'RR'
            string with the initials of the desired similarity index to calculate the medoid from. 
            See gen_sim_dict description for keys.
       
        c_threshold: {int, float, 'min', 'dissimilar', None}
            Default: None
            threshold for the counters. If not provided, it will be calculated with n_objects % 2.

        w_factor: 
            Default: 'fraction'
            desired weighing factors for the counters. 
        
        weight: string
            Default: 'nw'
            {'nw', 'w'} desired weighing method for the similarity index.

        c_total: np.array
            Default: None
            Columnwise sum, not necessary to provide
        
        ---------
        Return
        ---------
        total: list
            list of tuples with index and complementary similarity for each object"""

    # Check for input errors
    if n_ary not in ['AC', 'BUB', 'CT1', 'CT2', 'CT3', 'CT4', 'Fai', 'Gle', 'Ja', 'Ja0', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2']: 
        raise ValueError("Desired similarity index not available")

    if weight not in ['nw', 'w']: raise ValueError("weight must be 'nw' or 'w'")

    # Calculate and check columnwise sum
    # If not provided, calculate it. If provided, check if it is correct    
    if c_total is None: c_total = np.sum(data, axis = 0)
    elif c_total is not None and len(data[0]) != len(c_total): 
        raise ValueError("Dimensions of objects and columnwise sum differ")
    
    # Calculate necessary counters to find complemetary similarities
    n_objects = len(data)
    comp_sums = c_total - data
    total = []

    # Calculate complementary similarity for each object
    for i, obj in enumerate(comp_sums):
        sim_dict = gen_sim_dict(obj, n_objects = n_objects - 1, c_threshold = c_threshold, w_factor = w_factor)
        sim_index = sim_dict[weight][n_ary]
        total.append((i, sim_index))
    
    return total

def sorted_comp_sim(data, c_threshold = None, n_ary = 'RR', w_factor = 'fraction', weight = 'nw', c_total = None):
    """Return a sorted list of tuples with the complementary similarity for each object.
       For arguments, see calculate_comp_sim"""
    
    total = calculate_comp_sim(data, c_threshold = c_threshold, n_ary = n_ary, w_factor = w_factor, weight = weight, c_total = c_total)
    sort = sorted(total, key = lambda x: x[1])

    return sort

