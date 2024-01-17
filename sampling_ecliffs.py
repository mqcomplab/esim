import numpy as np
import glob as glob
from esim import *
import pickle
import random 

"""
                    Extended activity cliff sampling methods 
    
    ------------------------------------------------------------------------------

    Miranda-Quintana Group, Department of Chemistry, University of Florida
    ------------------------------------------------------------------------------
    
    Please, cite the extended activity cliffs analysis paper if you use this code:
    DOI:
    
    ------------------------------------------------------------------------------
    
    Github repository: @mqcomplab
    Latest update: 01/15/2024 by @klopezperez"""


""" Functions to calculate eSALI values:
    - calculate_esali_init: calculates the initial esali value of a set of molecules
    - calculate_esali: calculates the esali value after adding a new molecule to a set of molecules"""

def calculate_esali_init(data, prop, n_ary):
    """ Function to get the initial esali value of a set of molecules
        --------------------------------------------------------------
        data: np.array
            Array of arrays containing the binary string objects, same length as prop

        prop: np.array
            Array containing the property associated with each object, same length as data

        n_ary: str
            similarity index desired for the esali metric """
    
    if len(data) == 1 and len(prop) == 1:
        raise ValueError("esali and esim cannot be calculated to only one object")
    elif len(data) != len(prop):
        raise ValueError("Number of objects and properties differ")
    else:
        c_sum = np.sum(np.array(data), axis = 0)
        n = len(data)
        esim = gen_sim_dict(c_sum, n_objects = n)[n_ary]

        prop_avg = np.mean(prop)
        sum_sq = np.sum((np.array(prop) - prop_avg)**2)
        p_i_2_sum = np.sum(np.square(prop))
        esali = sum_sq / (n * (1 - esim))

    return esali, c_sum, prop_avg, p_i_2_sum, n

def calculate_esali(fp_sum_ch, avg_prop_ch, prop_sq_ch, n_ch, fp_new, prop_new, n_ary):

    """ Function to get the esali value after adding a new molecule to an existing data set
        -----------------------------------------------------------------------------------
        fp_sum_ch: np.array
            Columnwise sum of the fingerprints of the chosen objects

        avg_prop_ch: float
            Average of the property

        prop_sq_ch: float
            Sum of squares of properties up to that point

        n_ch: int
            number of objects chosen up to that point

        fp_new: np.array
            fingerprint of bits of the object to add to the set

        prop_new: float
            numeric value of the property of the new object to add to the set

        n_ary:
            similarity index desired for the esali metric """

    if len(fp_sum_ch) != len(fp_new):
        raise ValueError("Number of bits in fingerprints differs between inputs")
 
    c_sum_ch = np.sum([fp_sum_ch, fp_new], axis = 0)
    n = n_ch + 1
    P_avg_ch = (avg_prop_ch * n_ch + prop_new) / n
    P_i_2_sum_ch = prop_sq_ch + prop_new**2
    esim_chosen = gen_sim_dict(c_sum_ch, n_objects = n)[n_ary]
    esali = (P_i_2_sum_ch - n * P_avg_ch**2) / (n * (1 - esim_chosen))
 
    return esali, c_sum_ch, P_avg_ch, P_i_2_sum_ch, n


def choose_eSALI(data, prop, percentage, start = 'medoid', n_ary = 'RR'):
    """choose_eSALI: function to pick molecules maximizing eSALI metric
      ----------------------------------------------------------------

     Arguments
     ---------
     data: array of arrays
         Array of array  contains the binary objects

     props: array of floats 
         Array containing property associated with each object
 
     start: str or list
        srt: key on what is used to start the selection 
        {'medoid', 'random', 'outlier'}  

     n_ary: str
         similarity index to use for the eSALI metric
    """ 

# This is function takes a list of fingerprints associated to a given property in the prop index. 
# It returns a list with the ordered index in order to minimize the eSALI metric.  

    if start =='medoid':
        seed = calculate_medoid(data,  n_ary = n_ary)
        chosen = [seed]
    elif start == 'random':
        n_total = len(data)
        seed = random.randint(0, n_total - 1)
        chosen = [seed]
    elif start == 'outlier':
        seed = calculate_outlier(data, n_ary = n_ary)
        chosen = [seed]
    elif isinstance(start, list):
        chosen = start
    else:
        raise ValueError('Select a correct starting point: medoid, random or outlier')
 
    fp = data
    prop = prop
    eSALIS = [0]
 
    fp_ch = np.array([fp[i] for i in chosen])
    prop_ch = np.array([prop[i] for i in chosen])
    n_chosen = len(fp_ch)
    c_sum_ch = np.sum(fp_ch, axis = 0)

    P_avg_ch = np.average(prop_ch) 
    P_i_2_sum_ch = np.sum(np.square(prop_ch)) 

    select_from = np.delete(np.array(range(len(fp))), chosen)
  
    while n_chosen < int(len(fp) * percentage / 100) :
        n = n_chosen + 1
        target = 0
        index = 0
        for i in select_from:
            c_sum = np.sum([c_sum_ch, fp[i]], axis = 0)
            P_i = prop[i]
            P_avg = (P_avg_ch * n_chosen + P_i) / n
            P_i_2_sum = P_i_2_sum_ch + P_i * P_i
            esim = gen_sim_dict(c_sum, n_objects = n)[n_ary] 
            eSALI = (P_i_2_sum - n * P_avg**2) / (n * (1 - esim))
            if eSALI > target:
                index = i
                target = eSALI
    
  
        select_from = np.delete(select_from, np.argwhere(select_from==index))
    
        P_i_ch = prop[index]
        c_sum_ch = np.sum([c_sum_ch, fp[index]], axis = 0)
        P_avg_ch = (P_avg_ch * n_chosen + P_i_ch) / n
        P_i_2_sum_ch += P_i_ch**2
        chosen.append(index)
        n_chosen += 1
        esim_chosen = gen_sim_dict(c_sum_ch, n_objects = n_chosen)[n_ary]
        eSALIS.append((P_i_2_sum_ch - n_chosen * P_avg_ch**2) / (n_chosen * (1 - esim_chosen)))
 
    return chosen, eSALIS

################################################################################################
################################### ANTI-esali #################################################
################################################################################################

def choose_anti(data, prop, percentage, start = 'medoid', n_ary = 'RR'):
    """choose_anti: function to pick molecules minimazing eSALI metric
      ----------------------------------------------------------------

     Arguments
     ---------
     data: array of arrays
         Array of array  contains the binary objects

     props: array of floats 
         Array containing property associated with each object
 
     start: str or list
        srt: key on what is used to start the selection 
        {'medoid', 'random', 'outlier'}  

     n_ary: str
         similarity index to use for the eSALI metric
    """ 

# This is function takes a list of fingerprints associated to a given property in the prop index. 
# It returns a list with the ordered index in order to minimize the eSALI metric.  

    if start =='medoid':
        seed = calculate_medoid(data,  n_ary = n_ary)
        chosen = [seed]
    elif start == 'random':
        n_total = len(data)
        seed = random.randint(0, n_total - 1)
        chosen = [seed]
    elif start == 'outlier':
        seed = calculate_outlier(data, n_ary = n_ary)
        chosen = [seed]
    elif isinstance(start, list):
        chosen = start
    else:
        raise ValueError('Select a correct starting point: medoid, random or outlier')
 
    fp = data
    prop = prop
    eSALIS = [0]
 
    fp_ch = np.array([fp[i] for i in chosen])
    prop_ch = np.array([prop[i] for i in chosen])
    n_chosen = len(fp_ch)
    c_sum_ch = np.sum(fp_ch, axis = 0)

    P_avg_ch = np.average(prop_ch) 
    P_i_2_sum_ch = np.sum(np.square(prop_ch)) 

    select_from = np.delete(np.array(range(len(fp))), chosen)
  
    while n_chosen < int(len(fp) * percentage / 100) :
        n = n_chosen + 1
        target = 10**10
        index = 0
        for i in select_from:
            c_sum = np.sum([c_sum_ch, fp[i]], axis = 0)
            P_i = prop[i]
            P_avg = (P_avg_ch * n_chosen + P_i) / n
            P_i_2_sum = P_i_2_sum_ch + P_i * P_i
            esim = gen_sim_dict(c_sum, n_objects = n)[n_ary]
            eSALI = (P_i_2_sum - n * P_avg**2) / (n * (1 - esim))
            if eSALI < target:
                index = i
                target = eSALI
    
  
        select_from = np.delete(select_from, np.argwhere(select_from==index))
    
        P_i_ch = prop[index]
        c_sum_ch = np.sum([c_sum_ch, fp[index]], axis = 0)
        P_avg_ch = (P_avg_ch * n_chosen + P_i_ch) / n
        P_i_2_sum_ch += P_i_ch**2
        chosen.append(index)
        n_chosen += 1
        esim_chosen = gen_sim_dict(c_sum_ch, n_objects = n_chosen)[n_ary]
        eSALIS.append((P_i_2_sum_ch - n_chosen * P_avg_ch**2) / (n_chosen * (1 - esim_chosen)))
 
    return chosen, eSALIS


def get_new_index_n(data, selected_condensed, n, select_from_n, n_ary = 'RR'):
    """Select a new diverse molecule"""
    """data: dictionary containing all the fingerprints"""
    """selected_condensed: columnwise sum of all the fingerprints selected so far"""
    """n: number of fingerprints selected so far"""
    """select_from_n: dictionary of fingerprints that have not been chosen yet"""
    """c_threshold, n_ary, weight: see gen_sim_dict() function definition"""

    # Number of total fingerprints, it's going to be the number of selected so far + 1, because we are looking for the next molecule that will maximize diversity
    n_total = n + 1

    # min value that is guaranteed to be higher than all the comparisons, this value should be a warranty that we have something lower than the max possible similarity value (1.00)
    min_value = 1.01

    # placeholder index, initiating variable with a number outside the possible index to select 
    index = len(data) + 1

    # for all indices that have not been selected
    for i in select_from_n:
        # column sum
        c_total = selected_condensed + data[i]

        # calculating similarity
        sim_index = gen_sim_dict(c_total, n_objects = n_total)[n_ary]

        # if the sim of the set is less than the similarity of the previous diverse set, update min_value and index
        if sim_index < min_value:
            index = i
            min_value = sim_index
    
    return index

def diversity(data, percentage: int, start = 'medoid', n_ary = 'RR'):
    """ diversity: function to select from a dataset the most diverse molecules
    -----------------------------------------------------------------------

    Arguments
    ---------
    data: np.array
        Array of arrays containing the binary string objects 
     
    percentaje: int
        Percentage of the provided data that wants to be sampled

    start: str or list
        srt: key on what is used to start the selection 
        {'medoid', 'random', 'outlier'}  
         
        list: contains the indexes of the molecules you want to start the selection

    n_ary: str
        Key with the abbreviation of the similarity index to perform the selection 
    """
 
    # total number of objects
    n_total = len(data)

    # indices of all the objects
    total_indices = np.array(range(n_total))

    if start =='medoid':
        seed = calculate_medoid(data,  n_ary = n_ary)
        selected_n = [seed]
    elif start == 'random':
        seed = random.randint(0, n_total - 1)
        selected_n = [seed]
    elif start == 'outlier':
        seed = calculate_outlier(data, n_ary = n_ary)
        selected_n = [seed]
    elif isinstance(start, list):
        selected_n = start
    else:
        raise ValueError('Select a correct starting point: medoid, random or outlier')
  
 
    # Number of initial objects
    n = len(selected_n)

    # Number of objects be selected
    n_max = int(n_total * percentage / 100)

	# Condensation of selected initial selection 
    selected_condensed = np.sum([data[i] for i in selected_n], axis = 0)  

    while len(selected_n) < n_max:
        # indices from which to select the new fingerprints
        select_from_n = np.delete(total_indices, selected_n)

        # new index selected
        new_index_n = get_new_index_n(data, selected_condensed, n, select_from_n, n_ary = n_ary)

        # updating column sum vector
        selected_condensed += data[new_index_n]

        # updating selected indices
        selected_n.append(new_index_n)
        n = len(selected_n)

    return selected_n

def kennardstone(data, n_ary = 'RR'):
    """ kennardstone: 
    function to find the two furtherst molecules in a dataset to then use them as initial for diversity using esim
    -----------------------------------------------------------------------
    data: np.array
        Array of arrays containing the binary string objects
    
    n_ary: str
        Key with the abbreviation of the similarity index to perform the selection 
    """

    medoid = calculate_medoid(data, n_ary = n_ary)

    target = 1
    for i in range(len(data)):
        c_tot = np.sum([data[medoid], data[i]], axis = 0)
        sim_index = gen_sim_dict(c_tot, n_objects = 2)[n_ary]
        if sim_index < target:
            target = sim_index
            target_1 = i
    
    target = 1
    for i in range(len(data)):
        c_tot = np.sum([data[target_1], data[i]], axis = 0)
        sim_index = gen_sim_dict(c_tot, n_objects = 2)[n_ary]
        if sim_index < target:
            target = sim_index
            target_2 = i

    return [target_1, target_2]

def batches(n_batches, order):
    """batches:
    Function to divide a list of molecules into batches based on a given order and number of batches
    For the uniform sampling, the order was calculated using complementary similarity
    -----------------------------------------------------------------------
    n_batches: int
        Number of batches to divide the list of molecules
    order: list
        List of molecules to be divided into batches"""
    
    n_molecules = len(order)
    k = n_molecules % n_batches
    z = k

    if k == 0: n_per_batch = int(n_molecules/n_batches)
    else: n_per_batch = int(math.floor(n_molecules/n_batches))

    new_order = []
    batches = []
    q = 0

    if k == 0:
        for i in range(n_batches):
            batch = order[i*n_per_batch:(i+1)*n_per_batch]
            batches.append(batch)
            print(len(batch))
    else:
        while k != 0:
            batch = order[q:q + n_per_batch + 1]
            q = q + n_per_batch + 1
            k -= 1
            batches.append(batch)
        while q < n_molecules:
            batch = order[q:q + n_per_batch]
            q = q + n_per_batch
            batches.append(batch)

    for i in range(n_per_batch):
        for j in range(n_batches):
            new_order.append(batches[j][i])

    for i in range(z):
        new_order.append(batches[i][-1])

    return new_order
