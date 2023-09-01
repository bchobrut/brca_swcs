import os
import numpy as np
import pandas as pd
from localcider.sequenceParameters import SequenceParameters
from tqdm import tqdm
from functools import partial
#from multiprocessing import Pool, cpu_count
# from ray.util.multiprocessing.pool import Pool
import ray
from multiprocessing import Pool
tqdm.pandas()

@ray.remote
def calc_cs(cdr3, epitope, frame_max = 0, frame_min = 1, types = 'all', reverse=False):
    def encode(sequence):
        ciderseq = SequenceParameters(str(sequence))
        h1 = ciderseq.get_linear_hydropathy(blobLen=3)
        h2 = ciderseq.get_linear_hydropathy(blobLen=1)
        hydro = np.maximum(h1[1],h2[1])
        n1 = ciderseq.get_linear_NCPR(blobLen=3)
        n2 = ciderseq.get_linear_NCPR(blobLen=1)
        ncpr = np.where(n2[1] == -1, -1, n1[1])
        ncpr = np.where(n2[1] == 1, 1, ncpr)
        return hydro, ncpr
    
    def calc_CS_lists(epitope_array, cdr3_array, epitope_sequence, cdr3_sequence, frame_max, frame_min):
        pairwise_list = []
        arrangement_list = []
        #if framze size == 0 use length of epitope as frame size
        if frame_max == 0 or frame_max >= len(epitope_array):
            epitope_array_cut = epitope_array
            max_len = max(len(epitope_array_cut), len(cdr3_array))
            min_len = min(len(epitope_array_cut), len(cdr3_array))
            for i in range(max_len+min_len-1):
                if len(epitope_array_cut) <= len(cdr3_array):
                    if len(cdr3_sequence[max(0,i-min_len+1):max(0,i +1)]) >= frame_min:
                        res = np.sum(np.multiply(epitope_array_cut[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], cdr3_array[max(0,i-min_len+1):max(0,i +1)]))
                        pairwise_list.append(res)
                        arrangement_list.append((("cdr3: " + cdr3_sequence[max(0,i-min_len+1):max(0,i +1)]), ("antigen: " + epitope_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)] + "[{0}:{1}]".format(max(0,i-min_len+1), max(0,i +1)))))
#                         arrangement_list.append((cdr3_sequence[max(0,i-min_len+1):max(0,i +1)], epitope_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)]))
                else:
                    if len(cdr3_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)]) >= frame_min:
                        res = np.sum(np.multiply(cdr3_array[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], epitope_array_cut[max(0,i-min_len+1):max(0,i +1)]))
                        pairwise_list.append(res)
                        arrangement_list.append((("cdr3: " + cdr3_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)]), ("antigen: " + epitope_sequence[max(0,i-min_len+1):max(0,i +1)] + "[{0}:{1}]".format(max(0,i-min_len+1), max(0,i +1)))))
        else:
            for j in range(0,len(epitope_array)-frame_max+1):
                epitope_array_cut = epitope_array[j:j+frame_max]
                epitope_sequence_cut = epitope_sequence[j:j+frame_max]
                max_len = max(len(epitope_array_cut), len(cdr3_array))
                min_len = min(len(epitope_array_cut), len(cdr3_array))
                for i in range(max_len+min_len-1):
                    if len(epitope_array_cut) <= len(cdr3_array):
                        if len(cdr3_sequence[max(0,i-min_len+1):max(0,i +1)]) >= frame_min:
                            res = np.sum(np.multiply(epitope_array_cut[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], cdr3_array[max(0,i-min_len+1):max(0,i +1)]))
                            pairwise_list.append(res)
                            arrangement_list.append((("cdr3: " + cdr3_sequence[max(0,i-min_len+1):max(0,i +1)] ), ("protein: " + epitope_sequence_cut[max(0,min_len-i-1):max(0,min_len+max_len - i-1)] + "[{0}:{1}]".format(max(0,i-min_len+1), max(0,i +1)))))
                    else:
                        if len(cdr3_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)]) >= frame_min:
                            res = np.sum(np.multiply(cdr3_array[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], epitope_array_cut[max(0,i-min_len+1):max(0,i +1)]))
                            pairwise_list.append(res)
                            arrangement_list.append((("cdr3: " + cdr3_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)]), ("antigen: " + epitope_sequence_cut[max(0,i-min_len+1):max(0,i +1)] + "[{0}:{1}]".format(max(0,i-min_len+1), max(0,i +1)))))
        return pairwise_list, arrangement_list   
    
    if frame_min ==0 and len(cdr3) >= len(epitope):
        frame_min = len(epitope)
        
    if frame_min ==0 and len(cdr3) < len(epitope):
        frame_min = len(cdr3)
        
    if types == 'all' or types == 'combo':
        try:
            h1, n1 = encode(epitope)
            h2, n2, = encode(cdr3)
            pairwise_list_hydro, arrangement_list_hydro = calc_CS_lists(h1, h2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
            pairwise_list_ncpr, arrangement_list_ncpr = calc_CS_lists(n1, n2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min) 
            if reverse == True:
                pairwise_list_hydro_2, arrangement_list_hydro_2 = calc_CS_lists(h1[::-1], h2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
                pairwise_list_ncpr_2, arrangement_list_ncpr_2 = calc_CS_lists(n1[::-1], n2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
                arrangement_list_hydro_2 = [(sub[0],sub[1] + " (REVERSED)") for sub in arrangement_list_hydro_2] 
                arrangement_list_ncpr_2 = [(sub[0],sub[1] + " (REVERSED)") for sub in arrangement_list_ncpr_2] 
                pairwise_list_ncpr.extend(pairwise_list_ncpr_2)
                arrangement_list_ncpr.extend(arrangement_list_ncpr_2)
                pairwise_list_hydro.extend(pairwise_list_hydro_2)
                arrangement_list_hydro.extend(arrangement_list_hydro_2)
            pairwise_list_ncpr_fixed = [i * -1 for i in pairwise_list_ncpr]
            pairwise_list_combo = [pairwise_list_hydro[i] + pairwise_list_ncpr_fixed[i] for i in range(len(pairwise_list_hydro))]
            hydro_argmax = np.argmax(pairwise_list_hydro)
            hydro_CS = pairwise_list_hydro[hydro_argmax]
            hydro_arrangement = arrangement_list_hydro[hydro_argmax]
            ncpr_argmax = np.argmax(pairwise_list_ncpr_fixed)
            ncpr_CS = pairwise_list_ncpr_fixed[ncpr_argmax]
            ncpr_arrangement = arrangement_list_ncpr[ncpr_argmax]
            combo_argmax = np.argmax(pairwise_list_combo)
            combo_CS = pairwise_list_combo[combo_argmax]
            combo_arrangement = arrangement_list_ncpr[combo_argmax]
            return hydro_CS, hydro_arrangement, ncpr_CS, ncpr_arrangement, combo_CS, combo_arrangement
        except:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    elif types == 'ncpr':
        try:
            h1, n1 = encode(epitope)
            h2, n2, = encode(cdr3)
            pairwise_list_ncpr, arrangement_list_ncpr = calc_CS_lists(n1, n2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
            if reverse == True:
                pairwise_list_ncpr_2, arrangement_list_ncpr_2 = calc_CS_lists(n1[::-1], n2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
                arrangement_list_ncpr_2 = [(sub[0],sub[1] + " (REVERSED)") for sub in arrangement_list_ncpr_2] 
                pairwise_list_ncpr.extend(pairwise_list_ncpr_2)
                arrangement_list_ncpr.extend(arrangement_list_ncpr_2)
            pairwise_list_ncpr_fixed = [i * -1 for i in pairwise_list_ncpr]
            ncpr_argmax = np.argmax(pairwise_list_ncpr_fixed)
            ncpr_CS = pairwise_list_ncpr_fixed[ncpr_argmax]
            ncpr_arrangement = arrangement_list_ncpr[ncpr_argmax]
            return ncpr_CS, ncpr_arrangement
        except:
            return np.nan, np.nan
    
    elif types == 'hydro':
        try:
            h1, n1 = encode(epitope)
            h2, n2, = encode(cdr3)
            pairwise_list_hydro, arrangement_list_hydro = calc_CS_lists(h1, h2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
            if reverse == True:
                pairwise_list_hydro_2, arrangement_list_hydro_2 = calc_CS_lists(h1[::-1], h2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
                arrangement_list_hydro_2 = [sub + " (REVERSED)" for sub in arrangement_list_hydro_2] 
                arrangement_list_hydro_2 = [(sub[0],sub[1] + " (REVERSED)") for sub in arrangement_list_hydro_2] 
                pairwise_list_hydro.extend(pairwise_list_hydro_2)
                arrangement_list_hydro.extend(arrangement_list_hydro_2)
            hydro_argmax = np.argmax(pairwise_list_hydro)
            hydro_CS = pairwise_list_hydro[hydro_argmax]
            hydro_arrangement = arrangement_list_hydro[hydro_argmax]
            return hydro_CS, hydro_arrangementcombo_arrangement
        except:
            return np.nan, np.nan
        
    
def gen_cs(crd3_list, epitope_list, frame_max = 0, frame_min = 1, types = 'all', reverse=False):
    ray.init()
    cs_list = ray.get([calc_cs.remote(cdr3, epitope, frame_max = frame_max, frame_min = frame_min, types = types, reverse=reverse) for cdr3, epitope in zip(crd3_list, epitope_list)])
    ray.shutdown()
    return cs_list
         
def calc_hs(epitope, cdr3, epitope_ncpr, epitope_hydro, cdr3_ncpr, cdr3_hydro, frame_max = 0, frame_min = 4, types = 'all'):
    
    if epitope == cdr3:
        return 0, 0, 0
    
    def calc_CS_lists(epitope_array, cdr3_array, epitope_sequence, cdr3_sequence, frame_max, frame_min):
        pairwise_list = []
        #if framze size == 0 use length of epitope as frame size
        if frame_max == 0 or frame_max >= len(epitope_array):
            epitope_array_cut = epitope_array
            max_len = max(len(epitope_array_cut), len(cdr3_array))
            min_len = min(len(epitope_array_cut), len(cdr3_array))
            for i in range(max_len+min_len-1):
                if len(epitope_array_cut) <= len(cdr3_array):
                    if len(cdr3_sequence[max(0,i-min_len+1):max(0,i +1)]) >= frame_min:
                        res = np.sum(np.multiply(epitope_array_cut[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], cdr3_array[max(0,i-min_len+1):max(0,i +1)]))
                        pairwise_list.append(res)
                    
                else:
                    if len(cdr3_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)]) >= frame_min:
                        res = np.sum(np.multiply(cdr3_array[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], epitope_array_cut[max(0,i-min_len+1):max(0,i +1)]))
                        pairwise_list.append(res)
                       
        else:
            for j in range(0,len(epitope_array)-frame_max+1):
                epitope_array_cut = epitope_array[j:j+frame_max]
                epitope_sequence_cut = epitope_sequence[j:j+frame_max]
                max_len = max(len(epitope_array_cut), len(cdr3_array))
                min_len = min(len(epitope_array_cut), len(cdr3_array))
                for i in range(max_len+min_len-1):
                    if len(epitope_array_cut) <= len(cdr3_array):
                        if len(cdr3_sequence[max(0,i-min_len+1):max(0,i +1)]) >= frame_min:
                            res = np.sum(np.multiply(epitope_array_cut[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], cdr3_array[max(0,i-min_len+1):max(0,i +1)]))
                            pairwise_list.append(res)
                           
                    else:
                        if len(cdr3_sequence[max(0,min_len-i-1):max(0,min_len+max_len - i-1)]) >= frame_min:
                            res = np.sum(np.multiply(cdr3_array[max(0,min_len-i-1):max(0,min_len+max_len - i-1)], epitope_array_cut[max(0,i-min_len+1):max(0,i +1)]))
                            pairwise_list.append(res)
                           
        return pairwise_list   
    
    if frame_min ==0 and len(cdr3) >= len(epitope):
        frame_min = len(epitope)
        
    if frame_min ==0 and len(cdr3) < len(epitope):
        frame_min = len(cdr3)
        

    n1 = epitope_ncpr
    h1 = epitope_hydro
    n2 = cdr3_ncpr
    h2 = cdr3_hydro
    pairwise_list_hydro = calc_CS_lists(h1, h2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min)
    pairwise_list_ncpr = calc_CS_lists(n1, n2, epitope, cdr3, frame_max=frame_max, frame_min=frame_min) 
    pairwise_list_ncpr_fixed = pairwise_list_ncpr.copy()
    pairwise_list_combo = [pairwise_list_hydro[i] + pairwise_list_ncpr_fixed[i] for i in range(len(pairwise_list_hydro))]
    hydro_argmax = np.argmax(pairwise_list_hydro)
    hydro_CS = pairwise_list_hydro[hydro_argmax]
    ncpr_argmax = np.argmax(pairwise_list_ncpr_fixed)
    ncpr_CS = pairwise_list_ncpr_fixed[ncpr_argmax]
    combo_argmax = np.argmax(pairwise_list_combo)
    combo_CS = pairwise_list_combo[combo_argmax]
    return hydro_CS, ncpr_CS, combo_CS
#         except:
#             return np.nan, np.nan, np.nan
    

def encode(sequence):
    ciderseq = SequenceParameters(str(sequence))
    h1 = ciderseq.get_linear_hydropathy(blobLen=3)
    h2 = ciderseq.get_linear_hydropathy(blobLen=1)
    hydro = np.maximum(h1[1],h2[1])
    n1 = ciderseq.get_linear_NCPR(blobLen=3)
    n2 = ciderseq.get_linear_NCPR(blobLen=1)
    ncpr = np.where(n2[1] == -1, -1, n1[1])
    ncpr = np.where(n2[1] == 1, 1, ncpr)
    return hydro, ncpr
        
# @ray.remote
def gen_hs_matrix_mp_func(cdr3_col, cdr3_ncpr_col, cdr3_hydro_col, i, path):
    epitope = cdr3_col[i]
    epitope_ncpr = cdr3_ncpr_col[i]
    epitope_hydro = cdr3_hydro_col[i]
    cdr3_hs_list_combo = []
    cdr3_hs_list_hydro = []
    cdr3_hs_list_ncpr = []
    for j in cdr3_col[:i]:
        cdr3_hs_list_combo.append(0)
        cdr3_hs_list_hydro.append(0)
        cdr3_hs_list_ncpr.append(0)
    z = 0
    for cdr3 in cdr3_col[i:]:
        cdr3_ncpr = cdr3_ncpr_col[(i+z)]
        cdr3_hydro = cdr3_hydro_col[(i+z)]
        z+=1
        hydro_CS, ncpr_CS, combo_CS = calc_hs(epitope, cdr3, epitope_ncpr, epitope_hydro, cdr3_ncpr, cdr3_hydro)
        try:
            cdr3_hs_list_combo.append(int((1.0/combo_CS)*100))
        except:
            cdr3_hs_list_combo.append(int(combo_CS))
        try:
            cdr3_hs_list_hydro.append(int((1.0/hydro_CS)*100))
        except:
            cdr3_hs_list_hydro.append(int(hydro_CS))
        try:
            cdr3_hs_list_ncpr.append(int((1.0/ncpr_CS)*100))
        except:
            cdr3_hs_list_ncpr.append(int(ncpr_CS))
    
    cdr3_hs_list_combo_np = np.array(cdr3_hs_list_combo)
    cdr3_hs_list_hydro_np = np.array(cdr3_hs_list_hydro)
    cdr3_hs_list_ncpr_np = np.array(cdr3_hs_list_ncpr)
    
    np.save(os.path.join(path, "tmp", "combo_%s.npy"%i), cdr3_hs_list_combo_np)
    np.save(os.path.join(path, "tmp", "hydro_%s.npy"%i), cdr3_hs_list_hydro_np)
    np.save(os.path.join(path, "tmp", "ncpr_%s.npy"%i), cdr3_hs_list_ncpr_np)
    return
    
def gen_hs_matrix(df, path, types = 'all'):
    hs_matrix_combo = []
    hs_matrix_hydro = []
    hs_matrix_ncpr = []
    df[["cdr3_hydro", "cdr3_ncpr"]] = df.progress_apply(lambda x: encode(x["cdr3"]), axis=1, result_type='expand')
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    os.mkdir(os.path.join(path,"tmp"))
    
    with Pool() as pool:
#         results = pool.starmap(gen_hs_matrix_mp_func, [(df["cdr3"], df["cdr3_ncpr"], df["cdr3_hydro"], i, path) for i in range(len(df["cdr3"]))])
        results = [pool.apply_async(gen_hs_matrix_mp_func, args=(df["cdr3"], df["cdr3_ncpr"], df["cdr3_hydro"], i, path)) for i in range(len(df["cdr3"]))]
        results = [r.get() for r in tqdm(results)]
        
    
#     ray.init()
#     ray.get([gen_hs_matrix_mp_func.remote(df["cdr3"], df["cdr3_ncpr"], df["cdr3_hydro"], i, path) for i in range(len(df["cdr3"]))])
#     ray.shutdown()
    
#     for i in range(len(df["cdr3"])):
#         hs_matrix_combo.append(res[i][0])
#         hs_matrix_hydro.append(res[i][1])
#         hs_matrix_ncpr.append(res[i][2])
    return len(df["cdr3"])
