import os
import pandas as pd
import numpy as np
import itertools
import ray
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import pearsonr, mannwhitneyu, ttest_ind
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree
from . import calc_cs


@ray.remote
def survival_mp(full_df, antigen, cs_type):
#     antigen = product_combo[0]
#     cs_type = product_combo[1]
    length_cutoff = 10
    df = full_df[full_df["antigen"] == antigen]
    df = df.drop("antigen", axis=1)
    df = df.groupby("id").max()
    df_top = df[df["%s_cs"%cs_type] >= df["%s_cs"%cs_type].median()]
    df_bot = df[df["%s_cs"%cs_type] < df["%s_cs"%cs_type].median()]
    if len(df_top) >= length_cutoff and len(df_bot) >= length_cutoff:
        cph = CoxPHFitter()
        cph.fit(df[["%s_cs"%cs_type, "survival_time", "survival_status"]], duration_col='survival_time', event_col='survival_status')

        results = cph.summary.add_prefix('cox_')
        results.insert(0, "cs_type", cs_type)
        results.insert(0, "antigen", antigen)

        logrank = logrank_test(df_top["survival_time"], df_bot["survival_time"], event_observed_A = df_top["survival_status"], event_observed_B = df_bot["survival_status"])   
#         logrank_ncpr_control = logrank_test(df_top_control["%s_MONTHS"%survival], df_bot_control["%s_MONTHS"%survival], event_observed_A = df_top_control["%s_STATUS"%survival], event_observed_B = df_bot_control["%s_STATUS"%survival])   
        kmf_top = KaplanMeierFitter()       
        kmf_top.fit(df_top["survival_time"], event_observed=df_top["survival_status"])
        kmf_bot = KaplanMeierFitter()       
        kmf_bot.fit(df_bot["survival_time"], event_observed=df_bot["survival_status"])

        results["km_p"] = logrank.p_value
        results["km_median_top"] = kmf_top.median_survival_time_
        results["km_median_bot"] = kmf_bot.median_survival_time_
        results["km_median_difference"] = results["km_median_top"] - results["km_median_bot"]
        results["n_top"] = len(df_top)
        results["n_bot"] = len(df_bot)
        return results
    
    else:
        return None

@ray.remote
def rna_mp(full_df, antigen, rna, rna_gene):
    df = full_df[full_df["antigen"] == antigen]
    df = df.groupby("id").max().reset_index()
    rna = rna[["id", rna_gene]]
    df = df.merge(rna, on="id")
    df = df.dropna().reset_index(drop=True)
    if len(df) >=2: 
        cs_types = ["hydro_cs", "ncpr_cs", "combo_cs"]
        correlation_coeff = []
        correlation_p = []
        results = pd.DataFrame()
        for cs_type in cs_types:
            res = pearsonr(df[cs_type], df[rna_gene])
            correlation_coeff.append(res[0])
            correlation_p.append(res[1])
        results["cs_type"] = cs_types
        results["pearson_coeff"] = correlation_coeff
        results["pearson_p"] = correlation_p
        results["antigen"] = antigen
        results["rna_expression_gene"] = rna_gene
        results = results[["antigen", "rna_expression_gene", "cs_type", "pearson_coeff", "pearson_p"]]
        raw_df = None
        if min(correlation_p) <= 0.08:
            raw_df = df.copy()
        return results, raw_df
    else:
        return None, None


@ray.remote
def rna_survival_mp(full_df, rna_gene):
    try:
        length_cutoff = 10
        df = full_df[["survival_time", "survival_status", rna_gene]]
        df = df.dropna().reset_index(drop=True)
        df_top = df[df[rna_gene] >= df[rna_gene].median()]
        df_bot = df[df[rna_gene] < df[rna_gene].median()]
        if len(df_top) >= length_cutoff and len(df_bot) >= length_cutoff:
            cph = CoxPHFitter()
            cph.fit(df[[rna_gene, "survival_time", "survival_status"]], duration_col='survival_time', event_col='survival_status')
            results = cph.summary.add_prefix('cox_')
            results.insert(0, "rna_gene", rna_gene)
            logrank = logrank_test(df_top["survival_time"], df_bot["survival_time"], event_observed_A = df_top["survival_status"], event_observed_B = df_bot["survival_status"])   
            kmf_top = KaplanMeierFitter()       
            kmf_top.fit(df_top["survival_time"], event_observed=df_top["survival_status"])
            kmf_bot = KaplanMeierFitter()       
            kmf_bot.fit(df_bot["survival_time"], event_observed=df_bot["survival_status"])
            results["km_p"] = logrank.p_value
            results["km_median_top"] = kmf_top.median_survival_time_
            results["km_median_bot"] = kmf_bot.median_survival_time_
            results["km_median_difference"] = results["km_median_top"] - results["km_median_bot"]
            results["n_top"] = len(df_top)
            results["n_bot"] = len(df_bot)
            return results
        else:
            return None
    except:
        return None
    
@ray.remote
def hs_survival_mp(full_df, group):
    length_cutoff = 10
    df = full_df[["survival_time", "survival_status", group]]
    df = df.dropna().reset_index(drop=True)
    df_top = df[df[group] == 1]
    df_bot = df[df[group] == 0]
    
    results = pd.DataFrame(columns=["group", "km_p", "km_median_ingroup", "km_median_outgroup", "km_median_difference", "n_ingroup", "n_outgroup"])
    
    
    if len(df_top) >= length_cutoff and len(df_bot) >= length_cutoff:
        logrank = logrank_test(df_top["survival_time"], df_bot["survival_time"], event_observed_A = df_top["survival_status"], event_observed_B = df_bot["survival_status"])  
                           

        kmf_top = KaplanMeierFitter()       
        kmf_top.fit(df_top["survival_time"], event_observed=df_top["survival_status"])
        kmf_bot = KaplanMeierFitter()       
        kmf_bot.fit(df_bot["survival_time"], event_observed=df_bot["survival_status"])
                           
        results.loc[0,"group"] = group
        results.loc[0,"km_p"] = logrank.p_value
        results.loc[0,"km_median_ingroup"] = kmf_top.median_survival_time_
        results.loc[0,"km_median_outgroup"] = kmf_bot.median_survival_time_
        results["km_median_difference"] = results["km_median_ingroup"] - results["km_median_outgroup"]
        results.loc[0,"n_ingroup"] = len(df_top)
        results.loc[0,"n_outgroup"] = len(df_bot)
        return results
                           
    else:
        return None
    
@ray.remote
def hs_rna_mp(df, group, gene):
    df = df.dropna()
    df_top = df[df[group] == 1]
    df_bot = df[df[group] == 0]
    d = {}
    d["group"] = group
    d["gene"] = gene
    d["n_ingroup"] = len(df_top)
    d["n_outgroup"] = len(df_bot)
    d["mean_ingroup"] = df_top[gene].mean()
    d["mean_outgroup"] = df_bot[gene].mean()
    d["mean_difference"] = d["mean_ingroup"] -  d["mean_outgroup"]
    mwu_s, mwu_p = mannwhitneyu(df_top[gene], df_bot[gene])
    d["mwu_p"] = mwu_p
    t_s, t_p = ttest_ind(df_top[gene], df_bot[gene])
    d["ttest_p"] = t_p
    results = pd.DataFrame(d, index=[0])
    results = results[["group", "gene", "ttest_p", "mwu_p", "mean_ingroup", "mean_outgroup", "mean_difference", "n_ingroup", "n_outgroup"]]
    return results

class CSWT(object):
    def __init__(self, vdj, antigens, survival, rna):
        self.vdj = vdj
        self.antigens = antigens
        self.survival = survival
        self.rna = rna
        self.sanitize_vdj()
        self.sanitize_antigens()
        if self.survival is not None:
            self.sanitize_survival()
        if self.rna is not None:
            self.sanitize_rna()
        if "antigen_sequence" not in list(self.antigens):
            self.antigens = self.retreive_antigen_sequences(self.antigens)
            self.sanitize_antigens()
        self.combo_merge()
        pass
    
    def sanitize_vdj(self):
        self.vdj = self.vdj[["id", "cdr3"]]
        self.vdj = self.vdj.groupby(["id", "cdr3"]).first().reset_index()
        self.vdj["cdr3"] = self.vdj["cdr3"].str.upper()
        self.vdj = self.vdj[self.vdj["cdr3"].str.isalpha()]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("B")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("J")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("O")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("U")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("X")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("Z")]
        return
    
    def sanitize_antigens(self):
        self.antigens = self.antigens.drop_duplicates("antigen")
        if "antigen_sequence" in list(self.antigens):
            self.antigens["antigen_sequence"] = self.antigens["antigen_sequence"].str.upper()
            self.antigens = self.antigens[self.antigens["antigen_sequence"].str.isalpha()]
            self.antigens = self.antigens[~self.antigens["antigen_sequence"].str.contains("B")]
            self.antigens = self.antigens[~self.antigens["antigen_sequence"].str.contains("J")]
            self.antigens = self.antigens[~self.antigens["antigen_sequence"].str.contains("O")]
            self.antigens = self.antigens[~self.antigens["antigen_sequence"].str.contains("U")]
            self.antigens = self.antigens[~self.antigens["antigen_sequence"].str.contains("X")]
            self.antigens = self.antigens[~self.antigens["antigen_sequence"].str.contains("Z")]
        return
    
    def sanitize_survival(self):
        self.survival = self.survival[["id", "survival_time","survival_status"]]
        self.survival[["survival_time","survival_status"]] = self.survival[["survival_time", "survival_status"]].apply(pd.to_numeric, errors='coerce')
        self.survival = self.survival.dropna()
        return
    
    def sanitize_rna(self):
        self.rna_genes = list(self.rna)
        self.rna_genes.remove("id")
        self.rna[self.rna_genes] = self.rna[self.rna_genes].apply(pd.to_numeric, errors='coerce')
        return
    
    def retreive_antigen_sequences(self, antigens):
        sequence_db = pd.read_table("/config/csweb/webtool/uniprot.tab")
        sequence_db.columns = map(str.lower, sequence_db.columns)
        sequence_db = sequence_db.rename(columns={"gene names  (primary )":"antigen", "sequence":"antigen_sequence"})
        sequence_db = sequence_db.drop_duplicates("antigen")
        sequence_db = sequence_db[["antigen", "antigen_sequence"]]
        antigens = sequence_db[sequence_db["antigen"].isin(antigens["antigen"])]
        return antigens
    
    def combo_merge(self):
        vdj = self.vdj.copy()
        antigens = self.antigens.copy()
        vdj['key'] = 1
        antigens['key'] = 1
        self.combo_merge = pd.merge(vdj,antigens,on='key').drop('key',axis=1)
        if self.survival is not None:
            self.combo_merge = pd.merge(self.combo_merge, self.survival, on='id', how='left')
        if self.survival is not None and self.rna is not None:
            self.rna_survival = pd.merge(self.survival, self.rna, on='id', how='inner')
            self.rna_survival = self.rna_survival.drop_duplicates("id")
        return
    
    def calc_cs(self):
        self.combo_merge["result"] = calc_cs.gen_cs(self.combo_merge["cdr3"], self.combo_merge["antigen_sequence"], reverse=False)
        self.combo_merge[["hydro_cs","hydro_arrangement","ncpr_cs","ncpr_arrangement","combo_cs","combo_arrangement"]] = pd.DataFrame(self.combo_merge['result'].tolist(), index=self.combo_merge.index)
        self.combo_merge = self.combo_merge.drop("result", axis=1)
        self.report_full = self.combo_merge.copy()
        self.combo_merge = self.combo_merge.dropna()
        return
    
    def calc_survival(self):            
        ray.init()
        results_list = ray.get([survival_mp.remote(self.combo_merge, antigen, cs_type) for antigen, cs_type in list(itertools.product(set(self.combo_merge["antigen"]),["ncpr", "hydro", "combo"]))])
        ray.shutdown()
        self.survival_results = pd.concat(results_list, ignore_index=True)
        self.survival_results =  self.survival_results.sort_values("cox_p", ascending=True)
        return self.survival_results
        
    def calc_rna(self):
        full_df = self.combo_merge[["id", "antigen", "hydro_cs", "ncpr_cs", "combo_cs"]]
        ray.init()
        master_results_list = ray.get([rna_mp.remote(full_df, antigen, self.rna, rna_gene) for antigen, rna_gene in list(itertools.product(set(full_df["antigen"]),self.rna_genes))])
        results_list, self.rna_raw_dfs = zip(*master_results_list)
        ray.shutdown()
        self.rna_correlation_results = pd.concat(results_list, ignore_index=True)
        self.rna_correlation_results =  self.rna_correlation_results.sort_values("pearson_p", ascending=True)
        return self.rna_correlation_results
        
    def calc_rna_survival(self): 
#         results_list = []
#         for rna_gene in self.rna_genes:
#             rna_survival_mp(self.rna_survival, rna_gene)
        ray.init()
        results_list = ray.get([rna_survival_mp.remote(self.rna_survival, rna_gene) for rna_gene in self.rna_genes])
        ray.shutdown()
        self.rna_survival_results = pd.concat(results_list, ignore_index=True)
        self.rna_survival_results =  self.rna_survival_results.sort_values("cox_p", ascending=True)
        return self.rna_survival_results
        
        
class HSWT():
    def __init__(self, vdj, survival, rna, path):
        self.vdj = vdj
        self.survival = survival
        self.rna = rna
        self.path = path
        self.sanitize_vdj()
        if self.survival is not None:
            self.sanitize_survival()
        if self.rna is not None:
            self.sanitize_rna()
        pass
    
    def sanitize_vdj(self):
        self.vdj = self.vdj.groupby(["id", "cdr3"]).first().reset_index()
        self.vdj["cdr3"] = self.vdj["cdr3"].str.upper()
        self.vdj = self.vdj[self.vdj["cdr3"].str.isalpha()]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("B")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("J")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("O")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("U")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("X")]
        self.vdj = self.vdj[~self.vdj["cdr3"].str.contains("Z")]
        self.vdj = self.vdj.reset_index(drop=True)
        return


    def sanitize_survival(self):
        self.survival[["survival_time","survival_status"]] = self.survival[["survival_time", "survival_status"]].apply(pd.to_numeric, errors='coerce')
        self.survival = self.survival.dropna()
        return
    
    def sanitize_rna(self):
        self.rna_genes = list(self.rna)
        self.rna_genes.remove("id")
        self.rna[self.rna_genes] = self.rna[self.rna_genes].apply(pd.to_numeric, errors='coerce')               
        return
    
    def calc_hs_merge(self, df, hs_matrix):
        df["hs_matrix"] = hs_matrix
        distance_array = np.stack(df["hs_matrix"].to_numpy())
        distance_array += distance_array.T
        dm = DistanceMatrix(distance_array, list(df["cdr3"]))
        tree = nj(dm, result_constructor=str)
        t = Tree(tree)
        i = 1
        for node in t.traverse("levelorder"):
            leaves = node.get_leaf_names()
            if len(leaves) > 4:
                df["group_%s"%i] = 0
                df["group_%s"%i][df['cdr3'].isin(leaves)] = 1
                i += 1
        df = df.drop("id", axis=1)
        df = self.vdj.merge(df, on="cdr3")              
        return df
    
    def calc_hs(self):
        df = self.vdj.drop_duplicates("cdr3")
        num_files = calc_cs.gen_hs_matrix(df, self.path, types = 'all')
        for op in ["combo", "hydro", "ncpr"]:
            hs_matrix = []
            for i in range(num_files):
                tmp = np.load(os.path.join(self.path, "tmp", "{0}_{1}.npy".format(op, i)))
                hs_matrix.append(tmp)
            if op == "combo":
                self.homology_combo = self.calc_hs_merge(df, hs_matrix)
            elif op == "hydro":
                self.homology_hydro = self.calc_hs_merge(df, hs_matrix)
            elif op == "ncpr":
                self.homology_ncpr = self.calc_hs_merge(df, hs_matrix)
                           
    def hs_survival_merge(self):
        homology_combo = self.homology_combo.copy()
        groups_cols = list(homology_combo)
        groups_cols.remove("cdr3")
        groups_cols.remove("cdr3_ncpr")
        groups_cols.remove("cdr3_hydro")
        groups_cols.remove("hs_matrix")
        homology_combo = homology_combo[groups_cols]
        homology_combo = homology_combo.groupby("id").max().reset_index()
        self.homology_combo_survival = pd.merge(homology_combo, self.survival, on='id', how='inner')
        
        homology_hydro = self.homology_hydro.copy()
        groups_cols = list(homology_hydro)
        groups_cols.remove("cdr3")
        groups_cols.remove("cdr3_ncpr")
        groups_cols.remove("cdr3_hydro")
        groups_cols.remove("hs_matrix")
        homology_hydro = homology_hydro[groups_cols]
        homology_hydro = homology_hydro.groupby("id").max().reset_index()
        self.homology_hydro_survival = pd.merge(homology_hydro, self.survival, on='id', how='inner')

        homology_ncpr = self.homology_ncpr.copy()
        groups_cols = list(homology_ncpr)
        groups_cols.remove("cdr3")
        groups_cols.remove("cdr3_ncpr")
        groups_cols.remove("cdr3_hydro")
        groups_cols.remove("hs_matrix")
        homology_ncpr = homology_ncpr[groups_cols]
        homology_ncpr = homology_ncpr.groupby("id").max().reset_index()
        self.homology_ncpr_survival = pd.merge(homology_ncpr, self.survival, on='id', how='inner')

    def hs_rna_merge(self):
        self.homology_combo_rna = self.homology_combo.merge(self.rna, on="id")
        self.homology_hydro_rna = self.homology_hydro.merge(self.rna, on="id")
        self.homology_ncpr_rna = self.homology_ncpr.merge(self.rna, on="id")
        
    def calc_hs_survival(self):
        ray.init()
        combo_groups_list = [col for col in list(self.homology_combo_survival) if col.startswith("group")]
        combo_results_list = ray.get([hs_survival_mp.remote(self.homology_combo_survival, group) for group in combo_groups_list])
        ray.shutdown()
        self.homology_combo_survival_results = pd.concat(combo_results_list, ignore_index=True)
        
        self.homology_combo_survival_results =  self.homology_combo_survival_results.sort_values("km_p", ascending=True)
        
        ray.init()
        hydro_groups_list = [col for col in list(self.homology_hydro_survival) if col.startswith("group")]
        hydro_results_list = ray.get([hs_survival_mp.remote(self.homology_hydro_survival, group) for group in hydro_groups_list])
        ray.shutdown()
        self.homology_hydro_survival_results = pd.concat(hydro_results_list, ignore_index=True)
        
        self.homology_hydro_survival_results =  self.homology_hydro_survival_results.sort_values("km_p", ascending=True)
        
        ray.init()
        ncpr_groups_list = [col for col in list(self.homology_ncpr_survival) if col.startswith("group")]
        ncpr_results_list = ray.get([hs_survival_mp.remote(self.homology_ncpr_survival, group) for group in ncpr_groups_list])
        ray.shutdown()
        self.homology_ncpr_survival_results = pd.concat(ncpr_results_list, ignore_index=True)

        self.homology_ncpr_survival_results =  self.homology_ncpr_survival_results.sort_values("km_p", ascending=True)
        
        return
    
    def calc_hs_rna(self):
        combo_groups_genes_list = []
        for col in list(self.homology_combo):
            if col.startswith("group"):
                for gene in self.rna_genes:
                    combo_groups_genes_list.append((col, gene))
        hydro_groups_genes_list = []
        for col in list(self.homology_hydro):
            if col.startswith("group"):
                for gene in self.rna_genes:
                    hydro_groups_genes_list.append((col, gene))
        ncpr_groups_genes_list = []
        for col in list(self.homology_ncpr):
            if col.startswith("group"):
                for gene in self.rna_genes:
                    ncpr_groups_genes_list.append((col, gene))
        ray.init()
        combo_results_list = ray.get([hs_rna_mp.remote(self.homology_combo_rna[["id", col, gene]], col, gene) for col, gene in combo_groups_genes_list])
        ray.shutdown()
        self.homology_combo_rna_results = pd.concat(combo_results_list, ignore_index=True)
        self.homology_combo_rna_results = self.homology_combo_rna_results.dropna()
        self.homology_combo_rna_results = self.homology_combo_rna_results.sort_values("ttest_p", ascending=True)
        ray.init()
        hydro_results_list = ray.get([hs_rna_mp.remote(self.homology_hydro_rna[["id", col, gene]], col, gene) for col, gene in hydro_groups_genes_list])
        ray.shutdown()
        self.homology_hydro_rna_results = pd.concat(hydro_results_list, ignore_index=True)
        self.homology_hydro_rna_results = self.homology_hydro_rna_results.dropna()
        self.homology_hydro_rna_results = self.homology_hydro_rna_results.sort_values("ttest_p", ascending=True)
        ray.init()
        ncpr_results_list = ray.get([hs_rna_mp.remote(self.homology_ncpr_rna[["id", col, gene]], col, gene) for col, gene in ncpr_groups_genes_list])
        ray.shutdown()
        self.homology_ncpr_rna_results = pd.concat(ncpr_results_list, ignore_index=True)
        self.homology_ncpr_rna_results = self.homology_ncpr_rna_results.dropna()
        self.homology_ncpr_rna_results = self.homology_ncpr_rna_results.sort_values("ttest_p", ascending=True)