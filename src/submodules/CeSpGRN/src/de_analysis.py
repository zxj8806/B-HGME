import statsmodels.api as sm 
import statsmodels
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def GAM_pt(pse_t, expr, smooth = 'BSplines', df = 5, degree = 3, family = sm.families.NegativeBinomial()):

    from statsmodels.gam.api import GLMGam, BSplines, CyclicCubicSplines

    if smooth == 'BSplines':
        spline = BSplines(pse_t, df = [df], degree = [degree])
    elif smooth == 'CyclicCubicSplines':
        spline = CyclicCubicSplines(pse_t, df = [df])

    exog, endog = sm.add_constant(pse_t),expr
    # calculate full model
    model_full = sm.GLMGam(endog = endog, exog = exog, smoother = spline, family = family)
    try:
        res_full = model_full.fit()
    except:
        # print("The gene expression is mostly zero")
        return None, None, None
    else:
        # default is exog
        y_full = res_full.predict()
        # reduced model
        y_reduced = res_full.null

        # number of samples - number of paras (res_full.df_resid)
        df_full_residual = expr.shape[0] - df
        df_reduced_residual = expr.shape[0] - 1

        # likelihood of full model
        llf_full = res_full.llf
        # likelihood of reduced(null) model
        llf_reduced = res_full.llnull

        lrdf = (df_reduced_residual - df_full_residual)
        lrstat = -2*(llf_reduced - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
        return y_full, y_reduced, lr_pvalue
    

def de_analy(X, pseudo_order, p_val_t = 0.05, verbose = False, distri = "neg-binomial", fdr_correct = True):

    de_genes = {}
    for reconst_i in pseudo_order.columns:
        de_genes[reconst_i] = []
        sorted_pt = pseudo_order[reconst_i].dropna(axis = 0).sort_values()
        # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        ordering = sorted_pt.index.values.squeeze()
        X_traj = X.iloc[ordering, :]

        for idx, gene in enumerate(X_traj.columns.values):
            gene_dynamic = np.squeeze(X_traj.iloc[:,idx])
            pse_t = np.arange(gene_dynamic.shape[0])[:,None]
            if distri == "neg-binomial":
                gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='BSplines', df = 4, degree = 3, family=sm.families.NegativeBinomial())
            
            elif distri == "log-normal":                
                gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian(link = sm.families.links.log()))
            
            else:
                # treat as normal distribution
                gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian())

            if p_val is not None:
                if verbose:
                    print("gene: ", gene, ", pvalue = ", p_val)
                # if p_val <= p_val_t:
                de_genes[reconst_i].append({"gene": gene, "regression": gene_pred, "null": gene_null,"p_val": p_val})
        
        if fdr_correct:
            pvals = [x["p_val"] for x in de_genes[reconst_i]]
            is_de, pvals = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=p_val_t, method='indep', is_sorted=True)
            
            # update p-value
            for gene_idx in range(len(de_genes[reconst_i])):
                de_genes[reconst_i][gene_idx]["p_val"] = pvals[gene_idx]
            
            # remove the non-de genes
            de_genes[reconst_i] = [x for i,x in enumerate(de_genes[reconst_i]) if is_de[i] == True]

    # sort according to the p_val
    de_genes[reconst_i] = sorted(de_genes[reconst_i], key=lambda x: x["p_val"],reverse=False)

    return de_genes



def de_plot(X, pseudo_order, de_genes, figsize = (20,40), n_genes = 20):

    import os
    import errno
    # # turn off interactive mode for matplotlib
    # plt.ioff()

    ncols = 2
    nrows = np.ceil(n_genes/2).astype('int32')

    figs = []
    for reconst_i in de_genes.keys():
        # ordering of genes
        sorted_pt = pseudo_order[reconst_i].dropna(axis = 0).sort_values()
        # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        ordering = sorted_pt.index.values.squeeze()
        X_traj = X.iloc[ordering, :]


        # make plot
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
        colormap = plt.cm.get_cmap('tab20b', n_genes)
        for idx, gene in enumerate(de_genes[reconst_i][:n_genes]):
            # plot log transformed version
            gene_dynamic = np.squeeze(X_traj.loc[:,gene["gene"]].values)
            pse_t = np.arange(gene_dynamic.shape[0])[:,np.newaxis]

            gene_null = gene['null']
            gene_pred = gene['regression']

            axs[idx%nrows, idx//nrows].scatter(np.arange(gene_dynamic.shape[0]), gene_dynamic, color = colormap(idx), alpha = 0.7)
            axs[idx%nrows, idx//nrows].plot(pse_t, gene_pred, color = "black", alpha = 1)
            axs[idx%nrows, idx//nrows].plot(pse_t, gene_null, color = "red", alpha = 1)
            axs[idx%nrows, idx//nrows].set_title(gene['gene'])

        figs.append(fig)
    return figs
