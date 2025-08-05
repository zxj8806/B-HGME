import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import product, permutations, combinations, combinations_with_replacement

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay,roc_curve,auc,RocCurveDisplay, average_precision_score, roc_auc_score
import warnings

def kendalltau(G_inf, G_true, sparse = False):

    import scipy.stats as stats

    return stats.kendalltau(G_inf.reshape(-1), G_true.reshape(-1))
    
def NMSE(G_inf, G_true):

    return ((G_inf - G_true) ** 2).sum()/(G_true ** 2).sum()

def PS(G_inf, G_true):

    return int(np.all(np.sign(G_inf)==np.sign(G_true)))

def pearson(G_inf, G_true, sparse = False):

    import scipy.stats as stats
    return stats.pearsonr(G_inf.reshape(-1), G_true.reshape(-1))

def spearman(G_inf, G_true, sparse = False):

    import scipy.stats as stats
    return stats.spearmanr(G_inf.reshape(-1), G_true.reshape(-1))

def cossim(G_inf, G_true, sparse = False):

    G_inf_vector = G_inf.reshape(-1)
    G_true_vector = G_true.reshape(-1)
    G_inf_norm = np.sqrt((G_inf_vector ** 2).sum())
    G_true_norm = np.sqrt((G_true_vector ** 2).sum())
    return np.sum(G_inf_vector * G_true_vector)/G_inf_norm/G_true_norm

def compute_auc_signed(G_inf, G_true):

    gt_pos = G_true.copy()
    estm_pos = G_inf.copy()
    gt_pos[np.where(G_true < 0)] = 0
    estm_pos[np.where(estm_pos < 0)] = 0
    # binarize
    gt_pos = (gt_pos > 1e-6).astype(int)
    estm_pos = (estm_pos - np.min(estm_pos))/(np.max(estm_pos) - np.min(estm_pos) + 1e-12)
    _, _, _, _, AUPRC_pos, AUROC_pos, _ = _compute_auc(estm_pos, gt_pos)

    gt_neg = G_true.copy()
    estm_neg = G_inf.copy()
    gt_neg[np.where(G_true > 0)] = 0
    estm_neg[np.where(estm_neg > 0)] = 0
    # binarize
    gt_neg = (gt_neg < -1e-6).astype(int)
    estm_neg = - estm_neg
    estm_neg = (estm_neg - np.min(estm_neg))/(np.max(estm_neg) - np.min(estm_neg) + 1e-12)
    _, _, _, _, AUPRC_neg, AUROC_neg, _ = _compute_auc(estm_neg, gt_neg)

    return AUPRC_pos, AUPRC_neg

def compute_auc_abs(G_inf, G_true):

    G_inf_abs = np.abs(G_inf)
    G_true_abs = np.abs(G_true)
    G_true_abs = (G_true_abs > 1e-6).astype(int)
    G_inf_abs = (G_inf_abs - np.min(G_inf_abs))/(np.max(G_inf_abs) - np.min(G_inf_abs) + 1e-12)
    _, _, _, _, AUPRC, AUROC, _ = _compute_auc(G_inf_abs, G_true_abs)
    return AUPRC

def _compute_auc(estm_adj, gt_adj):
    
    if np.max(estm_adj) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        # show warning if no positive values in y_true
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fpr, tpr, thresholds = roc_curve(y_true=gt_adj.reshape(-1,), y_score=estm_adj.reshape(-1,), pos_label=1)
        
        if len(set(gt_adj.reshape(-1,))) == 1:
            prec, recall = np.array([0., 1.]), np.array([1., 0.])
        else:
            prec, recall, thresholds = precision_recall_curve(y_true=gt_adj.reshape(-1,), probas_pred=estm_adj.reshape(-1,), pos_label=1)

        # the same
        # AUPRC = average_precision_score(gt_adj.reshape(-1,), estm_adj.reshape(-1,)) 
        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), thresholds    

def compute_auc(estm_adj, gt_adj, directed = False):

    # make absolute and within (0,1)
    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    
    if np.max(estm_norm_adj) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        # assert np.abs(np.max(estm_norm_adj) - 1) < 1e-4
        if directed == False:
            gt_adj = ((gt_adj + gt_adj.T) > 0).astype(int)
        np.fill_diagonal(gt_adj, 0)
        np.fill_diagonal(estm_norm_adj, 0)
        rows, cols = np.where(gt_adj != 0)

        fpr, tpr, thresholds = roc_curve(y_true=gt_adj.reshape(-1,), y_score=estm_norm_adj.reshape(-1,), pos_label=1)
        
        ngenes, _ = gt_adj.shape
        
        if len(set(gt_adj.reshape(-1,))) == 1:
            prec, recall = np.array([0., 1.]), np.array([1., 0.])
        else:
            prec, recall, thresholds = precision_recall_curve(y_true=gt_adj.reshape(-1,), probas_pred=estm_norm_adj.reshape(-1,), pos_label=1)

        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), thresholds    

def compute_auc_ori(estm_adj, gt_adj, directed = False):

    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    
    if np.max(estm_norm_adj) == 0:
        return 0, 0, 0, 0, 0, 0
    else:
        # assert np.abs(np.max(estm_norm_adj) - 1) < 1e-4
        if directed == False:
            gt_adj = ((gt_adj + gt_adj.T) > 0).astype(np.int)
        np.fill_diagonal(gt_adj, 0)
        np.fill_diagonal(estm_norm_adj, 0)
        rows, cols = np.where(gt_adj != 0)

        trueEdgesDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        trueEdgesDF.Gene1 = [str(x) for x in rows]
        trueEdgesDF.Gene2 = [str(y) for y in cols]
        trueEdgesDF.EdgeWeight = 1

        rows, cols = np.where(estm_norm_adj != 0)
        predEdgeDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        predEdgeDF.Gene1 = [str(x) for x in rows]
        predEdgeDF.Gene2 = [str(y) for y in cols]
        predEdgeDF.EdgeWeight = np.array([estm_norm_adj[i,j] for i,j in zip(rows,cols)])

        order = np.argsort(predEdgeDF.EdgeWeight.values.squeeze())[::-1]
        predEdgeDF = predEdgeDF.iloc[order,:]

        prec, recall, fpr, tpr, AUPRC, AUROC = _computeScores(trueEdgesDF, predEdgeDF, directed = directed, selfEdges = False)

        return prec, recall, fpr, tpr, AUPRC, AUROC



def _computeScores(trueEdgesDF, predEdgeDF, 
directed = True, selfEdges = True):

    if directed:        
        if selfEdges:
            possibleEdges = list(product(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         repeat = 2))
        else:
            possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         r = 2))
        
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        
        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
                   (trueEdgesDF['Gene2'] == key.split('|')[1])])>0:
                    TrueEdgeDict[key] = 1
                
        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])]
            if len(subDF)>0:
                PredEdgeDict[key] = np.abs(subDF.EdgeWeight.values[0])

    else:
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        else:
            possibleEdges = list(combinations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}

        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene2'] == key.split('|')[1])) |
                              ((trueEdgesDF['Gene2'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene1'] == key.split('|')[1]))]) > 0:
                TrueEdgeDict[key] = 1  

        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[((predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])) |
                              ((predEdgeDF['Gene2'] == key.split('|')[0]) &
                               (predEdgeDF['Gene1'] == key.split('|')[1]))]
            if len(subDF)>0:
                PredEdgeDict[key] = max(np.abs(subDF.EdgeWeight.values))

    outDF = pd.DataFrame([TrueEdgeDict,PredEdgeDict]).T
    outDF.columns = ['TrueEdges','PredEdges']
    
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)
    
    return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)



def compute_earlyprec(estm_adj, gt_adj, directed = False, TFEdges = False):

    estm_norm_adj = np.abs(estm_adj)/np.max(np.abs(estm_adj) + 1e-12)
    if np.max(estm_norm_adj) == 0:
        return 0, 0
    else:
        rows, cols = np.where(gt_adj != 0)

        trueEdgesDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        trueEdgesDF.Gene1 = np.array([str(x) for x in rows], dtype = np.object)
        trueEdgesDF.Gene2 = np.array([str(y) for y in cols], dtype = np.object)
        trueEdgesDF.EdgeWeight = 1

        rows, cols = np.where(estm_norm_adj != 0)
        predEdgeDF = pd.DataFrame(columns = ["Gene1", "Gene2", "EdgeWeight"])
        predEdgeDF.Gene1 = np.array([str(x) for x in rows], dtype = np.object)
        predEdgeDF.Gene2 = np.array([str(y) for y in cols], dtype = np.object)
        predEdgeDF.EdgeWeight = np.array([estm_norm_adj[i,j] for i,j in zip(rows,cols)])

        order = np.argsort(predEdgeDF.EdgeWeight.values.squeeze())[::-1]
        predEdgeDF = predEdgeDF.iloc[order,:]


        trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
        trueEdgesDF.drop_duplicates(keep = 'first', inplace=True)
        trueEdgesDF.reset_index(drop=True, inplace=True)


        predEdgeDF = predEdgeDF.loc[(predEdgeDF['Gene1'] != predEdgeDF['Gene2'])]
        predEdgeDF.drop_duplicates(keep = 'first', inplace=True)
        predEdgeDF.reset_index(drop=True, inplace=True)

        if TFEdges:

            uniqueNodes = np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']])
            possibleEdges_TF = set(product(set(trueEdgesDF.Gene1),set(uniqueNodes)))

            possibleEdges_noSelf = set(permutations(uniqueNodes, r = 2))

            possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)

            TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}

            trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
            trueEdges = trueEdges[trueEdges.isin(TrueEdgeDict)]
            print("\nEdges considered ", len(trueEdges))
            numEdges = len(trueEdges)

            predEdgeDF['Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
            predEdgeDF = predEdgeDF[predEdgeDF['Edges'].isin(TrueEdgeDict)]

        else:
            trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
            trueEdges = set(trueEdges.values)
            numEdges = len(trueEdges)

        if not predEdgeDF.shape[0] == 0:

            predEdgeDF.EdgeWeight = predEdgeDF.EdgeWeight.round(6)
            predEdgeDF.EdgeWeight = predEdgeDF.EdgeWeight.abs()

            maxk = min(predEdgeDF.shape[0], numEdges)
            edgeWeightTopk = predEdgeDF.iloc[maxk-1].EdgeWeight

            nonZeroMin = np.nanmin(predEdgeDF.EdgeWeight.replace(0, np.nan).values)

            bestVal = max(nonZeroMin, edgeWeightTopk)

            newDF = predEdgeDF.loc[(predEdgeDF['EdgeWeight'] >= bestVal)]

            rankDict = set(newDF['Gene1'] + "|" + newDF['Gene2'])
        else:
            rankDict = []

        if len(rankDict) != 0:
            intersectionSet = rankDict.intersection(trueEdges)
            Eprec = len(intersectionSet)/(len(rankDict)+1e-12)
            Erec = len(intersectionSet)/(len(trueEdges)+1e-12)
        else:
            Eprec = 0
            Erec = 0

    return Eprec, Erec


def compute_eprec_signed(G_inf, G_true):

    gt_pos = G_true.copy()
    estm_pos = G_inf.copy()
    gt_pos[np.where(G_true < 0)] = 0
    estm_pos[np.where(estm_pos < 0)] = 0

    gt_pos = (gt_pos > 1e-6).astype(int)
    estm_pos = (estm_pos - np.min(estm_pos))/(np.max(estm_pos) - np.min(estm_pos) + 1e-12)
    Eprec_pos, Erec_pos = compute_earlyprec(estm_pos, gt_pos)

    gt_neg = G_true.copy()
    estm_neg = G_inf.copy()
    gt_neg[np.where(G_true > 0)] = 0
    estm_neg[np.where(estm_neg > 0)] = 0

    gt_neg = (gt_neg < -1e-6).astype(int)
    estm_neg = - estm_neg
    estm_neg = (estm_neg - np.min(estm_neg))/(np.max(estm_neg) - np.min(estm_neg) + 1e-12)
    Eprec_neg, Erec_neg = compute_earlyprec(estm_neg, gt_neg)

    return Eprec_pos, Eprec_neg

def compute_eprec_abs(G_inf, G_true):

    G_inf_abs = np.abs(G_inf)
    G_true_abs = np.abs(G_true)
    G_true_abs = (G_true_abs > 1e-6).astype(int)
    G_inf_abs = (G_inf_abs - np.min(G_inf_abs))/(np.max(G_inf_abs) - np.min(G_inf_abs) + 1e-12)
    Eprec, Erec = compute_earlyprec(G_inf_abs, G_true_abs)
    return Eprec