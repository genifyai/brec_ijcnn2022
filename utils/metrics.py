import numpy as np


def precision_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    """
    c = 0
    for p in preds[:k]:
        if p in gt:
            c += 1
    return c / k


def precision_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    """
    precs = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        precs.append(precision_k(k, g, p))
    return sum(precs) / len(gt)


def recall_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    """
    c = 0
    for p in preds[:k]:
        if p in gt:
            c += 1
    return c / len(gt)


def recall_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    """
    recalls = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        recalls.append(recall_k(k, g, p))
    return sum(recalls) / len(gt)


def mrr_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    """
    for i, p in enumerate(preds[:k]):
        if p in gt:
            return 1 / (i + 1)
    return 0.


def mrr_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    """
    mrr = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        mrr.append(mrr_k(k, g, p))
    return sum(mrr) / len(gt)


def ndcg_k(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[int], index of ground truth recommendations
    :param preds: list[int], index of predicted recommendations
    """
    c = 0
    j = 0
    for i, p in enumerate(preds[:k]):
        if p in gt:
            c += 1 / np.log(1 + (i + 1))
            j += 1
    d = 0
    for i in range(j):
        d += 1 / np.log(1 + (i + 1))
    if d == 0:
        return 0
    return c / d


def ndcg_k_batch(k, gt, preds):
    """
    :param k: int, scope of metric
    :param gt: list[list[int]], list of index of ground truth recommendations
    :param preds: list[list[int]], list of index of predicted recommendations
    """
    ndcg = []
    if len(gt) == 0:
        print("Error: no data")
        return 0
    for g, p in zip(gt, preds):
        ndcg.append(ndcg_k(k, g, p))
    return sum(ndcg) / len(gt)


def compute_metrics(l, p, verbose=True):
    """
        :param l: list[list[int]], list of index of ground truth recommendations
        :param p: list[list[int]], list of index of predicted recommendations
    """
    tot_prec1 = precision_k_batch(1, l, p)
    tot_prec3 = precision_k_batch(3, l, p)
    tot_prec5 = precision_k_batch(5, l, p)
    tot_prec10 = precision_k_batch(10, l, p)
    tot_recall1 = recall_k_batch(1, l, p)
    tot_recall3 = recall_k_batch(3, l, p)
    tot_recall5 = recall_k_batch(5, l, p)
    tot_recall10 = recall_k_batch(10, l, p)
    mrr20 = mrr_k_batch(20, l, p)
    ndcg20 = ndcg_k_batch(20, l, p)
    metrics_dict = {"prec1": tot_prec1, "prec3": tot_prec3, "prec5": tot_prec5, "prec10": tot_prec10,
                    "recall1": tot_recall1, "recall3": tot_recall3, "recall5": tot_recall5, "recall10": tot_recall10,
                    "mrr20": mrr20, "ndcg20": ndcg20}
    if verbose:
        print_metrics_dict(metrics_dict)
    return metrics_dict


def print_metrics_dict(metrics):
    keys = metrics.keys()
    if "prec1" in keys:
        print("Precision 1:", metrics["prec1"])
    if "prec3" in keys:
        print("Precision 3:", metrics["prec3"])
    if "prec5" in keys:
        print("Precision 5:", metrics["prec5"])
    if "prec10" in keys:
        print("Precision 10:", metrics["prec10"])
    if "recall1" in keys:
        print("Recall 1:", metrics["recall1"])
    if "recall3" in keys:
        print("Recall 3:", metrics["recall3"])
    if "recall5" in keys:
        print("Recall 5:", metrics["recall5"])
    if "recall10" in keys:
        print("Recall 10:", metrics["recall10"])
    if "mrr20" in keys:
        print("Mean Reciprocal Rank 20:", metrics["mrr20"])
    if "ndgc20" in keys:
        print("Normalized Discount Cumulative Gain 20:", metrics["ndcg20"])
