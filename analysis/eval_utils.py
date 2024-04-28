import numpy as np

def jaccard_similarity(list1, list2):
    """
    Compute the Jaccard similarity between two lists.
    """
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def get_topk_de_gene_ids(ctrl, post, k=20):
    """
    Get the top k differentially expressed genes from the results.
    """
    # Get the top k differentially expressed genes
    diff = post - ctrl
    diff_genes_ids = np.argsort(np.abs(diff))[-k:]
    return diff_genes_ids