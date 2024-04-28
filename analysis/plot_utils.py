import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import numpy as np

def letter_annotation(ax, xoffset, yoffset, letter, fontsize=12):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=fontsize, weight='bold')


def volcano_plot(x, y, row, gene_names, fontsize=12):
    selected_genes = row['Lead_genes'].split(';')
    m = gene_names.isin(selected_genes)
    idxs = np.argwhere(m).ravel()
    
    plt.title(row['Term'])
    plt.scatter(x[~m], y[~m], s=5, c='lightgray')
    plt.scatter(x[m], y[m], s=10, c='red')
    # plt.xlabel('Average expression')
    plt.xlabel('Log fold change', fontsize=fontsize)
    plt.ylabel('$-\log_{10}$(p-value)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize);
    plt.yticks(fontsize=fontsize);
    
    texts = []
    for i in idxs:
        txt = gene_names.values[i]
        texts.append(plt.text(x[i], y[i], txt, fontsize=fontsize))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5), min_arrow_len=5)
    return plt.gca()