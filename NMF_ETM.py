import numpy as np
import pandas as pd
import pickle
import anndata
import scanpy as sc
import torch
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from seaborn import heatmap, clustermap
from etm import ETM
from matplotlib.patches import Patch
import igraph as ig
import louvain


with open('/Users/user/Downloads/data/MP.pickle', 'rb') as f:
    df = pickle.load(f)
df.set_index('Unnamed: 0', inplace=True) # Cell Ids as indices
sample_info = pd.read_csv('/Users/user/Downloads/data/sample_info.csv')
with open('/Users/user/Downloads/data/MP_genes.pickle', 'rb') as f:
    genes = pickle.load(f)
sample_id = pickle.load(open('/Users/user/Downloads/data/cell_IDs.pkl', 'rb'))

df = df.loc[list(sample_id), :]
X = df[genes].values
N = X.shape[0]  # single-cell samples
M = X.shape[1]  # genes

mp_anndata = anndata.AnnData(X=X)
mp_anndata.obs['Celltype'] = sample_info['assigned_cluster'].values

K = 16  # topics

#-----------------------------------------------------Question 1--------------------------------------------------------

def evaluate_ari(cell_embed, adata):
    adata.obsm['cell_embed'] = cell_embed
    sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=30)
    sc.tl.louvain(adata, resolution=0.15)
    ari = adjusted_rand_score(adata.obs['Celltype'], adata.obs['louvain'])
    return ari

W_init = np.random.random((M, K))
H_init = np.random.random((K, N))

def nmf_sse(X, W, H, niter):

    perf = np.ndarray(shape=(niter, 3), dtype='float')
    for i in range(niter):
        H = H * ((np.matmul(W.T, X) / np.matmul(np.matmul(W.T, W), H)))
        W = W * ((np.matmul(X, H.T) / np.matmul(np.matmul(W, H), H.T)))

        MSE = np.sum(np.square(X - np.matmul(W, H))) / (N * M)
        ARI = evaluate_ari(H.T, mp_anndata)
        perf[i] = [i, MSE, ARI]

    return W, H, perf

W_nmf_sse, H_nmf_sse, nmf_sse_perf = nmf_sse(X.T, W_init, H_init, 100)
#print(W_nmf_sse, H_nmf_sse, nmf_sse_perf)

#-----------------------------------------------------Question 2--------------------------------------------------------
def monitor_perf(perf):

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(perf[:, 0], perf[:, 1])
    axes[1].plot(perf[:, 0], perf[:, 2])

monitor_perf(nmf_sse_perf)
#plt.title("SSE and ARI of NMF")
plt.savefig("figures/nmf_vs_scetm.pdf")

#-----------------------------------------------------Question 3--------------------------------------------------------
def nmf_psn(X, W, H, niter):

    perf = np.ndarray(shape=(niter, 3), dtype='float')
    W[W <= 0] = 1e-16
    H[H <= 0] = 1e-16

    for i in range(niter):

        ONE = np.ones((M, N))
        W = W + (W / np.matmul(ONE, H.T)) * (np.matmul(X / np.matmul(W, H), H.T) - np.matmul(ONE, H.T))
        H = H + (H / np.matmul(W.T, ONE)) * (np.matmul(W.T, X / np.matmul(W, H)) - np.matmul(W.T, ONE))

        W[W <= 0] = 1e-16
        H[H <= 0] = 1e-16

        ALPL = np.sum(X * np.log(np.matmul(W, H)) - np.matmul(W, H)) / (N * M)
        ARI = evaluate_ari(H.T, mp_anndata)
        #print("iteration: ", i, ALPL, ARI)
        perf[i] = [i, ALPL, ARI]

    return W, H, perf

W_nmf_psn, H_nmf_psn, nmf_psn_perf = nmf_psn(X.T, W_init, H_init,100)
#print(W_nmf_psn, H_nmf_psn, nmf_psn_perf)

monitor_perf(nmf_psn_perf)
#plt.title("Poisson vs Ari")
plt.savefig("figures/nmf_psn.pdf")

# compare NMF-SSE and NMF-Poisson
fig, ax = plt.subplots()
plt.xlabel("Iteration")
plt.ylabel("ARI")
nmf_sse_perf_df = pd.DataFrame(data=nmf_sse_perf, columns=['Iter', "SSE", 'ARI'])
nmf_psn_perf_df = pd.DataFrame(data=nmf_psn_perf, columns=['Iter', "Poisson", 'ARI'])

ax.plot(nmf_psn_perf_df["Iter"], nmf_psn_perf_df["ARI"], color='red', label='NMF-PSN')
ax.plot(nmf_sse_perf_df["Iter"], nmf_sse_perf_df["ARI"], color='blue', label='NMF-SSE')
ax.legend()
plt.savefig("figures/nmf_sse_vs_psn.pdf")

#-----------------------------------------------------Question 4--------------------------------------------------------

X_tensor = torch.from_numpy(np.array(X, dtype="float32"))
X_tensor_normalized = X_tensor / X_tensor.sum(1).unsqueeze(1)

model = ETM(num_topics=K,vocab_size=len(genes),t_hidden_size=256,rho_size=256,theta_act='relu',embeddings=None,
            train_embeddings=True,enc_drop=0.5)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1.2e-6)

def train_scETM_helper(model, X_tensor, X_tensor_normalized):

    model.train()
    optimizer.zero_grad()
    model.zero_grad()

    # fwd and bwd
    nll, kl_theta = model(X_tensor, X_tensor_normalized)
    loss = nll + kl_theta
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    return torch.sum(loss).item()

# get sample encoding theta from the trained encoder network
def get_theta(model, input_x):
    model.eval()
    with torch.no_grad():
        q_theta = model.q_theta(input_x)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta

def train_scETM(model, X_tensor, X_tensor_normalized, niter):
    perf = np.ndarray(shape=(niter, 3), dtype='float32')

    for i in range(niter):
        NELBO = train_scETM_helper(model, X_tensor, X_tensor_normalized)
        theta_embed_topic_mix = get_theta(model, X_tensor_normalized)

        with torch.no_grad():
            ARI = evaluate_ari(theta_embed_topic_mix, mp_anndata)
        perf[i] = [i, NELBO, ARI]
        #print("iteration: ", i, NELBO, ARI)

    return model, perf

#-----------------------------------------------------Question 5-6------------------------------------------------------

model, scetm_perf = train_scETM(model, X_tensor, X_tensor_normalized,1000)
monitor_perf(scetm_perf)
plt.savefig("figures/scETM_train.pdf")

# Compare NMF-Poisson and scETM
W_nmf_psn, H_nmf_psn, nmf_psn_perf = nmf_psn(X.T, W_init, H_init, 1000)
monitor_perf(nmf_psn_perf)

_, ax = plt.subplots()
plt.xlabel("Iteration")
plt.ylabel("ARI")
nmf_psn_perf_df = pd.DataFrame(data=nmf_psn_perf, columns=['Iter', "Poisson", 'ARI'])
scetm_perf_df = pd.DataFrame(data=scetm_perf, columns=['Iter', "NELBO", 'ARI'])
ax.plot(nmf_psn_perf_df["Iter"], nmf_psn_perf_df["ARI"], color='red', label='NMF-PSN')
ax.plot(scetm_perf_df["Iter"], scetm_perf_df["ARI"], color='black', label='scETM')
ax.legend()
plt.savefig("figures/nmf_vs_scetm.pdf")
ax.show()

#-----------------------------------------------------Question 7--------------------------------------------------------

mp_anndata.obsm['cell_embed'] = H_nmf_psn.T
sc.tl.tsne(mp_anndata, use_rep='cell_embed')
fig2, ax2 = plt.subplots(figsize=(10, 7))
sc.pl.tsne(mp_anndata, color='Celltype', ax=ax2)

mp_anndata.obsm['cell_embed'] = get_theta(model, X_tensor_normalized)
sc.tl.tsne(mp_anndata, use_rep='cell_embed')
fig1, ax1 = plt.subplots(figsize=(10, 7))
sc.pl.tsne(mp_anndata, color='Celltype', ax=ax1)

mp_anndata.obs["Celltype"] = mp_anndata.obs["Celltype"].astype('object')
cell_types = mp_anndata.obs["Celltype"].unique()
cls = ["hotpink", "darkkhaki", "turquoise","lightsteelblue", "peachpuff", "lightgreen", "pink", "royalblue", "orange", "green", "red", "blueviolet", "brown"]
lut = dict(zip(mp_anndata.obs["Celltype"].unique(), cls))
row_colors = mp_anndata.obs["Celltype"].map(lut)
g = clustermap(mp_anndata.obsm['cell_embed'].detach().numpy(), row_colors=row_colors.to_numpy(), cmap="Reds")

handles = [Patch(facecolor=lut[name]) for name in lut]
plt.legend(handles, lut, title="Cell Types", bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure, loc='upper right')

plt.savefig('figures/cells_heatmap_scetm.pdf')

#-----------------------------------------------------Question 8--------------------------------------------------------

gene_topic = ETM.get_beta(model)
gene_topic = gene_topic.detach().numpy()

df_gene_topic = pd.DataFrame(gene_topic.T)
df_gene_topic.insert(0, 'genes', genes)

df_results = pd.DataFrame()

for i in range(16):
    df_top5 = df_gene_topic.nlargest(5, i)
    df_results = df_results.append(df_top5, ignore_index=True, sort=False)

df_results = df_results.fillna(0)
df_results = df_results.set_index('genes')

fig, ax = plt.subplots(figsize=(10, 33))
ax = heatmap(df_results, vmax=0.2, cmap="Reds", linecolor="white")

plt.savefig('figures/topics_heatmap_scetm.pdf')
