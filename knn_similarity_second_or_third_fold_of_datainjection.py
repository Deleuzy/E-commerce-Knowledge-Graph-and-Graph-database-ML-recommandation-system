# This is running the same knn similarity but this time for each time we have a new data injection under the dv-id data split
# === knn_customer_similarity.py ===
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import joblib
import os
from gensim.models import Word2Vec

print("🔄 Recalibrating KNN-based customer similarity graph with NEW Node2Vec embeddings...")

# ———————————————————————
# CONFIG — MAKE IT DVID-AWARE
# ———————————————————————
DVID = 4  # 👈 ADD THIS — should match train_node2vec DVID
NODE2VEC_MODEL_PATH = f"models/node2vec_model_dvid{DVID}.pkl"  # 👈 Updated
GRAPH_PATH = f"graphs/graph_enriched_with_embeddings_dvid{DVID}.pkl"  # 👈 Updated
SIMILARITY_GRAPH_OUTPUT = f"graphs/G_similarity_dvid{DVID}.graphml"  # 👈 Version output
PARTITION_OUTPUT = f"models/customer_partition_dvid{DVID}.pkl"  # 👈 Version partition
TOP_K = 5
METRIC = 'cosine'

os.makedirs("graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ———————————————————————
# 1. LOAD NODE2VEC MODEL
# ———————————————————————
try:
   node2vec_model = Word2Vec.load(NODE2VEC_MODEL_PATH)
   print(f"✅ Loaded Node2Vec model from '{NODE2VEC_MODEL_PATH}'")
except Exception as e:
   raise FileNotFoundError(f"❌ Could not load Node2Vec model: {e}")


# ———————————————————————
# 1.5 LOAD GRAPH WITH EMBEDDINGS 👈 FIXED
# ———————————————————————
try:
    G = joblib.load(GRAPH_PATH)
    print(f"✅ Loaded enriched graph with {G.number_of_nodes():,} nodes")
except FileNotFoundError:
    raise FileNotFoundError("❌ Enriched graph not found. Run train_node2vec_full_fast.py first.")

# ———————————————————————
# 2. EXTRACT CUSTOMER EMBEDDINGS
# ———————————————————————
print("🔍 Extracting customer embeddings from G...")
customer_nodes = []
embeddings = []


for node_id, data in G.nodes(data=True):
   if data.get('label') == 'Customer':
       str_node = str(node_id)
       if str_node in node2vec_model.wv:
           customer_nodes.append(node_id)
           embeddings.append(node2vec_model.wv[str_node])
       else:
           print(f"⚠️ No embedding for customer: {node_id}")


if len(embeddings) == 0:
   raise ValueError("❌ No customer embeddings found. Check Node2Vec training.")


X = np.array(embeddings)
print(f"✅ Extracted embeddings for {len(customer_nodes)} customers.")


# ———————————————————————
# 3. FIT KNN (COSINE SIMILARITY)
# ———————————————————————
print(f"🧮 Fitting NearestNeighbors (k={TOP_K}, metric={METRIC})...")
nn_model = NearestNeighbors(n_neighbors=TOP_K, metric=METRIC, algorithm='brute')
nn_model.fit(X)


distances, indices = nn_model.kneighbors(X)


# ———————————————————————
# 4. BUILD SIMILARITY GRAPH
# ———————————————————————
G_similarity = nx.Graph()
G_similarity.add_nodes_from(customer_nodes)


print("🔗 Adding edges based on KNN similarity...")
for i in range(len(customer_nodes)):
   source = customer_nodes[i]
   for j in range(TOP_K):
       neighbor_idx = indices[i, j]
       target = customer_nodes[neighbor_idx]
       if source != target:
           similarity = 1 - distances[i, j]  # Convert distance → similarity [0,1]
           G_similarity.add_edge(source, target, weight=similarity, similarity=similarity)


# ———————————————————————
# 5. SAVE & VERIFY
# ———————————————————————
nx.write_graphml(G_similarity, SIMILARITY_GRAPH_OUTPUT)
joblib.dump(G_similarity, "models/G_similarity.pkl")
print(f"✅ Saved similarity graph to: {SIMILARITY_GRAPH_OUTPUT}")


print(f"\n✅ KNN Similarity Graph Recalibrated:")
print(f"   Nodes: {G_similarity.number_of_nodes()}")
print(f"   Edges: {G_similarity.number_of_edges()}")
print(f"   Top-K: {TOP_K}, Metric: {METRIC}")
print(f"   Max similarity: {np.max([d['similarity'] for u, v, d in G_similarity.edges(data=True)]) if G_similarity.edges() else 0:.3f}")


# ———————————————————————
# 6. Export for downstream use
print(f"💾 Saved to models/: G_similarity.pkl")
