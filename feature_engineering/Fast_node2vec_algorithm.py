# this is a fast version of the node2vec algorithm for when the data gets big and training the other one would take too much time 

# === train_node2vec_full_fast.py ===
import networkx as nx
import numpy as np
import os
from node2vec import Node2Vec
import joblib
import pandas as pd
from datetime import datetime
import multiprocessing

print("🎯 Starting PRODUCTION-GRADE Node2Vec Training (Scalable & Robust)...")

# ———————————————————————
# CONFIG - OPTIMIZED FOR LARGE SPARSE GRAPHS
# ———————————————————————
TRAINING_CUTOFF_STR = '2022-12-31'
DVID = 4  # 👈 CHANGED TO 2 — and now it matters!

# 🚀 PRODUCTION-GRADE PARAMETERS — BALANCED SPEED + QUALITY
WALK_LENGTH = 80
NUM_WALKS = 20
DIMENSIONS = 128
WINDOW = 10
P = 0.5
Q = 2.0
WORKERS = min(8, multiprocessing.cpu_count())
EPOCHS = 5
MIN_COUNT = 1

# Output — VERSIONED BY DVID 👇
MODEL_OUTPUT = f"models/node2vec_model_dvid{DVID}.pkl"
EMBEDDINGS_OUTPUT = f"models/node_embeddings_dvid{DVID}.pkl"
GRAPH_OUTPUT = f'graphs/graph_enriched_with_embeddings_dvid{DVID}.pkl'
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# ———————————————————————
# 1. LOAD GRAPH
# ———————————————————————
try:
    G = joblib.load('graphs/graph_enriched.pkl')
    print(f"✅ Loaded graph with {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
except FileNotFoundError:
    raise FileNotFoundError("❌ No graph found. Run build_graph_final.py first.")

# ———————————————————————
# 2. GRAPH PROCESSING
# ———————————————————————
def filter_graph_by_time(G, cutoff_str):
    print(f"🕒 Filtering graph to data ≤ {cutoff_str}...")
    G_filtered = G.copy()
    cutoff_date = datetime.fromisoformat(cutoff_str).date()

    edges_to_remove = []
    for u, v, d in G.edges(data=True):
        ts = d.get('timestamp')
        if isinstance(ts, str):
            try:
                event_date = datetime.fromisoformat(ts.split()[0]).date()
                if event_date > cutoff_date:
                    edges_to_remove.append((u, v))
            except:
                continue

    G_filtered.remove_edges_from(edges_to_remove)
    print(f"🕒 Removed {len(edges_to_remove)} future edges")
    return G_filtered

G_filtered = filter_graph_by_time(G, TRAINING_CUTOFF_STR)
print(f"📊 Final graph for training: {G_filtered.number_of_nodes():,} nodes, {G_filtered.number_of_edges():,} edges")

# ———————————————————————
# 3. TRAIN NODE2VEC
# ———————————————————————
print("🧱 Generating biased random walks (PRODUCTION-GRADE)...")
print(f"   Using {WORKERS} worker processes...")

node2vec = Node2Vec(
    graph=G_filtered,
    dimensions=DIMENSIONS,
    walk_length=WALK_LENGTH,
    num_walks=NUM_WALKS,
    p=P,
    q=Q,
    workers=WORKERS,
    quiet=False
)

print("🧠 Training embedding model (skip-gram)...")
print("⚠️  Training on FULL GRAPH — this may take 15-30 minutes (worth it)...")
model = node2vec.fit(
    vector_size=DIMENSIONS,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=1,
    workers=WORKERS,
    epochs=EPOCHS
)

# ———————————————————————
# 4. SAVE MODEL & EMBEDDINGS
# ———————————————————————
model.save(MODEL_OUTPUT)
print(f"✅ Saved Node2Vec model: {MODEL_OUTPUT}")

node_embeddings = {}
for node in G_filtered.nodes():
    str_node = str(node)
    if str_node in model.wv:
        node_embeddings[node] = model.wv[str_node]
    else:
        node_embeddings[node] = np.zeros(DIMENSIONS)

print(f"✅ Generated embeddings for {len(node_embeddings):,} nodes.")
joblib.dump(node_embeddings, EMBEDDINGS_OUTPUT)
print(f"✅ Saved node embeddings: {EMBEDDINGS_OUTPUT}")

# ———————————————————————
# 5. INJECT EMBEDDINGS INTO GRAPH
# ———————————————————————
print("💾 Injecting embeddings back into graph nodes...")
for node, emb in node_embeddings.items():
    for i in range(DIMENSIONS):
        G_filtered.nodes[node][f'embedding_{i}'] = float(emb[i])

joblib.dump(G_filtered, GRAPH_OUTPUT)  # 👈 VERSIONED FILENAME
print(f"✅ Saved graph with embeddings: {GRAPH_OUTPUT}")

# ———————————————————————
# 6. VERIFY RESULTS
# ———————————————————————
print("\n" + "="*60)
print("📊 PRODUCTION-GRADE NODE2VEC TRAINING COMPLETE")
print(f"   Final graph: {G_filtered.number_of_nodes():,} nodes, {G_filtered.number_of_edges():,} edges")
print(f"   Embedding dimensions: {DIMENSIONS}")
print(f"   Walk length: {WALK_LENGTH}, Num walks: {NUM_WALKS}")
print(f"   p={P}, q={Q} → optimized for sparse, community-rich graphs")
print(f"   Training cutoff: {TRAINING_CUTOFF_STR}")
print(f"   Data Version ID: {DVID}")  # 👈 Now meaningful!
print("="*60)
