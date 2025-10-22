# Initial node2vec algorithm inscription of the embeddings of the graph that capture the topological relations which will be used for further feature engineering
# === train_node2vec_robust.py === 6th step
# Leak-proof, noise-robust Node2Vec with time-safe walks
import networkx as nx
import numpy as np
import os
from node2vec import Node2Vec
import joblib
import pandas as pd
from datetime import datetime

print("🎯 Starting ROBUST, LEAK-PROOF Node2Vec...")

# === train_node2vec_robust.py ===
# UPDATE the graph loading section:

# 1. LOAD GRAPH
try:
    G = joblib.load('graphs/graph_enriched.pkl')  # ← This should now load 57K nodes
    print(f"✅ Loaded graph with {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
except FileNotFoundError:
    try:
        G = joblib.load('graphs/graph_enriched_dvid2.pkl')  # ← Fallback
        print(f"✅ Loaded DVID-2 graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    except FileNotFoundError:
        try:
            G = joblib.load('graphs/graph.pkl')
            print(f"⚠️  Loaded old graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        except FileNotFoundError:
            raise FileNotFoundError("❌ No graph found. Run build_graph_final.py first.")

# ———————————————————————
# CONFIG
# ———————————————————————
TRAINING_CUTOFF_STR = '2022-12-31'  # Must match ingestion
TRAINING_CUTOFF_TIME = datetime.fromisoformat(TRAINING_CUTOFF_STR)
DVID = 1

# Parameters
WALK_LENGTH = 30
NUM_WALKS = 200
DIMENSIONS = 64
WINDOW = 10
P = 1.0
Q = 0.5
WORKERS = 4
EPOCHS = 10
MIN_COUNT = 1

# Output
MODEL_OUTPUT = "models/node2vec_model.pkl"
os.makedirs("models", exist_ok=True)

# ———————————————————————
# 2. PRUNE INACTIVE NODES
# ———————————————————————
def prune_inactive_nodes(G, cutoff_time, max_inactive_days=730):
    print(f"✂️  Pruning nodes inactive for > {max_inactive_days} days...")
    remove_nodes = []
    cutoff = cutoff_time.date()  # Compare to date

    for node, data in G.nodes(data=True):
        if data.get('label') == 'Order':
            continue

        # Extract timestamps (now strings)
        timestamps = []
        for _, _, d in G.edges(node, data=True):
            ts = d.get('timestamp')
            if isinstance(ts, str):
                try:
                    t_date = datetime.fromisoformat(ts.split()[0]).date()
                    if t_date <= cutoff:
                        timestamps.append(t_date)
                except:
                    continue

        if not timestamps:
            remove_nodes.append(node)
        else:
            last_active = max(ts for ts in timestamps)
            if (cutoff - last_active).days > max_inactive_days:
                remove_nodes.append(node)

    G.remove_nodes_from(remove_nodes)
    print(f"✂️  Removed {len(remove_nodes)} inactive nodes")
    return G

G_pruned = prune_inactive_nodes(G, TRAINING_CUTOFF_TIME)

# ———————————————————————
# 3. ADD STRUCTURAL NOISE (Edge Dropout)
# ———————————————————————
def add_edge_dropout(G, drop_rate=0.15):
    print(f"🧱 Adding structural noise: dropping {drop_rate*100:.0f}% of edges...")
    G_noisy = G.copy()

    # Only drop edges from dvid=1
    edges_to_drop = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get('dvid') == DVID
    ]

    if len(edges_to_drop) == 0:
        print("⚠️ No edges with dvid=1 found.")
        return G_noisy

    n_drop = int(len(edges_to_drop) * drop_rate)
    if n_drop == 0:
        print("🧱 No edges to drop (drop_rate too low)")
        return G_noisy

    # Randomly select edges to remove
    indices = np.random.choice(len(edges_to_drop), size=n_drop, replace=False)
    edges_to_remove = [edges_to_drop[i] for i in indices]

    G_noisy.remove_edges_from(edges_to_remove)
    print(f"🧱 Removed {len(edges_to_remove)} edges for robustness")
    return G_noisy

G_noisy = add_edge_dropout(G_pruned, drop_rate=0.15)

# ———————————————————————
# 4. FILTER GRAPH TO TRAINING CUTOFF
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

G_filtered = filter_graph_by_time(G_noisy, TRAINING_CUTOFF_STR)

# ———————————————————————
# 5. TRAIN NODE2VEC
# ———————————————————————
print("🧱 Generating biased random walks (p=1.0, q=0.5)...")
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
model = node2vec.fit(
    vector_size=DIMENSIONS,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=1,
    workers=WORKERS,
    epochs=EPOCHS
)

# ———————————————————————
# 6. SAVE MODEL & EMBEDDINGS
# ———————————————————————
model.save(MODEL_OUTPUT)
print(f"✅ Saved Node2Vec model: {MODEL_OUTPUT}")

# Extract embeddings
node_embeddings = {}
for node in G_filtered.nodes():
    str_node = str(node)
    if str_node in model.wv:
        node_embeddings[node] = model.wv[str_node]
    else:
        print(f"⚠️ No embedding for node {node}")

print(f"✅ Generated embeddings for {len(node_embeddings)} nodes.")

# Save embeddings dictionary
joblib.dump(node_embeddings, "models/node_embeddings.pkl")
print("✅ Saved node embeddings dictionary")

# ———————————————————————
# 7. VERIFY
# ———————————————————————
print("\n" + "="*60)
print("📊 NODE2VEC TRAINING COMPLETE")
print(f"   Final graph: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
print(f"   Embedding dimensions: {DIMENSIONS}")
print(f"   Training cutoff: {TRAINING_CUTOFF_STR}")
print(f"   Data Version ID: {DVID}")
print("="*60)

# Example embedding
sample_node = list(G_filtered.nodes())[0]
if sample_node in node_embeddings:
    print(f"\n🔍 Example embedding (first 10 dims) for '{sample_node}':")
    print(node_embeddings[sample_node][:10])
else:
    print(f"\n🔍 No embedding for sample node '{sample_node}'")
