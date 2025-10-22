# Split of the data in different communities
# === louvain_communities.py ===
# Run Louvain on customer similarity graph and update G
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import joblib
import os

print("🧩 Recalibrating Louvain community detection on UPDATED customer similarity graph...")

# Load the UPDATED graph:
try:
    G = joblib.load('graphs/graph_enriched.pkl')  # ← Load updated graph
    print(f"✅ Loaded updated graph with {G.number_of_nodes():,} nodes")
except FileNotFoundError:
    try:
        G = joblib.load('graphs/graph_enriched_dvid2.pkl')  # ← Fallback
        print(f"✅ Loaded DVID-2 graph with {G.number_of_nodes():,} nodes")
    except FileNotFoundError:
        try:
            G = joblib.load('graphs/graph.pkl')
            print(f"⚠️  Loaded old graph with {G.number_of_nodes():,} nodes")
        except FileNotFoundError:
            raise FileNotFoundError("❌ No graph found. Run graph injection first.")

# ———————————————————————
# CONFIG
# ———————————————————————
G_SIMILARITY_PATH = "models/G_similarity.pkl"
COMMUNITY_OUTPUT = "models/customer_partition.pkl"
GRAPH_UPDATE = True  # Update original G with community IDs
VISUALIZE_LIMIT = 200  # Only visualize if < 200 nodes

# ———————————————————————
# 1. LOAD SIMILARITY GRAPH
# ———————————————————————
try:
    G_similarity = joblib.load(G_SIMILARITY_PATH)
    print(f"✅ Loaded G_similarity from '{G_SIMILARITY_PATH}'")
except Exception as e:
    raise FileNotFoundError(f"❌ Could not load G_similarity: {e}")

if G_similarity.number_of_nodes() == 0:
    raise ValueError("❌ G_similarity is empty. Run knn_customer_similarity.py first.")

if G_similarity.number_of_edges() == 0:
    raise ValueError("❌ G_similarity has no edges. Check TOP_K or embedding quality.")

print(f"📊 G_similarity: {G_similarity.number_of_nodes()} nodes, {G_similarity.number_of_edges()} edges")

# ———————————————————————
# 2. RUN LOUVAIN
# ———————————————————————
print("🔍 Running Louvain with 'similarity' edge weights...")
partition = community_louvain.best_partition(
    G_similarity,
    weight='similarity',
    random_state=42
)
num_communities = len(set(partition.values()))

print(f"✅ Louvain complete. Found {num_communities} distinct communities.")

# ———————————————————————
# 3. SAVE PARTITION
# ———————————————————————
joblib.dump(partition, COMMUNITY_OUTPUT)
print(f"💾 Saved community partition to: {COMMUNITY_OUTPUT}")

# ———————————————————————
# 4. UPDATE ORIGINAL GRAPH G + SAVE IT TO DISK
# ———————————————————————
if GRAPH_UPDATE and 'G' in globals():
    print("🏷️  Updating Customer nodes in G with new community IDs...")
    updated_count = 0
    for node_id, comm_id in partition.items():
        if node_id in G and G.nodes[node_id].get('label') == 'Customer':
            G.nodes[node_id]['communityId'] = int(comm_id)
            G.nodes[node_id]['segment'] = f'Segment-{comm_id}'
            updated_count += 1
    print(f"✅ Updated {updated_count} customers in G.")

    # ✅ CRITICAL: SAVE THE UPDATED GRAPH BACK TO DISK
    joblib.dump(G, 'graphs/graph_enriched.pkl')
    print("💾 Saved UPDATED graph with community IDs to: graphs/graph_enriched.pkl")
else:
    print("⚠️  Original graph G not available for update.")

# ———————————————————————
# 5. VISUALIZE (if small)
# ———————————————————————
if G_similarity.number_of_nodes() < VISUALIZE_LIMIT:
    print("🎨 Visualizing community structure...")
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_similarity, seed=42, k=3, iterations=50)

    # Color by community
    colors = [partition[node] for node in G_similarity.nodes()]
    cmap = plt.cm.viridis
    nx.draw_networkx_nodes(G_similarity, pos, node_size=150, node_color=colors, cmap=cmap, alpha=0.9)
    nx.draw_networkx_edges(G_similarity, pos, alpha=0.3, edge_color='gray', width=0.5)

    plt.title(f"Louvain Communities (k={len(set(partition.values()))})\nCustomer Similarity Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print(f"📊 Graph too large to visualize ({G_similarity.number_of_nodes()} nodes).")

# ———————————————————————
# 6. SUMMARY
# ———————————————————————
print("\n" + "="*60)
print("🧩 LOUVAIN COMPLETE")
print(f"   Communities found: {num_communities}")
print(f"   Average community size: {len(G_similarity.nodes()) / num_communities:.1f}")
print("="*60)
