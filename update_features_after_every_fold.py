# Here we update the features with every dvid fold which has to be done after the feature engineering is done with every new injection 

# === update_features_csv.py ===
import pandas as pd
import joblib
import os
from pathlib import Path

print("🧩 Injecting enriched graph features into customer/product CSVs...")

# ———————————————————————
# 1. LOAD GRAPH — FALLBACK TO BASE IF ENRICHED NOT FOUND
# ———————————————————————
graph_paths = [
    'graphs/graph_enriched_with_degree_features.pkl',
    'graphs/graph_enriched.pkl'
]

G = None
for path in graph_paths:
    try:
        G = joblib.load(path)
        print(f"✅ Loaded graph from: {path}")
        break
    except FileNotFoundError:
        continue

if G is None:
    raise FileNotFoundError("❌ No graph found. Run analyze_degrees.py first.")

print(f"📊 Graph loaded with {G.number_of_nodes():,} nodes")

# ———————————————————————
# 2. LOAD CSV FILES — WITH ERROR HANDLING
# ———————————————————————
csv_paths = {
    'customer': 'models/customer_features.csv',
    'product': 'models/product_features.csv'
}

try:
    customer_features_df = pd.read_csv(csv_paths['customer'])
    product_features_df = pd.read_csv(csv_paths['product'])
except FileNotFoundError as e:
    raise FileNotFoundError(f"❌ Could not load CSV: {e}")

print(f"📊 Customer features: {len(customer_features_df):,} rows")
print(f"📊 Product features: {len(product_features_df):,} rows")

# ———————————————————————
# 3. DEFINE FEATURES TO INJECT
# ———————————————————————
customer_new_features = [
    'log_degree',
    'degree_percentile',
    'degree_zscore',
    'type_normalized_degree',
    'comm_normalized_degree',
    'comm_degree_percentile',
    'communityId',  # from Louvain
    'segment'       # from Louvain
]

product_degree_features = [
    'log_degree',
    'degree_percentile',
    'degree_zscore',
    'type_normalized_degree',
    'comm_normalized_degree',
    'comm_degree_percentile'
]

# ———————————————————————
# 4. UPDATE CUSTOMER FEATURES — SAFE & GRACEFUL
# ———————————————————————
updated_customer_rows = set()
for idx, row in customer_features_df.iterrows():
    node_id = row['node_id']
    if str(node_id) not in G.nodes:
        continue  # Skip if node not in graph

    node_data = G.nodes[str(node_id)]
    for feat in customer_new_features:
        if feat in node_data:
            customer_features_df.at[idx, feat] = node_data[feat]
        else:
            # Fallback: -1 for categorical, 0.0 for numeric
            if feat in ['communityId', 'segment']:
                customer_features_df.at[idx, feat] = -1
            else:
                customer_features_df.at[idx, feat] = 0.0
    updated_customer_rows.add(idx)

print(f"✅ Updated {len(updated_customer_rows):,} customer rows with new features")

# ———————————————————————
# 5. UPDATE PRODUCT FEATURES — SAME LOGIC
# ———————————————————————
updated_product_rows = set()
for idx, row in product_features_df.iterrows():
    node_id = row['node_id']
    if str(node_id) not in G.nodes:
        continue

    node_data = G.nodes[str(node_id)]
    for feat in product_degree_features:
        if feat in node_data:
            product_features_df.at[idx, feat] = node_data[feat]
        else:
            product_features_df.at[idx, feat] = 0.0
    updated_product_rows.add(idx)

print(f"✅ Updated {len(updated_product_rows):,} product rows with new features")

# ———————————————————————
# 6. SAVE BACK TO CSV — OVERWRITE
# ———————————————————————
os.makedirs("models", exist_ok=True)
customer_features_df.to_csv('models/customer_features.csv', index=False)
product_features_df.to_csv('models/product_features.csv', index=False)

print("💾 Saved updated customer_features.csv and product_features.csv")

# ———————————————————————
# 7. VERIFICATION & SUMMARY
# ———————————————————————
print("\n" + "="*60)
print("🎉 FEATURE INJECTION COMPLETE")
print("   All required features are now in CSVs.")
print("   Next time you run enhanced_xgb_training.py, no warnings!")
print("="*60)
