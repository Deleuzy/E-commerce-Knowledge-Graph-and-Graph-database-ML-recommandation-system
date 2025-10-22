# Here we train the first variant of the xgbooster model in order to be able to use it to provide top k recommandations 

# === xgboost_temporal_for_recommendations.py ===
# Train XGBoost EXACTLY like the RF model for reshuffling recommendations
import networkx as nx
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from datetime import datetime
import random
import os


print("🚀 Training XGBoost for TEMPORAL RECOMMENDATIONS (Like RF Model)")


# ———————————————————————
# 1. CREATE OUTPUT FOLDER
# ———————————————————————
os.makedirs("models", exist_ok=True)
print("✅ Created 'models/' directory")


# ———————————————————————
# 2. LOAD ENRICHED GRAPH & FEATURES
# ———————————————————————
try:
   G = joblib.load('graphs/graph_enriched.pkl')
   print("✅ Loaded enriched graph")
except FileNotFoundError:
   raise FileNotFoundError("❌ Run category_enrich.py first")


try:
   node2vec_model = joblib.load('models/node2vec_model.pkl')
   print("✅ Loaded Node2Vec model")
except FileNotFoundError:
   raise FileNotFoundError("❌ Run train_node2vec_robust.py first")


customer_features_df = pd.read_csv('models/customer_features.csv')
product_features_df = pd.read_csv('models/product_features.csv')
embedding_dim = 64


print(f"✅ Loaded {len(customer_features_df)} customer features")
print(f"✅ Loaded {len(product_features_df)} product features")


# ———————————————————————
# 3. STANDARDIZE TIMESTAMPS
# ———————————————————————
print("🧹 Standardizing timestamps...")


for u, v, data in G.edges(data=True):
   if 'timestamp' in data:
       ts = data['timestamp']
       if isinstance(ts, str):
           try:
               data['timestamp'] = pd.to_datetime(ts).to_pydatetime()
           except:
               del data['timestamp']
       elif isinstance(ts, pd.Timestamp):
           data['timestamp'] = ts.to_pydatetime()
       elif not isinstance(ts, datetime):
           del data['timestamp']


# ———————————————————————
# 4. USE FIXED CURRENT_TIME (LIKE RF MODEL)
# ———————————————————————
current_time = datetime(2023, 1, 1)  # Fixed date for ALL features
print(f"📅 Using FIXED current_time for ALL features: {current_time.strftime('%Y-%m-%d')}")


# ———————————————————————
# 5. EXTRACT ALL POSITIVE LINKS (Customer → Product)
# ———————————————————————
print("🔗 Extracting all positive customer-product links...")


positive_links = set()


# Case 1: Direct Customer → Product (PURCHASED)
for u, v, data in G.edges(data=True):
   if G.nodes[u].get('label') == 'Customer' and G.nodes[v].get('label') == 'Product':
       if data.get('type') == 'PURCHASED':
           positive_links.add((u, v))


# Case 2: Through Order: Customer → Order → Product
for u, v, data in G.edges(data=True):
   if data.get('type') == 'PURCHASED' and G.nodes[v].get('label') == 'Order':
       for _, p, ed in G.edges(v, data=True):
           if ed.get('type') == 'CONTAINS' and G.nodes[p].get('label') == 'Product':
               positive_links.add((u, p))


print(f"✅ Found {len(positive_links)} positive links")


if len(positive_links) == 0:
   raise ValueError("❌ No positive links found. Check graph edges.")


# ———————————————————————
# 6. GENERATE RANDOM NEGATIVE SAMPLES
# ———————————————————————
print("📉 Generating random negative samples...")


all_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer']
all_products = [n for n, d in G.nodes(data=True) if d.get('label') == 'Product']


if not all_customers or not all_products:
   raise ValueError("❌ No customers or products found.")


negative_links = set()
max_attempts = len(positive_links) * 10
attempts = 0


while len(negative_links) < len(positive_links) and attempts < max_attempts:
   c = random.choice(all_customers)
   p = random.choice(all_products)
   if (c, p) not in positive_links:
       negative_links.add((c, p))
   attempts += 1


if len(negative_links) == 0:
   raise ValueError("❌ No negative samples generated.")


print(f"✅ Generated {len(negative_links)} negative links")


# ———————————————————————
# 7. FEATURE ENGINEERING (WITH FIXED current_time)
# ———————————————————————
def calculate_temporal_features(G, node, current_time):
   timestamps = [
       data['timestamp'] for _, _, data in G.edges(node, data=True)
       if 'timestamp' in data
       and isinstance(data['timestamp'], datetime)
       and data['timestamp'] <= current_time
   ]
   if not timestamps:
       return {'recency': 999, 'frequency': 0}
   latest = max(timestamps)
   return {
       'recency': (current_time - latest).days,
       'frequency': len(timestamps)
   }


def get_link_features(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time):
   try:
       c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
       p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
   except IndexError:
       return np.zeros(2 + 2*emb_dim + 4)  # 134 total


   # Embeddings
   c_emb = [c_row[f'embedding_{i}'] for i in range(emb_dim)]
   p_emb = [p_row[f'embedding_{i}'] for i in range(emb_dim)]


   # Category match
   cust_cat = G.nodes[cust_id].get('preferred_category', 'unknown').lower()
   prod_cat = G.nodes[prod_id].get('category', 'unknown')
   category_match = 1 if cust_cat == prod_cat else 0


   # Temporal features (USING FIXED current_time)
   temporal_feats = calculate_temporal_features(G, cust_id, current_time)


   # Popularity = product degree
   popularity = p_row['degree']


   # Build feature vector
   return np.array([
       c_row['degree'],           # 1
       p_row['degree'],           # 1
       *c_emb,                    # 64
       *p_emb,                    # 64
       category_match,            # 1
       popularity,                # 1
       temporal_feats['recency'], # 1
       temporal_feats['frequency'] # 1
   ])


# ———————————————————————
# 8. CREATE DATASET
# ———————————————————————
print("⚙️ Building dataset with FIXED current_time...")


all_links = list(positive_links) + list(negative_links)
all_labels = [1] * len(positive_links) + [0] * len(negative_links)


X_list = []
for c, p in all_links:
   features = get_link_features(c, p, customer_features_df, product_features_df, embedding_dim, current_time)
   X_list.append(features)


X = np.array(X_list)
y = np.array(all_labels)


print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")


# ———————————————————————
# 9. TRAIN-TEST SPLIT
# ———————————————————————
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"📊 X_train shape: {X_train.shape}")
print(f"📊 y_train shape: {y_train.shape}")
print(f"📊 X_test shape: {X_test.shape}")
print(f"📊 y_test shape: {y_test.shape}")


# ———————————————————————
# 10. TRAIN XGBOOST MODEL
# ———————————————————————
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


params = {
   'objective': 'binary:logistic',
   'eval_metric': 'auc',
   'max_depth': 8,
   'learning_rate': 0.1,
   'subsample': 0.8,
   'colsample_bytree': 0.8,
   'lambda': 1.0,
   'alpha': 0.1,
   'tree_method': 'hist',
   'seed': 42
}


print("🏋️ Training XGBoost model...")
model = xgb.train(
   params,
   dtrain,
   num_boost_round=100,
   evals=[(dtrain, 'train'), (dtest, 'eval')],
   early_stopping_rounds=10,
   verbose_eval=50
)


# ———————————————————————
# 9. EVALUATE
# ———————————————————————
y_pred_prob = model.predict(dtest, iteration_range=(0, model.best_iteration))
y_pred = (y_pred_prob > 0.5).astype(int)


auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


print("\n📈 Final Evaluation (Production-Ready):")
print(f"   AUC: {auc:.4f}")
print(f"   Accuracy: {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall: {rec:.4f}")


# ———————————————————————
# 12. SAVE MODEL
# ———————————————————————
model_path = "models/xgb_for_recommendations.json"
model.save_model(model_path)
joblib.dump(model, "models/xgb_for_recommendations.pkl")
print(f"✅ Model saved to '{model_path}'")


print("\n🎉 XGBoost model trained for temporal recommendations!")
print("➡️ Now use it with the recommendation engine below")
