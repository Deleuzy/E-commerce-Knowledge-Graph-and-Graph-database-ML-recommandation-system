# Here I assert an dynamic discount function and try to add it as a another feature to the calculation and I simulate 4 different dates to make it more shuffling and dynamic for production ready

# === xgb_temporal_discount_robust_fixed.py ===
# Fixed XGBoost training with temporal features + dynamic discounts (135 features)
import networkx as nx
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from datetime import datetime, timedelta
import random
import os


print("🚀 Training FIXED XGBoost with Temporal + Discount Features (135 features)")


# ———————————————————————
# 1. SETUP
# ———————————————————————
os.makedirs("models", exist_ok=True)


# Load required data
try:
   G = joblib.load('graphs/graph_enriched.pkl')
   print("✅ Loaded enriched graph")
except FileNotFoundError:
   try:
       G = joblib.load('graphs/graph.pkl')
       print("✅ Loaded basic graph")
   except:
       raise FileNotFoundError("❌ Run category_enrich.py first")


try:
   customer_features_df = pd.read_csv('models/customer_features.csv')
   product_features_df = pd.read_csv('models/product_features.csv')
   print(f"✅ Loaded customer features: {len(customer_features_df)}")
   print(f"✅ Loaded product features: {len(product_features_df)}")
except:
   raise FileNotFoundError("❌ Run feature generation script first")


embedding_dim = 64
print(f"✅ Using embedding dimension: {embedding_dim}")


# ———————————————————————
# 2. IMPROVED TIMESTAMP STANDARDIZATION
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
# 3. DYNAMIC DISCOUNT ASSIGNMENT (LIKE YOUR RF MODEL)
# ———————————————————————
def assign_dynamic_discounts_for_date(G, sim_date):
   """Assign realistic discounts based on date (like RF model)"""
   for node, data in G.nodes(data=True):
       if data.get('label') == 'Product':
           month = sim_date.month
           if month == 12:  # December: big sale
               discount = round(random.uniform(0.30, 0.50), 2)
           elif month in [6, 7, 11]:  # Summer + Black Friday
               discount = round(random.uniform(0.10, 0.20), 2)
           elif month in [1, 2, 8]:  # Clearance months
               discount = round(random.uniform(0.0, 0.05), 2)
           else:  # Regular discount
               discount = round(random.uniform(0.0, 0.05), 2)
           data['discount'] = discount


print("✅ Dynamic discount function defined.")


# ———————————————————————
# 4. EXTRACT POSITIVE LINKS (Customer → Product via Order)
# ———————————————————————
print("🔗 Extracting positive customer-product links...")


positive_links = set()
for u, v, data in G.edges(data=True):
   if data.get('type') == 'PURCHASED' and G.nodes[v].get('label') == 'Order':
       for _, p, ed in G.edges(v, data=True):
           if ed.get('type') == 'CONTAINS' and G.nodes[p].get('label') == 'Product':
               positive_links.add((u, p))


print(f"✅ Found {len(positive_links)} positive links")


if len(positive_links) == 0:
   raise ValueError("❌ No positive links found. Check graph edges.")


# ———————————————————————
# 5. GENERATE RANDOM NEGATIVE SAMPLES
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
# 6. IMPROVED TEMPORAL FEATURES
# ———————————————————————
def calculate_temporal_features(G, node, current_time):
   """Calculate temporal features"""
   timestamps = [
       data['timestamp'] for _, _, data in G.edges(node, data=True)
       if 'timestamp' in data
       and isinstance(data['timestamp'], datetime)
       and data['timestamp'] <= current_time
   ]
   if not timestamps:
       return {'recency': 999, 'frequency': 0}
   latest = max(timestamps)
   recency = (current_time - latest).days
   frequency = len(timestamps)
   return {'recency': recency, 'frequency': frequency}


# ———————————————————————
# 7. ENHANCED FEATURE EXTRACTION (EXACTLY 135 FEATURES)
# ———————————————————————
def get_link_features_with_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time):
   """Extract features - EXACTLY 135 features (134 + discount)"""
   try:
       c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
       p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
   except (IndexError, KeyError):
       return np.zeros(135)  # EXACTLY 135 features


   # Degrees (2)
   customer_degree = float(c_row.get('degree', 0))
   product_degree = float(p_row.get('degree', 0))


   # Embeddings (128 = 64 + 64)
   try:
       c_emb = [float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
       p_emb = [float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
   except:
       c_emb = [0.0] * emb_dim
       p_emb = [0.0] * emb_dim


   # Temporal features (4)
   c_temp = calculate_temporal_features(G, cust_id, current_time)
   p_temp = calculate_temporal_features(G, prod_id, current_time)


   # Discount feature (1) - THIS IS THE KEY DIFFERENCE!
   discount = float(G.nodes[prod_id].get('discount', 0.0))


   # Build feature vector - EXACTLY 135 features:
   features = np.array([
       customer_degree, product_degree,    # 2
       *c_emb, *p_emb,                     # 128 (64 + 64)
       c_temp['recency'], c_temp['frequency'],  # 2
       p_temp['recency'], p_temp['frequency'],   # 2
       discount  # ← THIS EXTRA FEATURE = 135 total
   ])


   # Ensure exactly 135 features
   if len(features) != 135:
       if len(features) < 135:
           features = np.pad(features, (0, 135 - len(features)), constant_values=0.0)
       else:
           features = features[:135]


   return features


# ———————————————————————
# 8. ROBUST TRAINING ACROSS MULTIPLE DATES
# ———————————————————————
print("📅 Training across multiple dates with dynamic discounts...")


# Multiple training dates for robustness
training_dates = [
   datetime(2022, 3, 1),   # Low discount period
   datetime(2022, 6, 15),  # Summer sale
   datetime(2022, 9, 10),  # Regular period
   datetime(2022, 12, 1)   # High discount period (Christmas)
]


all_training_data = []
all_training_labels = []


for sim_date in training_dates:
   print(f"  → Training on {sim_date.strftime('%Y-%m-%d')} (Month: {sim_date.strftime('%b')})")


   # Assign discounts for this date
   assign_dynamic_discounts_for_date(G, sim_date)


   # Create features for this date
   X_batch = []
   for c, p in (list(positive_links) + list(negative_links)):
       features = get_link_features_with_discount(
           c, p, customer_features_df, product_features_df, embedding_dim, sim_date
       )
       X_batch.append(features)


   all_training_data.extend(X_batch)
   all_training_labels.extend([1] * len(positive_links) + [0] * len(negative_links))


# Convert to arrays
X_full = np.array(all_training_data)
y_full = np.array(all_training_labels)


print(f"📊 Full training dataset: X={X_full.shape}, y={y_full.shape}")


# Add noise for regularization
noise_scale = 0.01
X_noisy = X_full + np.random.normal(0, noise_scale, X_full.shape)


# ———————————————————————
# 9. TRAIN-TEST SPLIT
# ———————————————————————
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
   X_noisy, y_full, test_size=0.2, random_state=42, stratify=y_full
)


print(f"📊 Final split - Train: {X_train.shape}, Test: {X_test.shape}")


# ———————————————————————
# 10. TRAIN XGBOOST MODEL (ROBUST PARAMETERS)
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
   'lambda': 1.0,      # L2 regularization
   'alpha': 0.1,       # L1 regularization
   'tree_method': 'hist',
   'seed': 42
}


print("🏋️ Training robust XGBoost model with discounts (135 features)...")
model = xgb.train(
   params,
   dtrain,
   num_boost_round=100,
   evals=[(dtrain, 'train'), (dtest, 'eval')],
   early_stopping_rounds=10,
   verbose_eval=25
)


# ———————————————————————
# 11. EVALUATE
# ———————————————————————
y_pred_prob = model.predict(dtest, iteration_range=(0, model.best_iteration))
y_pred = (y_pred_prob > 0.5).astype(int)


auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


print("\n📈 Final Evaluation (Robust with Discounts - 135 features):")
print(f"   AUC: {auc:.4f}")
print(f"   Accuracy: {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall: {rec:.4f}")


# ———————————————————————
# 12. SAVE MODEL
# ———————————————————————
model_path = "models/xgb_robust_discount.json"
model.save_model(model_path)
joblib.dump(model, "models/xgb_robust_discount.pkl")
print(f"✅ Model saved to '{model_path}'")


# Save feature names for reference
feature_names = (
   ['cust_degree', 'prod_degree'] +
   [f'cust_emb_{i}' for i in range(embedding_dim)] +
   [f'prod_emb_{i}' for i in range(embedding_dim)] +
   ['cust_recency', 'cust_frequency', 'prod_recency', 'prod_frequency', 'discount']
)
print(f"✅ Feature count: {len(feature_names)} (135 total)")


print("\n🎉 FIXED XGBOOST MODEL WITH DISCOUNTS TRAINED SUCCESSFULLY!")
print("➡️ Ready for recommendation engine with temporal + discount awareness")


# ———————————————————————
# 13. AUTO-TRAIN NO-DISCOUNT MODEL TOO (FOR COMPLETENESS)
# ———————————————————————
print("\n🔄 Auto-training NO-DISCOUNT model (134 features) for completeness...")


def get_link_features_no_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time):
   """Extract features - EXACTLY 134 features (NO discount)"""
   try:
       c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
       p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
   except (IndexError, KeyError):
       return np.zeros(134)  # EXACTLY 134 features (NO discount)


   # Degrees (2)
   customer_degree = float(c_row.get('degree', 0))
   product_degree = float(p_row.get('degree', 0))


   # Embeddings (128 = 64 + 64)
   try:
       c_emb = [float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
       p_emb = [float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
   except:
       c_emb = [0.0] * emb_dim
       p_emb = [0.0] * emb_dim


   # Temporal features (4)
   c_temp = calculate_temporal_features(G, cust_id, current_time)
   p_temp = calculate_temporal_features(G, prod_id, current_time)


   # Build feature vector - EXACTLY 134 features (NO discount):
   features = np.array([
       customer_degree, product_degree,    # 2
       *c_emb, *p_emb,                     # 128 (64 + 64)
       c_temp['recency'], c_temp['frequency'],  # 2
       p_temp['recency'], p_temp['frequency']   # 2
       # TOTAL: 134 features (NO discount feature!)
   ])


   # Ensure exactly 134 features
   if len(features) != 134:
       if len(features) < 134:
           features = np.pad(features, (0, 134 - len(features)), constant_values=0.0)
       else:
           features = features[:134]


   return features


# Create no-discount training data
print("⚙️ Creating no-discount training data...")


all_training_data_no_discount = []
all_training_labels_no_discount = []


for sim_date in training_dates:
   print(f"  → Creating no-discount features for {sim_date.strftime('%Y-%m-%d')}")


   # Assign discounts for this date (still needed for consistency)
   assign_dynamic_discounts_for_date(G, sim_date)


   # Create features for this date (NO DISCOUNT VERSION)
   X_batch = []
   for c, p in (list(positive_links) + list(negative_links)):
       features = get_link_features_no_discount(
           c, p, customer_features_df, product_features_df, embedding_dim, sim_date
       )
       X_batch.append(features)


   all_training_data_no_discount.extend(X_batch)
   all_training_labels_no_discount.extend([1] * len(positive_links) + [0] * len(negative_links))


# Convert to arrays
X_full_no_discount = np.array(all_training_data_no_discount)
y_full_no_discount = np.array(all_training_labels_no_discount)


print(f"📊 No-discount training dataset: X={X_full_no_discount.shape}, y={y_full_no_discount.shape}")


# Add noise for regularization
noise_scale = 0.01
X_noisy_no_discount = X_full_no_discount + np.random.normal(0, noise_scale, X_full_no_discount.shape)


# Train-test split
X_train_nd, X_test_nd, y_train_nd, y_test_nd = train_test_split(
   X_noisy_no_discount, y_full_no_discount, test_size=0.2, random_state=42, stratify=y_full_no_discount
)


# Train no-discount model
dtrain_nd = xgb.DMatrix(X_train_nd, label=y_train_nd)
dtest_nd = xgb.DMatrix(X_test_nd, label=y_test_nd)


print("🏋️ Training no-discount XGBoost model (134 features)...")
model_nd = xgb.train(
   params,
   dtrain_nd,
   num_boost_round=100,
   evals=[(dtrain_nd, 'train'), (dtest_nd, 'eval')],
   early_stopping_rounds=10,
   verbose_eval=25
)


# Evaluate no-discount model
y_pred_prob_nd = model_nd.predict(dtest_nd, iteration_range=(0, model_nd.best_iteration))
y_pred_nd = (y_pred_prob_nd > 0.5).astype(int)


auc_nd = roc_auc_score(y_test_nd, y_pred_prob_nd)
acc_nd = accuracy_score(y_test_nd, y_pred_nd)
prec_nd = precision_score(y_test_nd, y_pred_nd)
rec_nd = recall_score(y_test_nd, y_pred_nd)


print("\n📈 No-Discount Model Evaluation (134 features):")
print(f"   AUC: {auc_nd:.4f}")
print(f"   Accuracy: {acc_nd:.4f}")
print(f"   Precision: {prec_nd:.4f}")
print(f"   Recall: {rec_nd:.4f}")


# Save no-discount model
model_path_nd = "models/xgb_for_recommendations.json"
model_nd.save_model(model_path_nd)
joblib.dump(model_nd, "models/xgb_for_recommendations.pkl")
print(f"✅ No-discount model saved to '{model_path_nd}'")


print("\n🎉 BOTH MODELS TRAINED SUCCESSFULLY!")
print("   → With-discount: 135 features")
print("   → No-discount: 134 features")
print("   → Ready for DVID-3 incremental updates!")
