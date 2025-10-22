# Here I have the top k recommandations for the no discount one 

# === xgb_temporal_diverse_fixed.py ===
# Fixed version matching your 134-feature XGBoost model
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import xgboost as xgb
import random
import os


print("🎯 Starting: XGBoost TEMPORAL + DIVERSE Recommendations (134 features)")


# ———————————————————————
# 1. LOAD REQUIRED ARTIFACTS
# ———————————————————————
try:
   G = joblib.load('graphs/graph_enriched.pkl')
   print(f"✅ Loaded enriched graph with {G.number_of_nodes()} nodes")
except:
   try:
       G = joblib.load('graphs/graph.pkl')
       print(f"✅ Loaded basic graph with {G.number_of_nodes()} nodes")
   except:
       raise FileNotFoundError("❌ No graph found. Run graph building scripts first.")


try:
   customer_features_df = pd.read_csv('models/customer_features.csv')
   product_features_df = pd.read_csv('models/product_features.csv')
   print(f"✅ Loaded customer features: {len(customer_features_df)}")
   print(f"✅ Loaded product features: {len(product_features_df)}")
except:
   raise FileNotFoundError("❌ Run feature generation script first.")


# Load XGBoost model
try:
   xgb_model = xgb.Booster()
   xgb_model.load_model('models/xgb_for_recommendations.json')
   print("✅ Loaded XGBoost model")

   class XGBWrapper:
       def predict_proba(self, X):
           if X.ndim == 1:
               X = X.reshape(1, -1)
           dmat = xgb.DMatrix(X)
           pred = self.model.predict(dmat)
           return np.column_stack([1 - pred, pred])

   xgb_wrapper = XGBWrapper()
   xgb_wrapper.model = xgb_model

except Exception as e:
   raise FileNotFoundError(f"❌ Could not load XGBoost model: {e}")


# ———————————————————————
# 2. DIAGNOSE TIMESTAMPS
# ———————————————————————
def diagnose_timestamps(G):
   """Check what timestamps actually exist in the graph"""
   timestamps = []
   for u, v, data in G.edges(data=True):
       if 'timestamp' in data and data['timestamp'] is not None:
           ts = data['timestamp']
           if isinstance(ts, datetime):
               timestamps.append(ts)
           elif isinstance(ts, str):
               try:
                   timestamps.append(datetime.fromisoformat(ts.split()[0]))
               except:
                   pass
           elif isinstance(ts, pd.Timestamp):
               timestamps.append(ts.to_pydatetime())

   if timestamps:
       print(f"📊 Timestamp range: {min(timestamps)} to {max(timestamps)}")
       print(f"📊 Total timestamps: {len(timestamps)}")
       print(f"📊 Unique dates: {len(set(ts.date() for ts in timestamps))}")
   else:
       print("⚠️ No valid timestamps found in graph!")
   return timestamps


# Diagnose timestamps
print("\n🔍 Diagnosing timestamps in graph...")
timestamps_found = diagnose_timestamps(G)


# ———————————————————————
# 3. CONFIGURATION
# ———————————————————————
embedding_dim = 64  # From your Node2Vec training
print(f"✅ Using embedding dimension: {embedding_dim}")


# ———————————————————————
# 4. TEMPORAL FEATURES (EXACT COPY FROM YOUR TRAINING CODE)
# ———————————————————————
def calculate_temporal_features(G, node, current_time):
   """Calculate temporal features - EXACT COPY from training code"""
   times = []
   for _, _, data in G.edges(node, data=True):
       if 'timestamp' in data and data['timestamp'] is not None:
           ts = data['timestamp']
           # Handle different timestamp formats
           if isinstance(ts, str):
               try:
                   if ' ' in ts:
                       ts = datetime.fromisoformat(ts.split()[0])
                   else:
                       ts = datetime.fromisoformat(ts)
               except:
                   continue
           elif isinstance(ts, pd.Timestamp):
               ts = ts.to_pydatetime()

           if isinstance(ts, datetime) and ts <= current_time:
               times.append(ts)

   if not times:
       return {'recency': 999, 'frequency': 0}
   latest = max(times)
   recency = (current_time - latest).days
   frequency = len(times)
   return {'recency': recency, 'frequency': frequency}


# ———————————————————————
# 5. FEATURE EXTRACTION (EXACT MATCH TO TRAINING - 134 FEATURES)
# ———————————————————————
def get_link_features(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time):
   """Extract features - EXACT MATCH to training (134 features)"""
   try:
       c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
       p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
   except (IndexError, KeyError):
       # Return exactly 134 features (2 + 128 + 4)
       return np.zeros(134)


   # Degrees (2 features)
   customer_degree = float(c_row.get('degree', 0))
   product_degree = float(p_row.get('degree', 0))


   # Embeddings (128 features - 64 + 64)
   try:
       c_emb = [float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
       p_emb = [float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
   except:
       c_emb = [0.0] * emb_dim
       p_emb = [0.0] * emb_dim


   # Temporal features (4 features)
   c_temp = calculate_temporal_features(G, cust_id, current_time)
   p_temp = calculate_temporal_features(G, prod_id, current_time)


   # Build feature vector - EXACTLY 134 features:
   # 2 (degrees) + 128 (embeddings) + 4 (temporal) = 134
   features = np.array([
       customer_degree, product_degree,    # 2 features
       *c_emb, *p_emb,                     # 128 features (64 + 64)
       c_temp['recency'], c_temp['frequency'],  # 2 features
       p_temp['recency'], p_temp['frequency']   # 2 features
   ])

   # Double-check feature count
   if len(features) != 134:
       print(f"⚠️ Feature count mismatch: got {len(features)}, expected 134")
       if len(features) < 134:
           features = np.pad(features, (0, 134 - len(features)), constant_values=0.0)
       else:
           features = features[:134]

   return features


# ———————————————————————
# 6. GET VALID PRODUCTS (EXACT COPY)
# ———————————————————————
def get_valid_products(G):
   """Get valid products - EXACT COPY from RF code"""
   valid_products = []
   for node, data in G.nodes(data=True):
       if data.get('label') == 'Product':
           if (
               data.get('color') not in [None, 'N/A', 'Unknown', ''] and
               data.get('size') not in [None, 'N/A', 'Unknown', ''] and
               data.get('category') not in [None, 'N/A', 'Unknown', ''] and
               data.get('stock', 0) > 0
           ):
               valid_products.append(node)
   return valid_products


valid_products = get_valid_products(G)
print(f"✅ Filtered {len(valid_products)} valid products.")


# ———————————————————————
# 7. SOFTMAX SAMPLING RECOMMENDATION ENGINE
# ———————————————————————
def get_recommendations_for_customer(G, customer_id, current_time, top_n=3, temperature=2.5):
   """Recommendation engine with softmax sampling"""
   if customer_id not in G:
       return []


   # Get products customer hasn't bought
   purchased_products = set()
   for _, order_id, data in G.edges(customer_id, data=True):
       if data.get('type') == 'PURCHASED' and G.nodes[order_id].get('label') == 'Order':
           for _, product_id, order_data in G.edges(order_id, data=True):
               if order_data.get('type') == 'CONTAINS' and G.nodes[product_id].get('label') == 'Product':
                   purchased_products.add(product_id)


   potential_products = [p for p in valid_products if p not in purchased_products]


   if len(potential_products) == 0:
       return []


   # Calculate features for all potential products
   product_features = []
   for product_id in potential_products:
       link_features = get_link_features(
           customer_id, product_id,
           customer_features_df, product_features_df,
           embedding_dim, current_time  # Use the specific current_time
       )
       product_features.append((product_id, link_features))


   if not product_features:
       return []


   product_ids, feature_vectors = zip(*product_features)
   feature_vectors = np.array(feature_vectors)


   # Debug: Check feature dimensions
   print(f"🔍 Feature vectors shape: {feature_vectors.shape}")


   # Predict probabilities using XGBoost
   try:
       probs = xgb_wrapper.predict_proba(feature_vectors)[:, 1]
   except Exception as e:
       print(f"❌ Error in prediction: {e}")
       print(f"Feature shape: {feature_vectors.shape}")
       return []


   # Apply softmax with temperature for diversity
   logits = np.log(np.clip(probs, 1e-8, 1 - 1e-8))
   logits_scaled = logits / temperature
   exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
   sampling_probs = exp_logits / exp_logits.sum()


   # Sample with category diversity
   seen_categories = set()
   selected = []
   attempts = 0
   max_attempts = top_n * 50


   while len(selected) < top_n and attempts < max_attempts:
       if len(product_ids) == 0:
           break
       idx = np.random.choice(len(product_ids), p=sampling_probs)
       prod_id = product_ids[idx]
       prob = probs[idx]
       category = str(G.nodes[prod_id].get('category', 'Unknown')).upper()


       if category in seen_categories or G.nodes[prod_id].get('stock', 0) <= 0:
           attempts += 1
           continue


       selected.append((prod_id, prob))
       seen_categories.add(category)
       attempts += 1


   # Fallback to top items if sampling didn't work
   if len(selected) < top_n:
       fallback = sorted([(p, pr) for p, pr in zip(product_ids, probs) if G.nodes[p].get('stock', 0) > 0],
                         key=lambda x: x[1], reverse=True)
       for prod_id, prob in fallback:
           if len(selected) >= top_n:
               break
           cat = str(G.nodes[prod_id].get('category', '')).upper()
           if cat not in seen_categories:
               selected.append((prod_id, prob))
               seen_categories.add(cat)


   return selected


# ———————————————————————
# 8. ENRICH RECOMMENDATIONS (NO DISCOUNTS)
# ———————————————————————
def enrich_recommendations(G, recommendations):
   """Add human-readable information"""
   enriched_recs = []
   for prod_id, prob in recommendations:
       data = G.nodes[prod_id]
       enriched_recs.append({
           'product_id': prod_id,
           'probability': round(prob, 4),
           'color': data.get('color', 'N/A'),
           'size': data.get('size', 'N/A'),
           'category': data.get('category', 'N/A'),
           'stock': data.get('stock', 0),
           'message': f"🎯 You might like this {data.get('color', 'colorful')} {data.get('size', 'fit')} {data.get('category', 'item')}!"
       })
   return enriched_recs


# ———————————————————————
# 9. GET CUSTOMER DISPLAY NAME
# ———————————————————————
def get_display_name(G, cust_id):
   """Get readable customer name"""
   data = G.nodes[cust_id]
   if data.get('source') == 'international':
       return data.get('name', cust_id)
   elif data.get('source') == 'amazon':
       city = data.get('city', 'Unknown')
       country = data.get('country', 'Unknown')
       return f"{city}, {country}"
   else:
       return cust_id


# ———————————————————————
# 10. RUN SIMULATION WITH FIXED DATES
# ———————————————————————
sample_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer'][:3]


# Use the same fixed dates as in your working example
simulation_dates = [
   datetime(2022, 3, 1),
   datetime(2022, 7, 15),
   datetime(2022, 12, 1)
]


print("\n" + "="*80)
print("🎯 XGBOOST TEMPORAL RECOMMENDATIONS WITH DIVERSITY (134 FEATURES)")
print("   → Changing current_time changes recency/frequency → reshuffling occurs")
print("="*80)


for customer_id in sample_customers:
   display_name = get_display_name(G, customer_id)
   print(f"\n🧑 Customer: {display_name}")
   print("─" * 80)


   for sim_date in simulation_dates:
       print(f"\n📅 {sim_date.strftime('%Y-%m-%d')}")

       # Debug: Show temporal features
       if valid_products:
           debug_prods = valid_products[:2]
           for i, prod_id in enumerate(debug_prods, 1):
               c_temp = calculate_temporal_features(G, customer_id, sim_date)
               p_temp = calculate_temporal_features(G, prod_id, sim_date)
               print(f"🔍 Temporal Debug: Cust Recency={c_temp['recency']}, Freq={c_temp['frequency']} | Prod Recency={p_temp['recency']}, Freq={p_temp['frequency']}")


       # Get recommendations with softmax sampling
       recommendations = get_recommendations_for_customer(G, customer_id, sim_date, top_n=3, temperature=2.5)

       if not recommendations:
           print("  ⚠️ No recommendations available.")
           continue


       # Enrich and display
       enriched_recs = enrich_recommendations(G, recommendations)
       for i, rec in enumerate(enriched_recs, 1):
           print(f"  {i}. {rec['color']} {rec['size']} {rec['category']} (P={rec['probability']})")
           print(f"     📦 Stock: {rec['stock']} | {rec['message']}")


print("\n" + "="*80)
print("🎉 XGBOOST TEMPORAL + DIVERSE RECOMMENDATIONS COMPLETE")
print("="*80)
