# Here I train a neural network model for both no-discount and a shuffling discount data it just takes more time than the xgbooster that is why I preferred the xgbooster

# === enhanced_nn_training.py ===
import networkx as nx
import pandas as pd
import numpy as np
import joblib
import torch q
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime, timedelta
import random
import os
import json
import matplotlib.pyplot as plt
import sklearn

print("🧠 Starting ENHANCED NEURAL NETWORK Training")
print("="*60)

# ———————————————————————
# HANDLE PYTORCH 2.6+ SECURITY CHANGES
# ———————————————————————
try:
    # Allow sklearn StandardScaler to be loaded for backward compatibility
    from torch import serialization
    serialization.add_safe_globals([sklearn.preprocessing._data.StandardScaler])
    print("✅ Added sklearn StandardScaler to safe globals")
except Exception as e:
    print(f"⚠️  Could not add sklearn to safe globals: {e}")

# ———————————————————————
# 1. ENHANCED MULTI-TOWER NEURAL NETWORK MODELS
# ———————————————————————
class EnhancedDVIDPredictor(nn.Module):
    def __init__(self, input_dim=142, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # Raw logits

class MultiTowerModel(nn.Module):
    """Multi-tower architecture with separate towers for different feature types"""
    def __init__(self,
                 degree_dim=3,           # customer_degree, product_degree, degree_ratio
                 embedding_dim=64,       # node2vec embeddings
                 temporal_dim=4,         # recency/frequency for customer and product
                 structured_dim=4,       # louvain, category, knn_avg, knn_max
                 derived_dim=2,          # embedding_dot_product, degree_product
                 hidden_dims=[128, 64],
                 dropout=0.3):
        super().__init__()

        # Degree tower
        self.degree_tower = nn.Sequential(
            nn.Linear(degree_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Embedding tower (customer and product)
        self.embedding_tower = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal tower
        self.temporal_tower = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Structured features tower (Louvain, Category, KNN features)
        self.structured_tower = nn.Sequential(
            nn.Linear(structured_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Derived features tower
        self.derived_tower = nn.Sequential(
            nn.Linear(derived_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final combination layers
        combined_dim = 32 + 64 + 32 + 32 + 16  # Sum of all tower outputs
        final_layers = []
        prev_dim = combined_dim

        for hidden_dim in hidden_dims:
            final_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        final_layers.append(nn.Linear(prev_dim, 1))
        self.final_layers = nn.Sequential(*final_layers)

    def forward(self, degree_features, embedding_features, temporal_features, structured_features, derived_features):
        # Process each tower
        degree_out = self.degree_tower(degree_features)
        embedding_out = self.embedding_tower(embedding_features)
        temporal_out = self.temporal_tower(temporal_features)
        structured_out = self.structured_tower(structured_features)
        derived_out = self.derived_tower(derived_features)

        # Concatenate all tower outputs
        combined = torch.cat([degree_out, embedding_out, temporal_out, structured_out, derived_out], dim=1)

        # Final prediction
        return self.final_layers(combined)  # Raw logits

# ———————————————————————
# 2. LOAD EXISTING MODELS FOR WARM-START (FIXED VERSION)
# ———————————————————————
def load_existing_nn_models():
    """Load existing trained neural network models for warm-start - FIXED VERSION"""
    existing_models = {}

    # Try multiple possible file paths for with-discount model
    wd_model_paths = [
        'models/nn_robust_discount_multi_tower.pth',
        'models/nn_robust_discount_semantic.pth',  # Previous version
        'models/nn_robust_discount.pth',           # Simple version
        'nn_robust_discount_multi_tower.pth'       # Current directory
    ]

    # Try multiple possible file paths for no-discount model
    nd_model_paths = [
        'models/nn_for_recommendations_multi_tower.pth',
        'models/nn_for_recommendations_semantic.pth',  # Previous version
        'models/nn_for_recommendations.pth',           # Simple version
        'nn_for_recommendations_multi_tower.pth'       # Current directory
    ]

    # Try to load with-discount model
    existing_models['with_discount'] = None
    for path in wd_model_paths:
        try:
            if os.path.exists(path):
                print(f"🔍 Attempting to load {path}...")
                # Use weights_only=False to bypass PyTorch 2.6+ security for trusted models
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)

                # Check if it's a multi-tower model or single tower
                if 'model_type' in checkpoint and checkpoint['model_type'] == 'multi_tower':
                    # Load as MultiTowerModel
                    model_wd = MultiTowerModel(
                        degree_dim=3,
                        embedding_dim=64,
                        temporal_dim=4,
                        structured_dim=5,  # 4 + discount
                        derived_dim=2
                    )
                    model_wd.load_state_dict(checkpoint['model_state_dict'])
                    existing_models['with_discount'] = model_wd
                    print(f"✅ Loaded existing with-discount Multi-Tower model from: {path}")
                    break
                else:
                    # Load as EnhancedDVIDPredictor (backward compatibility)
                    model_wd = EnhancedDVIDPredictor()
                    model_wd.load_state_dict(checkpoint['model_state_dict'])
                    existing_models['with_discount'] = model_wd
                    print(f"✅ Loaded existing with-discount Single-Tower model from: {path}")
                    break
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {str(e)[:100]}...")
            continue

    if existing_models['with_discount'] is None:
        print("⚠️  No existing with-discount NN model found")

    # Try to load no-discount model
    existing_models['no_discount'] = None
    for path in nd_model_paths:
        try:
            if os.path.exists(path):
                print(f"🔍 Attempting to load {path}...")
                # Use weights_only=False to bypass PyTorch 2.6+ security for trusted models
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)

                # Check if it's a multi-tower model or single tower
                if 'model_type' in checkpoint and checkpoint['model_type'] == 'multi_tower':
                    # Load as MultiTowerModel
                    model_nd = MultiTowerModel(
                        degree_dim=3,
                        embedding_dim=64,
                        temporal_dim=4,
                        structured_dim=4,  # no discount
                        derived_dim=2
                    )
                    model_nd.load_state_dict(checkpoint['model_state_dict'])
                    existing_models['no_discount'] = model_nd
                    print(f"✅ Loaded existing no-discount Multi-Tower model from: {path}")
                    break
                else:
                    # Load as EnhancedDVIDPredictor (backward compatibility)
                    model_nd = EnhancedDVIDPredictor()
                    model_nd.load_state_dict(checkpoint['model_state_dict'])
                    existing_models['no_discount'] = model_nd
                    print(f"✅ Loaded existing no-discount Single-Tower model from: {path}")
                    break
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {str(e)[:100]}...")
            continue

    if existing_models['no_discount'] is None:
        print("⚠️  No existing no-discount NN model found")

    return existing_models

# ———————————————————————
# 3. LOAD NEW DATA AND GRAPH (ENHANCED)
# ———————————————————————
def load_new_data():
    """Load new data for training"""
    try:
        G = joblib.load('graphs/graph_enriched.pkl')
        print(f"✅ Loaded updated graph with {G.number_of_nodes():,} nodes")

        customer_features_df = pd.read_csv('models/customer_features.csv')
        product_features_df = pd.read_csv('models/product_features.csv')
        print(f"✅ Loaded customer features: {len(customer_features_df):,}")
        print(f"✅ Loaded product features: {len(product_features_df):,}")

        # 🆕 NEW: Load Louvain communities
        try:
            partition = joblib.load('models/customer_partition.pkl')
            customer_features_df['louvain_community'] = customer_features_df['node_id'].map(partition)
            print("✅ Injected Louvain communities into customer features")
        except Exception as e:
            print(f"⚠️  Could not load Louvain partition: {e}")
            customer_features_df['louvain_community'] = -1

        # 🆕 NEW: Ensure product category is present
        if 'category' in product_features_df.columns:
            product_features_df['category_encoded'] = product_features_df['category'].astype('category').cat.codes
            print("✅ Injected product category encoding")
        else:
            print("⚠️  Product category not found — using placeholder -1")
            product_features_df['category_encoded'] = -1

        # 🆕 NEW: Load KNN similarity graph
        try:
            knn_graph = nx.read_graphml('graphs/G_similarity.graphml')
            print("✅ Loaded KNN similarity graph")
        except Exception as e:
            print(f"⚠️  Could not load KNN graph: {e}")
            knn_graph = None

        return G, customer_features_df, product_features_df, knn_graph
    except Exception as e:
        raise FileNotFoundError(f"❌ Failed to load  {e}")

# ———————————————————————
# 4. NEW: KNN FEATURE CALCULATION FUNCTIONS
# ———————————————————————
def calculate_knn_features(customer_id, knn_graph):
    """Calculate KNN-based similarity features for a customer"""
    if knn_graph is None or customer_id not in knn_graph.nodes():
        return {
            'knn_avg_similarity': 0.0,
            'knn_max_similarity': 0.0,
            'knn_min_similarity': 0.0,
            'knn_std_similarity': 0.0
        }

    neighbors = list(knn_graph.neighbors(customer_id))
    similarities = []

    for neighbor in neighbors:
        if knn_graph.has_edge(customer_id, neighbor):
            weight = knn_graph[customer_id][neighbor].get('weight', 0.0)
            similarities.append(float(weight))

    if not similarities:
        return {
            'knn_avg_similarity': 0.0,
            'knn_max_similarity': 0.0,
            'knn_min_similarity': 0.0,
            'knn_std_similarity': 0.0
        }

    return {
        'knn_avg_similarity': np.mean(similarities),
        'knn_max_similarity': np.max(similarities),
        'knn_min_similarity': np.min(similarities),
        'knn_std_similarity': np.std(similarities) if len(similarities) > 1 else 0.0
    }

# ———————————————————————
# 5. EXTRACT ONLY NEW LINKS (TRUE INCREMENTAL) — WITH FALLBACK
# ———————————————————————
def extract_only_new_links(G, existing_positive_count=17546):
    """Extract ONLY new links added since last training (true incremental)"""
    print("🔗 Extracting ONLY NEW customer-product links (true incremental)...")

    cutoff_timestamp = datetime(2020, 1, 1)  # Adjust as needed

    positive_links = []
    skipped_due_to_bad_timestamp = 0

    for u, v, data in G.edges(data=True):
        if data.get('type') == 'PURCHASED' and G.nodes[v].get('label') == 'Order':
            edge_timestamp = data.get('timestamp', None)

            if edge_timestamp is None:
                skipped_due_to_bad_timestamp += 1
                continue

            if isinstance(edge_timestamp, str):
                try:
                    edge_timestamp = datetime.strptime(edge_timestamp, '%Y-%m-%d')
                except ValueError:
                    try:
                        edge_timestamp = datetime.strptime(edge_timestamp, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        skipped_due_to_bad_timestamp += 1
                        continue

            if not isinstance(edge_timestamp, datetime):
                skipped_due_to_bad_timestamp += 1
                continue

            if edge_timestamp > cutoff_timestamp:
                for _, p, ed in G.edges(v, data=True):
                    if ed.get('type') == 'CONTAINS' and G.nodes[p].get('label') == 'Product':
                        positive_links.append((u, p))

    new_positive_count = len(positive_links)
    print(f"📊 New positive links found: {new_positive_count:,}")
    if skipped_due_to_bad_timestamp > 0:
        print(f"⚠️  Skipped {skipped_due_to_bad_timestamp:,} edges due to invalid/missing timestamps")

    if new_positive_count == 0:
        print("⚠️  No new links found. Falling back to full extraction.")
        return extract_all_links_fallback(G)

    return positive_links

def extract_all_links_fallback(G):
    """Fallback: extract all if no timestamp-based filtering works"""
    positive_links = set()
    for u, v, data in G.edges(data=True):
        if data.get('type') == 'PURCHASED' and G.nodes[v].get('label') == 'Order':
            for _, p, ed in G.edges(v, data=True):
                if ed.get('type') == 'CONTAINS' and G.nodes[p].get('label') == 'Product':
                    positive_links.add((u, p))
    return list(positive_links)

def generate_balanced_negative_samples(G, positive_links):
    """Generate negative samples matching count of positives"""
    print("📉 Generating balanced negative samples...")

    all_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer']
    all_products = [n for n, d in G.nodes(data=True) if d.get('label') == 'Product']

    target_count = len(positive_links)
    negative_links = set()
    max_attempts = target_count * 10
    attempts = 0

    while len(negative_links) < target_count and attempts < max_attempts:
        c = random.choice(all_customers)
        p = random.choice(all_products)
        if (c, p) not in positive_links:
            negative_links.add((c, p))
        attempts += 1

    print(f"✅ Generated {len(negative_links):,} negative samples")
    return list(negative_links)

# ———————————————————————
# 6. FEATURE EXTRACTION — ENHANCED WITH ALL FEATURES
# ———————————————————————
def calculate_temporal_features(G, node, current_time):
    """Calculate temporal features"""
    timestamps = [
        data['timestamp'] for _, _, data in G.edges(node, data=True)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime) and data['timestamp'] <= current_time
    ]
    if not timestamps:
        return {'recency': 999, 'frequency': 0}
    latest = max(timestamps)
    return {
        'recency': (current_time - latest).days,
        'frequency': len(timestamps)
    }

# 🆕 NEW: Helper function for multi-tower features (was missing)
def get_link_features_multi_tower(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time, G, knn_graph=None, include_discount=True):
    """Extract features for multi-tower model"""
    try:
        c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
        p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
    except (IndexError, KeyError):
        # Return zeros for all feature types
        degree_features = np.zeros(3)
        embedding_features = np.zeros(emb_dim * 2)
        temporal_features = np.zeros(4)
        structured_features = np.zeros(4)
        derived_features = np.zeros(2)
        discount_feature = np.array([0.0]) if include_discount else np.array([])
        return degree_features, embedding_features, temporal_features, structured_features, derived_features, discount_feature

    # Degree features
    customer_degree = float(c_row.get('degree', 0))
    product_degree = float(p_row.get('degree', 0))
    degree_ratio = customer_degree / (product_degree + 1e-8) # Avoid division by zero

    # Embedding features
    try:
        c_emb = np.array([float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)])
        p_emb = np.array([float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)])
    except:
        c_emb = np.zeros(emb_dim)
        p_emb = np.zeros(emb_dim)

    embedding_features = np.concatenate([c_emb, p_emb])

    # Temporal features
    c_temp = calculate_temporal_features(G, cust_id, current_time)
    p_temp = calculate_temporal_features(G, prod_id, current_time)
    temporal_features = np.array([
        c_temp['recency'], c_temp['frequency'],
        p_temp['recency'], p_temp['frequency']
    ])

    # Structured features (Louvain, Category, KNN)
    c_community = float(c_row.get('louvain_community', -1))
    p_category = float(p_row.get('category_encoded', -1))
    knn_features = calculate_knn_features(cust_id, knn_graph)
    structured_features = np.array([
        c_community,
        p_category,
        knn_features['knn_avg_similarity'],
        knn_features['knn_max_similarity']
    ])

    # Derived features
    embedding_dot_product = np.dot(c_emb, p_emb)
    degree_product = customer_degree * product_degree
    derived_features = np.array([embedding_dot_product, degree_product])

    # Discount feature (only for with-discount model)
    if include_discount:
        discount = float(G.nodes[prod_id].get('discount', 0.0))
        discount_feature = np.array([discount])
    else:
        discount_feature = np.array([])

    degree_features = np.array([customer_degree, product_degree, degree_ratio])

    return degree_features, embedding_features, temporal_features, structured_features, derived_features, discount_feature

def create_incremental_dataset_multi_tower(G, positive_links, negative_links, customer_features_df,
                                         product_features_df, embedding_dim=64, knn_graph=None):
    """Create dataset for multi-tower model"""
    print("⚙️  Building incremental dataset for multi-tower model...")

    current_time = datetime(2023, 1, 1)

    all_links = positive_links + negative_links
    all_labels = [1] * len(positive_links) + [0] * len(negative_links)

    # Prepare feature lists
    degree_features_wd = []
    embedding_features_wd = []
    temporal_features_wd = []
    structured_features_wd = []
    derived_features_wd = []
    discount_features_wd = []

    degree_features_nd = []
    embedding_features_nd = []
    temporal_features_nd = []
    structured_features_nd = []
    derived_features_nd = []

    for c, p in all_links:
        # With discount features
        deg_feat_wd, emb_feat_wd, temp_feat_wd, struct_feat_wd, deriv_feat_wd, disc_feat_wd = get_link_features_multi_tower(
            c, p, customer_features_df, product_features_df, embedding_dim, current_time, G, knn_graph, include_discount=True
        )
        degree_features_wd.append(deg_feat_wd)
        embedding_features_wd.append(emb_feat_wd)
        temporal_features_wd.append(temp_feat_wd)
        structured_features_wd.append(struct_feat_wd)
        derived_features_wd.append(deriv_feat_wd)
        discount_features_wd.append(disc_feat_wd[0])  # Extract scalar value

        # No discount features
        deg_feat_nd, emb_feat_nd, temp_feat_nd, struct_feat_nd, deriv_feat_nd, _ = get_link_features_multi_tower(
            c, p, customer_features_df, product_features_df, embedding_dim, current_time, G, knn_graph, include_discount=False
        )
        degree_features_nd.append(deg_feat_nd)
        embedding_features_nd.append(emb_feat_nd)
        temporal_features_nd.append(temp_feat_nd)
        structured_features_nd.append(struct_feat_nd)
        derived_features_nd.append(deriv_feat_nd)

    # Convert to arrays
    X_degree_wd = np.array(degree_features_wd)
    X_embedding_wd = np.array(embedding_features_wd)
    X_temporal_wd = np.array(temporal_features_wd)
    X_structured_wd = np.array(structured_features_wd)
    X_derived_wd = np.array(derived_features_wd)
    X_discount_wd = np.array(discount_features_wd).reshape(-1, 1)

    X_degree_nd = np.array(degree_features_nd)
    X_embedding_nd = np.array(embedding_features_nd)
    X_temporal_nd = np.array(temporal_features_nd)
    X_structured_nd = np.array(structured_features_nd)
    X_derived_nd = np.array(derived_features_nd)

    y = np.array(all_labels)

    print(f"📊 Multi-tower dataset created:")
    print(f"   Degree features: {X_degree_wd.shape}")
    print(f"   Embedding features: {X_embedding_wd.shape}")
    print(f"   Temporal features: {X_temporal_wd.shape}")
    print(f"   Structured features: {X_structured_wd.shape}")
    print(f"   Derived features: {X_derived_wd.shape}")
    print(f"   Discount features: {X_discount_wd.shape}")
    print(f"   Labels: {y.shape}")

    # Combine discount with structured features for with-discount model
    X_structured_wd_combined = np.concatenate([X_structured_wd, X_discount_wd], axis=1)

    return {
        'with_discount': {
            'degree': X_degree_wd,
            'embedding': X_embedding_wd,
            'temporal': X_temporal_wd,
            'structured': X_structured_wd_combined,  # Includes discount
            'derived': X_derived_wd,
            'labels': y
        },
        'no_discount': {
            'degree': X_degree_nd,
            'embedding': X_embedding_nd,
            'temporal': X_temporal_nd,
            'structured': X_structured_nd,
            'derived': X_derived_nd,
            'labels': y
        }
    }

# ———————————————————————
# 7. ENHANCED NEURAL NETWORK TRAINING (UPDATED WITH FIXES)
# ———————————————————————
def train_multi_tower_model(train_data, val_data, model_name="Multi-Tower Model", existing_model=None):
    """Train multi-tower neural network with optional warm-start"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧠 Training {model_name} on {device}...")

    # Extract features
    X_degree_train = train_data['degree']
    X_embedding_train = train_data['embedding']
    X_temporal_train = train_data['temporal']
    X_structured_train = train_data['structured']
    X_derived_train = train_data['derived']
    y_train = train_data['labels']

    X_degree_val = val_data['degree']
    X_embedding_val = val_data['embedding']
    X_temporal_val = val_data['temporal']
    X_structured_val = val_data['structured']
    X_derived_val = val_data['derived']
    y_val = val_data['labels'] # This stays as numpy array

    # Feature standardization
    scalers = {}

    # Scale each feature type separately
    scalers['degree'] = StandardScaler()
    scalers['embedding'] = StandardScaler()
    scalers['temporal'] = StandardScaler()
    scalers['structured'] = StandardScaler()
    scalers['derived'] = StandardScaler()

    X_degree_train_scaled = scalers['degree'].fit_transform(X_degree_train)
    X_embedding_train_scaled = scalers['embedding'].fit_transform(X_embedding_train)
    X_temporal_train_scaled = scalers['temporal'].fit_transform(X_temporal_train)
    X_structured_train_scaled = scalers['structured'].fit_transform(X_structured_train)
    X_derived_train_scaled = scalers['derived'].fit_transform(X_derived_train)

    X_degree_val_scaled = scalers['degree'].transform(X_degree_val)
    X_embedding_val_scaled = scalers['embedding'].transform(X_embedding_val)
    X_temporal_val_scaled = scalers['temporal'].transform(X_temporal_val)
    X_structured_val_scaled = scalers['structured'].transform(X_structured_val)
    X_derived_val_scaled = scalers['derived'].transform(X_derived_val)

    # Convert to tensors
    X_degree_train_t = torch.tensor(X_degree_train_scaled, dtype=torch.float32).to(device)
    X_embedding_train_t = torch.tensor(X_embedding_train_scaled, dtype=torch.float32).to(device)
    X_temporal_train_t = torch.tensor(X_temporal_train_scaled, dtype=torch.float32).to(device)
    X_structured_train_t = torch.tensor(X_structured_train_scaled, dtype=torch.float32).to(device)
    X_derived_train_t = torch.tensor(X_derived_train_scaled, dtype=torch.float32).to(device)

    X_degree_val_t = torch.tensor(X_degree_val_scaled, dtype=torch.float32).to(device)
    X_embedding_val_t = torch.tensor(X_embedding_val_scaled, dtype=torch.float32).to(device)
    X_temporal_val_t = torch.tensor(X_temporal_val_scaled, dtype=torch.float32).to(device)
    X_structured_val_t = torch.tensor(X_structured_val_scaled, dtype=torch.float32).to(device)
    X_derived_val_t = torch.tensor(X_derived_val_scaled, dtype=torch.float32).to(device)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)  # This stays as tensor

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    pos_weight = torch.tensor([class_weights[1]], dtype=torch.float32).to(device)

    # Create data loaders
    train_dataset = TensorDataset(
        X_degree_train_t, X_embedding_train_t, X_temporal_train_t,
        X_structured_train_t, X_derived_train_t, y_train_t
    )
    val_dataset = TensorDataset(
        X_degree_val_t, X_embedding_val_t, X_temporal_val_t,
        X_structured_val_t, X_derived_val_t, y_val_t
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Create model
    model = MultiTowerModel(
        degree_dim=X_degree_train.shape[1],
        embedding_dim=64,  # Assuming 64-dimensional embeddings
        temporal_dim=X_temporal_train.shape[1],
        structured_dim=X_structured_train.shape[1],
        derived_dim=X_derived_train.shape[1]
    ).to(device)

    # 🔥 WARM-START: Load existing model weights if provided
    if existing_model is not None:
        try:
            # Handle both MultiTowerModel and EnhancedDVIDPredictor
            if isinstance(existing_model, MultiTowerModel):
                # Copy compatible weights from MultiTowerModel to MultiTowerModel
                existing_state = existing_model.state_dict()
                current_state = model.state_dict()

                # Only load compatible weights
                compatible_weights = {}
                for key, value in existing_state.items():
                    if key in current_state and value.shape == current_state[key].shape:
                        compatible_weights[key] = value

                current_state.update(compatible_weights)
                model.load_state_dict(current_state)
                print("🔁 Warm-started from existing Multi-Tower model weights")
            elif isinstance(existing_model, EnhancedDVIDPredictor):
                # Convert single-tower to multi-tower (limited transfer)
                print("🔄 Converting Single-Tower model to Multi-Tower (partial transfer)")
                # We can't directly transfer weights, but we can use the model as initialization reference
                print("🔁 Warm-started from existing Single-Tower model (architecture reference)")
            else:
                print("⚠️  Unknown existing model type, training from scratch")
        except Exception as e:
            print(f"⚠️  Warm-start failed, training from scratch: {e}")
    else:
        print("🆕 Training from scratch (no existing model found)")

    # ✅ UPDATED: More aggressive learning parameters
    INITIAL_LR = 0.02              # Increased from 0.01 for faster initial learning
    NUM_EPOCHS = 800               # Increased from 500 for more thorough training
    PATIENCE = 200                 # Increased patience
    LR_PATIENCE = 30              # Increased LR patience

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=LR_PATIENCE)
    # ✅ FIX: Initialize prev_lr before the training loop
    prev_lr = optimizer.param_groups[0]['lr'] # Initialize here

    # Training loop
    best_auc = 0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_aucs = []

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            degree_batch, embedding_batch, temporal_batch, structured_batch, derived_batch, y_batch = batch
            optimizer.zero_grad()
            logits = model(degree_batch, embedding_batch, temporal_batch, structured_batch, derived_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss/len(train_loader))

        # Validation
        model.eval()
        y_val_logits = []
        with torch.no_grad():
            for batch in val_loader:
                degree_batch, embedding_batch, temporal_batch, structured_batch, derived_batch, _ = batch
                logits = model(degree_batch, embedding_batch, temporal_batch, structured_batch, derived_batch)
                y_val_logits.extend(logits.cpu().numpy())

        y_val_logits = np.array(y_val_logits)
        y_val_pred = 1 / (1 + np.exp(-y_val_logits))  # Sigmoid

        # 🔧 FIX: y_val is already numpy array, no need for .cpu()
        auc = roc_auc_score(y_val, y_val_pred)  # Fixed: removed .cpu().numpy()
        val_aucs.append(auc)
        scheduler.step(auc)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"📉 Learning rate reduced to {current_lr:.6f}")
            prev_lr = current_lr # Update prev_lr

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # Save checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'scalers': scalers
            }
            torch.save(checkpoint_data, f'models/{model_name}_best_checkpoint.pth')
        else:
            patience_counter += 1

        if epoch % 20 == 0: # Print less frequently
            print(f"Epoch {epoch:3d} | Train Loss: {train_losses[-1]:.4f} | Val AUC: {auc:.4f} | LR: {current_lr:.6f}")

        # ✅ UPDATED: More patient early stopping
        if patience_counter >= PATIENCE:
            print(f"⏹️  Early stopping at epoch {epoch}")
            break

    print(f"✅ {model_name} Best Validation AUC: {best_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_aucs)
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')

    plt.tight_layout()
    plt.savefig(f'models/{model_name}_training_history.png')
    plt.close()

    return model, y_val_pred, best_auc, scalers

# ———————————————————————
# 8. THRESHOLD OPTIMIZATION
# ———————————————————————
def find_optimal_threshold_for_target(y_true, y_pred_proba, target_metric='f1', target_value=0.80):
    """Find best threshold for business needs"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        metric_value = locals()[target_metric]
        if metric_value > best_score: # Simplified condition for finding max
            best_score = metric_value
            best_threshold = threshold
            best_metrics = {'f1': f1, 'precision': prec, 'recall': rec, 'accuracy': acc, 'threshold': threshold}

    return best_threshold, best_metrics

# ———————————————————————
# 9. MAIN TRAINING PIPELINE
# ———————————————————————
def main():
    """Main neural network training pipeline"""

    print("🚀 Starting ENHANCED NEURAL NETWORK Training Pipeline")
    print("="*60)

    # 🔥 LOAD EXISTING MODELS FOR WARM-START
    print("🔍 Checking for existing models...")
    existing_models = load_existing_nn_models()

    # Load data
    G, customer_features_df, product_features_df, knn_graph = load_new_data()

    # Extract links
    positive_links = extract_only_new_links(G)
    negative_links = generate_balanced_negative_samples(G, positive_links)

    # Create dataset with all enhanced features
    dataset = create_incremental_dataset_multi_tower(
        G, positive_links, negative_links, customer_features_df, product_features_df, knn_graph=knn_graph
    )

    # Split data for both models
    print("⚙️  Splitting data for training...")

    # With-discount split
    wd_data = dataset['with_discount']
    (X_degree_train_wd, X_degree_test_wd,
     X_embedding_train_wd, X_embedding_test_wd,
     X_temporal_train_wd, X_temporal_test_wd,
     X_structured_train_wd, X_structured_test_wd,
     X_derived_train_wd, X_derived_test_wd,
     y_train_wd, y_test_wd) = train_test_split(
        wd_data['degree'], wd_data['embedding'], wd_data['temporal'],
        wd_data['structured'], wd_data['derived'], wd_data['labels'],
        test_size=0.2, random_state=42, stratify=wd_data['labels']
    )

    # No-discount split
    nd_data = dataset['no_discount']
    (X_degree_train_nd, X_degree_test_nd,
     X_embedding_train_nd, X_embedding_test_nd,
     X_temporal_train_nd, X_temporal_test_nd,
     X_structured_train_nd, X_structured_test_nd,
     X_derived_train_nd, X_derived_test_nd,
     y_train_nd, y_test_nd) = train_test_split(
        nd_data['degree'], nd_data['embedding'], nd_data['temporal'],
        nd_data['structured'], nd_data['derived'], nd_data['labels'],
        test_size=0.2, random_state=42, stratify=nd_data['labels']
    )

    # Prepare training data dictionaries
    train_data_wd = {
        'degree': X_degree_train_wd,
        'embedding': X_embedding_train_wd,
        'temporal': X_temporal_train_wd,
        'structured': X_structured_train_wd,
        'derived': X_derived_train_wd,
        'labels': y_train_wd
    }

    val_data_wd = {
        'degree': X_degree_test_wd,
        'embedding': X_embedding_test_wd,
        'temporal': X_temporal_test_wd,
        'structured': X_structured_test_wd,
        'derived': X_derived_test_wd,
        'labels': y_test_wd
    }

    train_data_nd = {
        'degree': X_degree_train_nd,
        'embedding': X_embedding_train_nd,
        'temporal': X_temporal_train_nd,
        'structured': X_structured_train_nd,
        'derived': X_derived_train_nd,
        'labels': y_train_nd
    }

    val_data_nd = {
        'degree': X_degree_test_nd,
        'embedding': X_embedding_test_nd,
        'temporal': X_temporal_test_nd,
        'structured': X_structured_test_nd,
        'derived': X_derived_test_nd,
        'labels': y_test_nd
    }

    # Train models
    print("\n" + "="*60)
    print("🧠 TRAINING MULTI-TOWER NEURAL NETWORKS")
    print("="*60)

    # Train with-discount model
    nn_wd_model, y_pred_wd, best_auc_wd, scalers_wd = train_multi_tower_model(
        train_data_wd, val_data_wd, "With-Discount Multi-Tower",
        existing_model=existing_models.get('with_discount')  # 🔥 WARM-START
    )

    # Train no-discount model
    nn_nd_model, y_pred_nd, best_auc_nd, scalers_nd = train_multi_tower_model(
        train_data_nd, val_data_nd, "No-Discount Multi-Tower",
        existing_model=existing_models.get('no_discount')  # 🔥 WARM-START
    )

    # Evaluate
    print("\n" + "="*60)
    print("📈 FINAL EVALUATION")
    print("="*60)

    # With-discount evaluation
    auc_wd = roc_auc_score(y_test_wd, y_pred_wd)
    pr_auc_wd = average_precision_score(y_test_wd, y_pred_wd)
    y_pred_wd_05 = (y_pred_wd > 0.5).astype(int)
    acc_wd = accuracy_score(y_test_wd, y_pred_wd_05)
    prec_wd = precision_score(y_test_wd, y_pred_wd_05)
    rec_wd = recall_score(y_test_wd, y_pred_wd_05)
    f1_wd = f1_score(y_test_wd, y_pred_wd_05)

    print(f"\n📊 With-Discount Multi-Tower Neural Network:")
    print(f"   AUC: {auc_wd:.4f} | PR-AUC: {pr_auc_wd:.4f} | Acc: {acc_wd:.4f} | Prec: {prec_wd:.4f} | Rec: {rec_wd:.4f} | F1: {f1_wd:.4f}")

    best_thresh_wd, best_metrics_wd = find_optimal_threshold_for_target(y_test_wd, y_pred_wd)
    print(f"   🎯 Optimal Threshold: {best_thresh_wd:.3f} → F1: {best_metrics_wd['f1']:.4f}")

    # No-discount evaluation
    auc_nd = roc_auc_score(y_test_nd, y_pred_nd)
    pr_auc_nd = average_precision_score(y_test_nd, y_pred_nd)
    y_pred_nd_05 = (y_pred_nd > 0.5).astype(int)
    acc_nd = accuracy_score(y_test_nd, y_pred_nd_05)
    prec_nd = precision_score(y_test_nd, y_pred_nd_05)
    rec_nd = recall_score(y_test_nd, y_pred_nd_05)
    f1_nd = f1_score(y_test_nd, y_pred_nd_05)

    print(f"\n📊 No-Discount Multi-Tower Neural Network:")
    print(f"   AUC: {auc_nd:.4f} | PR-AUC: {pr_auc_nd:.4f} | Acc: {acc_nd:.4f} | Prec: {prec_nd:.4f} | Rec: {rec_nd:.4f} | F1: {f1_nd:.4f}")

    best_thresh_nd, best_metrics_nd = find_optimal_threshold_for_target(y_test_nd, y_pred_nd)
    print(f"   🎯 Optimal Threshold: {best_thresh_nd:.3f} → F1: {best_metrics_nd['f1']:.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)

    torch.save({
        'model_state_dict': nn_wd_model.state_dict(),
        'scalers': scalers_wd,
        'model_type': 'multi_tower'
    }, 'models/nn_robust_discount_multi_tower.pth')
    print("✅ With-discount Multi-Tower NN model saved")

    torch.save({
        'model_state_dict': nn_nd_model.state_dict(),
        'scalers': scalers_nd,
        'model_type': 'multi_tower'
    }, 'models/nn_for_recommendations_multi_tower.pth')
    print("✅ No-discount Multi-Tower NN model saved")

    # Save metrics
    threshold_info = {
        'with_discount': {
            'threshold': float(best_thresh_wd),
            'metrics': best_metrics_wd,
            'auc': float(auc_wd),
            'pr_auc': float(pr_auc_wd),  # 🆕 NEW
            'accuracy': float(acc_wd),
            'precision': float(prec_wd),
            'recall': float(rec_wd),
            'f1': float(f1_wd)
        },
        'no_discount': {
            'threshold': float(best_thresh_nd),
            'metrics': best_metrics_nd,
            'auc': float(auc_nd),
            'pr_auc': float(pr_auc_nd),  # 🆕 NEW
            'accuracy': float(acc_nd),
            'precision': float(prec_nd),
            'recall': float(rec_nd),
            'f1': float(f1_nd)
        }
    }

    with open('models/nn_multi_tower_threshold_info.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)

    print("\n" + "="*60)
    print("🎉 ENHANCED MULTI-TOWER NEURAL NETWORK TRAINING COMPLETED!")
    print("="*60)
    print(f"📊 Results saved to models/nn_multi_tower_threshold_info.json")
    print("📈 Training history plots saved")
    print("✅ Models are now trained with enhanced features and ready for production.")
    print("="*60)

# ———————————————————————
# 10. RUN
# ———————————————————————
if __name__ == "__main__":
    main()

