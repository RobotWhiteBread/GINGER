# FILE: GINGER_pipeline.py (Definitive, Self-Documenting & Corrected Version)

import os
import yaml
import argparse
import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- OPTIONAL IMPORTS ---
# Scientific: Importing libraries for numerical operations, machine learning, and data visualization.
# Layman's:  Getting all the specialized tools we need from our toolbox to work with data and create charts.
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from statsmodels.multivariate.manova import MANOVA
    from scipy.spatial import ConvexHull
    import plotly.express as px
    import plotly.graph_objects as go
    import umap
    import hdbscan
    import shap
except ImportError as e:
    logging.warning(f"A data analysis library is not installed: {e}. Visualization stage may be limited.")

# --- PROFESSIONAL LOGGING SETUP ---
# Scientific: Configures a logger to output timestamped messages with severity levels (INFO, WARNING, ERROR).
# Layman's:  Sets up a detailed diary for the script, so it can write down everything it does and when.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(path):
    # --- LOAD CONFIGURATION ---
    # Scientific: Parses a YAML file to load pipeline parameters into a Python dictionary.
    # Layman's:  Reads the main settings file (config.yaml) that tells the script what to do.
    logging.info(f"Loading configuration from: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- DATA HANDLING ---
def find_image_files(root_dir, exclude_dirs, exclude_files):
    # --- DISCOVER IMAGE FILES ---
    # Scientific: Traverses a directory tree to find all image files, filtering against exclusion lists. It maps class names (folder names) to integer labels.
    # Layman's:  Searches through all the folders to find every valid picture, ignoring any files or folders we've told it to skip.
    logging.info("Discovering image files...")
    samples, class_names = [], set()
    for dirpath, _, filenames in os.walk(root_dir):
        if any(os.path.normpath(dirpath).startswith(os.path.normpath(p)) for p in exclude_dirs):
            continue
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.lower() not in exclude_files:
                class_name = os.path.basename(dirpath)
                class_names.add(class_name)
                samples.append((os.path.join(dirpath, filename), class_name))
    class_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}
    samples_with_idx = [(path, class_to_idx[name]) for path, name in samples]
    logging.info(f"Found {len(samples)} images across {len(class_names)} classes.")
    return samples_with_idx, sorted(list(class_names))

class ImageDataset(torch.utils.data.Dataset):
    # --- PYTORCH IMAGE DATASET ---
    # Scientific: A custom PyTorch Dataset class that loads an image from a given path, applies transformations, and handles potential file loading errors.
    # Layman's:  A recipe book for the computer. It tells PyTorch how to find an image, open it, resize it for the model, and what to do if an image file is broken.
    def __init__(self, all_samples, transform=None):
        self.samples = all_samples
        self.transform = transform
        self.targets = [s[1] for s in all_samples]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = datasets.folder.default_loader(path).convert("RGB")
            if self.transform: image = self.transform(image)
        except Exception as e:
            logging.error(f"Could not load image {path}: {e}")
            return torch.randn(3, 224, 224), -1, path
        return image, target, path

# --- STAGE 1: MODEL TRAINING ---
def run_training(config, all_samples, class_names):
    # --- TRAIN THE NEURAL NETWORK ---
    # Scientific: Trains a Convolutional Neural Network (CNN) using labeled images. It minimizes cross-entropy loss via backpropagation and saves the model weights that achieve the highest validation accuracy.
    # Layman's:  Teaches the computer to recognize different ginger species by showing it thousands of labeled examples. It learns by guessing, getting corrected, and slowly getting smarter. We save the "smartest" version of its brain.
    logging.info("--- STAGE 1: MODEL TRAINING ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = config['output_dir']
    model_path = os.path.join(output_dir, f"{config['project_name']}_best_model.pth")
    
    # Scientific: Performs a stratified train/validation split to maintain class distribution in both sets, which is critical for unbiased evaluation.
    # Layman's:  Splits the data into a large "study pile" and a smaller "quiz pile," making sure both piles have the same mix of species.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    full_dataset = ImageDataset(all_samples)
    train_indices, val_indices = next(sss.split(full_dataset.samples, full_dataset.targets))

    # Scientific: Defines image augmentation and normalization pipelines for training and validation data.
    # Layman's:  Sets up rules to randomly flip, rotate, and tweak the colors of training images so the computer learns the important features, not just the specific orientation of a leaf.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset.transform = train_transform
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = ImageDataset([all_samples[i] for i in val_indices], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)

    # Scientific: Initializes a pre-trained ResNet18 model, replacing its final classification layer to match the number of target classes.
    # Layman's:  Gets a powerful, pre-trained "vision brain" (ResNet18) and attaches a new final layer customized for our specific ginger species.
    model = models.get_model(config['training']['model_architecture'], weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    best_acc = 0.0
    for epoch in range(config['training']['epochs']):
        model.train()
        for images, labels, _ in train_loader:
            valid_mask = (labels != -1)
            if not valid_mask.any(): continue
            images, labels = images[valid_mask].to(device), labels[valid_mask].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                valid_mask = (labels != -1)
                if not valid_mask.any(): continue
                images, labels = images[valid_mask].to(device), labels[valid_mask].to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']} | Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            logging.info(f"  -> New best model saved with accuracy: {best_acc:.4f}")
            
    logging.info(f"Training complete. Best model saved to {model_path}")
    return model_path

# --- STAGE 2: FEATURE EXTRACTION ---
def run_feature_extraction(config, model_path, all_samples, class_names):
    # --- EXTRACT DEEP FEATURES ---
    # Scientific: Loads the trained model, removes the final classification layer, and performs a forward pass on all images. The output from the penultimate layer (a high-dimensional vector) is captured as the feature embedding.
    # Layman's:  Takes the "smartest brain" we saved, and for each image, asks "What do you think of this?" but stops it right before it gives a final name. The result is a list of numbers—a unique "digital fingerprint"—for each image.
    logging.info("--- STAGE 2: FEATURE EXTRACTION ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = config['output_dir']
    output_csv_path = os.path.join(output_dir, config['feature_extraction']['output_csv_name'])

    model = models.get_model(config['training']['model_architecture'], weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.fc = nn.Identity()
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(all_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)

    features, paths = [], []
    with torch.no_grad():
        for images, _, path_batch in tqdm(loader, desc="Extracting Features"):
            valid_mask = (images.sum(dim=[1,2,3]) != 0)
            if not valid_mask.any(): continue
            images = images[valid_mask].to(device)
            feats = model(images)
            features.append(feats.cpu())
            paths.extend([p for i, p in enumerate(path_batch) if valid_mask[i]])

    features_tensor = torch.cat(features).numpy()
    df = pd.DataFrame(features_tensor, columns=[f'deep_feat_{i}' for i in range(features_tensor.shape[1])])
    df['filepath'] = paths
    df['class_name'] = [os.path.basename(os.path.dirname(p)) for p in paths]
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Feature extraction complete. Data saved to {output_csv_path}")
    return df

# --- STAGE 3: VISUALIZATION & ANALYSIS ---
def run_visualization(config, features_df):
    logging.info("--- STAGE 3: VISUALIZATION & ANALYSIS (FULL SUITE) ---")
    output_dir = config['output_dir']
    project_name = config['project_name']
    
    # --- 1. Prepare Data ---
    # Scientific: Standardizes the high-dimensional feature data by scaling to zero mean and unit variance, a prerequisite for many ML algorithms.
    # Layman's:  Puts all the numbers in the "digital fingerprints" on the same scale (e.g., from -1 to 1) so that no single number unfairly dominates the analysis.
    logging.info("Preparing data for analysis...")
    features_df['image_name'] = features_df['filepath'].apply(os.path.basename)
    feature_columns = features_df.filter(regex='deep_feat_').columns
    X = features_df[feature_columns].values
    y = features_df['class_name'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_int = le.fit_transform(y)

    # --- 2. Dimensionality Reduction ---
    # Scientific: Applies algorithms like PCA, LDA, UMAP, and t-SNE to project the 512-dimensional feature vectors into a 2D or 3D space for visualization.
    # Layman's:  Takes the 512-number "fingerprint" for each image and creates a simple 2D map (like a scatter plot) where similar images are placed close together.
    logging.info("Performing dimensionality reduction...")
    pca = PCA(n_components=3); features_df[['PC1', 'PC2', 'PC3']] = pca.fit_transform(X_scaled)
    lda = LDA(n_components=min(len(np.unique(y)) - 1, 2)); features_df[['LDA1', 'LDA2']] = lda.fit_transform(X_scaled, y)
    tsne = TSNE(n_components=2, random_state=42); features_df[['TSNE1', 'TSNE2']] = tsne.fit_transform(X_scaled)
    reducer = umap.UMAP(**config['visualization']['umap']); features_df[['UMAP1', 'UMAP2']] = reducer.fit_transform(X_scaled)
    
    # --- 3. Statistical Analysis ---
    # Scientific: Performs a Multivariate Analysis of Variance (MANOVA) to test whether there are statistically significant differences between the means of the species groups based on their principal components.
    # Layman's:  Runs a powerful statistical test to see if the different species groups are mathematically distinct from each other, or if the differences could be due to random chance.
    try:
        logging.info("Performing MANOVA statistical test...")
        manova = MANOVA.from_formula('PC1 + PC2 + PC3 ~ class_name', data=features_df)
        with open(os.path.join(output_dir, f"{project_name}_manova_results.txt"), 'w') as f:
            f.write(str(manova.mv_test()))
        logging.info("  -> MANOVA results saved.")
    except Exception as e:
        logging.warning(f"MANOVA analysis skipped: {e}")

    # --- 4. Plotting & Chart Generation ---
    # Scientific: Generates and saves various visualizations, including 2D scatter plots of the reduced-dimension data and interactive 3D plots.
    # Layman's:  Creates all the charts and graphs, like the 2D maps (PCA, UMAP) and an interactive 3D model of the data that you can spin around.
    logging.info("Generating core visualizations...")
    plot_list = {'PCA': ('PC1', 'PC2'), 'LDA': ('LDA1', 'LDA2'), 'UMAP': ('UMAP1', 'UMAP2')}
    for name, (x_ax, y_ax) in plot_list.items():
        plt.figure(figsize=(12, 8)); sns.scatterplot(data=features_df, x=x_ax, y=y_ax, hue='class_name', s=60); plt.title(f'{name} Plot');
        plt.savefig(os.path.join(output_dir, f"{project_name}_{name}_2D.png")); plt.close()
        logging.info(f"  -> Saved {name} 2D plot.")
    
    logging.info("Generating interactive 3D PCA plot with convex hulls...")
    try:
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, name in enumerate(features_df['class_name'].unique()):
            class_df = features_df[features_df['class_name'] == name]
            points = class_df[['PC1', 'PC2', 'PC3']].values
            fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(color=colors[i]), name=name))
            if len(points) >= 4:
                hull = ConvexHull(points)
                fig.add_trace(go.Mesh3d(x=points[:,0], y=points[:,1], z=points[:,2], i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2], opacity=0.1, color=colors[i]))
        fig.update_layout(title_text='Interactive 3D PCA with Convex Hulls')
        fig.write_html(os.path.join(output_dir, f"{project_name}_pca_3D_interactive.html"))
        logging.info("  -> Saved interactive 3D PCA plot.")
    except Exception as e:
        logging.warning(f"Could not generate 3D plot: {e}")

    # --- 5. Clustering and Outlier Analysis ---
    # Scientific: Applies unsupervised clustering algorithms (KMeans, HDBSCAN) to find natural groupings in the data and uses methods like Isolation Forest to identify anomalous data points (outliers).
    # Layman's:  Tells the computer to find natural "clumps" in the data without looking at the labels. It also flags any images that seem to be the "odd one out" compared to the rest.
    logging.info("Performing clustering and outlier detection...")
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=10)
    features_df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
    logging.info(f"KMeans Silhouette Score: {silhouette_score(X_scaled, features_df['kmeans_cluster']):.3f}")

    clusterer = hdbscan.HDBSCAN(**config['visualization']['hdbscan'])
    features_df['hdbscan_cluster'] = clusterer.fit_predict(X_scaled)
    logging.info(f"HDBSCAN found {len(np.unique(features_df['hdbscan_cluster']))} clusters.")

    iso_forest = IsolationForest(**config['visualization']['outlier_detection'], random_state=42); features_df['isolation_outlier'] = iso_forest.fit_predict(X_scaled)
    lof = LocalOutlierFactor(**config['visualization']['outlier_detection']); features_df['lof_outlier'] = lof.fit_predict(X_scaled)
    logging.info(f"Detected {(features_df['isolation_outlier'] == -1).sum()} potential outliers.")
    
    logging.info("Generating t-SNE plot with outliers...")
    plt.figure(figsize=(12, 8)); sns.scatterplot(data=features_df, x='TSNE1', y='TSNE2', hue='class_name', s=60);
    outlier_mask = (features_df['isolation_outlier'] == -1) | (features_df['lof_outlier'] == -1)
    plt.scatter(features_df.loc[outlier_mask, 'TSNE1'], features_df.loc[outlier_mask, 'TSNE2'], c='red', marker='x', s=100, label='Outlier')
    plt.title('t-SNE with Outliers Highlighted'); plt.legend(); plt.savefig(os.path.join(output_dir, f"{project_name}_tsne_with_outliers.png")); plt.close()
    logging.info("  -> Saved t-SNE plot with outliers.")

    # --- 6. Hybrid and Consistency Analysis ---
    # Scientific: Uses a Gaussian Mixture Model (GMM) to identify samples with high classification ambiguity, which may be indicative of hybrids. Also checks label consistency against each sample's nearest neighbors in the feature space.
    # Layman's:  Looks for plants that seem to be "in-between" two known species. It also double-checks each plant's label by asking, "Are this plant's closest look-alikes also the same species?"
    logging.info("Running GMM for hybrid detection and checking neighbor consistency...")
    try:
        gmm = GaussianMixture(n_components=len(np.unique(y)), random_state=42, reg_covar=1e-6)
        gmm.fit(X_scaled)
        probs = gmm.predict_proba(X_scaled)
        features_df['gmm_hybrid_candidate'] = (probs.max(axis=1) < 0.6)
        logging.info(f"Found {features_df['gmm_hybrid_candidate'].sum()} potential hybrid candidates.")
    except ValueError as e:
        logging.warning(f"GMM analysis for hybrid detection failed. This can happen with highly separated data. Error: {e}")
        features_df['gmm_hybrid_candidate'] = False # Ensure column exists even if it fails
    
    nn = NearestNeighbors(n_neighbors=6); nn.fit(X_scaled)
    neighbors = nn.kneighbors(return_distance=False)
    label_consistency = [np.mean(y[nbrs[1:]] == y[i]) for i, nbrs in enumerate(neighbors)]
    features_df['nn_label_consistency'] = label_consistency
    logging.info(f"Found {len(features_df[features_df['nn_label_consistency'] < 0.5])} samples with low neighbor consistency.")
    
    # --- 7. Explainable AI (SHAP) ---
    # Scientific: Calculates SHAP (SHapley Additive exPlanations) values to determine the contribution of each deep feature to the model's predictions, providing insight into the model's decision-making process.
    # Layman's:  Asks the computer's brain: "To decide this was Species A, which parts of its 'digital fingerprint' were most important to you?" This helps us understand what the computer is actually "looking" at.
    logging.info("Calculating SHAP feature importances...")
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42); rf.fit(X_scaled, y)
        explainer = shap.TreeExplainer(rf); shap_values = explainer.shap_values(X_scaled)
        plt.figure(); shap.summary_plot(shap_values, features_df[feature_columns], show=False, plot_size=(10, 8));
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{project_name}_shap_summary.png")); plt.close()
        logging.info("  -> Saved SHAP summary plot.")
    except Exception as e:
        logging.warning(f"SHAP analysis failed. Is `shap` installed? Error: {e}")

    # --- 8. Final Reports & Clean-up Classifier ---
    # Scientific: Generates a final confusion matrix and a summary CSV for manual review. Also trains a final classifier on data with outliers removed to report a "clean" accuracy score.
    # Layman's:  Creates a final "report card" (confusion matrix) to see where the computer got confused. It also creates a master spreadsheet that flags all the weird or interesting plants for a human to look at.
    logging.info("Generating final summary reports...")
    cm = confusion_matrix(y_int, features_df['kmeans_cluster'])
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("KMeans Cluster vs. True Label Confusion Matrix"); plt.xlabel("Predicted Cluster"); plt.ylabel("True Label")
    plt.savefig(os.path.join(output_dir, f"{project_name}_kmeans_confusion_matrix.png")); plt.close()
    
    review_cols = ['filepath', 'class_name', 'kmeans_cluster', 'hdbscan_cluster', 'isolation_outlier', 'lof_outlier', 'nn_label_consistency', 'gmm_hybrid_candidate']
    features_df[review_cols].to_csv(os.path.join(output_dir, f"{project_name}_suspect_samples_for_review.csv"), index=False)
    
    clean_idx = (features_df['isolation_outlier'] != -1) & (features_df['lof_outlier'] != -1)
    rf_clean = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled[clean_idx], y[clean_idx])
    acc_clean = rf_clean.score(X_scaled[clean_idx], y[clean_idx])
    logging.info(f"Random Forest accuracy after removing outliers: {acc_clean:.3f}")
    
    logging.info("--- Visualization and Analysis Complete ---")


# --- MAIN ORCHESTRATOR ---
def main(config_path):
    # --- PIPELINE CONDUCTOR ---
    # Scientific: The main function that orchestrates the entire pipeline, calling the training, extraction, and visualization stages based on the provided configuration and command-line flags.
    # Layman's:  The "manager" of the whole project. It reads the settings, then tells the training, feature extraction, and visualization workers when to start their jobs in the correct order.
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at '{config_path}'. Please create it or check the path.")
        return

    os.makedirs(config['output_dir'], exist_ok=True)
    
    all_samples, class_names = find_image_files(
        config['image_data_root'], config.get('exclude_dirs', []), config.get('exclude_files', [])
    )
    
    model_path = os.path.join(config['output_dir'], f"{config['project_name']}_best_model.pth")
    features_df = None

    # Determine which stages to run based on flags from the command line
    run_train = args.train
    run_extract = args.extract
    run_visualize = args.visualize
    
    if run_train:
        model_path = run_training(config, all_samples, class_names)
    elif not os.path.exists(model_path) and (run_extract or run_visualize):
        logging.error(f"Training is skipped, but no model found at {model_path}. Please train a model first or provide a valid path.")
        return

    if run_extract:
        features_df = run_feature_extraction(config, model_path, all_samples, class_names)

    if run_visualize:
        if features_df is None:
            features_path = os.path.join(config['output_dir'], config['feature_extraction']['output_csv_name'])
            if not os.path.exists(features_path):
                logging.error(f"Visualization is enabled, but no features CSV found at {features_path}. Please run the feature extraction stage first.")
                return
            logging.info(f"Loading features from {features_path} for visualization.")
            features_df = pd.read_csv(features_path)
        
        run_visualization(config, features_df)
        
    logging.info("GINGER pipeline has finished.")

if __name__ == "__main__":
    # --- SCRIPT ENTRY POINT ---
    # Scientific: This block executes when the script is run directly. It parses command-line arguments to determine which pipeline stages to activate.
    # Laymen's:  This is where everything starts. When you run the Python script, this part reads the flags you added (like --train or --visualize) to know what to do.
    parser = argparse.ArgumentParser(description="GINGER: Unified Deep Learning Pipeline for Species Analysis.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--train', action='store_true', help='Run the training stage.')
    parser.add_argument('--extract', action='store_true', help='Run the feature extraction stage.')
    parser.add_argument('--visualize', action='store_true', help='Run the visualization and analysis stage.')
    args = parser.parse_args()
    main(args.config)