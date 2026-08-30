import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

# Set font for plots
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EmbeddingVisualizer:
    def __init__(self, file1_path, file2_path, file3_path=None, label_path=None):
        """
        Initialize embedding visualizer
        
        Args:
            file1_path: Path to the first embedding file
            file2_path: Path to the second embedding file
            file3_path: Path to the third embedding file (optional)
            label_path: Path to the label file (optional, for coloring)
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.file3_path = file3_path
        self.label_path = label_path
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.labels = None
        
    def load_data(self):
        """Load tensor data"""
        print("Loading data...")
        
        # Load first file
        if os.path.exists(self.file1_path):
            self.data1 = torch.load(self.file1_path)
            if isinstance(self.data1, torch.Tensor):
                if self.data1.requires_grad:
                    self.data1 = self.data1.detach().numpy()
                else:
                    self.data1 = self.data1.numpy()
            print(f"File 1 loaded successfully: {self.file1_path}")
            print(f"Data 1 shape: {self.data1.shape}")
        else:
            print(f"File 1 does not exist: {self.file1_path}")
            
        # Load second file
        if os.path.exists(self.file2_path):
            self.data2 = torch.load(self.file2_path)
            if isinstance(self.data2, torch.Tensor):
                if self.data2.requires_grad:
                    self.data2 = self.data2.detach().numpy()
                else:
                    self.data2 = self.data2.numpy()
            print(f"File 2 loaded successfully: {self.file2_path}")
            print(f"Data 2 shape: {self.data2.shape}")
        else:
            print(f"File 2 does not exist: {self.file2_path}")
        
        # Load third file
        if self.file3_path and os.path.exists(self.file3_path):
            self.data3 = torch.load(self.file3_path)
            if isinstance(self.data3, torch.Tensor):
                if self.data3.requires_grad:
                    self.data3 = self.data3.detach().numpy()
                else:
                    self.data3 = self.data3.numpy()
            print(f"File 3 loaded successfully: {self.file3_path}")
            print(f"Data 3 shape: {self.data3.shape}")
        elif self.file3_path:
            print(f"File 3 does not exist: {self.file3_path}")
        
        # Load label file
        if self.label_path and os.path.exists(self.label_path):
            self.labels = torch.load(self.label_path)
            if isinstance(self.labels, torch.Tensor):
                if self.labels.requires_grad:
                    self.labels = self.labels.detach().numpy()
                else:
                    self.labels = self.labels.numpy()
            print(f"Labels loaded successfully: {self.label_path}")
            print(f"Labels shape: {self.labels.shape}")
            print(f"Unique labels: {np.unique(self.labels)}")
        elif self.label_path:
            print(f"Warning: Label file does not exist: {self.label_path}")
    
    def preprocess_data(self, data):
        """Preprocess data"""
        if data is None:
            return None
            
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        return data_scaled
    
    def visualize_embeddings(self, data, name, dim_reduction='pca'):
        """
        Visualize embeddings with dimensionality reduction
        
        Args:
            data: The embedding data (numpy array)
            name: Name for the embedding (for file naming)
            dim_reduction: 'pca' or 'tsne'
        """
        if data is None:
            return
        
        print(f"\nVisualizing {name} using {dim_reduction.upper()}...")
        
        # Preprocess data
        data_processed = self.preprocess_data(data)
        
        # Apply dimensionality reduction
        if dim_reduction.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            data_2d = reducer.fit_transform(data_processed)
            method_name = "PCA"
            explained_var = reducer.explained_variance_ratio_.sum()
            print(f"  PCA explained variance ratio: {explained_var:.4f}")
        elif dim_reduction.lower() == 'tsne':
            print("  Computing t-SNE (this may take a while)...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            data_2d = reducer.fit_transform(data_processed)
            method_name = "t-SNE"
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {dim_reduction}")
        
        # Get labels for coloring
        use_labels = self.labels is not None and len(self.labels) == len(data_2d)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if use_labels:
            # Color by labels: label 0 -> red, label 1 -> light gray (high contrast)
            label_0_mask = self.labels == 0
            label_1_mask = self.labels == 1
            
            # Plot label 0 points (red for high contrast)
            if np.any(label_0_mask):
                ax.scatter(data_2d[label_0_mask, 0], data_2d[label_0_mask, 1], 
                          c='#FF3333', s=20, alpha=0.7, label='Label 0', edgecolors='none')
            
            # Plot label 1 points (light gray)
            if np.any(label_1_mask):
                ax.scatter(data_2d[label_1_mask, 0], data_2d[label_1_mask, 1], 
                          c='#888888', s=20, alpha=0.7, label='Label 1', edgecolors='none')
            
            ax.legend()
            title_suffix = " (colored by labels)"
        else:
            # No labels, use single color
            ax.scatter(data_2d[:, 0], data_2d[:, 1], c='gray', s=20, alpha=0.6, edgecolors='none')
            title_suffix = ""
        
        ax.set_title(f'{name} - {method_name} Visualization{title_suffix}', fontsize=14)
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f'{name}_{dim_reduction}_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved to: {output_file}")
        plt.close()
    
    def visualize_combined(self, dim_reduction='pca'):
        """
        Visualize all three files in one figure with three subplots
        
        Args:
            dim_reduction: 'pca' or 'tsne'
        """
        # Collect all data files
        data_list = []
        name_list = []
        
        if self.data1 is not None:
            data_list.append(self.data1)
            name_list.append("valueGraph embedding")
        if self.data2 is not None:
            data_list.append(self.data2)
            name_list.append("Baseline embedding (Roberta)")
        if self.data3 is not None:
            data_list.append(self.data3)
            name_list.append("Baseline embedding (BotRGCN,BotGCN,BotGAT)")
        
        if len(data_list) == 0:
            print("No data to visualize!")
            return
        
        print(f"\nVisualizing {len(data_list)} files together using {dim_reduction.upper()}...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(data_list), figsize=(6*len(data_list), 6))
        if len(data_list) == 1:
            axes = [axes]
        
        # Define consistent label colors (same across all subplots)
        label_colors = {0: '', 1: '#888888'}  # Label 0: bright blue, Label 1: gray
        label_names = {0: 'Human users', 1: 'Bot users'}
        
        # Colors for different files (when no labels)
        file_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        # Check if we can use labels (all data should have same length as labels)
        can_use_labels = False
        if self.labels is not None:
            # Check if all data have the same length as labels
            all_match = all(len(data) == len(self.labels) for data in data_list)
            if all_match:
                can_use_labels = True
                print(f"  Using labels for coloring (consistent colors across all subplots)")
        
        for idx, (data, name, ax) in enumerate(zip(data_list, name_list, axes)):
            # Preprocess data
            data_processed = self.preprocess_data(data)
            
            # Apply dimensionality reduction
            if dim_reduction.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                data_2d = reducer.fit_transform(data_processed)
                method_name = "PCA"
                explained_var = reducer.explained_variance_ratio_.sum()
                print(f"  {name} PCA explained variance ratio: {explained_var:.4f}")
            elif dim_reduction.lower() == 'tsne':
                print(f"  Computing t-SNE for {name} (this may take a while)...")
                reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                data_2d = reducer.fit_transform(data_processed)
                method_name = "t-SNE"
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {dim_reduction}")
            
            if can_use_labels:
                # Color by labels: use consistent colors across all subplots
                unique_labels = np.unique(self.labels)
                
                for label_val in unique_labels:
                    label_mask = self.labels == label_val
                    color = label_colors.get(label_val, '#CCCCCC')  # Default gray if label not in dict
                    label_name = label_names.get(label_val, f'Label {label_val}')
                    
                    # Always add label to legend for each subplot
                    ax.scatter(data_2d[label_mask, 0], data_2d[label_mask, 1], 
                              c=color, s=20, alpha=0.7, label=label_name, edgecolors='none')
                
                # Show legend on every subplot
                ax.legend()
                title_suffix = " (colored by labels)"
            else:
                # No labels, use file-specific color
                ax.scatter(data_2d[:, 0], data_2d[:, 1], c=file_colors[idx], s=20, alpha=0.6, edgecolors='none')
                title_suffix = ""
            
            ax.set_title(f'{name} - {method_name}{title_suffix}', fontsize=12)
            ax.set_xlabel(f'{method_name} Component 1')
            ax.set_ylabel(f'{method_name} Component 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f'combined_{dim_reduction}_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Combined visualization saved to: {output_file}")
        plt.close()
    
    def run_visualization(self):
        """Run visualization with t-SNE"""
        # Load data
        self.load_data()
        
        # Visualize all three files together with t-SNE
        self.visualize_combined(dim_reduction='tsne')
        
        print("\nVisualization completed!")

def main():
    """Main function"""
    # File paths
    file1_path = "workshop/enron_spam_data/TwiBot-22/src/BotRGCN/twibot_22/processed_data/tweets_tensorour.pt"
    file2_path = "workshop/enron_spam_data/TwiBot-22/src/BotRGCN/twibot_22/processed_data/user_embedding512.pt"
    file3_path = "workshop/enron_spam_data/TwiBot-22/src/BotRGCN/twibot_22/processed_data/tweets_tensor.pt"
    label_path = "workshop/enron_spam_data/TwiBot-22/src/BotRGCN/twibot_22/processed_data/label.pt"
    
    # Create visualizer
    visualizer = EmbeddingVisualizer(file1_path, file2_path, file3_path, label_path)
    
    # Run visualization
    visualizer.run_visualization()

if __name__ == "__main__":
    main()
