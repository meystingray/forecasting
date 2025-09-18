import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dtaidistance import dtw
import warnings
warnings.filterwarnings('ignore')

class DTWTimeSeriesClustering:
    """
    Complete implementation of DTW-based hierarchical clustering for time series.
    Optimized for datasets with ~4000 series of <100 measurements each.
    """
    
    def __init__(self, sakoe_chiba_radius=None, linkage_method='ward'):
        """
        Initialize the clustering object.
        
        Parameters:
        -----------
        sakoe_chiba_radius : int or None
            Constraint radius for DTW. If None, will be set to 15% of series length
        linkage_method : str
            Linkage method for hierarchical clustering ('ward', 'complete', 'average')
        """
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.linkage_method = linkage_method
        self.distance_matrix_ = None
        self.linkage_matrix_ = None
        self.cluster_labels_ = None
        self.silhouette_scores_ = {}
        
    def preprocess_data(self, time_series_data, normalize=True, smooth_window=None):
        """
        Preprocess time series data.
        
        Parameters:
        -----------
        time_series_data : array-like or list of arrays
            Time series data. Can be 2D array (n_series, n_timepoints) or list of 1D arrays
        normalize : bool
            Whether to apply z-score normalization to each series
        smooth_window : int or None
            If provided, applies moving average smoothing with given window size
            
        Returns:
        --------
        processed_data : list of arrays
            Preprocessed time series data
        """
        print("Preprocessing time series data...")
        
        # Convert to list of arrays if needed
        if isinstance(time_series_data, np.ndarray):
            if time_series_data.ndim == 2:
                processed_data = [time_series_data[i] for i in range(len(time_series_data))]
            else:
                processed_data = [time_series_data]
        else:
            processed_data = [np.array(ts) for ts in time_series_data]
        
        # Apply smoothing if requested
        if smooth_window is not None:
            print(f"Applying smoothing with window size {smooth_window}")
            smoothed_data = []
            for ts in processed_data:
                # Simple moving average
                if len(ts) >= smooth_window:
                    smoothed = np.convolve(ts, np.ones(smooth_window)/smooth_window, mode='valid')
                    # Pad to maintain original length
                    pad_size = len(ts) - len(smoothed)
                    smoothed = np.pad(smoothed, (pad_size//2, pad_size - pad_size//2), mode='edge')
                else:
                    smoothed = ts
                smoothed_data.append(smoothed)
            processed_data = smoothed_data
        
        # Apply normalization if requested
        if normalize:
            print("Applying z-score normalization")
            normalized_data = []
            for ts in processed_data:
                if np.std(ts) > 1e-8:  # Avoid division by zero
                    normalized = (ts - np.mean(ts)) / np.std(ts)
                else:
                    normalized = ts - np.mean(ts)  # Just center if std is too small
                normalized_data.append(normalized)
            processed_data = normalized_data
        
        print(f"Preprocessed {len(processed_data)} time series")
        return processed_data
    
    def compute_distance_matrix(self, time_series_data, use_multiprocessing=True, n_jobs=-1):
        """
        Compute DTW distance matrix with memory optimization.
        
        Parameters:
        -----------
        time_series_data : list of arrays
            Preprocessed time series data
        use_multiprocessing : bool
            Whether to use multiprocessing for faster computation
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        print("Computing DTW distance matrix...")
        n_series = len(time_series_data)
        series_length = len(time_series_data[0])
        
        # Set Sakoe-Chiba radius if not provided
        if self.sakoe_chiba_radius is None:
            self.sakoe_chiba_radius = max(1, int(0.15 * series_length))
            print(f"Using Sakoe-Chiba radius: {self.sakoe_chiba_radius}")
        
        print(f"Computing distances for {n_series} series...")
        print(f"Estimated memory usage: ~{(n_series**2 * 4) / (1024**2):.1f} MB")
        
        # Compute distance matrix using dtaidistance
        if use_multiprocessing:
            self.distance_matrix_ = dtw.distance_matrix_fast(
                time_series_data,
                window=self.sakoe_chiba_radius,
                use_pruning=True,
                use_mp=True,
                max_length_diff=series_length // 4  # Allow some length difference
            )
        else:
            self.distance_matrix_ = dtw.distance_matrix_fast(
                time_series_data,
                window=self.sakoe_chiba_radius,
                use_pruning=True
            )
        
        print("Distance matrix computation completed!")
        return self.distance_matrix_
    
    def perform_clustering(self, n_clusters_range=None):
        """
        Perform hierarchical clustering and evaluate different numbers of clusters.
        
        Parameters:
        -----------
        n_clusters_range : tuple or None
            Range of cluster numbers to evaluate (min, max). If None, uses (2, 15)
        """
        if self.distance_matrix_ is None:
            raise ValueError("Distance matrix not computed. Call compute_distance_matrix first.")
        
        print("Performing hierarchical clustering...")
        
        # Convert to condensed distance matrix for scipy
        condensed_distances = squareform(self.distance_matrix_, checks=False)
        
        # Perform hierarchical clustering
        self.linkage_matrix_ = linkage(condensed_distances, method=self.linkage_method)
        
        # Evaluate different numbers of clusters
        if n_clusters_range is None:
            n_clusters_range = (2, min(15, len(self.distance_matrix_) // 10))
        
        print(f"Evaluating cluster numbers from {n_clusters_range[0]} to {n_clusters_range[1]}...")
        
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            labels = fcluster(self.linkage_matrix_, n_clusters, criterion='maxclust')
            
            # Compute silhouette score
            try:
                sil_score = silhouette_score(self.distance_matrix_, labels, metric='precomputed')
                self.silhouette_scores_[n_clusters] = sil_score
                print(f"  {n_clusters} clusters: silhouette score = {sil_score:.3f}")
            except:
                print(f"  {n_clusters} clusters: silhouette score computation failed")
        
        # Find optimal number of clusters
        if self.silhouette_scores_:
            optimal_n_clusters = max(self.silhouette_scores_.keys(), 
                                   key=lambda k: self.silhouette_scores_[k])
            self.cluster_labels_ = fcluster(self.linkage_matrix_, optimal_n_clusters, criterion='maxclust')
            print(f"\nOptimal number of clusters: {optimal_n_clusters}")
            print(f"Best silhouette score: {self.silhouette_scores_[optimal_n_clusters]:.3f}")
        
        return self.linkage_matrix_
    
    def get_cluster_labels(self, n_clusters=None):
        """
        Get cluster labels for a specific number of clusters.
        
        Parameters:
        -----------
        n_clusters : int or None
            Number of clusters. If None, uses the optimal number found during clustering
            
        Returns:
        --------
        labels : array
            Cluster labels for each time series
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Clustering not performed. Call perform_clustering first.")
        
        if n_clusters is None:
            if self.cluster_labels_ is None:
                raise ValueError("No optimal clustering found. Specify n_clusters explicitly.")
            return self.cluster_labels_
        else:
            return fcluster(self.linkage_matrix_, n_clusters, criterion='maxclust')
    
    def plot_dendrogram(self, max_display=50, figsize=(12, 8)):
        """
        Plot dendrogram for hierarchical clustering results.
        
        Parameters:
        -----------
        max_display : int
            Maximum number of series to display in dendrogram
        figsize : tuple
            Figure size for the plot
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Clustering not performed. Call perform_clustering first.")
        
        plt.figure(figsize=figsize)
        
        # If too many series, truncate the dendrogram
        if len(self.distance_matrix_) > max_display:
            dendrogram(self.linkage_matrix_, truncate_mode='level', p=10)
            plt.title(f'Hierarchical Clustering Dendrogram (Truncated)')
        else:
            dendrogram(self.linkage_matrix_)
            plt.title('Hierarchical Clustering Dendrogram')
        
        plt.xlabel('Time Series Index or Cluster Size')
        plt.ylabel('DTW Distance')
        plt.tight_layout()
        plt.show()
    
    def plot_silhouette_analysis(self, figsize=(10, 6)):
        """
        Plot silhouette scores for different numbers of clusters.
        """
        if not self.silhouette_scores_:
            print("No silhouette scores available. Run perform_clustering first.")
            return
        
        plt.figure(figsize=figsize)
        n_clusters = list(self.silhouette_scores_.keys())
        scores = list(self.silhouette_scores_.values())
        
        plt.plot(n_clusters, scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Optimal Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        # Highlight the best score
        best_n = max(self.silhouette_scores_.keys(), key=lambda k: self.silhouette_scores_[k])
        best_score = self.silhouette_scores_[best_n]
        plt.plot(best_n, best_score, 'ro', markersize=12, alpha=0.7)
        plt.annotate(f'Best: {best_n} clusters\nScore: {best_score:.3f}', 
                    xy=(best_n, best_score), xytext=(best_n + 1, best_score + 0.02),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.show()
    
    def plot_cluster_examples(self, time_series_data, n_examples=5, figsize=(15, 10)):
        """
        Plot example time series from each cluster.
        
        Parameters:
        -----------
        time_series_data : list of arrays
            Original time series data
        n_examples : int
            Number of examples to show per cluster
        figsize : tuple
            Figure size for the plot
        """
        if self.cluster_labels_ is None:
            raise ValueError("No cluster labels available. Run perform_clustering first.")
        
        unique_clusters = np.unique(self.cluster_labels_)
        n_clusters = len(unique_clusters)
        
        fig, axes = plt.subplots(n_clusters, 1, figsize=figsize, sharex=True)
        if n_clusters == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_indices = np.where(self.cluster_labels_ == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # Select random examples from this cluster
            n_show = min(n_examples, cluster_size)
            selected_indices = np.random.choice(cluster_indices, n_show, replace=False)
            
            for idx in selected_indices:
                axes[i].plot(time_series_data[idx], color=colors[i], alpha=0.7, linewidth=1)
            
            # Plot cluster centroid (mean)
            cluster_data = [time_series_data[idx] for idx in cluster_indices]
            # Compute mean (simple average - more sophisticated methods could be used)
            min_len = min(len(ts) for ts in cluster_data)
            truncated_data = [ts[:min_len] for ts in cluster_data]
            centroid = np.mean(truncated_data, axis=0)
            axes[i].plot(centroid, color='black', linewidth=3, linestyle='--', 
                        label=f'Centroid')
            
            axes[i].set_title(f'Cluster {cluster_id} (n={cluster_size})')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()
    
    def get_cluster_summary(self):
        """
        Get summary statistics for each cluster.
        
        Returns:
        --------
        summary : dict
            Dictionary with cluster statistics
        """
        if self.cluster_labels_ is None:
            raise ValueError("No cluster labels available. Run perform_clustering first.")
        
        unique_clusters, counts = np.unique(self.cluster_labels_, return_counts=True)
        
        summary = {
            'n_clusters': len(unique_clusters),
            'cluster_sizes': dict(zip(unique_clusters, counts)),
            'total_series': len(self.cluster_labels_),
            'silhouette_scores': self.silhouette_scores_
        }
        
        return summary


# Example usage and helper functions
def load_sample_data(n_series=100, series_length=50):
    """
    Generate sample time series data for testing.
    Creates three types of patterns: linear trends, sine waves, and random walks.
    """
    np.random.seed(42)
    time_series = []
    
    for i in range(n_series):
        if i < n_series // 3:
            # Linear trends with noise
            t = np.linspace(0, 1, series_length)
            slope = np.random.uniform(-2, 2)
            ts = slope * t + np.random.normal(0, 0.1, series_length)
        elif i < 2 * n_series // 3:
            # Sine waves with different frequencies and phases
            t = np.linspace(0, 4*np.pi, series_length)
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, np.pi)
            ts = np.sin(freq * t + phase) + np.random.normal(0, 0.1, series_length)
        else:
            # Random walks
            steps = np.random.normal(0, 0.1, series_length)
            ts = np.cumsum(steps)
        
        time_series.append(ts)
    
    return time_series


def run_complete_analysis(time_series_data, sakoe_chiba_radius=None, 
                         normalize=True, smooth_window=None):
    """
    Run complete DTW clustering analysis pipeline.
    
    Parameters:
    -----------
    time_series_data : array-like or list of arrays
        Time series data
    sakoe_chiba_radius : int or None
        DTW constraint radius
    normalize : bool
        Whether to normalize the data
    smooth_window : int or None
        Smoothing window size
        
    Returns:
    --------
    clustering_model : DTWTimeSeriesClustering
        Fitted clustering model
    """
    # Initialize clustering model
    clustering_model = DTWTimeSeriesClustering(sakoe_chiba_radius=sakoe_chiba_radius)
    
    # Preprocess data
    processed_data = clustering_model.preprocess_data(
        time_series_data, 
        normalize=normalize, 
        smooth_window=smooth_window
    )
    
    # Compute distance matrix
    clustering_model.compute_distance_matrix(processed_data)
    
    # Perform clustering
    clustering_model.perform_clustering()
    
    # Display results
    print("\n" + "="*50)
    print("CLUSTERING RESULTS SUMMARY")
    print("="*50)
    summary = clustering_model.get_cluster_summary()
    print(f"Total time series: {summary['total_series']}")
    print(f"Optimal number of clusters: {summary['n_clusters']}")
    print(f"Cluster sizes: {summary['cluster_sizes']}")
    
    # Plot results
    clustering_model.plot_silhouette_analysis()
    clustering_model.plot_dendrogram()
    clustering_model.plot_cluster_examples(processed_data)
    
    return clustering_model


# Example usage
if __name__ == "__main__":
    print("DTW Time Series Clustering - Example Usage")
    print("="*50)
    
    # Generate sample data (replace with your actual data loading)
    print("Loading sample data...")
    sample_data = load_sample_data(n_series=200, series_length=80)
    
    # Run complete analysis
    model = run_complete_analysis(
        time_series_data=sample_data,
        normalize=True,
        smooth_window=3  # Optional smoothing
    )
    
    # Get cluster labels for a specific number of clusters
    labels_5_clusters = model.get_cluster_labels(n_clusters=5)
    print(f"\nCluster labels for 5 clusters: {np.unique(labels_5_clusters, return_counts=True)}")