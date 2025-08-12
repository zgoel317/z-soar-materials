#!/usr/bin/env python3
"""
Apply UMAP to latent activation patterns using Jaccard metric.

This script:
1. Loads cached latent activations 
2. Builds a sparse matrix where rows are latents and columns are token positions
3. Sums across all batches to get overall activation patterns
4. Applies UMAP to these patterns using Jaccard metric
5. Optionally filters latents based on Jaccard similarity threshold

Usage:
    python umap_latent_patterns.py --cache_dir path/to/cache --output_path umap_embeddings.pt
    python umap_latent_patterns.py --cache_dir path/to/cache --jaccard_threshold 0.1 --create_plots

Network Graph Performance Tips:
    - Use --max_nodes 5000 for very large graphs
    - Use --auto_connectivity_threshold to find largest connected component using top 10% of most activated latents and its optimal threshold
    - Use --top_k_edges 20 to keep only top 20 highest Jaccard similarities per latent (dramatically reduces edges, default: 20)
    - After graph stabilizes, disable physics for snappy panning/zooming
    - Open HTML in modern browser for best performance
"""

import argparse
import gc
import os
from pathlib import Path
from typing import Optional

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from safetensors.numpy import load_file
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import time
from safetensors import safe_open
import networkx as nx

# Try to import pyvis for interactive network visualization
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Warning: pyvis not available. Network graphs will use matplotlib fallback.")

# Check if HDBSCAN is available
HDBSCAN_AVAILABLE = True







def load_and_build_coactivation_matrix(cache_dir: Path, token_batch_size: int = 100_000, use_gpu: bool = False) -> torch.Tensor:
    """
    Load cached latents and build a coactivation matrix using the same code as neighbours.py.
    
    Args:
        cache_dir: Directory containing cached latent files (.safetensors)
        token_batch_size: Batch size for processing tokens to manage memory
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        torch.Tensor: Coactivation matrix where (i,j) = how often latents i and j activate together
    """
    print("Computing co-occurrence matrix (using neighbours.py code)")
    assert (
        cache_dir is not None
    ), "Cache directory is required for co-occurrence-based neighbours"
    paths = os.listdir(cache_dir)

    all_locations = []
    for path in paths:
        if path.endswith(".safetensors"):
            print(f"Loading {path}...")
            split_data = load_file(cache_dir / path)
            first_feature = int(path.split("/")[-1].split("_")[0])
            locations = torch.tensor(split_data["locations"].astype(np.int64))
            locations[:, 2] = locations[:, 2] + first_feature

            all_locations.append(locations)

    # concatenate the locations and activations
    locations = torch.cat(all_locations)
    print(f"Total activations loaded: {len(locations):,}")

    batch_index = locations[:, 0]
    ctx_index = locations[:, 1]
    latent_index = locations[:, 2]

    n_latents = int(latent_index.max().item()) + 1
    ctx_len = locations[:, 1].max().item() + 1
    n_batches = locations[:, 0].max().item() + 1
    
    print(f"Dataset stats:")
    print(f"  - Number of latents: {n_latents:,}")
    print(f"  - Context length: {ctx_len}")
    print(f"  - Number of batches: {n_batches:,}")
    
    # Convert from (batch_id, ctx_pos) to a unique 1D index
    idx_cantor = (
        (batch_index + ctx_index) * (batch_index + ctx_index + 1)
    ) // 2 + ctx_index

    # Sort the indices, because they are not sorted after concatenation
    idx_cantor, idx_cantor_sorted_idx = idx_cantor.sort(dim=0, stable=True)
    latent_index = latent_index[idx_cantor_sorted_idx]

    current_batch_size = token_batch_size
    co_occurrence_matrix = None
    done = False
    while not done:
        try:
            print(f"Building coactivation matrix with batch size {current_batch_size:,}...")
            
            # Find indices where idx_cantor crosses each batch boundary
            batch_boundaries = torch.arange(
                0, n_batches, current_batch_size // ctx_len
            )
            cantor_batch_boundaries = (
                (batch_boundaries + ctx_len) * (batch_boundaries + ctx_len + 1)
            ) // 2 + ctx_len

            batch_boundaries_tensor = torch.searchsorted(
                idx_cantor, cantor_batch_boundaries
            )
            batch_boundaries = [0] + batch_boundaries_tensor.tolist()

            if batch_boundaries[-1] != len(idx_cantor):
                batch_boundaries.append(len(idx_cantor))
                
            co_occurrence_matrix = torch.zeros(
                (n_latents, n_latents), dtype=torch.int64
            )
            
            total_batches = len(batch_boundaries) - 1
            print(f"Processing {total_batches} batches...")
            
            # Estimate time
            estimated_time_per_batch = 0.3  # seconds per batch
            total_estimated_time = total_batches * estimated_time_per_batch
            print(f"Estimated time: ~{total_estimated_time/60:.1f} minutes ({total_estimated_time:.0f}s)")

            for start, end in tqdm(
                zip(batch_boundaries[:-1], batch_boundaries[1:]),
                total=total_batches,
                desc="Building coactivation matrix",
                unit="batch"
            ):
                # get all ind_cantor values between start and start
                selected_idx_cantor = idx_cantor[start:end]
                # shift the indices to start from 0
                selected_idx_cantor = selected_idx_cantor - selected_idx_cantor[0]
                selected_latent_index = latent_index[start:end]
                number_positions = int(selected_idx_cantor.max().item()) + 1

                # create a sparse matrix of the selected indices
                sparse_matrix_indices = torch.stack(
                    [selected_latent_index, selected_idx_cantor], dim=0
                )
                sparse_matrix = torch.sparse_coo_tensor(
                    sparse_matrix_indices,
                    torch.ones(len(selected_latent_index)),
                    (n_latents, number_positions),
                    check_invariants=True,
                )
                sparse_matrix = sparse_matrix.cuda()
                partial_cooc = (sparse_matrix @ sparse_matrix.T).cpu()
                co_occurrence_matrix += partial_cooc.int()

                del sparse_matrix, partial_cooc
                torch.cuda.empty_cache()
            done = True

        except RuntimeError:  # Out of memory
            current_batch_size = current_batch_size // 2
            if current_batch_size < 2:
                raise ValueError(
                    "Batch size is too small to compute similarity matrix. "
                    "You don't have enough memory."
                )
            print(f"Out of memory, reducing batch size to {current_batch_size:,}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Coactivation matrix construction completed!")
    print(f"Matrix shape: {co_occurrence_matrix.shape}")
    print(f"Total co-occurrences: {co_occurrence_matrix.sum().item():,}")
    print(f"Non-zero entries: {(co_occurrence_matrix > 0).sum().item():,}")
    
    # Convert to float for UMAP processing
    coactivation_matrix = co_occurrence_matrix.float()
    
    # Estimate time for next steps
    n_latents = coactivation_matrix.shape[0]
    estimated_umap_time = max(30, n_latents // 1000)  # Rough estimate: 30s minimum, +1s per 1000 latents
    estimated_hdbscan_time = max(10, n_latents // 2000)  # Rough estimate: 10s minimum, +1s per 2000 latents
    
    print(f"\nTime estimates for next steps:")
    print(f"  - UMAP computation: ~{estimated_umap_time}s")
    print(f"  - HDBSCAN clustering: ~{estimated_hdbscan_time}s")
    print(f"  - Total remaining time: ~{estimated_umap_time + estimated_hdbscan_time}s")
    
    return coactivation_matrix





def apply_umap_to_patterns(coactivation_matrix: torch.Tensor, 
                          n_components: int = 2,
                          n_neighbors: int = 15,
                          min_dist: float = 0.1,
                          random_state: int = 42,
                          use_gpu: bool = False,
                          use_coactivation_directly: bool = False,
                          jaccard_threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply UMAP to the coactivation matrix using either Jaccard metric or direct coactivation distances.
    
    Args:
        coactivation_matrix: Coactivation matrix where (i,j) = co-occurrence count of latents i and j
        n_components: Number of dimensions for UMAP embedding
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed
        use_gpu: Whether to use GPU acceleration
        use_coactivation_directly: If True, use coactivation matrix directly as distance (1/entry + epsilon)
        jaccard_threshold: Jaccard similarity threshold for filtering latents (0.0 = no filtering)
        
    Returns:
        tuple: (UMAP embeddings, active mask indicating which latents were kept)
    """
    
    # Convert to numpy and ensure proper dtype
    coactivation_data = coactivation_matrix.numpy().astype(np.float32)
    
    print(f"Using all {coactivation_matrix.shape[0]} latents")
    
    # Always compute Jaccard matrix for filtering purposes
    print("Computing Jaccard similarity matrix for filtering...")
    
    # Copy the exact compute_jaccard function from neighbours.py
    def compute_jaccard(cooc_matrix):
        self_occurrence = cooc_matrix.diagonal()
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        jaccard_matrix = cooc_matrix / (
            self_occurrence[:, None] + self_occurrence - cooc_matrix + epsilon
        )
        # remove the diagonal and keep the upper triangle
        del cooc_matrix, self_occurrence
        torch.cuda.empty_cache()
        return jaccard_matrix
    
    # Compute Jaccard similarity matrix
    jaccard_matrix = compute_jaccard(coactivation_data)
    
    # Print distribution of Jaccard matrix entries
    jaccard_values = jaccard_matrix[jaccard_matrix > 0]  # Exclude zeros (diagonal and no overlap)
    if len(jaccard_values) > 0:
        print(f"Jaccard matrix distribution:")
        print(f"  Non-zero values: {len(jaccard_values):,}")
        print(f"  Min: {jaccard_values.min():.4f}")
        print(f"  Max: {jaccard_values.max():.4f}")
        print(f"  Mean: {jaccard_values.mean():.4f}")
        print(f"  Median: {np.median(jaccard_values):.4f}")
        print(f"  Std: {jaccard_values.std():.4f}")
        print(f"  Percentiles: 25th={np.percentile(jaccard_values, 25):.4f}, 75th={np.percentile(jaccard_values, 75):.4f}")
    else:
        print("Jaccard matrix has no non-zero values (only diagonal)")
    
    # Apply Jaccard threshold filtering if specified
    if jaccard_threshold > 0.0:
        print(f"Applying Jaccard threshold filtering (threshold: {jaccard_threshold})...")
        
        # Create mask for entries that don't meet the threshold
        # Exclude self-connections from threshold checking
        jaccard_matrix_no_self = jaccard_matrix.copy()
        np.fill_diagonal(jaccard_matrix_no_self, 0.0)
        
        # Create mask for entries below or equal to threshold
        below_threshold_mask = jaccard_matrix_no_self <= jaccard_threshold
        
        # Count how many entries will be zeroed out
        n_entries_below_threshold = np.sum(below_threshold_mask)
        total_entries = below_threshold_mask.size - below_threshold_mask.shape[0]  # Exclude diagonal
        
        print(f"Jaccard filtering: zeroing out {n_entries_below_threshold:,} entries below or equal to threshold")
        print(f"  - Total entries (excluding diagonal): {total_entries:,}")
        print(f"  - Entries below or equal to threshold: {100 * n_entries_below_threshold / total_entries:.1f}%")
        
        # Set entries below threshold to 0 in both matrices
        coactivation_data[below_threshold_mask] = 0.0
        jaccard_matrix[below_threshold_mask] = 0.0
        
        # Create active_mask for compatibility (all latents are kept, but some connections are zeroed)
        active_mask = np.ones(coactivation_matrix.shape[0], dtype=bool)
        print(f"Matrices preserved with original shape: {coactivation_data.shape}")
    else:
        # No filtering - keep all latents and connections
        active_mask = np.ones(coactivation_matrix.shape[0], dtype=bool)
        print("No Jaccard threshold filtering applied")
    
    if use_coactivation_directly:
        print("Using coactivation matrix directly as distance matrix...")
        
        # Convert coactivation matrix to distance matrix: 1/entry + epsilon
        epsilon = 1e-8
        distance_matrix = 1.0 / (coactivation_data + epsilon)
        
        # Remove diagonal (self-distance) and ensure symmetry
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
        
        print("Applying UMAP with precomputed coactivation distances...")
        
        # Apply UMAP with precomputed distances
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='precomputed',
            verbose=True
        )
        
        embeddings = umap_model.fit_transform(distance_matrix)
        
    else:
        print("Computing Jaccard similarity from coactivation matrix...")
        
        # Convert similarity to distance for UMAP
        jaccard_distance = 1.0 - jaccard_matrix
        del jaccard_matrix
        torch.cuda.empty_cache()
        
        # Remove diagonal (self-similarity) and ensure symmetry
        np.fill_diagonal(jaccard_distance, 0.0)
        jaccard_distance = np.maximum(jaccard_distance, jaccard_distance.T)
        
        print("Applying UMAP with precomputed distances...")
        
        # Apply UMAP with precomputed distances
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='precomputed',
            verbose=True
        )
        
        embeddings = umap_model.fit_transform(jaccard_distance)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings, active_mask


def apply_hdbscan_clustering(embeddings: np.ndarray,
                           min_cluster_size: int = 10,
                           min_samples: int = 5,
                           cluster_selection_epsilon: float = 0.0,
                           alpha: float = 1.0) -> tuple[np.ndarray, Optional]:
    """
    Apply HDBSCAN clustering to UMAP embeddings.
    
    Args:
        embeddings: UMAP embeddings
        min_cluster_size: Minimum size of clusters
        min_samples: Number of samples in neighborhood for core points
        cluster_selection_epsilon: Distance threshold for cluster selection
        alpha: Distance scaling parameter
        
    Returns:
        tuple: (cluster_labels, hdbscan_model or None)
    """
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available, returning all points as noise")
        return np.array([-1] * len(embeddings)), None
    print("Applying HDBSCAN clustering...")
    
    # Initialize HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
        prediction_data=True,
        core_dist_n_jobs=-1  # Use all available cores
    )
    
    # Fit and predict
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Print clustering statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"HDBSCAN clustering completed!")
    print(f"  - Number of clusters: {n_clusters}")
    print(f"  - Number of noise points: {n_noise}")
    print(f"  - Clustered points: {len(cluster_labels) - n_noise}")
    print(f"  - Clustering percentage: {100 * (len(cluster_labels) - n_noise) / len(cluster_labels):.1f}%")
    
    return cluster_labels, clusterer


def create_interactive_3d_plot_no_clustering(embeddings: np.ndarray, output_dir: Path, 
                                           prefix: str = "latent_umap", title_suffix: str = "", 
                                           latent_indices: np.ndarray = None):
    """
    Create an interactive 3D HTML plot using Plotly without clustering.
    
    Args:
        embeddings: UMAP embeddings (n_samples, n_dims)
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        title_suffix: Additional text for the title
        latent_indices: Array of latent indices for coloring and hover info (optional)
    """
    if embeddings.shape[1] < 3:
        print("Skipping 3D plot - embeddings have fewer than 3 dimensions")
        return
    
    # Create color gradient based on latent indices if available
    if latent_indices is not None:
        # Create color gradient from latent indices (indigo to red)
        trace = go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=latent_indices,
                colorscale='RdBu',  # Red to Blue (indigo-like)
                colorbar=dict(title="Latent Index"),
                opacity=0.7
            ),
            name='All Latents',
            hovertemplate='<b>All Latents</b><br>' +
                        'Latent Index: %{marker.color}<br>' +
                        'X: %{x:.3f}<br>' +
                        'Y: %{y:.3f}<br>' +
                        'Z: %{z:.3f}<br>' +
                        '<extra></extra>'
        )
    else:
        # Single trace for all points without color gradient
        trace = go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.7
            ),
            name='All Latents',
            hovertemplate='<b>All Latents</b><br>' +
                        'X: %{x:.3f}<br>' +
                        'Y: %{y:.3f}<br>' +
                        'Z: %{z:.3f}<br>' +
                        '<extra></extra>'
        )
    
    traces = [trace]
    title = f'UMAP Embeddings (3D Interactive){title_suffix}<br>{embeddings.shape[0]:,} Latents'
    
    # Create the figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2', 
            zaxis_title='UMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save as interactive HTML
    html_filename = f"{prefix}_3d_interactive_no_clustering.html"
    fig.write_html(output_dir / html_filename)
    print(f"Interactive 3D plot (no clustering) saved to {output_dir / html_filename}")
    print(f"  - Open this file in a web browser to rotate and explore the 3D visualization")


def create_interactive_3d_plot(embeddings: np.ndarray, cluster_labels: np.ndarray, output_dir: Path, 
                              prefix: str = "latent_umap", title_suffix: str = "", latent_indices: np.ndarray = None):
    """
    Create an interactive 3D HTML plot using Plotly with clustering.
    
    Args:
        embeddings: UMAP embeddings (n_samples, n_dims)
        cluster_labels: Cluster labels from HDBSCAN
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        title_suffix: Additional text for the title
        latent_indices: Array of latent indices for coloring and hover info (optional)
    """
    if embeddings.shape[1] < 3:
        print("Skipping 3D plot - embeddings have fewer than 3 dimensions")
        return
    
    # Create color map for clusters
    unique_labels = sorted(set(cluster_labels))
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Set1
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Create traces for each cluster
    traces = []
    for label in unique_labels:
        mask = cluster_labels == label
        if label == -1:
            name = 'Noise'
            color = 'black'
        else:
            name = f'Cluster {label}'
            color = color_map[label]
        
        # Add latent index info to hover template if available
        if latent_indices is not None:
            hover_template = '<b>%{fullData.name}</b><br>' + \
                           'Latent Index: %{customdata}<br>' + \
                           'X: %{x:.3f}<br>' + \
                           'Y: %{y:.3f}<br>' + \
                           'Z: %{z:.3f}<br>' + \
                           '<extra></extra>'
            customdata = latent_indices[mask]
        else:
            hover_template = '<b>%{fullData.name}</b><br>' + \
                           'X: %{x:.3f}<br>' + \
                           'Y: %{y:.3f}<br>' + \
                           'Z: %{z:.3f}<br>' + \
                           '<extra></extra>'
            customdata = None
        
        trace = go.Scatter3d(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1], 
            z=embeddings[mask, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                opacity=0.7
            ),
            name=name,
            hovertemplate=hover_template,
            customdata=customdata
        )
        traces.append(trace)
    
    title = f'UMAP Embeddings with HDBSCAN Clustering (3D Interactive){title_suffix}<br>' + \
            f'{len(unique_labels) - (1 if -1 in unique_labels else 0)} Clusters, {embeddings.shape[0]:,} Latents'
    
    # Create the figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2', 
            zaxis_title='UMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save as interactive HTML
    html_filename = f"{prefix}_3d_interactive.html"
    fig.write_html(output_dir / html_filename)
    print(f"Interactive 3D plot saved to {output_dir / html_filename}")
    print(f"  - Open this file in a web browser to rotate and explore the 3D visualization")


def plot_umap_embeddings_only(embeddings: np.ndarray, output_dir: Path, prefix: str = "latent_umap", latent_indices: np.ndarray = None):
    """
    Plot UMAP embeddings without clustering.
    
    Args:
        embeddings: UMAP embeddings (n_samples, n_dims)
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        latent_indices: Array of latent indices for coloring and hover info (optional)
    """
    print("Creating UMAP embeddings plot (no clustering)...")
    
    n_dims = embeddings.shape[1]
    
    if n_dims == 2:
        # 2D plot
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                    alpha=0.6, s=1, c='blue', edgecolors='none')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP Embeddings of Latent Activation Patterns\n(All Latents)')
        plt.grid(True, alpha=0.3)
        
        # Add text with statistics
        stats_text = f'Total Latents: {embeddings.shape[0]:,}\nEmbedding Dimensions: {embeddings.shape[1]}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_embeddings_only.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    elif n_dims == 4:
        # Create multiple 2D projections for 4D data
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('UMAP Embeddings of Latent Activation Patterns (4D)\nAll Latents - Multiple 2D Projections', fontsize=16)
        
        # Plot all 2D combinations
        combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for idx, (dim1, dim2) in enumerate(combinations):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            ax.scatter(embeddings[:, dim1], embeddings[:, dim2], 
                      alpha=0.6, s=1, c='blue', edgecolors='none')
            ax.set_xlabel(f'UMAP Dimension {dim1+1}')
            ax.set_ylabel(f'UMAP Dimension {dim2+1}')
            ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Total Latents: {embeddings.shape[0]:,}\nEmbedding Dimensions: {embeddings.shape[1]}'
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_embeddings_only.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a 3D plot using first 3 dimensions
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
                           alpha=0.6, s=1, c='blue', edgecolors='none')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_zlabel('UMAP Dimension 3')
        ax.set_title('UMAP Embeddings (3D View - First 3 Dimensions)\nAll Latents')
        
        # Add statistics text
        stats_text = f'Total Latents: {embeddings.shape[0]:,}\nShowing: Dimensions 1-3 of {embeddings.shape[1]}'
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_embeddings_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # For other dimensions, create a grid of 2D projections
        n_plots = min(6, n_dims * (n_dims - 1) // 2)  # Max 6 plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'UMAP Embeddings of Latent Activation Patterns ({n_dims}D)\nAll Latents - Multiple 2D Projections', fontsize=16)
        
        plot_idx = 0
        for dim1 in range(n_dims):
            for dim2 in range(dim1 + 1, n_dims):
                if plot_idx >= n_plots:
                    break
                row, col = plot_idx // 3, plot_idx % 3
                ax = axes[row, col]
                
                ax.scatter(embeddings[:, dim1], embeddings[:, dim2], 
                          alpha=0.6, s=1, c='blue', edgecolors='none')
                ax.set_xlabel(f'UMAP Dimension {dim1+1}')
                ax.set_ylabel(f'UMAP Dimension {dim2+1}')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, 6):
            row, col = idx // 3, idx % 3
            axes[row, col].set_visible(False)
        
        # Add statistics text
        stats_text = f'Total Latents: {embeddings.shape[0]:,}\nEmbedding Dimensions: {embeddings.shape[1]}'
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_embeddings_only.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"UMAP embeddings plot saved to {output_dir / f'{prefix}_embeddings_only.png'}")
    if n_dims >= 3:
        print(f"3D UMAP embeddings plot saved to {output_dir / f'{prefix}_embeddings_3d.png'}")
        # Create interactive 3D plot
        create_interactive_3d_plot_no_clustering(embeddings, output_dir, prefix, " (No Clustering)", latent_indices)


def plot_umap_results(embeddings: np.ndarray, cluster_labels: np.ndarray, output_dir: Path, prefix: str = "latent_umap", latent_indices: np.ndarray = None):
    """
    Plot UMAP embeddings with HDBSCAN clustering results.
    
    Args:
        embeddings: UMAP embeddings (n_samples, n_dims)
        cluster_labels: Cluster labels from HDBSCAN
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        latent_indices: Array of latent indices for coloring and hover info (optional)
    """
    print("Creating UMAP visualization plots...")
    
    n_dims = embeddings.shape[1]
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    # Create color map for clusters
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    if n_dims == 2:
        # 2D clustered plot
        plt.figure(figsize=(12, 10))
        
        for label in unique_labels:
            mask = cluster_labels == label
            if label == -1:
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                           alpha=0.6, s=1, c='black', label='Noise', edgecolors='none')
            else:
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                           alpha=0.6, s=1, c=color_map[label], label=f'Cluster {label}', edgecolors='none')
        
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title(f'UMAP Embeddings with HDBSCAN Clustering\n{n_clusters} Clusters, {embeddings.shape[0]:,} Latents')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_clustered.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    elif n_dims == 4:
        # Create multiple 2D projections for 4D data
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'UMAP Embeddings with HDBSCAN Clustering (4D)\n{n_clusters} Clusters, {embeddings.shape[0]:,} Latents - Multiple 2D Projections', fontsize=16)
        
        # Plot all 2D combinations
        combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for idx, (dim1, dim2) in enumerate(combinations):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            for label in unique_labels:
                mask = cluster_labels == label
                if label == -1:
                    ax.scatter(embeddings[mask, dim1], embeddings[mask, dim2], 
                              alpha=0.6, s=1, c='black', label='Noise' if idx == 0 else "", edgecolors='none')
                else:
                    ax.scatter(embeddings[mask, dim1], embeddings[mask, dim2], 
                              alpha=0.6, s=1, c=color_map[label], label=f'Cluster {label}' if idx == 0 else "", edgecolors='none')
            
            ax.set_xlabel(f'{method} Dimension {dim1+1}')
            ax.set_ylabel(f'{method} Dimension {dim2+1}')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_clustered.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a 3D plot using first 3 dimensions
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for label in unique_labels:
            mask = cluster_labels == label
            if label == -1:
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1], embeddings[mask, 2], 
                          alpha=0.6, s=1, c='black', label='Noise', edgecolors='none')
            else:
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1], embeddings[mask, 2], 
                          alpha=0.6, s=1, c=color_map[label], label=f'Cluster {label}', edgecolors='none')
        
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_zlabel('UMAP Dimension 3')
        ax.set_title(f'UMAP Embeddings with HDBSCAN Clustering (3D View)\n{n_clusters} Clusters, {embeddings.shape[0]:,} Latents')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_clustered_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # For other dimensions, create a grid of 2D projections
        n_plots = min(6, n_dims * (n_dims - 1) // 2)  # Max 6 plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'UMAP Embeddings with HDBSCAN Clustering ({n_dims}D)\n{n_clusters} Clusters, {embeddings.shape[0]:,} Latents - Multiple 2D Projections', fontsize=16)
        
        plot_idx = 0
        for dim1 in range(n_dims):
            for dim2 in range(dim1 + 1, n_dims):
                if plot_idx >= n_plots:
                    break
                row, col = plot_idx // 3, plot_idx % 3
                ax = axes[row, col]
                
                for label in unique_labels:
                    mask = cluster_labels == label
                    if label == -1:
                        ax.scatter(embeddings[mask, dim1], embeddings[mask, dim2], 
                                  alpha=0.6, s=1, c='black', label='Noise' if plot_idx == 0 else "", edgecolors='none')
                    else:
                        ax.scatter(embeddings[mask, dim1], embeddings[mask, dim2], 
                                  alpha=0.6, s=1, c=color_map[label], label=f'Cluster {label}' if plot_idx == 0 else "", edgecolors='none')
                
                ax.set_xlabel(f'UMAP Dimension {dim1+1}')
                ax.set_ylabel(f'UMAP Dimension {dim2+1}')
                ax.grid(True, alpha=0.3)
                if plot_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, 6):
            row, col = idx // 3, idx % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_clustered.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create cluster size distribution plot
    plt.figure(figsize=(10, 6))
    cluster_ids = [label for label in unique_labels if label != -1]
    cluster_sizes = [np.sum(cluster_labels == label) for label in cluster_ids]
    
    plt.bar(cluster_ids, cluster_sizes, alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Latents')
    plt.title('Cluster Size Distribution')
    plt.xticks(cluster_ids)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{prefix}_cluster_sizes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    
    if n_dims >= 3:
        print(f"3D clustered plot saved to {output_dir / f'{prefix}_clustered_3d.png'}")
        # Create interactive 3D plot with clustering
        create_interactive_3d_plot(embeddings, cluster_labels, output_dir, prefix, " (After Clustering)", latent_indices)


def create_network_graph(matrix: np.ndarray, 
                        output_dir: Path, 
                        prefix: str = "latent_network",
                        edge_threshold: float = 0.0,
                        matrix_type: str = "coactivation",
                        max_nodes: int = 10000) -> None:
    """
    Create a network graph visualization using Pyvis.
    
    Args:
        matrix: The matrix to create the graph from (coactivation or Jaccard)
        output_dir: Directory to save the graph visualization
        prefix: Prefix for output filenames
        edge_threshold: Threshold for including edges (0.0 = include all non-zero edges)
        matrix_type: Type of matrix ("coactivation" or "jaccard")
        max_nodes: Maximum number of nodes to include (for performance)
    """
    print(f"Creating network graph from {matrix_type} matrix using Pyvis...")
    
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()
    
    # Remove diagonal (self-connections)
    matrix_no_self = matrix.copy()
    np.fill_diagonal(matrix_no_self, 0.0)
    
    # Apply edge threshold
    if edge_threshold > 0.0:
        print(f"Applying edge threshold: {edge_threshold}")
        matrix_no_self[matrix_no_self < edge_threshold] = 0.0
    
    # Find non-zero edges
    edge_indices = np.where(matrix_no_self > 0.0)
    edge_weights = matrix_no_self[edge_indices]
    
    print(f"Found {len(edge_weights)} edges above threshold")
    
    if len(edge_weights) == 0:
        print("No edges found above threshold, skipping network graph creation")
        return
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add edges with weights
    print("Creating NetworkX graph from matrix...")
    with tqdm(total=len(edge_indices[0]), desc="  - Adding edges to graph", unit="edge") as pbar:
        for i, (row, col) in enumerate(zip(edge_indices[0], edge_indices[1])):
            if row != col:  # Avoid self-loops
                G.add_edge(int(row), int(col), weight=float(edge_weights[i]))
            pbar.update(1)
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # If too many nodes, sample a subset for visualization
    if G.number_of_nodes() > max_nodes:
        print(f"Graph has {G.number_of_nodes()} nodes, sampling {max_nodes} for visualization...")
        # Sample nodes with highest degree
        print("  - Computing node degrees...")
        node_degrees = dict(G.degree())
        print("  - Sorting nodes by degree...")
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        sampled_nodes = [node for node, _ in top_nodes]
        print("  - Creating subgraph...")
        G = G.subgraph(sampled_nodes).copy()
        print(f"Sampled graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    

    
    # Check if graph is connected and report status
    is_connected = nx.is_connected(G)
    if is_connected:
        print("Graph is connected.")
    else:
        n_components = nx.number_connected_components(G)
        print(f"Graph is not connected. Found {n_components} components.")
        print("Visualizing all components together...")
    
    # Create Pyvis network
    if PYVIS_AVAILABLE:
        print("Creating Pyvis network...")
        try:
            net = Network(height='800px', width='100%', bgcolor='#ffffff', 
                         font_color='#000000', directed=False, notebook=False)
        except Exception as e:
            print(f"Error creating Pyvis network: {e}")
            print("Falling back to matplotlib...")
            create_matplotlib_network_graph(G, output_dir, prefix, matrix_type, edge_threshold)
            return
        
        # Configure physics for better performance
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "smooth": {
              "type": "continuous"
            }
          }
        }
        """)
        
        # Color legend will be added after network is fully created
        
        # Add nodes
        print("Adding nodes to Pyvis network...")
        
        # Get node indices for color gradient
        # Map NetworkX node indices back to original latent indices
        if hasattr(G, 'original_indices'):
            # Use original latent indices if available
            original_indices = G.original_indices
            node_indices = [original_indices[node] for node in G.nodes()]
        else:
            # Fallback to NetworkX node indices
            node_indices = sorted(list(G.nodes()))
        
        min_index = min(node_indices)
        max_index = max(node_indices)
        index_range = max_index - min_index
        
        with tqdm(total=G.number_of_nodes(), desc="  - Adding nodes", unit="node") as pbar:
            for node in G.nodes():
                degree = G.degree(node)
                
                # Get the original latent index for this node
                if hasattr(G, 'original_indices'):
                    original_latent_index = G.original_indices[node]
                else:
                    original_latent_index = node
                
                # Color nodes by latent index (ultra-smooth continuous gradient from blue to red)
                if index_range > 0:
                    # Normalize index to [0, 1] range
                    normalized_index = (original_latent_index - min_index) / index_range
                    
                    # Create ultra-smooth continuous gradient using full RGB spectrum
                    # Blue (0.0) -> Cyan -> Green -> Yellow -> Orange -> Red (1.0)
                    
                    # Use a more sophisticated color mapping for smoother transitions
                    if normalized_index <= 0.25:
                        # Blue to Cyan (0.0 to 0.25)
                        r = 0
                        g = int(255 * (normalized_index * 4))
                        b = 255
                    elif normalized_index <= 0.5:
                        # Cyan to Green (0.25 to 0.5)
                        r = 0
                        g = 255
                        b = int(255 * (1 - (normalized_index - 0.25) * 4))
                    elif normalized_index <= 0.75:
                        # Green to Yellow (0.5 to 0.75)
                        r = int(255 * (normalized_index - 0.5) * 4)
                        g = 255
                        b = 0
                    else:
                        # Yellow to Red (0.75 to 1.0)
                        r = 255
                        g = int(255 * (1 - (normalized_index - 0.75) * 4))
                        b = 0
                    
                    color = f'rgb({r}, {g}, {b})'
                else:
                    # Fallback if all nodes have same index
                    color = 'rgb(0, 0, 255)'
                
                # Add border for better visibility
                border_width = 2 if degree > 5 else 1
                
                net.add_node(node, 
                            label=str(original_latent_index),
                            title=f'Latent Index: {original_latent_index}\nDegree: {degree}\nConnections: {degree}\nColor: Index-based gradient',
                            color=color,
                            size=min(10 + degree, 30),  # Size based on degree
                            borderWidth=border_width,
                            borderWidthSelected=3,
                            font={'size': 10, 'face': 'Arial'})
                
                pbar.update(1)
                pbar.set_postfix({"degree": degree, "index": node})
        
        # Add edges
        print("Adding edges to Pyvis network...")
        edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
        if edge_weights_list:
            max_weight = max(edge_weights_list)
            with tqdm(total=G.number_of_edges(), desc="  - Adding edges", unit="edge") as pbar:
                for u, v in G.edges():
                    weight = G[u][v]['weight']
                    # Normalize edge width based on weight
                    width = max(1, int(5 * weight / max_weight))
                    # Color edges by weight (darker = higher weight)
                    weight_intensity = min(weight / max_weight, 1.0)
                    edge_color = f'rgba(0, 0, 0, {0.3 + 0.7 * weight_intensity})'
                    
                    net.add_edge(u, v, 
                               title=f'Weight: {weight:.4f}\nNormalized: {weight_intensity:.3f}',
                               width=width,
                               color=edge_color,
                               smooth={'type': 'continuous'},
                               selectionWidth=width + 2)
                    
                    pbar.update(1)
                    pbar.set_postfix({"weight": f"{weight:.3f}"})
        
        # Add color legend to the network
        try:
            # Create a simple legend using Pyvis's built-in features
            legend_text = f"Color Legend: Blue (low index {min_index}) → Green → Yellow → Red (high index {max_index})"
            net.add_node("legend", label=legend_text, color="white", size=0, physics=False, 
                        x=0, y=0, title="Node colors represent latent index progression")
        except:
            pass  # Skip legend if it fails
        
        # Save Pyvis network as HTML
        html_file = output_dir / f"{prefix}_network_pyvis.html"
        try:
            # Try to save with explicit notebook=False and open_browser=False
            net.write_html(str(html_file), notebook=False, open_browser=False)
            print(f"Pyvis network graph saved to {html_file}")
            print(f"  - Open this file in a web browser to interact with the network")
            print(f"  - Use mouse to drag nodes, zoom, and pan")
            print(f"  - Hover over nodes and edges for detailed information")
            print(f"  - Use navigation buttons for zoom and fit operations")
            print(f"  - Color gradient: Blue → Cyan → Green → Yellow → Red (ultra-smooth)")
        except Exception as e:
            print(f"Error saving Pyvis HTML: {e}")
            print("Falling back to matplotlib...")
            create_matplotlib_network_graph(G, output_dir, prefix, matrix_type, edge_threshold)
            return
        
    if not PYVIS_AVAILABLE:
        print("Pyvis not available, falling back to matplotlib...")
        # Fallback to matplotlib if Pyvis is not available
        create_matplotlib_network_graph(G, output_dir, prefix, matrix_type, edge_threshold)
        return
    
    # Save graph statistics
    stats_file = output_dir / f"{prefix}_network_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Network Graph Statistics\n")
        f.write(f"========================\n")
        f.write(f"Matrix type: {matrix_type}\n")
        f.write(f"Edge threshold: {edge_threshold}\n")
        f.write(f"Number of nodes: {G.number_of_nodes()}\n")
        f.write(f"Number of edges: {G.number_of_edges()}\n")
        f.write(f"Average degree: {np.mean(list(dict(G.degree()).values())):.2f}\n")
        f.write(f"Average clustering coefficient: {nx.average_clustering(G):.4f}\n")
        
        # Handle disconnected graphs gracefully
        f.write(f"Graph is connected: {'Yes' if is_connected else 'No'}\n")
        if is_connected:
            f.write(f"Average shortest path length: {nx.average_shortest_path_length(G):.4f}\n")
        else:
            f.write(f"Number of connected components: {nx.number_connected_components(G)}\n")
            # Show component sizes
            component_sizes = [len(c) for c in nx.connected_components(G)]
            f.write(f"Component sizes: {sorted(component_sizes, reverse=True)[:10]}\n")
        
        f.write(f"Density: {nx.density(G):.6f}\n")
    
    print(f"Network statistics saved to {stats_file}")



def find_connectivity_threshold_for_component(matrix: np.ndarray, min_threshold: float = 0.0, max_threshold: float = 1.0, tolerance: float = 0.001) -> float:
    """
    Find the maximum threshold that keeps a component connected using binary search.
    
    Args:
        matrix: Jaccard similarity matrix for a single component
        min_threshold: Minimum threshold to start from
        max_threshold: Maximum threshold to test up to
        tolerance: Tolerance for threshold precision
        
    Returns:
        float: The highest threshold that keeps the component connected
    """
    print(f"  - Finding connectivity threshold between {min_threshold:.3f} and {max_threshold:.3f}...")
    
    # First, verify that the component is actually connected at min_threshold
    print("  - Verifying component connectivity at minimum threshold...")
    
    # Debug: Check matrix statistics and edge counts
    non_zero_values = matrix[matrix > 0]
    total_possible_edges = matrix.shape[0] * (matrix.shape[0] - 1) // 2
    non_zero_edges = len(non_zero_values) - matrix.shape[0]  # Exclude diagonal
    
    print(f"  - Matrix statistics: shape={matrix.shape}, non-zero values={len(non_zero_values)}")
    print(f"  - Total possible edges (excluding diagonal): {total_possible_edges:,}")
    print(f"  - Non-zero edges: {non_zero_edges:,}")
    if len(non_zero_values) > 0:
        print(f"  - Non-zero value range: [{non_zero_values.min():.6f}, {non_zero_values.max():.6f}]")
    
    adjacency_min = matrix > min_threshold
    G_min = nx.from_numpy_array(adjacency_min)
    edges_at_min = G_min.number_of_edges()
    is_connected_at_min = nx.is_connected(G_min)
    print(f"  - Component connected at threshold {min_threshold:.6f}: {is_connected_at_min}")
    print(f"  - Edges at threshold {min_threshold:.6f}: {edges_at_min:,}")
    
    if not is_connected_at_min:
        print(f"  - Warning: Component not connected even at threshold {min_threshold:.6f}")
        return min_threshold
    
    # Estimate number of iterations for progress bar
    max_iterations = int(np.log2((max_threshold - min_threshold) / tolerance)) + 1
    print(f"  - Estimated iterations: {max_iterations}")
    
    # Binary search for the connectivity threshold
    left, right = min_threshold, max_threshold
    best_threshold = min_threshold
    iteration = 0
    
    print(f"  - Starting binary search: left={left:.6f}, right={right:.6f}, tolerance={tolerance:.6f}")
    
    with tqdm(total=max_iterations, desc="  - Binary search", unit="iter") as pbar:
        while right - left > tolerance:
            iteration += 1
            mid = (left + right) / 2
            
            # Create binary adjacency matrix above threshold
            adjacency = matrix > mid
            
            # Create graph from adjacency matrix
            G_test = nx.from_numpy_array(adjacency)
            
            # Check connectivity
            is_connected = nx.is_connected(G_test)
            
            if is_connected:
                # This threshold works, try a higher one
                best_threshold = mid
                left = mid
                pbar.set_postfix({"threshold": f"{mid:.6f}", "status": "connected", "best": f"{best_threshold:.6f}", "edges": G_test.number_of_edges()})
            else:
                # This threshold doesn't work, try a lower one
                right = mid
                pbar.set_postfix({"threshold": f"{mid:.6f}", "status": "disconnected", "best": f"{best_threshold:.6f}", "edges": G_test.number_of_edges()})
            
            pbar.update(1)
    
    print(f"  - Binary search completed: best_threshold={best_threshold:.6f}")
    
    # Debug: Check if we actually found a better threshold
    if best_threshold == min_threshold:
        print(f"  - Warning: best_threshold equals min_threshold ({min_threshold:.6f})")
        print(f"  - This means the component is only connected at the minimum threshold")
        print(f"  - Checking if we can find a slightly higher threshold...")
        
        # Try a few more iterations with smaller tolerance
        test_thresholds = [min_threshold + tolerance * i for i in range(1, 6)]
        for test_thresh in test_thresholds:
            if test_thresh > max_threshold:
                break
            test_adjacency = matrix > test_thresh
            G_test = nx.from_numpy_array(test_adjacency)
            if nx.is_connected(G_test):
                best_threshold = test_thresh
                print(f"  - Found higher working threshold: {best_threshold:.6f}")
                break
            else:
                print(f"  - Threshold {test_thresh:.6f} breaks connectivity")
    
    # Final verification
    print(f"  - Final verification: testing threshold {best_threshold:.6f}")
    final_adjacency = matrix > best_threshold
    G_final = nx.from_numpy_array(final_adjacency)
    final_connected = nx.is_connected(G_final)
    print(f"  - Component connected at final threshold {best_threshold:.6f}: {final_connected}")
    
    if not final_connected:
        print(f"  - Error: Final threshold {best_threshold:.6f} does not keep component connected!")
        # Fallback to min_threshold
        return min_threshold
    
    print(f"  - Component is connected at threshold {best_threshold:.6f}")
    return best_threshold


def find_largest_component_top_latents(matrix: np.ndarray, max_nodes: int = 10000, output_dir: Path = None, prefix: str = "debug") -> tuple[np.ndarray, float, list]:
    """
    Find the largest connected component using only the top decile of most activated latents,
    then find the maximum threshold that keeps it connected.
    
    Args:
        matrix: Jaccard similarity matrix
        max_nodes: Maximum number of nodes to select
        output_dir: Directory to save debug plots (optional)
        prefix: Prefix for debug plot filenames
        
    Returns:
        tuple: (component_matrix, threshold, component_nodes) - matrix for largest component, threshold, and node indices
    """
    print("Finding largest connected component using top activated latents...")
    
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(matrix):
        matrix_np = matrix.cpu().numpy()
    else:
        matrix_np = matrix
    
    # Count original edges (excluding diagonal)
    print("  - Analyzing original matrix...")
    total_possible_edges = matrix_np.shape[0] * (matrix_np.shape[0] - 1) // 2
    non_zero_edges = np.sum(matrix_np > 0) - matrix_np.shape[0]  # Exclude diagonal
    print(f"  - Total possible edges (excluding diagonal): {total_possible_edges:,}")
    print(f"  - Non-zero Jaccard edges: {non_zero_edges:,}")
    print(f"  - Zero Jaccard edges: {total_possible_edges - non_zero_edges:,}")
    
    # Filter out zero edges first
    print("  - Filtering out zero Jaccard edges...")
    matrix_filtered = matrix_np.copy()
    matrix_filtered[matrix_np == 0] = 0  # This is redundant but explicit
    
    # Count edges after zero filtering
    edges_after_zero_filter = np.sum(matrix_filtered > 0) - matrix_filtered.shape[0]
    print(f"  - Edges after zero filtering: {edges_after_zero_filter:,}")
    
    # Get top decile of latents based on activation (sum of Jaccard similarities)
    print("  - Finding top decile of most activated latents...")
    latent_activation_scores = np.sum(matrix_filtered, axis=1)  # Sum of Jaccard similarities per latent
    print(f"  - Processing {len(latent_activation_scores):,} latent activation scores...")
    
    # Find the threshold for top 10% of latents
    top_decile_threshold = np.percentile(latent_activation_scores, 90)  # Top 10%
    print(f"  - Top decile activation threshold: {top_decile_threshold:.6f}")
    
    # Select top 10% of latents
    top_latent_indices = np.where(latent_activation_scores >= top_decile_threshold)[0]
    print(f"  - Selected {len(top_latent_indices):,} top latents (top 10%)")
    
    # Debug: Show index distribution
    print(f"  - Index range: {top_latent_indices.min():,} to {top_latent_indices.max():,}")
    print(f"  - First 10 indices: {top_latent_indices[:10].tolist()}")
    print(f"  - Last 10 indices: {top_latent_indices[-10:].tolist()}")
    
    # Create distribution plot of selected indices
    if output_dir is not None:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Histogram of selected indices
            plt.subplot(1, 2, 1)
            plt.hist(top_latent_indices, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Original Latent Index')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Selected Latent Indices\n(Top 10% - {len(top_latent_indices):,} latents)')
            plt.grid(True, alpha=0.3)
            
            # Cumulative distribution
            plt.subplot(1, 2, 2)
            sorted_indices = np.sort(top_latent_indices)
            plt.plot(sorted_indices, np.arange(1, len(sorted_indices) + 1), 'r-', linewidth=2)
            plt.xlabel('Original Latent Index')
            plt.ylabel('Cumulative Count')
            plt.title('Cumulative Distribution of Selected Indices')
            plt.grid(True, alpha=0.3)
            
            # Add reference lines for full range
            full_range = matrix_filtered.shape[0]
            plt.axhline(y=len(top_latent_indices), color='gray', linestyle='--', alpha=0.7, label=f'Total selected: {len(top_latent_indices):,}')
            plt.axvline(x=full_range-1, color='gray', linestyle='--', alpha=0.7, label=f'Full range: 0-{full_range-1:,}')
            plt.legend()
            
            plt.tight_layout()
            
            # Save the plot
            indices_plot_path = output_dir / f"{prefix}_selected_indices_distribution.png"
            plt.savefig(indices_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  - Selected indices distribution plot saved to: {indices_plot_path}")
            
        except Exception as e:
            print(f"  - Warning: Could not create indices distribution plot: {e}")
    else:
        print("  - Skipping indices distribution plot (no output directory provided)")
    
    # Extract submatrix for top latents
    print("  - Extracting submatrix for top latents...")
    matrix_top_latents = matrix_filtered[np.ix_(top_latent_indices, top_latent_indices)]
    
    # Count edges in top latents matrix
    edges_after_top_latents = np.sum(matrix_top_latents > 0) - matrix_top_latents.shape[0]
    print(f"  - Edges after top latents filtering: {edges_after_top_latents:,}")
    print(f"  - Removed {edges_after_zero_filter - edges_after_top_latents:,} edges (from bottom 90% of latents)")
    
    # Create graph from top latents
    print("  - Creating graph from top latents...")
    adjacency = matrix_top_latents > 0
    G_top_latents = nx.from_numpy_array(adjacency)
    
    # Store original latent indices for proper coloring
    G_top_latents.original_indices = top_latent_indices
    
    print(f"  - Top latents graph: {G_top_latents.number_of_nodes()} nodes, {G_top_latents.number_of_edges()} edges")
    
    # Find all connected components
    print("  - Finding connected components...")
    components = list(nx.connected_components(G_top_latents))
    component_sizes = [len(c) for c in components]
    
    print(f"  - Found {len(components)} components with sizes: {component_sizes}")
    
    # Get the largest component
    largest_component = max(components, key=len)
    largest_component_nodes = sorted(list(largest_component))
    
    print(f"  - Largest component has {len(largest_component_nodes)} nodes")
    
    # Extract submatrix for the largest component
    print("  - Extracting submatrix for largest component...")
    component_matrix = matrix_top_latents[np.ix_(largest_component_nodes, largest_component_nodes)]
    
    # Count edges in component matrix
    component_total_edges = component_matrix.shape[0] * (component_matrix.shape[0] - 1) // 2
    component_non_zero_edges = np.sum(component_matrix > 0) - component_matrix.shape[0]
    print(f"  - Component matrix: {component_matrix.shape[0]} nodes, {component_non_zero_edges:,} non-zero edges")
    
    # Find connectivity threshold for just this component
    print(f"  - Finding connectivity threshold for largest component...")
    threshold = find_connectivity_threshold_for_component(component_matrix, min_threshold=0.0, max_threshold=0.002, tolerance=0.00001)
    
    return component_matrix, threshold, largest_component_nodes








def create_matplotlib_network_graph(G: nx.Graph, output_dir: Path, prefix: str, 
                                  matrix_type: str, edge_threshold: float) -> None:
    """
    Fallback function to create network graph using matplotlib if Pyvis is not available.
    """
    print("Creating matplotlib fallback network graph...")
    
    # Calculate layout
    print("Computing graph layout...")
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    print("Using spring layout")
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_size=50, 
                          node_color='lightblue',
                          alpha=0.7)
    
    # Draw edges with weights as thickness
    edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights_list:
        # Normalize edge weights for visualization
        max_weight = max(edge_weights_list)
        edge_widths = [2.0 * w / max_weight for w in edge_weights_list]
    else:
        edge_widths = [1.0] * G.number_of_edges()
    
    nx.draw_networkx_edges(G, pos, 
                          width=edge_widths,
                          alpha=0.5,
                          edge_color='gray')
    
    # Add labels for a subset of nodes (to avoid clutter)
    if G.number_of_nodes() <= 100:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(f'Network Graph from {matrix_type.capitalize()} Matrix\n'
              f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    plt.axis('off')
    
    # Save the plot
    output_file = output_dir / f"{prefix}_network_matplotlib.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matplotlib network graph saved to {output_file}")


def plot_edge_distribution_by_latent(matrix: np.ndarray, 
                                   output_dir: Path, 
                                   prefix: str = "latent_network",
                                   edge_threshold: float = 0.0,
                                   matrix_type: str = "coactivation") -> None:
    """
    Create a plot showing the distribution of edges (degree) with respect to latent indices.
    
    Args:
        matrix: The matrix to create the graph from (coactivation or Jaccard)
        output_dir: Directory to save the plot
        prefix: Prefix for output filenames
        edge_threshold: Threshold for including edges (0.0 = include all non-zero edges)
        matrix_type: Type of matrix ("coactivation" or "jaccard")
    """
    print(f"Creating edge distribution plot from {matrix_type} matrix...")
    
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(matrix):
        matrix = matrix.cpu().numpy()
    
    # Remove diagonal (self-connections)
    matrix_no_self = matrix.copy()
    np.fill_diagonal(matrix_no_self, 0.0)
    
    # Apply edge threshold
    if edge_threshold > 0.0:
        print(f"Applying edge threshold: {edge_threshold}")
        matrix_no_self[matrix_no_self < edge_threshold] = 0.0
    
    # Count edges for each latent (sum of non-zero entries in each row)
    edge_counts = np.sum(matrix_no_self > 0.0, axis=1)
    
    # Determine what we're counting based on matrix type
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        edge_description = "co-activations"
        print(f"Co-activation distribution statistics:")
    else:
        edge_description = "edges"
        print(f"Edge distribution statistics:")
    
    print(f"  Total latents: {len(edge_counts):,}")
    print(f"  Latents with {edge_description}: {np.sum(edge_counts > 0):,}")
    print(f"  Latents without {edge_description}: {np.sum(edge_counts == 0):,}")
    print(f"  Min {edge_description} per latent: {edge_counts.min()}")
    print(f"  Max {edge_description} per latent: {edge_counts.max()}")
    print(f"  Mean {edge_description} per latent: {edge_counts.mean():.2f}")
    print(f"  Median {edge_description} per latent: {np.median(edge_counts):.2f}")
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Edge count vs Latent index (scatter plot)
    latent_indices = np.arange(len(edge_counts))
    ax1.scatter(latent_indices, edge_counts, alpha=0.6, s=20, color='blue')
    ax1.set_xlabel('Latent Index')
    # Set y-axis label based on matrix type
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        ax1.set_ylabel('Number of Co-activations (Degree)')
    else:
        ax1.set_ylabel('Number of Edges (Degree)')
    # Make title clearer about what's being shown
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        if edge_threshold > 0.0:
            title = f'Co-activation Distribution by Latent Index\n{matrix_type.capitalize()} Matrix, Threshold: {edge_threshold}'
        else:
            title = f'Co-activation Distribution by Latent Index\n{matrix_type.capitalize()} Matrix - ALL Co-activations (No Threshold)'
    else:
        if edge_threshold > 0.0:
            title = f'Edge Distribution by Latent Index\n{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}'
        else:
            title = f'Edge Distribution by Latent Index\n{matrix_type.capitalize()} Matrix - ALL Edges (No Threshold)'
    
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if np.sum(edge_counts > 0) > 1:
        # Only fit trend line if there are edges
        z = np.polyfit(latent_indices[edge_counts > 0], edge_counts[edge_counts > 0], 1)
        p = np.poly1d(z)
        ax1.plot(latent_indices, p(latent_indices), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (slope: {z[0]:.4f})')
        ax1.legend()
    
    # Plot 2: Histogram of edge counts
    ax2.hist(edge_counts, bins=min(50, len(set(edge_counts))), alpha=0.7, color='green', edgecolor='black')
    # Set x-axis label based on matrix type
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        ax2.set_xlabel('Number of Co-activations (Degree)')
    else:
        ax2.set_xlabel('Number of Edges (Degree)')
    ax2.set_ylabel('Frequency (Number of Latents)')
    # Make histogram title clearer
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        if edge_threshold > 0.0:
            hist_title = f'Distribution of Co-activation Counts\n{matrix_type.capitalize()} Matrix, Threshold: {edge_threshold}'
        else:
            hist_title = f'Distribution of Co-activation Counts\n{matrix_type.capitalize()} Matrix - ALL Co-activations (No Threshold)'
    else:
        if edge_threshold > 0.0:
            hist_title = f'Distribution of Edge Counts\n{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}'
        else:
            hist_title = f'Distribution of Edge Counts\n{matrix_type.capitalize()} Matrix - ALL Edges (No Threshold)'
    
    ax2.set_title(hist_title)
    ax2.grid(True, alpha=0.3)
    
    # Add vertical line for mean
    mean_edges = edge_counts.mean()
    ax2.axvline(mean_edges, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_edges:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / f"{prefix}_edge_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Edge distribution plot saved to {output_file}")
    
    # Create interactive Plotly version
    print("Creating interactive HTML edge distribution plot...")
    
    # Create interactive scatter plot
    fig_scatter = go.Figure()
    
    # Add scatter trace
    fig_scatter.add_trace(go.Scatter(
        x=latent_indices,
        y=edge_counts,
        mode='markers',
        marker=dict(
            size=6,
            color=edge_counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=f"Number of {edge_description.title()}", x=1.1)
        ),
        text=[f'Latent Index: {i}<br>{edge_description.title()}: {count}' for i, count in zip(latent_indices, edge_counts)],
        hoverinfo='text',
        name='Latent Nodes'
    ))
    
    # Add trend line if there are edges
    if np.sum(edge_counts > 0) > 1:
        z = np.polyfit(latent_indices[edge_counts > 0], edge_counts[edge_counts > 0], 1)
        p = np.poly1d(z)
        fig_scatter.add_trace(go.Scatter(
            x=latent_indices,
            y=p(latent_indices),
            mode='lines',
            line=dict(dash='dash', color='red', width=3),
            name=f'Trend Line (slope: {z[0]:.4f})'
        ))
    
    # Set title and labels based on matrix type
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        if edge_threshold > 0.0:
            scatter_title = f'Co-activation Distribution by Latent Index<br>{matrix_type.capitalize()} Matrix, Threshold: {edge_threshold}'
            y_axis_title = 'Number of Co-activations (Degree)'
        else:
            scatter_title = f'Co-activation Distribution by Latent Index<br>{matrix_type.capitalize()} Matrix - ALL Co-activations (No Threshold)'
            y_axis_title = 'Number of Co-activations (Degree)'
    else:
        if edge_threshold > 0.0:
            scatter_title = f'Edge Distribution by Latent Index<br>{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}'
            y_axis_title = 'Number of Edges (Degree)'
        else:
            scatter_title = f'Edge Distribution by Latent Index<br>{matrix_type.capitalize()} Matrix - ALL Edges (No Threshold)'
            y_axis_title = 'Number of Edges (Degree)'
    
    fig_scatter.update_layout(
        title=scatter_title,
        xaxis_title='Latent Index',
        yaxis_title=y_axis_title,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    # Save interactive scatter plot
    html_scatter_file = output_dir / f"{prefix}_edge_distribution_scatter.html"
    fig_scatter.write_html(html_scatter_file)
    print(f"Interactive scatter plot saved to {html_scatter_file}")
    
    # Create interactive histogram
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=edge_counts,
        nbinsx=min(50, len(set(edge_counts))),
        marker_color='green',
        opacity=0.7,
        name='Edge Count Distribution'
    ))
    
    # Add vertical line for mean
    fig_hist.add_vline(
        x=mean_edges, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: {mean_edges:.2f}",
        annotation_position="top right"
    )
    
    # Set title and labels based on matrix type
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        if edge_threshold > 0.0:
            hist_title_interactive = f'Distribution of Co-activation Counts<br>{matrix_type.capitalize()} Matrix, Threshold: {edge_threshold}'
            x_axis_title = 'Number of Co-activations (Degree)'
        else:
            hist_title_interactive = f'Distribution of Co-activation Counts<br>{matrix_type.capitalize()} Matrix - ALL Co-activations (No Threshold)'
            x_axis_title = 'Number of Co-activations (Degree)'
    else:
        if edge_threshold > 0.0:
            hist_title_interactive = f'Distribution of Edge Counts<br>{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}'
            x_axis_title = 'Number of Edges (Degree)'
        else:
            hist_title_interactive = f'Distribution of Edge Counts<br>{matrix_type.capitalize()} Matrix - ALL Edges (No Threshold)'
            x_axis_title = 'Number of Edges (Degree)'
    
    fig_hist.update_layout(
        title=hist_title_interactive,
        xaxis_title=x_axis_title,
        yaxis_title='Frequency (Number of Latents)',
        plot_bgcolor='white'
    )
    
    # Save interactive histogram
    html_hist_file = output_dir / f"{prefix}_edge_distribution_histogram.html"
    fig_hist.write_html(html_hist_file)
    print(f"Interactive histogram saved to {html_hist_file}")
    
    # Save edge count data as CSV for further analysis
    if matrix_type == "coactivation" or matrix_type == "coactivation_original":
        csv_file = output_dir / f"{prefix}_coactivation_counts.csv"
        column_name = 'coactivation_count'
        print(f"Co-activation count data saved to {csv_file}")
    else:
        csv_file = output_dir / f"{prefix}_edge_counts.csv"
        column_name = 'edge_count'
        print(f"Edge count data saved to {csv_file}")
    
    import pandas as pd
    df = pd.DataFrame({
        'latent_index': latent_indices,
        column_name: edge_counts
    })
    df.to_csv(csv_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Apply UMAP to latent activation patterns")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Path to directory containing cached latent files")
    parser.add_argument("--output_path", type=str, default="umap_embeddings.pt",
                       help="Path to save the UMAP embeddings")
    parser.add_argument("--n_components", type=int, default=2,
                       help="Number of UMAP components")
    parser.add_argument("--n_neighbors", type=int, default=15,
                       help="Number of neighbors for UMAP")
    parser.add_argument("--min_dist", type=float, default=0.1,
                       help="Minimum distance for UMAP")
    parser.add_argument("--token_batch_size", type=int, default=100_000,
                       help="Batch size for processing tokens (adjust for memory)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_activation_matrix", action="store_true",
                       help="Save the activation matrix")
    parser.add_argument("--create_plots", action="store_true",
                       help="Create visualization plots")
    parser.add_argument("--min_cluster_size", type=int, default=10,
                       help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=5,
                       help="Minimum samples for HDBSCAN core points")
    parser.add_argument("--cluster_selection_epsilon", type=float, default=0.0,
                       help="Distance threshold for cluster selection")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Distance scaling parameter for HDBSCAN")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU acceleration where available")
    parser.add_argument("--cache_dir_intermediate", type=str, default=None,
                       help="Directory to cache intermediate results (UMAP embeddings)")

    parser.add_argument("--skip_hdbscan", action="store_true",
                       help="Skip HDBSCAN computation and only run UMAP")
    parser.add_argument("--use_coactivation_directly", action="store_true",
                       help="Use coactivation matrix directly as distance (1/entry + epsilon) instead of Jaccard similarity")
    parser.add_argument("--html_prefix", type=str, default="latent_umap",
                       help="Prefix for HTML output filenames (default: latent_umap)")
    parser.add_argument("--jaccard_threshold", type=float, default=0.0,
                       help="Jaccard similarity threshold for filtering latents and network graph edges (0.0 = no filtering)")
    parser.add_argument("--create_network_graph", action="store_true",
                       help="Create a network graph visualization using Pyvis (optimized for performance)")
    parser.add_argument("--create_edge_distribution", action="store_true",
                       help="Create a plot showing edge distribution by latent index")
    parser.add_argument("--max_nodes", type=int, default=10000,
                       help="Maximum number of nodes to include in network graph (default: 10000)")
    parser.add_argument("--auto_connectivity_threshold", action="store_true",
                       help="Use top 10% of most activated latents to find largest connected component and its optimal threshold")
    
    parser.add_argument("--top_k_edges", type=int, default=20,
                       help="Keep only top K highest Jaccard similarities per row/column before filtering (default: 20)")

    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise ValueError(f"Cache directory {cache_dir} does not exist")
    
    output_path = Path(args.output_path)
    output_dir = output_path.parent
    
    # Set up intermediate caching
    if args.cache_dir_intermediate is None:
        cache_dir_intermediate = output_dir / "intermediate_cache"
    else:
        cache_dir_intermediate = Path(args.cache_dir_intermediate)
    
    cache_dir_intermediate.mkdir(parents=True, exist_ok=True)
    
    # Build coactivation matrix
    coactivation_matrix = load_and_build_coactivation_matrix(cache_dir, args.token_batch_size, args.use_gpu)
    
    # Optionally save coactivation matrix
    if args.save_activation_matrix:
        matrix_path = output_dir / "coactivation_matrix.pt"
        torch.save(coactivation_matrix, matrix_path)
        print(f"Coactivation matrix saved to {matrix_path}")
    
    # Create edge distribution plot from coactivation matrix BEFORE any filtering
    if args.create_edge_distribution or args.create_network_graph:
        print("\nCreating edge distribution plot from coactivation matrix BEFORE any filtering...")
        print("  - This shows the true distribution of coactivations BEFORE any UMAP or filtering")
        print("  - X-axis: Latent Index (0 to {coactivation_matrix.shape[0]-1:,})")
        print("  - Y-axis: Number of co-activations per latent")
        print("  - Shows how many times each latent co-activates with other latents")
        plot_edge_distribution_by_latent(
            coactivation_matrix,
            output_dir,
            args.html_prefix,
            0.0,  # No edge threshold for coactivation matrix
            "coactivation_original"
        )
    
    # Check if UMAP embeddings are cached
    umap_cache_file = cache_dir_intermediate / "umap_embeddings.pt"
    active_mask_cache_file = cache_dir_intermediate / "active_mask.pt"
    umap_params_cache_file = cache_dir_intermediate / "umap_params.pt"
    
    if umap_cache_file.exists() and active_mask_cache_file.exists() and umap_params_cache_file.exists():
        # Check if UMAP parameters match
        cached_umap_params = torch.load(umap_params_cache_file)
        current_umap_params = {
            'n_components': args.n_components,
            'n_neighbors': args.n_neighbors,
            'min_dist': args.min_dist,
            'random_state': args.random_state,
            'use_coactivation_directly': args.use_coactivation_directly,
            'jaccard_threshold': args.jaccard_threshold
        }
        
        if cached_umap_params == current_umap_params:
            print("Loading UMAP embeddings from cache...")
            embeddings = torch.load(umap_cache_file).numpy()
            active_mask = torch.load(active_mask_cache_file)
            print(f"Loaded embeddings shape: {embeddings.shape}")
            
            # Generate latent indices based on active mask for plotting
            latent_indices = np.arange(coactivation_matrix.shape[0])[active_mask]
        else:
            print("UMAP parameters changed, recomputing...")
            # Apply UMAP
            embeddings, active_mask = apply_umap_to_patterns(
                coactivation_matrix,
                n_components=args.n_components,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                random_state=args.random_state,
                use_gpu=args.use_gpu,
                use_coactivation_directly=args.use_coactivation_directly,
                jaccard_threshold=args.jaccard_threshold
            )
            
            # Cache UMAP results
            print(f"Caching UMAP embeddings to {umap_cache_file}...")
            torch.save(torch.from_numpy(embeddings), umap_cache_file)
            torch.save(active_mask, active_mask_cache_file)
            
            # Cache UMAP parameters
            torch.save(current_umap_params, umap_params_cache_file)
            print("UMAP embeddings and parameters cached successfully!")
            
            # Generate latent indices for plotting (now all latents are kept)
            latent_indices = np.arange(coactivation_matrix.shape[0])
            
                    # Create embeddings plot (before clustering)
        if args.create_plots:
            plot_umap_embeddings_only(embeddings, output_dir, args.html_prefix, latent_indices=latent_indices)
    else:
        # Apply UMAP
        embeddings, active_mask = apply_umap_to_patterns(
            coactivation_matrix,
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
            use_gpu=args.use_gpu,
            use_coactivation_directly=args.use_coactivation_directly,
            jaccard_threshold=args.jaccard_threshold
        )
        
        # Cache UMAP results
        print(f"Caching UMAP embeddings to {umap_cache_file}...")
        torch.save(torch.from_numpy(embeddings), umap_cache_file)
        torch.save(active_mask, active_mask_cache_file)
        
        # Cache UMAP parameters
        current_umap_params = {
            'n_components': args.n_components,
            'n_neighbors': args.n_neighbors,
            'min_dist': args.min_dist,
            'random_state': args.random_state,
            'use_coactivation_directly': args.use_coactivation_directly,
            'jaccard_threshold': args.jaccard_threshold
        }
        torch.save(current_umap_params, umap_params_cache_file)
        print("UMAP embeddings and parameters cached successfully!")
        
        # Generate latent indices for plotting (now all latents are kept)
        latent_indices = np.arange(coactivation_matrix.shape[0])
        
        # Create embeddings plot (before clustering)
        if args.create_plots:
            plot_umap_embeddings_only(embeddings, output_dir, args.html_prefix, latent_indices=latent_indices)
    
    # Check if HDBSCAN results are cached
    hdbscan_cache_file = cache_dir_intermediate / "hdbscan_clusters.pt"
    hdbscan_params_cache_file = cache_dir_intermediate / "hdbscan_params.pt"
    
    if args.skip_hdbscan:
        print("Skipping HDBSCAN clustering as requested...")
        cluster_labels = np.array([-1] * len(embeddings))  # All noise
        clusterer = None
    elif hdbscan_cache_file.exists() and hdbscan_params_cache_file.exists():
        # Check if parameters match
        cached_params = torch.load(hdbscan_params_cache_file)
        current_params = {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples,
            'cluster_selection_epsilon': args.cluster_selection_epsilon,
            'alpha': args.alpha,
            'jaccard_threshold': args.jaccard_threshold
        }
        
        if cached_params == current_params:
            print("Loading HDBSCAN results from cache...")
            cluster_labels = torch.load(hdbscan_cache_file).numpy()
            clusterer = None  # We don't cache the full clusterer object
            print(f"Loaded cluster labels, found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
        else:
            print("HDBSCAN parameters changed, recomputing...")
            cluster_labels, clusterer = apply_hdbscan_clustering(
                embeddings,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_epsilon=args.cluster_selection_epsilon,
                alpha=args.alpha
            )
            # Cache HDBSCAN results
            print(f"Caching HDBSCAN results to {hdbscan_cache_file}...")
            torch.save(torch.from_numpy(cluster_labels), hdbscan_cache_file)
            torch.save(current_params, hdbscan_params_cache_file)
    else:
        # Define current_params for this case
        current_params = {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples,
            'cluster_selection_epsilon': args.cluster_selection_epsilon,
            'alpha': args.alpha,
            'jaccard_threshold': args.jaccard_threshold
        }
        
        cluster_labels, clusterer = apply_hdbscan_clustering(
            embeddings,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            alpha=args.alpha
        )
        # Cache HDBSCAN results
        print(f"Caching HDBSCAN results to {hdbscan_cache_file}...")
        torch.save(torch.from_numpy(cluster_labels), hdbscan_cache_file)
        torch.save(current_params, hdbscan_params_cache_file)
    
    # Save results
    results = {
        'embeddings': embeddings,
        'cluster_labels': cluster_labels,
        'active_mask': active_mask,
        'n_latents_total': coactivation_matrix.shape[0],
        'umap_params': {
            'n_components': args.n_components,
            'n_neighbors': args.n_neighbors,
            'min_dist': args.min_dist,
            'metric': 'coactivation_direct' if args.use_coactivation_directly else 'jaccard',
            'random_state': args.random_state,
            'use_coactivation_directly': args.use_coactivation_directly,
            'jaccard_threshold': args.jaccard_threshold
        },
        'hdbscan_params': {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples,
            'cluster_selection_epsilon': args.cluster_selection_epsilon,
            'alpha': args.alpha
        }
    }
    
    torch.save(results, output_path)
    print(f"UMAP + HDBSCAN results saved to {output_path}")
    
    # Create plots if requested
    if args.create_plots:
        # Generate latent indices based on active mask
        latent_indices = np.arange(coactivation_matrix.shape[0])[active_mask]
        
        if args.skip_hdbscan:
            # Use the no-clustering plotting function
            print("Creating plots without clustering...")
            plot_umap_embeddings_only(embeddings, output_dir, args.html_prefix, latent_indices=latent_indices)
            # Also create the interactive 3D plot without clustering
            if embeddings.shape[1] >= 3:
                create_interactive_3d_plot_no_clustering(embeddings, output_dir, args.html_prefix, " (No Clustering)", latent_indices)
        else:
            # Use the clustering plotting function
            plot_umap_results(embeddings, cluster_labels, output_dir, args.html_prefix, latent_indices=latent_indices)
    
    # Create network graph if requested
    if args.create_network_graph:
        print("\nCreating network graph...")
        print("Computing Jaccard matrix for network graph...")
        
        # Convert coactivation matrix to numpy for Jaccard computation
        coactivation_np = coactivation_matrix.cpu().numpy() if torch.is_tensor(coactivation_matrix) else coactivation_matrix
        
        # Compute Jaccard similarity
        def compute_jaccard_for_network(cooc_matrix):
            self_occurrence = cooc_matrix.diagonal()
            epsilon = 1e-8
            jaccard_matrix = cooc_matrix / (
                self_occurrence[:, None] + self_occurrence - cooc_matrix + epsilon
            )
            return jaccard_matrix
        
        jaccard_matrix = compute_jaccard_for_network(coactivation_np)
        
        # Print distribution of Jaccard matrix entries for network graph
        jaccard_values = jaccard_matrix[jaccard_matrix > 0]  # Exclude zeros (diagonal and no overlap)
        if len(jaccard_values) > 0:
            print(f"Network graph Jaccard matrix distribution:")
            print(f"  Non-zero values: {len(jaccard_values):,}")
            print(f"  Min: {jaccard_values.min():.4f}")
            print(f"  Max: {jaccard_values.max():.4f}")
            print(f"  Mean: {jaccard_values.mean():.4f}")
            print(f"  Median: {np.median(jaccard_values):.4f}")
            print(f"  Std: {jaccard_values.std():.4f}")
            print(f"  Percentiles: 25th={np.percentile(jaccard_values, 25):.4f}, 75th={np.percentile(jaccard_values, 75):.4f}")
        else:
            print("Network graph Jaccard matrix has no non-zero values (only diagonal)")
        
        # Note: Edge distribution plot already created from coactivation matrix BEFORE any filtering
        print(f"\nEdge distribution plot already created from coactivation matrix BEFORE any filtering")
        print("  - This shows the true distribution BEFORE any UMAP or filtering")
        print("  - X-axis: Latent Index (0 to {coactivation_np.shape[0]-1:,})")
        print("  - Y-axis: Number of co-activations per latent")
        
        # Find largest component using top decile edges if requested
        if args.auto_connectivity_threshold:
            print("\nFinding largest connected component using top decile edges...")
            print(f"  - Computing on FULL graph with {jaccard_matrix.shape[0]:,} nodes")
            print(f"  - Will extract largest component using top 10% of most activated latents")
            
            component_matrix, connectivity_threshold, component_nodes = find_largest_component_top_latents(
                jaccard_matrix, args.max_nodes, output_dir, args.html_prefix
            )
            
            print(f"  - Largest component has {len(component_nodes)} nodes")
            print(f"  - Component found using top 10% of most activated latents")
            print(f"  - Using connectivity threshold: {connectivity_threshold:.6f}")
            
            # Use the component matrix and threshold
            jaccard_matrix = component_matrix
            args.jaccard_threshold = connectivity_threshold
            
            # Update max_nodes if the component is smaller
            if len(component_nodes) < args.max_nodes:
                print(f"  - Component size ({len(component_nodes)}) is smaller than max_nodes ({args.max_nodes})")
                print(f"  - Using component size as max_nodes")
                args.max_nodes = len(component_nodes)
        else:
            # Use the full matrix and specified threshold
            component_nodes = list(range(jaccard_matrix.shape[0]))
        
        # Apply top-K edge filtering if requested (AFTER selecting top latents)
        if args.top_k_edges is not None:
            print(f"\nApplying top-{args.top_k_edges} edge filtering per latent...")
            print(f"  - Current Jaccard matrix: {jaccard_matrix.shape[0]:,} × {jaccard_matrix.shape[1]:,}")
            
            # Count current edges
            current_edges = np.sum(jaccard_matrix > 0) - jaccard_matrix.shape[0]  # Exclude diagonal
            print(f"  - Current non-zero edges: {current_edges:,}")
            
            # Create a copy for filtering
            jaccard_filtered = jaccard_matrix.copy()
            
            # For each row, keep only top K values (excluding diagonal)
            for i in range(jaccard_matrix.shape[0]):
                row_values = jaccard_matrix[i, :].copy()
                row_values[i] = 0  # Exclude diagonal
                
                # Find top K values
                if np.sum(row_values > 0) > args.top_k_edges:
                    # Get indices of top K values
                    top_k_indices = np.argsort(row_values)[-args.top_k_edges:]
                    # Zero out everything except top K
                    mask = np.ones_like(row_values, dtype=bool)
                    mask[top_k_indices] = False
                    jaccard_filtered[i, mask] = 0
            
            # For each column, keep only top K values (excluding diagonal)
            for j in range(jaccard_matrix.shape[1]):
                col_values = jaccard_matrix[:, j].copy()
                col_values[j] = 0  # Exclude diagonal
                
                # Find top K values
                if np.sum(col_values > 0) > args.top_k_edges:
                    # Get indices of top K values
                    top_k_indices = np.argsort(col_values)[-args.top_k_edges:]
                    # Zero out everything except top K
                    mask = np.ones_like(col_values, dtype=bool)
                    mask[top_k_indices] = False
                    jaccard_filtered[mask, j] = 0
            
            # Count edges after filtering
            filtered_edges = np.sum(jaccard_filtered > 0) - jaccard_filtered.shape[0]  # Exclude diagonal
            print(f"  - Edges after top-{args.top_k_edges} filtering: {filtered_edges:,}")
            print(f"  - Removed {current_edges - filtered_edges:,} edges ({((current_edges - filtered_edges) / current_edges * 100):.1f}% reduction)")
            
            # Use the filtered matrix
            jaccard_matrix = jaccard_filtered
        
        # Apply Jaccard threshold if specified
        if args.jaccard_threshold > 0.0:
            jaccard_matrix_no_self = jaccard_matrix.copy()
            np.fill_diagonal(jaccard_matrix_no_self, 0.0)
            below_threshold_mask = jaccard_matrix_no_self <= args.jaccard_threshold
            jaccard_matrix[below_threshold_mask] = 0.0
        
        # Note: Edge distribution plot already created from coactivation matrix BEFORE any filtering
        print(f"\nEdge distribution plot already created from coactivation matrix BEFORE any filtering")
        print("  - This shows the true distribution BEFORE any Jaccard computation or filtering")
        
        # Create network graph from component matrix
        if args.auto_connectivity_threshold:
            print(f"\nCreating network graph from largest component matrix ({len(component_nodes)} nodes)...")
            print(f"  - Component found using top 10% of most activated latents")
            print(f"  - Using connectivity threshold: {args.jaccard_threshold:.6f}")
        else:
            print(f"\nCreating network graph from FULL matrix with max_nodes sampling...")
        
        create_network_graph(
            jaccard_matrix, 
            output_dir, 
            args.html_prefix,
            args.jaccard_threshold,
            "jaccard",
            args.max_nodes
        )
    
    # Note: Edge distribution plot already created from coactivation matrix BEFORE any filtering
    if args.create_edge_distribution and not args.create_network_graph:
        print("\nEdge distribution plot already created from coactivation matrix BEFORE any filtering")
        print("  - This shows the true distribution BEFORE any UMAP or filtering")
    
    print("\nSummary:")
    print(f"  - Total latents: {coactivation_matrix.shape[0]:,}")
    print(f"  - UMAP dimensions: {embeddings.shape[1]}")
    print(f"  - Activation matrix shape: {coactivation_matrix.shape}")
    if args.jaccard_threshold > 0.0:
        print(f"  - Jaccard threshold applied: {args.jaccard_threshold} (connections <= threshold zeroed out)")
    print(f"  - Number of clusters: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"  - Noise points: {list(cluster_labels).count(-1)}")
    
    # Add edge distribution summary if network graph or edge distribution was created
    if args.create_network_graph or args.create_edge_distribution:
        print(f"  - Edge distribution analysis: Enabled")
        print(f"  - Created edge distribution plot from coactivation matrix BEFORE any filtering")
        print(f"    - Shows true distribution BEFORE any UMAP or filtering")
        print(f"    - X-axis: Latent Index (0 to {coactivation_matrix.shape[0]-1:,})")
        print(f"    - Y-axis: Number of co-activations per latent")
        if args.jaccard_threshold > 0.0:
            print(f"  - Jaccard threshold for edges: {args.jaccard_threshold}")


if __name__ == "__main__":
    main()