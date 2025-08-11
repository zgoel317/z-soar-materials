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
    
    print(f"UMAP embeddings shape: {embeddings.shape}")
    
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
    print("Creating visualization plots...")
    
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
            
            ax.set_xlabel(f'UMAP Dimension {dim1+1}')
            ax.set_ylabel(f'UMAP Dimension {dim2+1}')
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
                        max_nodes: int = 1000) -> None:
    """
    Create a network graph visualization using NetworkX.
    
    Args:
        matrix: The matrix to create the graph from (coactivation or Jaccard)
        output_dir: Directory to save the graph visualization
        prefix: Prefix for output filenames
        edge_threshold: Threshold for including edges (0.0 = include all non-zero edges)
        matrix_type: Type of matrix ("coactivation" or "jaccard")
        max_nodes: Maximum number of nodes to include (for performance)
    """
    print(f"Creating network graph from {matrix_type} matrix...")
    
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
    for i, (row, col) in enumerate(zip(edge_indices[0], edge_indices[1])):
        if row != col:  # Avoid self-loops
            G.add_edge(int(row), int(col), weight=float(edge_weights[i]))
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # If too many nodes, sample a subset for visualization
    if G.number_of_nodes() > max_nodes:
        print(f"Graph has {G.number_of_nodes()} nodes, sampling {max_nodes} for visualization...")
        # Sample nodes with highest degree
        node_degrees = dict(G.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        sampled_nodes = [node for node, _ in top_nodes]
        G = G.subgraph(sampled_nodes).copy()
        print(f"Sampled graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Check if graph is connected and report status
    is_connected = nx.is_connected(G)
    if not is_connected:
        print(f"Graph is not connected. Found {nx.number_connected_components(G)} components.")
        print(f"Visualizing all components together...")
    else:
        print("Graph is connected.")
    
    # Calculate layout
    print("Computing graph layout...")
    # Use spring layout
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
    output_file = output_dir / f"{prefix}_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network graph saved to {output_file}")
    
    # Create interactive Plotly version
    print("Creating interactive HTML graph...")
    
    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Prepare node data with hover information
    node_x = []
    node_y = []
    node_text = []
    node_colors_degree = []
    node_colors_index = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Show latent index and degree in hover
        degree = G.degree(node)
        node_text.append(f'Latent Index: {node}<br>Degree: {degree}')
        # Color nodes by degree (more connections = darker blue)
        node_colors_degree.append(min(degree / 10, 1.0))  # Normalize to 0-1 range
        # Color nodes by latent index (sequential rainbow colors)
        node_colors_index.append(node)
    
    # Create node trace colored by latent index
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=8,
            color=node_colors_index,
            colorscale='Viridis',  # Rainbow-like colorscale
            showscale=True,
            colorbar=dict(title="Latent Index", x=1.1),
            line=dict(width=2, color='white')
        ))
    
    # Create the layout
    layout = go.Layout(
        title=f'Network Graph - Colored by Latent Index<br>Edge Threshold: {edge_threshold}',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    # Create the figure and save as HTML
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    html_file = output_dir / f"{prefix}_network_interactive.html"
    fig.write_html(html_file)
    print(f"Interactive graph saved to {html_file}")
    
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
    
    print(f"Edge distribution statistics:")
    print(f"  Total latents: {len(edge_counts):,}")
    print(f"  Latents with edges: {np.sum(edge_counts > 0):,}")
    print(f"  Latents without edges: {np.sum(edge_counts == 0):,}")
    print(f"  Min edges per latent: {edge_counts.min()}")
    print(f"  Max edges per latent: {edge_counts.max()}")
    print(f"  Mean edges per latent: {edge_counts.mean():.2f}")
    print(f"  Median edges per latent: {np.median(edge_counts):.2f}")
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Edge count vs Latent index (scatter plot)
    latent_indices = np.arange(len(edge_counts))
    ax1.scatter(latent_indices, edge_counts, alpha=0.6, s=20, color='blue')
    ax1.set_xlabel('Latent Index')
    ax1.set_ylabel('Number of Edges (Degree)')
    ax1.set_title(f'Edge Distribution by Latent Index\n{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}')
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
    ax2.set_xlabel('Number of Edges (Degree)')
    ax2.set_ylabel('Frequency (Number of Latents)')
    ax2.set_title(f'Distribution of Edge Counts\n{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}')
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
            colorbar=dict(title="Number of Edges", x=1.1)
        ),
        text=[f'Latent Index: {i}<br>Edges: {count}' for i, count in zip(latent_indices, edge_counts)],
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
    
    fig_scatter.update_layout(
        title=f'Edge Distribution by Latent Index<br>{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}',
        xaxis_title='Latent Index',
        yaxis_title='Number of Edges (Degree)',
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
    
    fig_hist.update_layout(
        title=f'Distribution of Edge Counts<br>{matrix_type.capitalize()} Matrix, Edge Threshold: {edge_threshold}',
        xaxis_title='Number of Edges (Degree)',
        yaxis_title='Frequency (Number of Latents)',
        plot_bgcolor='white'
    )
    
    # Save interactive histogram
    html_hist_file = output_dir / f"{prefix}_edge_distribution_histogram.html"
    fig_hist.write_html(html_hist_file)
    print(f"Interactive histogram saved to {html_hist_file}")
    
    # Save edge count data as CSV for further analysis
    csv_file = output_dir / f"{prefix}_edge_counts.csv"
    import pandas as pd
    df = pd.DataFrame({
        'latent_index': latent_indices,
        'edge_count': edge_counts
    })
    df.to_csv(csv_file, index=False)
    print(f"Edge count data saved to {csv_file}")


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
    parser.add_argument("--skip_umap", action="store_true",
                       help="Skip UMAP computation and load from cache")
    parser.add_argument("--skip_hdbscan", action="store_true",
                       help="Skip HDBSCAN computation and only run UMAP")
    parser.add_argument("--use_coactivation_directly", action="store_true",
                       help="Use coactivation matrix directly as distance (1/entry + epsilon) instead of Jaccard similarity")
    parser.add_argument("--html_prefix", type=str, default="latent_umap",
                       help="Prefix for HTML output filenames (default: latent_umap)")
    parser.add_argument("--jaccard_threshold", type=float, default=0.0,
                       help="Jaccard similarity threshold for filtering latents and network graph edges (0.0 = no filtering)")
    parser.add_argument("--create_network_graph", action="store_true",
                       help="Create a network graph visualization using NetworkX")
    parser.add_argument("--create_edge_distribution", action="store_true",
                       help="Create a plot showing edge distribution by latent index")
    
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
    
    # Check if UMAP embeddings are cached
    umap_cache_file = cache_dir_intermediate / "umap_embeddings.pt"
    active_mask_cache_file = cache_dir_intermediate / "active_mask.pt"
    umap_params_cache_file = cache_dir_intermediate / "umap_params.pt"
    
    if args.skip_umap and umap_cache_file.exists() and active_mask_cache_file.exists() and umap_params_cache_file.exists():
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
            args.skip_umap = False
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
        
        # Create UMAP embeddings plot (before clustering)
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
        
        # Apply Jaccard threshold if specified
        if args.jaccard_threshold > 0.0:
            jaccard_matrix_no_self = jaccard_matrix.copy()
            np.fill_diagonal(jaccard_matrix_no_self, 0.0)
            below_threshold_mask = jaccard_matrix_no_self <= args.jaccard_threshold
            jaccard_matrix[below_threshold_mask] = 0.0
        
        create_network_graph(
            jaccard_matrix, 
            output_dir, 
            args.html_prefix,
            args.jaccard_threshold,
            "jaccard"
        )
        
        # Also create edge distribution plot
        print("\nCreating edge distribution plot...")
        plot_edge_distribution_by_latent(
            jaccard_matrix,
            output_dir,
            args.html_prefix,
            args.jaccard_threshold,
            "jaccard"
        )
    
    # Create edge distribution plot if requested (even without full network graph)
    if args.create_edge_distribution and not args.create_network_graph:
        print("\nCreating edge distribution plot from coactivation matrix...")
        plot_edge_distribution_by_latent(
            coactivation_matrix,
            output_dir,
            args.html_prefix,
            0.0,  # No edge threshold for coactivation matrix
            "coactivation"
        )
    
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
        if args.jaccard_threshold > 0.0:
            print(f"  - Jaccard threshold for edges: {args.jaccard_threshold}")


if __name__ == "__main__":
    main()