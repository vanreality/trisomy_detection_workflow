#!/usr/bin/env python3
"""
Beta to episcore Conversion for Single Sample Trisomy Detection

This script processes methylation beta values for a single sample to calculate
chromosome-level statistics and Z-scores for trisomy detection.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import click
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Initialize rich console for formatted output
console = Console()


def parse_chr_list(spec: str) -> List[str]:
    """Parse a chromosome spec like '1-22', '1,2,X,Y', or mixed spec like '1-22,X'."""
    spec = spec.strip()
    tokens = [s.strip() for s in spec.split(",") if s.strip()]
    result = []
    for token in tokens:
        if "-" in token and not token.startswith("chr"):
            try:
                start, end = token.split("-")
                start_int = int(start)
                end_int = int(end)
                result.extend([f"chr{i}" for i in range(start_int, end_int + 1)])
            except ValueError:
                result.append(token if token.startswith("chr") else f"chr{token}")
        else:
            result.append(token if token.startswith("chr") else f"chr{token}")
    return result


def read_beta(
    beta_path: str,
    usecols: List[str],
    filter_depth: Optional[int] = None,
    depth_col: Optional[str] = None,
    cpg_filter: Optional[List[str]] = None,
    chr_list: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Read and filter beta value file for a single sample.
    
    Args:
        beta_path: Path to the gzipped TSV beta value file.
        usecols: List of column names to read from the file.
        filter_depth: Minimum depth threshold for filtering (optional).
        depth_col: Column name containing depth information (optional).
        cpg_filter: List of CpG site identifiers to retain (optional).
        chr_list: List of chromosome names to include (optional).
        
    Returns:
        Tuple containing:
            - hypo: DataFrame with hypomethylated CpG sites (meandiff < 0)
            - hyper: DataFrame with hypermethylated CpG sites (meandiff > 0)
            - report: Dictionary with filtering statistics
            
    Raises:
        FileNotFoundError: If beta file doesn't exist.
        ValueError: If required columns are missing.
    """
    try:
        # Read beta value file
        df = pd.read_csv(
            beta_path,
            sep='\t',
            compression='gzip',
            usecols=usecols
        )
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Beta file not found: {beta_path}")
        raise
    except Exception as e:
        console.print(f"[bold red]Error reading beta file {beta_path}:[/bold red] {str(e)}")
        raise

    # Filter by chromosome list if provided
    if chr_list is not None:
        df = df[df['chr'].isin(chr_list)]

    # Separate into hypo- and hyper-methylated sites
    hypo = df[df['meandiff'] < 0].copy()
    hyper = df[df['meandiff'] > 0].copy()

    raw_hypo_count = hypo.shape[0]
    raw_hyper_count = hyper.shape[0]

    # Apply depth filtering if specified
    if filter_depth is not None and depth_col is not None:
        hypo = hypo[hypo[depth_col] > filter_depth]
        hyper = hyper[hyper[depth_col] > filter_depth]

    depth_filtered_hypo_count = hypo.shape[0]
    depth_filtered_hyper_count = hyper.shape[0]

    # Apply CpG site list filtering if specified
    if cpg_filter is not None:
        hypo['filter'] = (hypo['chr'].astype(str) + ':' + 
                         hypo['start'].astype(str) + '-' + 
                         hypo['end'].astype(str))
        hyper['filter'] = (hyper['chr'].astype(str) + ':' + 
                          hyper['start'].astype(str) + '-' + 
                          hyper['end'].astype(str))
        hypo = hypo[hypo['filter'].isin(cpg_filter)].drop(columns=['filter'])
        hyper = hyper[hyper['filter'].isin(cpg_filter)].drop(columns=['filter'])

    cpg_list_filtered_hypo_count = hypo.shape[0]
    cpg_list_filtered_hyper_count = hyper.shape[0]

    # Compile filtering statistics
    report = {
        'raw_hypo_count': raw_hypo_count,
        'raw_hyper_count': raw_hyper_count,
        'depth_filtered_hypo_count': depth_filtered_hypo_count,
        'depth_filtered_hyper_count': depth_filtered_hyper_count,
        'cpg_list_filtered_hypo_count': cpg_list_filtered_hypo_count,
        'cpg_list_filtered_hyper_count': cpg_list_filtered_hyper_count,
    }

    return hypo, hyper, report


def calculate_chr_level_beta(
    hypo: pd.DataFrame,
    hyper: pd.DataFrame,
    chr_list: List[str],
    meth_col: str,
    unmeth_col: Optional[str] = None,
    total_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Calculate chromosome-level beta values from CpG-level data.

    Chromosome beta is ``sum(meth) / sum(total)`` when *total_col* is set,
    otherwise ``sum(meth) / (sum(meth) + sum(unmeth))``.
    """
    if total_col is None and unmeth_col is None:
        raise ValueError("Need unmeth_col or total_col to compute chromosome-level beta")

    hypo_chr_counts = hypo['chr'].value_counts().reindex(chr_list, fill_value=0).to_dict()
    hyper_chr_counts = hyper['chr'].value_counts().reindex(chr_list, fill_value=0).to_dict()

    def chr_level_beta(df: pd.DataFrame) -> np.ndarray:
        if total_col is not None:
            grouped = df.groupby('chr').agg({meth_col: 'sum', total_col: 'sum'})
            grouped['chr_level_beta'] = grouped[meth_col] / grouped[total_col]
        else:
            grouped = df.groupby('chr').agg({meth_col: 'sum', unmeth_col: 'sum'})
            grouped['chr_level_beta'] = grouped[meth_col] / (
                grouped[meth_col] + grouped[unmeth_col]
            )
        grouped = grouped.loc[grouped.index.intersection(chr_list)]
        grouped = grouped.reindex(chr_list)
        return grouped['chr_level_beta'].to_numpy()

    return (
        chr_level_beta(hypo),
        chr_level_beta(hyper),
        hypo_chr_counts,
        hyper_chr_counts,
    )


def compute_zscore_frame(
    hypo: pd.DataFrame,
    hyper: pd.DataFrame,
    chr_list: List[str],
    meth_col: str,
    unmeth_col: Optional[str] = None,
    total_col: Optional[str] = None,
    reference_matrix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the wide one-row z-score table for a hypo/hyper CpG split."""
    hypo_chr_beta, hyper_chr_beta, hypo_chr_counts, hyper_chr_counts = (
        calculate_chr_level_beta(
            hypo, hyper, chr_list, meth_col, unmeth_col, total_col
        )
    )
    hypo_z_intra, hyper_z_intra, s_intra = calculate_s_intra(
        hypo_chr_beta, hyper_chr_beta, hypo_chr_counts, hyper_chr_counts, chr_list
    )
    hypo_z_inter = hyper_z_inter = s_inter = None
    if reference_matrix is not None:
        hypo_z_inter, hyper_z_inter, s_inter = calculate_s_inter(
            hypo_z_intra,
            hyper_z_intra,
            hypo_chr_counts,
            hyper_chr_counts,
            chr_list,
            reference_matrix,
        )
    return build_output_dataframe(
        chr_list,
        hypo_chr_beta,
        hyper_chr_beta,
        hypo_z_intra,
        hyper_z_intra,
        s_intra,
        hypo_chr_counts,
        hyper_chr_counts,
        hypo_z_inter,
        hyper_z_inter,
        s_inter,
    )


def calculate_s_intra(
    hypo_chr_beta: np.ndarray,
    hyper_chr_beta: np.ndarray,
    hypo_chr_counts: Dict[str, int],
    hyper_chr_counts: Dict[str, int],
    chr_list: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate intra-sample Z-scores (s_intra) for a single sample.
    
    This performs within-sample normalization by calculating z-scores
    across chromosomes for both hypo and hyper, then combining them
    using weighted averaging based on CpG counts.
    
    Args:
        hypo_chr_beta: Array of chromosome-level beta values for hypo.
        hyper_chr_beta: Array of chromosome-level beta values for hyper.
        hypo_chr_counts: Dictionary of CpG counts per chromosome for hypo.
        hyper_chr_counts: Dictionary of CpG counts per chromosome for hyper.
        chr_list: List of chromosome names.
        
    Returns:
        Tuple containing:
            - hypo_z_intra: Array of intra-sample Z-scores for hypo
            - hyper_z_intra: Array of intra-sample Z-scores for hyper
            - s_intra: Array of combined weighted Z-scores
    """
    # Intra-sample normalization (z-score within sample across chromosomes)
    if np.isnan(hypo_chr_beta).all():
        hypo_z_intra = np.full(len(chr_list), np.nan)
    else:
        hypo_z_intra = (hypo_chr_beta - np.nanmean(hypo_chr_beta)) / np.nanstd(hypo_chr_beta)
    
    if np.isnan(hyper_chr_beta).all():
        hyper_z_intra = np.full(len(chr_list), np.nan)
    else:
        hyper_z_intra = (hyper_chr_beta - np.nanmean(hyper_chr_beta)) / np.nanstd(hyper_chr_beta)
    
    # Combine hypo and hyper using weighted averaging
    hypo_counts_array = np.array([hypo_chr_counts[chr_name] for chr_name in chr_list])
    hyper_counts_array = np.array([hyper_chr_counts[chr_name] for chr_name in chr_list])
    
    # Calculate weights (square root of CpG counts)
    w_hypo = np.sqrt(hypo_counts_array)
    w_hyper = np.sqrt(hyper_counts_array)
    total_weight = np.sqrt(w_hypo**2 + w_hyper**2)
    
    # Compute weighted combined Z-score (s_intra)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_intra = (hyper_z_intra * w_hyper - hypo_z_intra * w_hypo) / total_weight
        s_intra = np.where(np.isnan(s_intra), 0, s_intra)
    
    return hypo_z_intra, hyper_z_intra, s_intra


def calculate_s_inter(
    hypo_z_intra: np.ndarray,
    hyper_z_intra: np.ndarray,
    hypo_chr_counts: Dict[str, int],
    hyper_chr_counts: Dict[str, int],
    chr_list: List[str],
    reference_matrix: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate inter-sample Z-scores (s_inter) using reference matrix.
    
    This performs normalization using reference sample statistics by
    extracting chr*_hypo_z_intra and chr*_hyper_z_intra columns from the reference matrix to
    calculate mean and standard deviation for each chromosome.
    
    Args:
        hypo_z_intra: Array of intra-sample Z-scores for hypo.
        hyper_z_intra: Array of intra-sample Z-scores for hyper.
        hypo_chr_counts: Dictionary of CpG counts per chromosome for hypo.
        hyper_chr_counts: Dictionary of CpG counts per chromosome for hyper.
        chr_list: List of chromosome names.
        reference_matrix: DataFrame with reference samples containing chr*_hypo_z_intra and chr*_hyper_z_intra columns.
        
    Returns:
        Tuple containing:
            - hypo_z_inter: Array of inter-sample Z-scores for hypo
            - hyper_z_inter: Array of inter-sample Z-scores for hyper
            - s_inter: Array of combined weighted Z-scores normalized to reference
            
    Raises:
        ValueError: If required columns are missing in reference matrix.
    """
    # Extract reference mean and std for each chromosome
    ref_hypo_means = []
    ref_hypo_stds = []
    ref_hyper_means = []
    ref_hyper_stds = []

    for chr_name in chr_list:
        col_name = f"{chr_name}_hypo_z_intra"
        if col_name not in reference_matrix.columns:
            raise ValueError(
                f"Reference matrix missing required column: {col_name}. "
                f"Available columns: {list(reference_matrix.columns)}"
            )
        
        # Calculate mean and std from reference samples
        ref_values = reference_matrix[col_name].dropna()
        if len(ref_values) == 0:
            console.print(f"[yellow]Warning:[/yellow] No valid reference values for {chr_name}")
            ref_hypo_means.append(0.0)
            ref_hypo_stds.append(1.0)
        else:
            ref_hypo_means.append(ref_values.mean())
            ref_hypo_stds.append(ref_values.std(ddof=0))

        col_name = f"{chr_name}_hyper_z_intra"
        if col_name not in reference_matrix.columns:
            raise ValueError(
                f"Reference matrix missing required column: {col_name}. "
                f"Available columns: {list(reference_matrix.columns)}"
            )
        
        # Calculate mean and std from reference samples
        ref_values = reference_matrix[col_name].dropna()
        if len(ref_values) == 0:
            console.print(f"[yellow]Warning:[/yellow] No valid reference values for {chr_name}")
            ref_hyper_means.append(0.0)
            ref_hyper_stds.append(1.0)
        else:
            ref_hyper_means.append(ref_values.mean())
            ref_hyper_stds.append(ref_values.std(ddof=0))
    
    ref_hypo_mean_array = np.array(ref_hypo_means)
    ref_hypo_std_array = np.array(ref_hypo_stds)
    ref_hyper_mean_array = np.array(ref_hyper_means)
    ref_hyper_std_array = np.array(ref_hyper_stds)
    
    # Apply reference normalization to hypo and hyper z_intra
    with np.errstate(divide='ignore', invalid='ignore'):
        hypo_z_inter = (hypo_z_intra - ref_hypo_mean_array) / ref_hypo_std_array
        hyper_z_inter = (hyper_z_intra - ref_hyper_mean_array) / ref_hyper_std_array
    
    # Combine hypo and hyper using weighted averaging
    hypo_counts_array = np.array([hypo_chr_counts[chr_name] for chr_name in chr_list])
    hyper_counts_array = np.array([hyper_chr_counts[chr_name] for chr_name in chr_list])
    
    # Calculate weights (square root of CpG counts)
    w_hypo = np.sqrt(hypo_counts_array)
    w_hyper = np.sqrt(hyper_counts_array)
    total_weight = np.sqrt(w_hypo**2 + w_hyper**2)
    
    # Compute weighted combined Z-score (s_inter)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_inter = (hyper_z_inter * w_hyper - hypo_z_inter * w_hypo) / total_weight
        s_inter = np.where(np.isnan(s_inter), 0, s_inter)
    
    return hypo_z_inter, hyper_z_inter, s_inter


def build_output_dataframe(
    chr_list: List[str],
    hypo_chr_beta: np.ndarray,
    hyper_chr_beta: np.ndarray,
    hypo_z_intra: np.ndarray,
    hyper_z_intra: np.ndarray,
    s_intra: np.ndarray,
    hypo_chr_counts: Dict[str, int],
    hyper_chr_counts: Dict[str, int],
    hypo_z_inter: Optional[np.ndarray] = None,
    hyper_z_inter: Optional[np.ndarray] = None,
    s_inter: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Build output DataFrame with all calculated metrics.
    
    Args:
        chr_list: List of chromosome names.
        hypo_chr_beta: Array of chromosome-level beta values for hypo.
        hyper_chr_beta: Array of chromosome-level beta values for hyper.
        hypo_z_intra: Array of intra-sample Z-scores for hypo.
        hyper_z_intra: Array of intra-sample Z-scores for hyper.
        s_intra: Array of combined weighted Z-scores (intra-sample).
        hypo_chr_counts: Dictionary of CpG counts per chromosome for hypo.
        hyper_chr_counts: Dictionary of CpG counts per chromosome for hyper.
        hypo_z_inter: Array of inter-sample Z-scores for hypo (optional).
        hyper_z_inter: Array of inter-sample Z-scores for hyper (optional).
        s_inter: Array of combined weighted Z-scores (inter-sample, optional).
        
    Returns:
        DataFrame with columns for each metric organized by chromosome.
    """
    output_data = {}
    
    # Add metrics for each chromosome
    for idx, chr_name in enumerate(chr_list):
        output_data[f"{chr_name}_hypo_beta"] = hypo_chr_beta[idx]
        output_data[f"{chr_name}_hyper_beta"] = hyper_chr_beta[idx]
        output_data[f"{chr_name}_hypo_z_intra"] = hypo_z_intra[idx]
        output_data[f"{chr_name}_hyper_z_intra"] = hyper_z_intra[idx]
        output_data[f"{chr_name}_s_intra"] = s_intra[idx]
        output_data[f"{chr_name}_hypo_cpgs_count"] = hypo_chr_counts[chr_name]
        output_data[f"{chr_name}_hyper_cpgs_count"] = hyper_chr_counts[chr_name]
        
        # Add inter-sample metrics if provided
        if hypo_z_inter is not None and hyper_z_inter is not None and s_inter is not None:
            output_data[f"{chr_name}_hypo_z_inter"] = hypo_z_inter[idx]
            output_data[f"{chr_name}_hyper_z_inter"] = hyper_z_inter[idx]
            output_data[f"{chr_name}_s_inter"] = s_inter[idx]
    
    # Create DataFrame with single row
    df = pd.DataFrame([output_data])
    
    return df


@click.command()
@click.option(
    '--beta-value',
    required=True,
    type=click.Path(exists=True),
    help='Path to gzipped TSV beta value file for single sample'
)
@click.option(
    '--reference-episcore-matrix',
    type=click.Path(exists=True),
    default=None,
    help='Path to reference episcore matrix (TSV) with chr*_s_intra columns (optional)'
)
@click.option(
    '--output-prefix',
    required=True,
    type=str,
    help='Output prefix for z-score file (TSV, can be gzipped if ends with .gz)'
)
@click.option(
    '--depth',
    type=int,
    default=None,
    help='Minimum depth threshold for filtering CpG sites (optional)'
)
@click.option(
    '--cpg-list',
    type=click.Path(exists=True),
    default=None,
    help='Path to CpG list file (filtered CpG sites, optional)'
)
@click.option(
    '--chr-list',
    default='1-22',
    type=str,
    help='Chromosomes to analyze (e.g., "1-22", "1,2,3,X,Y", or "1-22,X")'
)
@click.option(
    '--beta-cols',
    default='chr,start,end,target_meth_count,target_unmeth_count,raw_meth_count,raw_total_count,meandiff',
    type=str,
    help='Comma-separated list of columns to read from beta file'
)
@click.option(
    '--depth-col',
    default='raw_total_count',
    type=str,
    help='Column name for depth filtering'
)
@click.option(
    '--ncpus',
    default=1,
    type=int,
    help='Number of CPUs (currently not used, reserved for future parallelization)'
)
def main(
    beta_value: str,
    reference_episcore_matrix: Optional[str],
    output_prefix: str,
    depth: Optional[int],
    cpg_list: Optional[str],
    chr_list: str,
    beta_cols: str,
    depth_col: str,
    ncpus: int
) -> None:
    """
    Calculate beta values and Z-scores for a single sample trisomy detection.
    
    This script processes methylation beta values from CpG sites to calculate
    chromosome-level statistics and Z-scores:
    
    \b
    1. Chromosome-level mean beta values (hypo/hyper)
    2. Intra-sample Z-scores (normalized within sample)
    3. Inter-sample Z-scores (normalized using reference matrix, if provided)
    4. Combined weighted Z-scores (hypo + hyper)
    
    The output is a TSV file with one row containing all calculated metrics.
    
    If --reference-episcore-matrix is not provided, only s_intra is calculated.
    If --reference-episcore-matrix is provided, both s_intra and s_inter are calculated.
    
    Example (without reference):
    \b
        beta_to_episcore.py \\
            --beta-value sample.beta.tsv.gz \\
            --output sample_zscore.tsv.gz \\
            --depth 100 \\
            --cpg-list cpg_list.txt
    
    Example (with reference):
    \b
        beta_to_episcore.py \\
            --beta-value sample.beta.tsv.gz \\
            --reference-episcore-matrix reference_matrix.tsv \\
            --output sample_zscore.tsv.gz \\
            --depth 100 \\
            --cpg-list cpg_list.txt
    """
    console.rule("[bold blue]Beta to episcore Conversion for Single Sample")
    console.print(f"\n[bold]Input Parameters:[/bold]")
    console.print(f"  Beta value file: {beta_value}")
    console.print(f"  Reference matrix: {reference_episcore_matrix if reference_episcore_matrix else 'None (s_intra only)'}")
    console.print(f"  Output prefix: {output_prefix}")
    console.print(f"  Depth threshold: {depth if depth else 'None'}")
    console.print(f"  CpG list file: {cpg_list if cpg_list else 'None'}")
    console.print(f"  Chromosomes: {chr_list}")
    console.print(f"  Depth column: {depth_col}")
    
    output = f"{output_prefix}_zscore.tsv"
    try:
        # Parse chromosome list
        chromosomes = parse_chr_list(chr_list)
        
        console.print(f"  Analyzing {len(chromosomes)} chromosomes")
        
        # Parse beta columns
        beta_columns = [col.strip() for col in beta_cols.split(',')]
        console.print(f"  Beta columns: {', '.join(beta_columns)}")
        
        # Load CpG list if provided
        cpg_filter_list = None
        if cpg_list:
            console.print("\n[bold cyan]Loading CpG list...[/bold cyan]")
            cpg_list_df = pd.read_csv(cpg_list, sep='\t', usecols=['chr', 'start', 'end'])
            cpg_list_df['filter'] = (cpg_list_df['chr'].astype(str) + ':' + 
                                     cpg_list_df['start'].astype(str) + '-' + 
                                     cpg_list_df['end'].astype(str))
            cpg_filter_list = cpg_list_df['filter'].tolist()
            console.print(f"[green]✓[/green] Loaded {len(cpg_filter_list):,} CpG sites")
        
        # Read beta value file
        console.print("\n[bold cyan]Step 1: Reading beta value file[/bold cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Reading beta file...", total=None)
            
            hypo, hyper, report = read_beta(
                beta_value,
                usecols=beta_columns,
                filter_depth=depth,
                depth_col=depth_col if depth else None,
                cpg_filter=cpg_filter_list,
                chr_list=chromosomes
            )
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✓[/green] Hypo CpG sites: {report['cpg_list_filtered_hypo_count']:,}")
        console.print(f"[green]✓[/green] Hyper CpG sites: {report['cpg_list_filtered_hyper_count']:,}")

        ref_matrix = None
        if reference_episcore_matrix:
            console.print("\n[bold cyan]Loading reference matrix[/bold cyan]")
            ref_matrix = pd.read_csv(reference_episcore_matrix, sep='\t')
            console.print(f"[green]✓[/green] Loaded reference matrix: {len(ref_matrix)} samples")

        # After-MQ (primary): target meth / unmeth. Fall back to raw if target is absent.
        console.print("\n[bold cyan]Step 2: After-MQ z-scores (target counts)[/bold cyan]")
        if 'target_meth_count' in hypo.columns and 'target_unmeth_count' in hypo.columns:
            after_meth, after_unmeth, after_total = (
                'target_meth_count',
                'target_unmeth_count',
                None,
            )
        elif 'raw_meth_count' in hypo.columns and 'raw_unmeth_count' in hypo.columns:
            after_meth, after_unmeth, after_total = (
                'raw_meth_count',
                'raw_unmeth_count',
                None,
            )
        else:
            raise ValueError(
                "Cannot find methylation columns. Expected "
                "'target_meth_count/target_unmeth_count' "
                "(or raw_meth_count/raw_unmeth_count as fallback)"
            )
        console.print(f"  Using columns: {after_meth}, {after_unmeth or after_total}")
        output_df = compute_zscore_frame(
            hypo,
            hyper,
            chromosomes,
            after_meth,
            unmeth_col=after_unmeth,
            total_col=after_total,
            reference_matrix=ref_matrix,
        )
        compression = 'gzip' if output.endswith('.gz') else None
        output_df.to_csv(output, sep='\t', index=False, float_format='%.6f', compression=compression)
        console.print(f"[green]✓[/green] After-MQ output saved to: {output}")
        console.print(f"    Columns: {len(output_df.columns)}")

        # Before-MQ: raw_meth_count / raw_total_count, same site filter and columns.
        has_raw = (
            'raw_meth_count' in hypo.columns
            and 'raw_total_count' in hypo.columns
        )
        if has_raw:
            console.print("\n[bold cyan]Step 3: Before-MQ z-scores (raw meth / raw total)[/bold cyan]")
            before_df = compute_zscore_frame(
                hypo,
                hyper,
                chromosomes,
                'raw_meth_count',
                total_col='raw_total_count',
                reference_matrix=ref_matrix,
            )
            if list(before_df.columns) != list(output_df.columns):
                raise ValueError(
                    "before-MQ columns do not match after-MQ: "
                    f"{list(before_df.columns)} vs {list(output_df.columns)}"
                )
            before_path = f"{output_prefix}_zscore_before_mq.tsv"
            before_df.to_csv(
                before_path, sep='\t', index=False, float_format='%.6f'
            )
            console.print(f"[green]✓[/green] Before-MQ output saved to: {before_path}")
            console.print(f"    Columns: {len(before_df.columns)}")
        else:
            console.print(
                "\n[yellow]Note:[/yellow] raw_meth_count/raw_total_count not in beta file; "
                "skipping _zscore_before_mq.tsv"
            )

        console.rule("[bold green]✓ Analysis Complete")
        console.print(f"\n[bold]Output file:[/bold] {output}\n")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}", style="bold red")
        console.print_exception()
        sys.exit(1)


if __name__ == '__main__':
    main()
