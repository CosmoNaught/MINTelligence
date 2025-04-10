#!/usr/bin/env python3
import duckdb
import pandas as pd
import numpy as np
import argparse
from tabulate import tabulate

def analyze_metrics(db_path, sample_size=10):
    """
    Query the database to analyze prevalence and cases metrics.
    
    Args:
        db_path (str): Path to the DuckDB database
        sample_size (int): Number of random parameter combinations to sample
    """
    try:
        # Connect to the database
        conn = duckdb.connect(db_path, read_only=True)
        print(f"Connected to database: {db_path}")
        
        # Get distinct parameter combinations
        distinct_params_query = """
        SELECT DISTINCT parameter_index, simulation_index 
        FROM simulation_results
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        distinct_params = conn.execute(distinct_params_query, [sample_size]).fetchall()
        
        if not distinct_params:
            print("No data found in the simulation_results table.")
            return
        
        print(f"\nAnalyzing {len(distinct_params)} random parameter/simulation combinations")
        
        # Create placeholders for the WHERE clause
        param_conditions = []
        for param_idx, sim_idx in distinct_params:
            param_conditions.append(f"(parameter_index = {param_idx} AND simulation_index = {sim_idx})")
        
        where_clause = " OR ".join(param_conditions)
        
        # Query to get metrics for selected parameter combinations
        metrics_query = f"""
        SELECT 
            parameter_index,
            simulation_index,
            timesteps,
            -- Prevalence calculation
            n_detect_lm_0_1825,
            n_age_0_1825,
            CASE WHEN n_age_0_1825 = 0 THEN NULL
                 ELSE CAST(n_detect_lm_0_1825 AS DOUBLE) / n_age_0_1825
            END AS prevalence,
            
            -- Cases calculation
            n_inc_clinical_0_36500,
            n_age_0_36500,
            CASE WHEN n_age_0_36500 = 0 THEN NULL
                 ELSE CAST(n_inc_clinical_0_36500 AS DOUBLE) / n_age_0_36500
            END AS cases
        FROM 
            simulation_results
        WHERE 
            {where_clause}
        ORDER BY 
            parameter_index, simulation_index, timesteps
        """
        
        # Execute the query and convert to DataFrame
        results_df = conn.execute(metrics_query).fetchdf()
        
        if results_df.empty:
            print("No results found for the selected parameters.")
            return
        
        # Calculate statistics for each parameter/simulation combination
        stats = []
        for (param_idx, sim_idx), group in results_df.groupby(['parameter_index', 'simulation_index']):
            # Calculate statistics for prevalence
            prevalence_mean = group['prevalence'].mean()
            prevalence_min = group['prevalence'].min()
            prevalence_max = group['prevalence'].max()
            prevalence_null_count = group['prevalence'].isna().sum()
            prevalence_null_pct = (prevalence_null_count / len(group)) * 100
            
            # Calculate statistics for cases
            cases_mean = group['cases'].mean()
            cases_min = group['cases'].min()
            cases_max = group['cases'].max()
            cases_null_count = group['cases'].isna().sum()
            cases_null_pct = (cases_null_count / len(group)) * 100
            
            # Add row to stats
            stats.append({
                'parameter_index': param_idx,
                'simulation_index': sim_idx,
                'record_count': len(group),
                'prevalence_mean': prevalence_mean,
                'prevalence_min': prevalence_min,
                'prevalence_max': prevalence_max,
                'prevalence_null_count': prevalence_null_count,
                'prevalence_null_pct': prevalence_null_pct,
                'cases_mean': cases_mean,
                'cases_min': cases_min,
                'cases_max': cases_max,
                'cases_null_count': cases_null_count,
                'cases_null_pct': cases_null_pct
            })
        
        stats_df = pd.DataFrame(stats)
        
        # Print the statistics
        print("\nMetrics Statistics per Parameter/Simulation Combination:")
        print(tabulate(stats_df, headers='keys', tablefmt='psql', floatfmt='.5f'))
        
        # Print overall summary
        print("\nOverall Summary:")
        print(f"Total records analyzed: {len(results_df)}")
        print(f"Prevalence - Mean: {results_df['prevalence'].mean():.5f}, Min: {results_df['prevalence'].min():.5f}, Max: {results_df['prevalence'].max():.5f}")
        print(f"Prevalence - Null values: {results_df['prevalence'].isna().sum()} ({results_df['prevalence'].isna().mean()*100:.2f}%)")
        print(f"Cases - Mean: {results_df['cases'].mean():.5f}, Min: {results_df['cases'].min():.5f}, Max: {results_df['cases'].max():.5f}")
        print(f"Cases - Null values: {results_df['cases'].isna().sum()} ({results_df['cases'].isna().mean()*100:.2f}%)")
        
        # Sample of raw data
        print("\nSample of Raw Data (first 10 rows):")
        print(tabulate(results_df.head(10), headers='keys', tablefmt='psql', floatfmt='.5f'))
        
        conn.close()
        
    except Exception as e:
        print(f"Error analyzing metrics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prevalence and cases metrics in DuckDB")
    parser.add_argument("--db-path", default="/home/cosmo/net/malaria/Cosmo/segMINT/db3.duckdb", 
                        help="Path to the DuckDB database file")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Number of random parameter combinations to sample")
    args = parser.parse_args()
    
    analyze_metrics(args.db_path, args.sample_size)