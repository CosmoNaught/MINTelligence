#!/usr/bin/env python3
import duckdb
import argparse

def export_metrics(db_path):
    """
    Query the database to export prevalence and cases metrics to a CSV file.
    Args:
        db_path (str): Path to the DuckDB database
    """
    try:
        # Connect to the database
        conn = duckdb.connect(db_path, read_only=True)
        print(f"Connected to database: {db_path}")
        
        # Query to get the first 10 parameters with cases multiplied by 1000 and aggregated every 30 timesteps
        metrics_query = """
        WITH first_10_params AS (
            SELECT DISTINCT parameter_index
            FROM simulation_results
            ORDER BY parameter_index
            LIMIT 10
        ),
        -- Group by parameter_index, simulation_index, and 30-timestep groups
        aggregated_data AS (
            SELECT
                r.parameter_index,
                r.simulation_index,
                (r.timesteps / 30) * 30 AS timestep_group,
                SUM(r.n_detect_lm_0_1825) AS total_detect_lm,
                SUM(r.n_age_0_1825) AS total_age_0_1825,
                SUM(r.n_inc_clinical_0_36500) AS total_inc_clinical,
                SUM(r.n_age_0_36500) AS total_age_0_36500
            FROM
                simulation_results r
            INNER JOIN
                first_10_params p ON r.parameter_index = p.parameter_index
            GROUP BY
                r.parameter_index, r.simulation_index, (r.timesteps / 30) * 30
        )
        SELECT
            parameter_index,
            simulation_index,
            timestep_group AS timesteps,
            total_detect_lm,
            total_age_0_1825,
            CASE WHEN total_age_0_1825 = 0 THEN NULL
                ELSE CAST(total_detect_lm AS DOUBLE) / total_age_0_1825
            END AS prevalence,
            total_inc_clinical,
            total_age_0_36500,
            CASE WHEN total_age_0_36500 = 0 THEN NULL
                ELSE (CAST(total_inc_clinical AS DOUBLE) / total_age_0_36500) * 1000
            END AS cases_per_1000
        FROM
            aggregated_data
        ORDER BY
            parameter_index, simulation_index, timesteps
        LIMIT 100 -- Still limiting to approximately 10 rows per parameter
        """
        
        # Execute the query and save directly to CSV
        conn.execute(f"COPY ({metrics_query}) TO 'query.csv' (HEADER, DELIMITER ',')")
        print(f"Successfully exported data to query.csv")
        conn.close()
    except Exception as e:
        print(f"Error exporting metrics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export prevalence and cases metrics to CSV")
    parser.add_argument("--db-path", default="/home/cosmo/net/malaria/Cosmo/segMINT/db3.duckdb",
                        help="Path to the DuckDB database file")
    args = parser.parse_args()
    export_metrics(args.db_path)