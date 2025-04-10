#!/usr/bin/env python3
import argparse
import duckdb
import pandas as pd
import time
import os

def query_zero_future_interventions(db_path, table_name, output_dir=None):
    """
    Query the database to find parameters where both itn_future and irs_future are set to 0
    """
    print(f"[INFO] Connecting to DuckDB at {db_path}")
    start_time = time.time()
    
    try:
        # Connect to the database
        con = duckdb.connect(db_path, read_only=True)
        con.execute("PRAGMA threads=8;")
        
        # Build the query to find parameters with zero future interventions
        query = f"""
            SELECT DISTINCT 
                parameter_index, 
                global_index,
                itn_future,
                irs_future,
                -- Include other key parameters for context
                eir,  
                phi_bednets,
                seasonal, 
                lsm
            FROM {table_name}
            WHERE itn_future = 0 AND irs_future = 0 AND itn_use = 0 AND irs_use = 0 AND routine = 0 AND Q0 = 1
            ORDER BY parameter_index
        """
        
        # Execute the query
        result = con.execute(query).df()
        con.close()
        
        query_time = time.time() - start_time
        print(f"[INFO] Query completed in {query_time:.2f} seconds")
        
        if result.empty:
            print("[INFO] No parameters found with both itn_future=0 and irs_future=0")
            return None
        
        # Print summary results
        param_count = len(result)
        print(f"\n[RESULTS] Found {param_count} parameters with itn_future=0 and irs_future=0:")
        
        for i, row in result.iterrows():
            print(f"  - Parameter Index: {row['parameter_index']}, Global Index: {row['global_index']}")
        
        # Save results to CSV if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "zero_future_interventions.csv")
            result.to_csv(csv_path, index=False)
            print(f"\n[INFO] Results saved to {csv_path}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Query DuckDB to find parameters with zero future interventions")
    parser.add_argument("--db-path", default="/home/cosmo/net/malaria/Cosmo/segMINT/db3.duckdb", 
                       help="Path to DuckDB database file")
    parser.add_argument("--table-name", default="simulation_results", 
                       help="Table name inside DuckDB")
    parser.add_argument("--output-dir", default="./query_results", 
                       help="Directory to save query results")
    args = parser.parse_args()
    
    # Execute the query
    query_zero_future_interventions(args.db_path, args.table_name, args.output_dir)
    
    print("\n[INFO] Query execution completed")

if __name__ == "__main__":
    main()