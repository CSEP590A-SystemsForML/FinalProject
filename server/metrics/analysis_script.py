import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "metrics.db"
OUTPUT_IMAGE = BASE_DIR / "cost_by_optimizations.png"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def calculate_costs_and_plot():
    """
    Function 1: Calculates total cost for each run_id and displays it 
    under the activated optimizations for each as a bar graph.
    """
    conn = get_db_connection()
    try:
        # Check if cost column exists in problem_solving table
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(problem_solving)")
        columns = [col[1] for col in cursor.fetchall()]
        
        has_cost_column = "cost" in columns
        
        # We will query cost from problem_solving if it exists
        if has_cost_column:
            query = """
            SELECT ps.run_id, ps.cost, 
                   o.caveman, o.quantized_local_lm, o.quantized_kv_cache, 
                   o.web_search_compression, o.local_model_solves, 
                   o.long_context_compression_lemma, o.long_context_compression_ai
            FROM problem_solving ps
            JOIN optimizations o ON ps.run_id = o.run_id
            """
            df = pd.read_sql_query(query, conn)
        else:
            # Fallback/mock if the cost column is not present in problem_solving yet
            print("Warning: 'cost' column not found in 'problem_solving' table. Using mock costs for visualization.")
            query = """
            SELECT ps.run_id, 
                   o.caveman, o.quantized_local_lm, o.quantized_kv_cache, 
                   o.web_search_compression, o.local_model_solves, 
                   o.long_context_compression_lemma, o.long_context_compression_ai
            FROM problem_solving ps
            JOIN optimizations o ON ps.run_id = o.run_id
            """
            df = pd.read_sql_query(query, conn)
            # Generate dummy costs based on some heuristic or fixed values for display
            df['cost'] = 0.05 + 0.02 * df['caveman'] + 0.1 * df['local_model_solves']
            
        if df.empty:
            print("No run data found in the database. Plot cannot be generated.")
            return
            
        # Determine active optimizations for each run
        optimization_cols = [
            'caveman', 'quantized_local_lm', 'quantized_kv_cache', 
            'web_search_compression', 'local_model_solves', 
            'long_context_compression_lemma', 'long_context_compression_ai'
        ]
        
        active_opts_list = []
        for idx, row in df.iterrows():
            active = [col for col in optimization_cols if row[col]]
            active_str = ", ".join(active) if active else "none"
            active_opts_list.append(active_str)
            
        df['active_optimizations'] = active_opts_list
        
        # Create labels combining run_id and active optimizations
        df['label'] = df.apply(lambda r: f"Run: {r['run_id']}\n({r['active_optimizations']})", axis=1)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['label'], df['cost'], color='skyblue', edgecolor='black')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f"${height:.4f}",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
                     
        plt.title("Total Cost by Run and Active Optimizations", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Run ID & Active Optimizations", fontsize=12, labelpad=10)
        plt.ylabel("Total Cost ($)", fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(OUTPUT_IMAGE, dpi=300)
        print(f"Cost plot successfully saved to {OUTPUT_IMAGE}")
        plt.close()
        
    finally:
        conn.close()

def determine_outliers():
    """
    Function 2: Determines outliers/routing problems based on specific rules:
    - If a problem has been labeled as anything other than "easy" and goes to gpt-oss-20b that's a problem.
    - If it is a medium and doesn't go to oss 120b or deepseek v4 flash, that's a problem.
    - If a problem that is hard goes to gpt oss 20b or 120b, that's a problem.
    - If a very hard goes to any model other than kimi k2.6 that's also a problem.
    """
    conn = get_db_connection()
    try:
        query = "SELECT run_id, problem_id, model_id, difficulty FROM routing"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No routing data found to analyze for outliers.")
            return []
            
        outliers = []
        for idx, row in df.iterrows():
            run_id = row['run_id']
            prob_id = row['problem_id']
            model_id = row['model_id']
            difficulty = row['difficulty']
            
            diff_clean = str(difficulty).strip().lower()
            model_clean = str(model_id).strip().lower()
            
            is_outlier = False
            reason = ""
            
            # 1. Anything other than "easy" going to gpt-oss-20b is a problem
            if diff_clean != "easy" and "gpt-oss-20b" in model_clean:
                is_outlier = True
                reason = "Non-easy problem routed to gpt-oss-20b"
                
            # 2. Medium problem must go to oss 120b or deepseek v4 flash
            elif diff_clean == "medium" and not ("gpt-oss-120b" in model_clean or "deepseek-v4-flash" in model_clean):
                is_outlier = True
                reason = "Medium problem not routed to gpt-oss-120b or deepseek-v4-flash"
                
            # 3. Hard problem going to gpt-oss-20b or 120b is a problem
            elif diff_clean == "hard" and ("gpt-oss-20b" in model_clean or "gpt-oss-120b" in model_clean):
                is_outlier = True
                reason = "Hard problem routed to gpt-oss-20b or gpt-oss-120b"
                
            # 4. Very hard problem going to any model other than kimi k2.6 is a problem
            elif diff_clean in ("very_hard", "very hard") and "kimi-k2.6" not in model_clean:
                is_outlier = True
                reason = "Very hard problem not routed to kimi-k2.6"
                
            if is_outlier:
                outliers.append({
                    'run_id': run_id,
                    'problem_id': prob_id,
                    'model_id': model_id,
                    'difficulty': difficulty,
                    'reason': reason
                })
                
        outliers_df = pd.DataFrame(outliers)
        print("\n=== ROUTING OUTLIERS DETECTED ===")
        if not outliers_df.empty:
            print(outliers_df.to_string(index=False))
        else:
            print("No outliers or routing anomalies detected!")
        print("=================================\n")
        
        return outliers
    finally:
        conn.close()

if __name__ == "__main__":
    print("Running stand-alone analysis script...")
    calculate_costs_and_plot()
    determine_outliers()