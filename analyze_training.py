import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def analyze_custom(log_dir, duration_seconds=0, run_id=None):
    training_file = os.path.join(log_dir, "training_log.csv")
    steps_file = os.path.join(log_dir, "steps_log.csv")
    
    # Suffix setup
    suffix = f"_{run_id}" if run_id is not None else ""
    
    if not os.path.exists(training_file):
        print("No training log found.")
        return

    print(f"Generating Report for {log_dir} (Run {run_id})...")
    try:
        df = pd.read_csv(training_file)
    except Exception as e:
        print(f"Error reading training log: {e}")
        return
    
    # ... (Data processing remains same)
    
    # Fix: Rename 'reward' to 'total_reward' if needed
    if 'reward' in df.columns and 'total_reward' not in df.columns:
        df['total_reward'] = df['reward']

    # --- Metrics Calculation ---
    if 'episode' in df.columns:
        df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
        df.dropna(subset=['episode'], inplace=True)
        df.sort_values('episode', inplace=True)
        
    total_episodes = len(df)
    avg_reward = df['total_reward'].mean() if 'total_reward' in df.columns else 0
    
    win_rate = 0
    if 'win' in df.columns:
        win_rate = df['win'].mean() * 100
    
    # Stability
    stability = 0
    if 'total_reward' in df.columns and len(df) > 50:
        stability = df['total_reward'].rolling(50).std().iloc[-1]

    # --- Plots ---
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Phases
        phase_lines = [10000, 25000]
        def add_phases(ax):
            for p in phase_lines:
                ax.axvline(x=p, color='k', linestyle='--', alpha=0.3)
        
        # 1. Learning State [0,0]
        ax1 = axes[0,0]
        has_q = False
        if 'avg_q' in df.columns:
            ax1.plot(df['episode'], df['avg_q'], color='blue', label='Avg Max Q')
            ax1.set_ylabel('Avg Q-Value', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            has_q = True
        
        if 'q_size' in df.columns:
            ax1r = ax1.twinx()
            ax1r.plot(df['episode'], df['q_size'], color='red', alpha=0.5, label='Q-Table Size')
            ax1r.set_ylabel('States Discovered', color='red')
            ax1r.tick_params(axis='y', labelcolor='red')
            has_q = True
            
        if has_q: ax1.set_title("Learning State")
        else: ax1.text(0.5, 0.5, "No Q-Data", ha='center')
        add_phases(ax1)
        ax1.grid(True)

        # 2. Win Rate [0,1]
        if 'win' in df.columns:
            axes[0,1].plot(df['episode'], df['win'].rolling(10).mean() * 100, color='green')
            axes[0,1].set_title(f"Win Rate % (Avg: {win_rate:.1f}%)")
        else: axes[0,1].set_title("Win Rate (Data Missing)")
        axes[0,1].set_ylim(0, 105)
        add_phases(axes[0,1])
        axes[0,1].grid(True)
        
        # 3. Rewards [1,0]
        ax3 = axes[1,0]
        if 'total_reward' in df.columns:
            ax3.plot(df['episode'], df['total_reward'], alpha=0.2, color='gray')
            ax3.plot(df['episode'], df['total_reward'].rolling(10).mean(), color='blue')
            ax3.set_ylabel('Reward', color='blue')
            ax3.set_title(f"Rewards")
        
        # 3. Rewards [1,0]
        ax3 = axes[1,0]
        if 'total_reward' in df.columns:
            ax3.plot(df['episode'], df['total_reward'], alpha=0.2, color='gray')
            ax3.plot(df['episode'], df['total_reward'].rolling(10).mean(), color='blue', label='Rew (Avg 10)')
            ax3.set_ylabel('Reward', color='blue')
            ax3.set_title(f"Rewards")
            ax3.axhline(0, color='black', linestyle='--', alpha=0.3) # Zero line
        
        if 'steps' in df.columns:
            ax3r = ax3.twinx()
            ax3r.plot(df['episode'], df['steps'].rolling(10).mean(), color='orange', alpha=0.8, label='Steps')
            ax3r.set_ylabel('Steps', color='orange')
            # Optional: Start steps from 0
            ax3r.set_ylim(bottom=0)
        
        add_phases(ax3)
        ax3.grid(True)
        
        # 4. Epsilon [1,1]
        if 'epsilon' in df.columns:
            axes[1,1].plot(df['episode'], df['epsilon'], color='purple')
            axes[1,1].set_title("Epsilon Decay")
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"metrics{suffix}.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting metrics: {e}")

    # --- Heatmaps ---
    if os.path.exists(steps_file):
        try:
            dfs = pd.read_csv(steps_file)
            # Detect map size from data or default to 20
            max_val = max(dfs['px'].max(), dfs['py'].max()) if not dfs.empty else 20
            map_size = 20 if max_val <= 20 else int(max_val) + 1
            
            # Deaths
            if 'died' in dfs.columns and 'px' in dfs.columns and 'py' in dfs.columns:
                df_deaths = dfs[dfs['died'] == 1]
                
                # Zawsze generuj heatmapę śmierci, nawet pustą
                plt.figure(figsize=(6,6))
                if not df_deaths.empty:
                    plt.hist2d(df_deaths['px'], df_deaths['py'], bins=[map_size, map_size], range=[[0,map_size],[0,map_size]], cmap='Reds')
                else:
                    plt.text(map_size/2, map_size/2, "No Deaths", ha='center', va='center', fontsize=12)
                    plt.xlim(0, map_size)
                    plt.ylim(0, map_size)
                
                plt.colorbar(label='Deaths')
                plt.title(f'Deaths (Run {run_id})')
                plt.gca().invert_yaxis()
                plt.savefig(os.path.join(log_dir, f"heatmap_deaths{suffix}.png"))
                plt.close()
            
            # Movement
            if 'px' in dfs.columns and 'py' in dfs.columns:
                plt.figure(figsize=(6,6))
                plt.hist2d(dfs['px'], dfs['py'], bins=[map_size, map_size], range=[[0,map_size],[0,map_size]], cmap='viridis')
                plt.colorbar(label='Visits')
                plt.title(f'Movement (Run {run_id})')
                plt.gca().invert_yaxis()
                plt.savefig(os.path.join(log_dir, f"heatmap_movement{suffix}.png"))
                plt.close()
            
        except Exception as e:
            print(f"Error generating heatmaps: {e}")

    # --- Text Report ---
    report = f"""
# Training Report: {log_dir} (Run {run_id})

## Statistics
- **Total Episodes**: {total_episodes}
- **Total Duration**: {duration_seconds:.1f} seconds
- **Average Reward**: {avg_reward:.2f}
- **Win Rate**: {win_rate:.1f}%
- **Stability**: {stability:.2f}

## Visuals
![Metrics](metrics{suffix}.png)
![Movement](heatmap_movement{suffix}.png)
![Deaths](heatmap_deaths{suffix}.png)
    """
    
    with open(os.path.join(log_dir, f"report{suffix}.md"), "w") as f:
        f.write(report)
    print(f"Report saved to {log_dir}/report{suffix}.md")

if __name__ == "__main__":
    # Test run on existing log if called directly
    if len(sys.argv) > 1:
        analyze_custom(sys.argv[1])
    else:
        print("Usage: python analyze_training.py <log_dir>")

