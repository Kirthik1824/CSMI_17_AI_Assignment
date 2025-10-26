import numpy as np
import matplotlib.pyplot as plt
import heapq
from typing import List, Tuple, Set
import time
import json

class Node:
    def __init__(self, pos: Tuple[int, int], g: float, h: float, parent=None):
        self.pos = pos
        self.g = g  # Cost from start
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f < other.f

class RobotPathfinder:
    def __init__(self, grid_size: Tuple[int, int], obstacle_prob: float = 0.2):
        self.rows, self.cols = grid_size
        self.grid = self.create_grid(obstacle_prob)
        self.start = self.get_random_free_cell()
        self.goal = self.get_random_free_cell()
        while self.start == self.goal:
            self.goal = self.get_random_free_cell()
    
    def create_grid(self, obstacle_prob: float) -> np.ndarray:
        """Create grid with random obstacles (0=free, 1=obstacle)"""
        return np.random.choice([0, 1], size=(self.rows, self.cols), 
                               p=[1-obstacle_prob, obstacle_prob])
    
    def get_random_free_cell(self) -> Tuple[int, int]:
        """Get random free cell in grid"""
        while True:
            pos = (np.random.randint(0, self.rows), 
                   np.random.randint(0, self.cols))
            if self.grid[pos] == 0:
                return pos
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (4-directional movement)"""
        r, c = pos
        neighbors = []
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                self.grid[nr, nc] == 0):
                neighbors.append((nr, nc))
        return neighbors
    
    def manhattan_distance(self, pos: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
    
    def euclidean_distance(self, pos: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((pos[0] - self.goal[0])**2 + (pos[1] - self.goal[1])**2)
    
    def chebyshev_distance(self, pos: Tuple[int, int]) -> float:
        """Chebyshev distance heuristic (diagonal distance)"""
        return max(abs(pos[0] - self.goal[0]), abs(pos[1] - self.goal[1]))
    
    def a_star_search(self, heuristic_name: str) -> dict:
        """A* search with specified heuristic"""
        # Select heuristic function
        heuristics = {
            'Manhattan': self.manhattan_distance,
            'Euclidean': self.euclidean_distance,
            'Chebyshev': self.chebyshev_distance
        }
        heuristic_func = heuristics[heuristic_name]
        
        # Initialize
        start_time = time.time()
        open_set = []
        heapq.heappush(open_set, Node(self.start, 0, heuristic_func(self.start)))
        closed_set: Set[Tuple[int, int]] = set()
        g_scores = {self.start: 0}
        
        nodes_expanded = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            
            if current.pos in closed_set:
                continue
            
            nodes_expanded += 1
            
            if current.pos == self.goal:
                # Reconstruct path
                path = []
                node = current
                while node:
                    path.append(node.pos)
                    node = node.parent
                path.reverse()
                
                return {
                    'path': path,
                    'path_length': len(path) - 1,
                    'nodes_expanded': nodes_expanded,
                    'time': time.time() - start_time,
                    'found': True
                }
            
            closed_set.add(current.pos)
            
            for neighbor in self.get_neighbors(current.pos):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current.g + 1
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h = heuristic_func(neighbor)
                    heapq.heappush(open_set, 
                                 Node(neighbor, tentative_g, h, current))
        
        return {
            'path': None,
            'path_length': float('inf'),
            'nodes_expanded': nodes_expanded,
            'time': time.time() - start_time,
            'found': False
        }
    
    def visualize_path(self, path: List[Tuple[int, int]], 
                      heuristic_name: str, save_name: str = None):
        """Visualize the grid and path"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create visualization grid
        vis_grid = self.grid.copy().astype(float)
        vis_grid[vis_grid == 1] = 0.3  # Obstacles in dark
        
        # Mark path
        if path:
            for pos in path[1:-1]:
                vis_grid[pos] = 0.7
        
        # Mark start and goal
        vis_grid[self.start] = 1.0
        vis_grid[self.goal] = 0.5
        
        im = ax.imshow(vis_grid, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Labels
        ax.text(self.start[1], self.start[0], 'S', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        
        ax.set_title(f'A* Search with {heuristic_name} Heuristic\n' +
                    f'Path Length: {len(path)-1 if path else "No Path"}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.colorbar(im, ax=ax, label='Obstacle=Dark, Path=Yellow, Start=Green, Goal=Orange')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close()

def run_experiments(num_trials: int = 30, grid_size: Tuple[int, int] = (20, 20)):
    """Run multiple trials and collect statistics"""
    heuristics = ['Manhattan', 'Euclidean', 'Chebyshev']
    results = {h: {'path_lengths': [], 'nodes_expanded': [], 
                   'times': [], 'success_rate': 0} 
              for h in heuristics}
    
    print(f"Running {num_trials} trials with grid size {grid_size}...\n")
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        pathfinder = RobotPathfinder(grid_size, obstacle_prob=0.25)
        
        for heuristic in heuristics:
            result = pathfinder.a_star_search(heuristic)
            
            if result['found']:
                results[heuristic]['path_lengths'].append(result['path_length'])
                results[heuristic]['nodes_expanded'].append(result['nodes_expanded'])
                results[heuristic]['times'].append(result['time'])
                results[heuristic]['success_rate'] += 1
            
            # Visualize first 3 successful trials
            if trial < 3 and result['found']:
                pathfinder.visualize_path(result['path'], heuristic, 
                                         f'path_trial{trial+1}_{heuristic}.png')
        
        print(f"  Results: Manhattan={results['Manhattan']['success_rate']}, "
              f"Euclidean={results['Euclidean']['success_rate']}, "
              f"Chebyshev={results['Chebyshev']['success_rate']}")
    
    # Calculate success rates
    for h in heuristics:
        results[h]['success_rate'] = results[h]['success_rate'] / num_trials * 100
    
    return results

def plot_comparison(results: dict):
    """Plot performance comparison graphs"""
    heuristics = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('A* Heuristics Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Average Path Length
    ax = axes[0, 0]
    avg_lengths = [np.mean(results[h]['path_lengths']) if results[h]['path_lengths'] 
                   else 0 for h in heuristics]
    std_lengths = [np.std(results[h]['path_lengths']) if results[h]['path_lengths'] 
                   else 0 for h in heuristics]
    ax.bar(heuristics, avg_lengths, yerr=std_lengths, capsize=5, 
           color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax.set_ylabel('Average Path Length')
    ax.set_title('Path Length Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Average Nodes Expanded
    ax = axes[0, 1]
    avg_nodes = [np.mean(results[h]['nodes_expanded']) if results[h]['nodes_expanded'] 
                 else 0 for h in heuristics]
    std_nodes = [np.std(results[h]['nodes_expanded']) if results[h]['nodes_expanded'] 
                 else 0 for h in heuristics]
    ax.bar(heuristics, avg_nodes, yerr=std_nodes, capsize=5, 
           color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax.set_ylabel('Average Nodes Expanded')
    ax.set_title('Computational Efficiency')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Average Time
    ax = axes[1, 0]
    avg_times = [np.mean(results[h]['times']) * 1000 if results[h]['times'] 
                 else 0 for h in heuristics]
    std_times = [np.std(results[h]['times']) * 1000 if results[h]['times'] 
                 else 0 for h in heuristics]
    ax.bar(heuristics, avg_times, yerr=std_times, capsize=5, 
           color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax.set_ylabel('Average Time (ms)')
    ax.set_title('Execution Time Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Success Rate
    ax = axes[1, 1]
    success_rates = [results[h]['success_rate'] for h in heuristics]
    ax.bar(heuristics, success_rates, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Path Finding Success Rate')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heuristics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nComparison graph saved as 'heuristics_comparison.png'")

def print_statistics(results: dict):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
    
    for heuristic, data in results.items():
        print(f"\n{heuristic} Heuristic:")
        print(f"  Success Rate: {data['success_rate']:.2f}%")
        if data['path_lengths']:
            print(f"  Avg Path Length: {np.mean(data['path_lengths']):.2f} ± {np.std(data['path_lengths']):.2f}")
            print(f"  Avg Nodes Expanded: {np.mean(data['nodes_expanded']):.2f} ± {np.std(data['nodes_expanded']):.2f}")
            print(f"  Avg Time: {np.mean(data['times'])*1000:.4f} ± {np.std(data['times'])*1000:.4f} ms")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run experiments
    results = run_experiments(num_trials=30, grid_size=(20, 20))
    
    # Plot comparisons
    plot_comparison(results)
    
    # Print statistics
    print_statistics(results)
    
    # Save results to JSON
    json_results = {h: {k: v if not isinstance(v, list) else 
                       [float(x) for x in v] 
                       for k, v in data.items()} 
                   for h, data in results.items()}
    
    with open('pathfinding_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to 'pathfinding_results.json'")
    print("Path visualizations saved for first 3 trials")
    print("\nAll done!")