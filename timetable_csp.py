import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple, Set
import random

class TimetableCSP:
    def __init__(self, num_courses: int = 8, num_rooms: int = 3, 
                 num_timeslots: int = 5, num_days: int = 5):
        """
        Initialize timetable CSP problem
        Variables: Courses (need to be scheduled)
        Domain: (day, timeslot, room) combinations
        Constraints: No conflicts, room capacity, teacher availability
        """
        self.num_courses = num_courses
        self.num_rooms = num_rooms
        self.num_timeslots = num_timeslots
        self.num_days = num_days
        
        # Course properties
        self.courses = [f"Course{i+1}" for i in range(num_courses)]
        self.teachers = {course: f"Teacher{(i % (num_courses//2)) + 1}" 
                        for i, course in enumerate(self.courses)}
        self.course_duration = {course: 1 for course in self.courses}  # All 1 slot
        
        # Room capacities
        self.room_capacity = {f"Room{i+1}": random.randint(30, 100) 
                             for i in range(num_rooms)}
        self.course_students = {course: random.randint(20, 80) 
                               for course in self.courses}
        
        # Initialize domain for each course
        self.domains = self._initialize_domains()
        
        # Statistics
        self.backtracks = 0
        self.constraint_checks = 0
    
    def _initialize_domains(self) -> Dict[str, List[Tuple]]:
        """Initialize domain for each course (all possible timeslots)"""
        domains = {}
        for course in self.courses:
            domain = []
            for day in range(self.num_days):
                for slot in range(self.num_timeslots):
                    for room in range(self.num_rooms):
                        room_name = f"Room{room+1}"
                        # Check room capacity constraint
                        if self.course_students[course] <= self.room_capacity[room_name]:
                            domain.append((day, slot, room_name))
            domains[course] = domain
        return domains
    
    def is_consistent(self, course: str, assignment: Tuple, 
                     current_assignment: Dict) -> bool:
        """Check if assignment is consistent with current assignments"""
        self.constraint_checks += 1
        day, slot, room = assignment
        teacher = self.teachers[course]
        
        for assigned_course, (a_day, a_slot, a_room) in current_assignment.items():
            assigned_teacher = self.teachers[assigned_course]
            
            # Room conflict: same room, same time
            if day == a_day and slot == a_slot and room == a_room:
                return False
            
            # Teacher conflict: same teacher, same time
            if day == a_day and slot == a_slot and teacher == assigned_teacher:
                return False
        
        return True
    
    def select_unassigned_variable_mrv(self, current_assignment: Dict, 
                                       domains: Dict) -> str:
        """Select variable with Minimum Remaining Values (MRV)"""
        unassigned = [c for c in self.courses if c not in current_assignment]
        if not unassigned:
            return None
        
        # Choose variable with smallest domain
        return min(unassigned, key=lambda c: len(domains[c]))
    
    def order_domain_values_lcv(self, course: str, domains: Dict, 
                                current_assignment: Dict) -> List[Tuple]:
        """Order domain values by Least Constraining Value (LCV)"""
        def count_conflicts(value):
            """Count how many values this assignment would eliminate"""
            conflicts = 0
            day, slot, room = value
            teacher = self.teachers[course]
            
            for other_course in self.courses:
                if other_course in current_assignment or other_course == course:
                    continue
                
                other_teacher = self.teachers[other_course]
                for other_value in domains[other_course]:
                    o_day, o_slot, o_room = other_value
                    
                    # Would conflict with this value
                    if ((day == o_day and slot == o_slot and room == o_room) or
                        (day == o_day and slot == o_slot and teacher == other_teacher)):
                        conflicts += 1
            
            return conflicts
        
        return sorted(domains[course], key=count_conflicts)
    
    def forward_check(self, course: str, assignment: Tuple, 
                     domains: Dict, current_assignment: Dict) -> Dict:
        """Forward checking: prune inconsistent values from future variables"""
        new_domains = {c: list(d) for c, d in domains.items()}
        day, slot, room = assignment
        teacher = self.teachers[course]
        
        for other_course in self.courses:
            if other_course in current_assignment or other_course == course:
                continue
            
            other_teacher = self.teachers[other_course]
            values_to_remove = []
            
            for value in new_domains[other_course]:
                o_day, o_slot, o_room = value
                
                # Check conflicts
                if ((day == o_day and slot == o_slot and room == o_room) or
                    (day == o_day and slot == o_slot and teacher == other_teacher)):
                    values_to_remove.append(value)
            
            for value in values_to_remove:
                new_domains[other_course].remove(value)
            
            # Domain wipeout
            if not new_domains[other_course]:
                return None
        
        return new_domains
    
    def backtrack_basic(self, current_assignment: Dict, domains: Dict, 
                       use_heuristics: bool = False) -> Dict:
        """Backtracking with optional heuristics"""
        if len(current_assignment) == len(self.courses):
            return current_assignment
        
        # Select variable
        if use_heuristics:
            course = self.select_unassigned_variable_mrv(current_assignment, domains)
        else:
            course = next(c for c in self.courses if c not in current_assignment)
        
        # Order domain values
        if use_heuristics:
            domain_values = self.order_domain_values_lcv(course, domains, 
                                                         current_assignment)
        else:
            domain_values = domains[course]
        
        for value in domain_values:
            if self.is_consistent(course, value, current_assignment):
                current_assignment[course] = value
                
                result = self.backtrack_basic(current_assignment, domains, use_heuristics)
                if result is not None:
                    return result
                
                del current_assignment[course]
                self.backtracks += 1
        
        return None
    
    def backtrack_forward_checking(self, current_assignment: Dict, 
                                   domains: Dict) -> Dict:
        """Backtracking with forward checking"""
        if len(current_assignment) == len(self.courses):
            return current_assignment
        
        # MRV heuristic
        course = self.select_unassigned_variable_mrv(current_assignment, domains)
        
        # LCV heuristic
        domain_values = self.order_domain_values_lcv(course, domains, 
                                                     current_assignment)
        
        for value in domain_values:
            if self.is_consistent(course, value, current_assignment):
                current_assignment[course] = value
                
                # Forward checking
                new_domains = self.forward_check(course, value, domains, 
                                                current_assignment)
                
                if new_domains is not None:
                    result = self.backtrack_forward_checking(current_assignment, 
                                                            new_domains)
                    if result is not None:
                        return result
                
                del current_assignment[course]
                self.backtracks += 1
        
        return None
    
    def solve(self, method: str = "basic") -> Tuple[Dict, dict]:
        """Solve timetable CSP with specified method"""
        self.backtracks = 0
        self.constraint_checks = 0
        start_time = time.time()
        
        if method == "basic":
            solution = self.backtrack_basic({}, self.domains, use_heuristics=False)
        elif method == "heuristics":
            solution = self.backtrack_basic({}, self.domains, use_heuristics=True)
        elif method == "forward_checking":
            solution = self.backtrack_forward_checking({}, self.domains)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        elapsed_time = time.time() - start_time
        
        stats = {
            'time': elapsed_time,
            'backtracks': self.backtracks,
            'constraint_checks': self.constraint_checks,
            'solved': solution is not None
        }
        
        return solution, stats
    
    def visualize_timetable(self, solution: Dict, method: str, save_name: str = None):
        """Visualize the timetable solution"""
        if solution is None:
            print(f"No solution found for {method}")
            return
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        fig, axes = plt.subplots(1, self.num_rooms, figsize=(15, 8))
        if self.num_rooms == 1:
            axes = [axes]
        
        fig.suptitle(f'Timetable Solution - {method}', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.courses)))
        course_colors = {course: colors[i] for i, course in enumerate(self.courses)}
        
        for room_idx in range(self.num_rooms):
            ax = axes[room_idx]
            room_name = f"Room{room_idx + 1}"
            
            # Create grid
            grid = np.zeros((self.num_timeslots, self.num_days))
            
            for course, (day, slot, room) in solution.items():
                if room == room_name:
                    grid[slot, day] = 1
                    # Add text
                    teacher = self.teachers[course]
                    ax.text(day, slot, f"{course}\n{teacher}", 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor=course_colors[course], 
                                   alpha=0.7))
            
            ax.set_xlim(-0.5, self.num_days - 0.5)
            ax.set_ylim(-0.5, self.num_timeslots - 0.5)
            ax.set_xticks(range(self.num_days))
            ax.set_yticks(range(self.num_timeslots))
            ax.set_xticklabels(days[:self.num_days], rotation=45)
            ax.set_yticklabels([f"Slot {i+1}" for i in range(self.num_timeslots)])
            ax.set_title(f'{room_name}\nCapacity: {self.room_capacity[room_name]}')
            ax.grid(True, linewidth=1.5, color='black')
            ax.invert_yaxis()
        
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close()

def run_experiments(num_trials: int = 10):
    """Run multiple trials and compare methods"""
    methods = {
        'Basic Backtracking': 'basic',
        'Backtracking + Heuristics': 'heuristics',
        'Backtracking + Forward Checking': 'forward_checking'
    }
    
    results = {name: {'times': [], 'backtracks': [], 'constraint_checks': [], 
                     'success_rate': 0} 
              for name in methods.keys()}
    
    print(f"Running {num_trials} trials...\n")
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        
        # Create same problem instance for all methods
        csp = TimetableCSP(num_courses=8, num_rooms=3, num_timeslots=5, num_days=5)
        
        for method_name, method_code in methods.items():
            # Reset statistics
            solution, stats = csp.solve(method_code)
            
            if stats['solved']:
                results[method_name]['times'].append(stats['time'])
                results[method_name]['backtracks'].append(stats['backtracks'])
                results[method_name]['constraint_checks'].append(stats['constraint_checks'])
                results[method_name]['success_rate'] += 1
                
                # Visualize first trial
                if trial == 0:
                    csp.visualize_timetable(solution, method_name, 
                                           f'timetable_{method_code}.png')
            
            print(f"  {method_name}: {'Success' if stats['solved'] else 'Failed'} - "
                  f"Time: {stats['time']:.4f}s, Backtracks: {stats['backtracks']}, "
                  f"Checks: {stats['constraint_checks']}")
    
    # Calculate success rates
    for method_name in methods.keys():
        results[method_name]['success_rate'] = \
            results[method_name]['success_rate'] / num_trials * 100
    
    return results

def plot_comparison(results: Dict):
    """Plot performance comparison"""
    methods = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CSP Methods Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    # 1. Average Time
    ax = axes[0, 0]
    avg_times = [np.mean(results[m]['times']) if results[m]['times'] else 0 
                 for m in methods]
    std_times = [np.std(results[m]['times']) if results[m]['times'] else 0 
                 for m in methods]
    ax.bar(range(len(methods)), avg_times, yerr=std_times, capsize=5, 
           color=colors, tick_label=methods)
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Execution Time Comparison')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 2. Average Backtracks
    ax = axes[0, 1]
    avg_backtracks = [np.mean(results[m]['backtracks']) if results[m]['backtracks'] 
                     else 0 for m in methods]
    std_backtracks = [np.std(results[m]['backtracks']) if results[m]['backtracks'] 
                     else 0 for m in methods]
    ax.bar(range(len(methods)), avg_backtracks, yerr=std_backtracks, capsize=5, 
           color=colors, tick_label=methods)
    ax.set_ylabel('Average Backtracks')
    ax.set_title('Number of Backtracks')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 3. Average Constraint Checks
    ax = axes[1, 0]
    avg_checks = [np.mean(results[m]['constraint_checks']) if results[m]['constraint_checks'] 
                  else 0 for m in methods]
    std_checks = [np.std(results[m]['constraint_checks']) if results[m]['constraint_checks'] 
                  else 0 for m in methods]
    ax.bar(range(len(methods)), avg_checks, yerr=std_checks, capsize=5, 
           color=colors, tick_label=methods)
    ax.set_ylabel('Average Constraint Checks')
    ax.set_title('Constraint Checks Comparison')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 4. Success Rate
    ax = axes[1, 1]
    success_rates = [results[m]['success_rate'] for m in methods]
    ax.bar(range(len(methods)), success_rates, color=colors, tick_label=methods)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Solution Success Rate')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig('csp_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nComparison graph saved as 'csp_methods_comparison.png'")

def print_statistics(results: Dict):
    """Print detailed statistics"""
    print("\n" + "="*70)
    print("PERFORMANCE STATISTICS - CSP METHODS")
    print("="*70)
    
    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Success Rate: {data['success_rate']:.2f}%")
        if data['times']:
            print(f"  Avg Time: {np.mean(data['times']):.4f} ± {np.std(data['times']):.4f} s")
            print(f"  Avg Backtracks: {np.mean(data['backtracks']):.2f} ± {np.std(data['backtracks']):.2f}")
            print(f"  Avg Constraint Checks: {np.mean(data['constraint_checks']):.2f} ± {np.std(data['constraint_checks']):.2f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    # Run experiments
    results = run_experiments(num_trials=10)
    
    # Plot comparison
    plot_comparison(results)
    
    # Print statistics
    print_statistics(results)
    
    # Save results to JSON
    json_results = {method: {k: v if not isinstance(v, list) else 
                            [float(x) for x in v] 
                            for k, v in data.items()} 
                   for method, data in results.items()}
    
    with open('timetable_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to 'timetable_results.json'")
    print("Timetable visualizations saved for first trial")
    print("\nAll done!")