# CSMI17 - Artificial Intelligence Assignment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Implementation of Robot Pathfinding using A* Search and Timetable Generation using Constraint Satisfaction Problem (CSP) solving techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [Problems Implemented](#problems-implemented)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [Author](#author)

## 🎯 Overview

This repository contains implementations of two fundamental AI problems:

1. **Robot Path-finding Problem**: Finding optimal paths in grid environments using A* search with multiple heuristics
2. **Timetable Generation Problem**: Solving course scheduling as a Constraint Satisfaction Problem using backtracking algorithms

Both implementations include comprehensive performance analysis, visualization, and comparison of different algorithmic approaches.

## 🤖 Problems Implemented

### Problem 1: Robot Path-finding using A* Search

Implements A* search algorithm with three different heuristics to find optimal paths for a robot navigating through a grid with obstacles.

**Heuristics Compared:**
- Manhattan Distance (L1 norm)
- Euclidean Distance (L2 norm)
- Chebyshev Distance (L∞ norm)

**Key Features:**
- Random grid generation with configurable obstacle probability
- Automated testing across 30 trials
- Performance metrics: path length, nodes expanded, execution time, success rate
- Visual path representations with color-coded grids

### Problem 2: Timetable Generation as CSP

Formulates course scheduling as a CSP and solves it using three backtracking variants.

**Methods Compared:**
- Basic Backtracking
- Backtracking with Variable/Value Ordering Heuristics (MRV + LCV)
- Backtracking with Forward Checking

**Constraints Handled:**
- Room capacity constraints
- No room conflicts (same room, same time)
- No teacher conflicts (same teacher, same time)

**Key Features:**
- Realistic timetable generation with multiple courses, rooms, and timeslots
- Automated testing across 10 trials
- Performance metrics: execution time, backtracks, constraint checks, success rate
- Visual timetable layouts for each room

## ✨ Features

- 🔄 **Automated Execution**: Single file execution runs complete experiments
- 📊 **Comprehensive Metrics**: Multiple performance indicators tracked
- 📈 **Visual Analysis**: Automatic generation of comparison graphs and visualizations
- 💾 **Data Export**: Results saved in JSON format for further analysis
- 📝 **Detailed Statistics**: Mean and standard deviation calculations
- 🎨 **Color-coded Visualizations**: Clear, publication-ready graphs

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-assignment.git
cd ai-assignment
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Problem 1: Robot Pathfinding
```bash
python robot_pathfinding.py
```

**Output Files Generated:**
- `heuristics_comparison.png` - Performance comparison graph
- `path_trial1_Manhattan.png` - Sample path visualization (Manhattan)
- `path_trial1_Euclidean.png` - Sample path visualization (Euclidean)
- `path_trial1_Chebyshev.png` - Sample path visualization (Chebyshev)
- `pathfinding_results.json` - Detailed numerical results

### Problem 2: Timetable Generation
```bash
python timetable_csp.py
```

**Output Files Generated:**
- `csp_methods_comparison.png` - Performance comparison graph
- `timetable_basic.png` - Timetable using basic backtracking
- `timetable_heuristics.png` - Timetable using heuristics
- `timetable_forward_checking.png` - Timetable using forward checking
- `timetable_results.json` - Detailed numerical results

### Configuration

Both scripts can be customized by modifying parameters in the `if __name__ == "__main__":` section:

**Robot Pathfinding:**
```python
results = run_experiments(num_trials=30, grid_size=(20, 20))
```

**Timetable CSP:**
```python
csp = TimetableCSP(num_courses=8, num_rooms=3, num_timeslots=5, num_days=5)
```

## 📁 Project Structure

```
ai-assignment/
│
├── robot_pathfinding.py          # Problem 1: A* search implementation
├── timetable_csp.py               # Problem 2: CSP implementation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── report.pdf                     # Assignment report (LaTeX compiled)
│
├── results/                       # Generated output directory
│   ├── pathfinding_results.json
│   ├── timetable_results.json
│   ├── *.png                      # All visualization images
│
└── docs/                          # Documentation
    └── assignment_report.tex      # LaTeX source for report
```

## 📊 Results

### Robot Pathfinding Performance

| Heuristic | Avg Path Length | Avg Nodes Expanded | Avg Time (ms) | Success Rate |
|-----------|----------------|-------------------|---------------|--------------|
| Manhattan | Optimal        | Lowest            | Fastest       | ~95%         |
| Euclidean | Optimal        | Medium            | Medium        | ~95%         |
| Chebyshev | Optimal        | Highest           | Slowest       | ~95%         |

**Key Finding**: Manhattan distance is most efficient for 4-directional grid movement.

### Timetable CSP Performance

| Method | Avg Backtracks | Avg Constraint Checks | Avg Time (s) | Success Rate |
|--------|---------------|----------------------|--------------|--------------|
| Basic Backtracking | Highest | Medium | Slowest | 100% |
| With Heuristics | Medium | Medium | Medium | 100% |
| Forward Checking | Lowest | Lowest | Fastest | 100% |

**Key Finding**: Forward checking provides best performance through early failure detection.

## 📖 Documentation

Detailed documentation is available in:
- **Assignment Report** (`report.pdf`): Complete analysis with algorithms, experimental setup, and results
- **Code Comments**: Inline documentation in both Python files
- **Docstrings**: Function-level documentation for all major methods

## 🔧 Dependencies

```
numpy>=1.21.0
matplotlib>=3.4.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🎓 Academic Context

**Course**: CSMI17 – Artificial Intelligence  
**Institution**: NIT Tiruchirappalli
**Semester**: 7

## 🤝 Contributing

This is an academic assignment project. However, suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**  
- GitHub: [@Kirthik1824](https://github.com/Kirthik1824)
- Email: kirthik.nitt@gmail.com

## 🙏 Acknowledgments

- Course instructor and teaching assistants
- Russell & Norvig's "Artificial Intelligence: A Modern Approach"
- Python scientific computing community (NumPy, Matplotlib)

## 📞 Contact

For questions or discussions about this project:
- Open an issue on GitHub
- Email: kirthik.nitt@gmail.com

---

⭐ Star this repository if you found it helpful!
