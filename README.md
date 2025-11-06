# Portfolio Optimization Project

## Project Structure
```
.
├── input_data/                   # Input data files
│   └── Sharpe-ratio-picked-instruments-data.csv
├── src/                          # Source code
│   ├── data_processing/          # Data cleaning and preparation
│   │   ├── __init__.py
│   │   └── data_cleaning.py
│   ├── calculations/             # Financial calculations
│   │   ├── __init__.py
│   │   └── financial_metrics.py
│   ├── optimization/             # Portfolio optimization
│   │   ├── __init__.py
│   │   └── portfolio_optimization.py
│   ├── visualization/            # Data visualization
│   │   ├── __init__.py
│   │   └── portfolio_plots.py
│   ├── __init__.py
│   └── main.py                   # Main entry point
├── XLSX files/                   # Excel output files
├── Visualization Graphs/        # Plot output files
├── config.json                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python src/main.py
```

## Configuration
Edit `config.json` to modify:
- Input data path
- Output directories for Excel files and plots

## Outputs
The script generates:
- Excel files in `XLSX files/` directory
- Portfolio allocation plots in `Visualization Graphs/` directory
