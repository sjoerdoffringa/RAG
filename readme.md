# Project Setup Guide

## Installation

1. Install Poetry (if you don't have it already):
   ```curl -sSL https://install.python-poetry.org | python3 -```

2. Install project dependencies:
   ```poetry install```

## Initialization

1. Copy the ```.env template``` file as ```.env``` and insert at least one API key.

## Running the UI

Start the Streamlit application with:
```poetry run streamlit run ui.py```

The application should automatically open in your default browser. If not, check the terminal for the local URL (typically http://localhost:8501).
