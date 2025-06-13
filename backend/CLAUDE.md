# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GridAnalytics is a full-stack application for analyzing electrical grid data from CEB (Central Eléctrica Brasileira). The system provides:

- **Backend**: FastAPI-based REST API for grid data analysis, machine learning predictions, and service quality assessment
- **Frontend**: Next.js application with interactive maps and data visualization
- **Data Processing**: Python scripts for grid topology analysis, ML modeling, and quality prediction

## Architecture

### Backend Structure (FastAPI)
- **main.py**: Core FastAPI application with REST endpoints for grid stats, topology, nodes, pain points, and substations
- **Data Models**: Pydantic models for GridNode, GridStats, and PainPoint
- **Analysis Modules**:
  - `analyze_grid_data.py`: Grid topology reconstruction and hierarchical analysis
  - `service_quality_matrix.py`: Service quality assessment and node analysis
  - `correlate_quality_analysis.py`: Quality correlation analysis
  - `enhanced_ml_predictor.py`: ML models for service quality prediction
  - `data_simulator.py`: Grid data simulation for improved ML training

### Data Flow
1. Raw grid data is loaded from CSV files in `../public/`
2. Topology is reconstructed into hierarchical structure (Subestaciones → Alimentadores → Transformadores → Circuitos)
3. Quality metrics are calculated based on user/power ratios, geographic distribution, and load analysis
4. ML models predict service quality and identify pain points
5. Results are served via REST API endpoints

### Frontend Integration
- Frontend runs on port 3000 (Next.js with TypeScript)
- Backend API runs on port 8000 (FastAPI with uvicorn)
- CORS configured for localhost:3000 communication

## Development Commands

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python test_grid_analysis.py

# Run analysis scripts
python analyze_grid_data.py
python enhanced_ml_predictor.py
python correlate_quality_analysis.py
```

### Frontend Development
```bash
cd ../frontend
npm install
npm run dev          # Development server
npm run build        # Production build
npm run start        # Production server
npm run lint         # ESLint check
```

## Key Data Structures

### Grid Hierarchy
- **Subestaciones**: Top-level electrical substations
- **Alimentadores**: Distribution feeders connected to substations
- **Transformadores**: Distribution transformers
- **Circuitos**: Low-voltage circuits serving end users

### Pain Point Analysis
System identifies problematic areas based on:
- User-to-power ratio (>1.0 users/kVA indicates overload)
- Low installed capacity with high user count
- Geographic clustering of issues

### ML Feature Engineering
- Real grid data features: coordinates, power, users, substation hierarchy
- Quality scoring based on interruption patterns and load analysis
- Correlation analysis between infrastructure and service quality

## Important File Locations

- **Grid Data**: `../public/Mediciones Originales CEB .csv`
- **Topology Output**: `grid_topology.json`
- **ML Models**: `enhanced_ml_model.json`, `ml_simulation_model.json`
- **Analysis Reports**: `quality_correlation_report.json`, `service_quality_report.json`

## Testing

Run the main test suite:
```bash
python test_grid_analysis.py
```

This runs async tests for grid statistics, pain point analysis, and validates the complete data processing pipeline.