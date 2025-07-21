# AI-Powered Math Equation Solver and Visualizer

A math equation solver with step-by-step symbolic simplification.

## Features

- **LaTeX/Math Input Parsing**: Parse mathematical expressions into Abstract Syntax Trees (AST).
- **Symbolic Simplification**: Step-by-step rule-based simplification using SymPy.
- **Visual Expression Trees**: Live rendering of mathematical expression trees with D3.js.
- **LLM-Powered Explanations**: AI-generated explanations for solution steps.
- **REST API**: Professional backend with FastAPI.
- **Modern Frontend**: React/TypeScript with responsive design.

## Architecture

The project is organized into a monorepo with a separate backend and frontend.

- **`backend/`**: A Python-based backend powered by FastAPI, responsible for handling API requests, performing mathematical computations, and interacting with the database.
- **`frontend/`**: A modern frontend built with React and TypeScript, providing a user-friendly interface for solving equations and visualizing results.

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Symbolic Math**: SymPy
- **Data Validation**: Pydantic
- **Database**: SQLAlchemy, Alembic
- **Caching**: Redis
- **HTTP Client**: HTTPX
- **AI/LLM**: OpenAI, Tiktoken
- **Production Server**: Gunicorn

### Frontend
- **Framework**: React 18 with TypeScript
- **Routing**: React Router
- **Data Fetching**: Axios
- **UI/Animation**: Framer Motion, Lucide React
- **Notifications**: React Hot Toast
- **LaTeX Rendering**: KaTeX
- **Visualization**: D3.js
- **Code Editor**: Monaco Editor
- **Styling**: Tailwind CSS
- **Build Tool**: Vite

### DevOps
- **Testing**: Pytest (backend), Jest (frontend)
- **Linting/Formatting**: Black, isort, Flake8, ESLint
- **CI/CD**: GitHub Actions
- **Pre-commit Hooks**: For maintaining code quality

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+

### Development Setup

1. **Clone and set up the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   # Backend
   cd backend && pip install -r requirements.txt
   
   # Frontend
   cd ../frontend && npm install
   ```

3. **Start the development servers**:
   ```bash
   # Terminal 1: Backend
   cd backend && uvicorn app.main:app --reload
   
   # Terminal 2: Frontend
   cd frontend && npm run dev
   ```

4. **Access the application**:
   - **Frontend**: `http://localhost:5173`
   - **Backend API**: `http://localhost:8000`
   - **API Documentation**: `http://localhost:8000/docs`

## API Reference

### Solve Equation
```http
POST /api/v1/solve
Content-Type: application/json

{
  "expression": "x^2 + 2*x + 1",
  "variable": "x",
  "steps": true,
  "explanation": true
}
```

### Parse Expression
```http
POST /api/v1/parse
Content-Type: application/json

{
  "expression": "\\frac{x^2 + 1}{x - 1}",
  "format": "latex"
}
```

## Testing

```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test
```