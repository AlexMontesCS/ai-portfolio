# Math Engine Frontend

Modern React/TypeScript frontend for the AI-Powered Math Engine API.

## 🚀 Quick Start

### Windows
```bash
.\setup.bat
npm run dev
```

### Linux/macOS
```bash
npm install
npm run dev
```

## 📱 Features

- **Modern UI**: Clean, responsive design with Tailwind CSS
- **Real-time Calculations**: Live mathematical expression solving
- **Step-by-step Solutions**: Detailed solving process visualization
- **Expression Visualization**: Interactive mathematical tree structures
- **LaTeX Support**: Beautiful mathematical notation rendering
- **Calculation History**: Track and revisit previous calculations
- **API Integration**: Seamless connection to Math Engine backend

## 🛠️ Technology Stack

- **React 18** - Modern React with hooks
- **TypeScript** - Type safety and better developer experience
- **Vite** - Fast development and building
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations
- **KaTeX** - Mathematical notation rendering
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API communication

## 📋 Requirements

- Node.js 16+ 
- npm or yarn
- Math Engine API running on http://localhost:8000

## 🔧 Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run type-check

# Linting
npm run lint
```

## 🌐 API Integration

The frontend automatically connects to the Math Engine API running on `http://localhost:8000`. Make sure the backend is running before starting the frontend.

### API Endpoints Used
- `GET /health` - Health check and status
- `POST /api/v1/parse` - Expression parsing
- `POST /api/v1/solve` - Equation solving
- `POST /api/v1/visualize/tree` - Visualization generation

## 📊 Features Overview

### Expression Input
- Text and LaTeX input formats
- Real-time validation
- Example expressions
- Keyboard shortcuts (Ctrl+Enter to calculate)

### Results Display
- Parsing results with AST information
- Step-by-step solution breakdown
- Execution time tracking
- Error handling and display

### Visualization
- Mathematical expression trees
- D3.js integration ready
- Interactive visualizations
- Metadata display (complexity, depth, etc.)

### History Management
- Last 10 calculations stored
- Quick access to previous results
- Re-run previous calculations
- Clear history functionality

### Status Monitoring
- Real-time API health status
- Connection monitoring
- Last calculation tracking
- System information display

## 🎨 UI Components

### Core Components
- `Header` - Navigation and branding
- `MathInput` - Expression input with format selection
- `ResultsPanel` - Solution display with steps
- `VisualizationPanel` - Expression tree visualization
- `HistoryPanel` - Calculation history management
- `StatusBar` - System status and monitoring

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimized
- Touch-friendly interfaces
- Accessible components

## 🔗 Integration with Backend

The frontend is designed to work seamlessly with the Math Engine API:

1. **Health Monitoring**: Automatic API health checks every 30 seconds
2. **Error Handling**: Graceful handling of API errors with user feedback
3. **Loading States**: Visual feedback during calculations
4. **Real-time Updates**: Immediate display of results and status changes

## 🚦 Available Scripts

- `npm run dev` - Start development server (http://localhost:3000)
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript compiler check

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   ├── services/       # API service layer
│   ├── types/          # TypeScript type definitions
│   ├── App.tsx         # Main application component
│   ├── main.tsx        # Application entry point
│   └── index.css       # Global styles
├── public/             # Static assets
├── index.html          # HTML template
├── package.json        # Dependencies and scripts
├── vite.config.ts      # Vite configuration
├── tailwind.config.js  # Tailwind CSS configuration
└── tsconfig.json       # TypeScript configuration
```

## 🎯 Usage Examples

### Basic Calculation
1. Enter expression: `x^2 + 2*x + 1 = 0`
2. Click "Calculate" or press Ctrl+Enter
3. View step-by-step solution and result

### LaTeX Input
1. Select "LaTeX" format
2. Enter: `\frac{x^2 + 1}{x - 1}`
3. Get parsed expression and visualization

### Visualization
1. Calculate any expression
2. Switch to "Visualization" tab
3. View expression tree structure and metadata

## 🔧 Configuration

### API Base URL
The API base URL can be configured in `src/services/api.ts`:

```typescript
const mathEngineAPI = new MathEngineAPI('http://localhost:8000');
```

### Proxy Configuration
Vite proxy is configured to forward `/api` requests to the backend:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

## 🌟 Production Deployment

For production deployment:

1. Build the application:
   ```bash
   npm run build
   ```

2. Serve the `dist` folder using any static file server

3. Ensure the backend API is accessible from your production domain

4. Update API base URL if needed

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for type safety
3. Add proper error handling
4. Include loading states for async operations
5. Ensure responsive design
6. Add appropriate accessibility features

---

**Frontend Status**: 🟢 **Ready for Development**  
**API Integration**: 🟢 **Fully Compatible**  
**Responsive Design**: 🟢 **Mobile & Desktop Ready**
