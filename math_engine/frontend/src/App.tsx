import { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import {
  Header,
  MathInput,
  ResultsPanel,
  VisualizationPanel,
  HistoryPanel,
  StatusBar
} from './components';
import { MathInput as MathInputType, CalculationResult, AppState } from './types';
import { mathEngineAPI } from './services/api';
import toast from 'react-hot-toast';

function App() {
  const [state, setState] = useState<AppState>({
    currentInput: {
      expression: '',
      format: 'text',
      isValid: false,
    },
    results: [],
    isLoading: false,
    activeTab: 'solve',
    theme: 'light',
  });

  const [apiStatus, setApiStatus] = useState<'healthy' | 'degraded' | 'unhealthy'>('healthy');

  // Check API health on mount
  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    try {
      const health = await mathEngineAPI.checkHealth();
      setApiStatus(health.status);
    } catch (error) {
      setApiStatus('unhealthy');
    }
  };

  const handleInputChange = (input: MathInputType) => {
    setState(prev => ({
      ...prev,
      currentInput: input,
    }));
  };

  const handleCalculate = async () => {
    if (!state.currentInput.expression.trim()) {
      toast.error('Please enter a mathematical expression');
      return;
    }

    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Parse the expression first
      const parseResult = await mathEngineAPI.parseExpression({
        expression: state.currentInput.expression,
        format: state.currentInput.format,
      });

      if (!parseResult.success) {
        toast.error(`Parse error: ${parseResult.error_message}`);
        setState(prev => ({ ...prev, isLoading: false }));
        return;
      }

      // Solve the expression
      const solveResult = await mathEngineAPI.solveExpression({
        expression: state.currentInput.expression,
        steps: true,
        approximation: true,
      });

      // Generate visualization if solving succeeded
      let visualizationResult;
      if (parseResult.success) {
        try {
          visualizationResult = await mathEngineAPI.generateVisualization({
            expression: state.currentInput.expression,
            type: 'tree',
          });
        } catch (error) {
          console.warn('Visualization failed:', error);
        }
      }

      // Create result object
      const result: CalculationResult = {
        input: { ...state.currentInput },
        parseResult,
        solveResult,
        visualizationResult,
        timestamp: Date.now(),
      };

      // Add to results and update state
      setState(prev => ({
        ...prev,
        results: [result, ...prev.results.slice(0, 9)], // Keep last 10 results
        isLoading: false,
      }));

      if (solveResult.success) {
        toast.success('Calculation completed successfully!');
      } else {
        toast.error(`Solve error: ${solveResult.error_message}`);
      }
    } catch (error) {
      console.error('Calculation error:', error);
      toast.error('Failed to calculate. Please check your connection.');
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const handleGenerateExplanation = async () => {
    if (!state.currentInput.expression.trim()) {
      toast.error('Please enter a mathematical expression first');
      return;
    }

    setState(prev => ({ ...prev, isLoading: true }));

    try {
      const completeSolution = await mathEngineAPI.generateCompleteSolution({
        expression: state.currentInput.expression,
        steps: true,
        approximation: true,
      });

      if (completeSolution.success) {
        // Create enhanced result object with AI explanations
        const result: CalculationResult = {
          input: { ...state.currentInput },
          parseResult: {
            success: true,
            parsed_expression: completeSolution.original_expression,
            ast_tree: null,
            variables: [],
            functions: [],
            constants: [],
            complexity_score: null,
            error_message: null,
          },
          solveResult: {
            success: true,
            result: completeSolution.final_result,
            solutions: [],
            steps: completeSolution.solution_steps,
            execution_time_ms: completeSolution.execution_time_ms,
            method_used: 'AI-Enhanced Solution',
            error_message: null,
            explanations: completeSolution.step_explanations,
          },
          aiExplanation: {
            overall_explanation: completeSolution.overall_explanation,
            key_concepts: completeSolution.key_concepts,
            step_explanations: completeSolution.step_explanations,
          },
          timestamp: Date.now(),
        };

        // Add to results
        setState(prev => ({
          ...prev,
          results: [result, ...prev.results.slice(0, 9)],
          isLoading: false,
          activeTab: 'solve', // Switch to solve tab to show results
        }));

        toast.success('AI explanation generated successfully!');
      } else {
        toast.error(`Failed to generate explanation: ${completeSolution.error_message}`);
        setState(prev => ({ ...prev, isLoading: false }));
      }
    } catch (error) {
      console.error('AI explanation error:', error);
      toast.error('Failed to generate explanation. Make sure your API key is configured.');
      setState(prev => ({ ...prev, isLoading: false }));
    }
  };

  const handleClearHistory = () => {
    setState(prev => ({
      ...prev,
      results: [],
    }));
    toast.success('History cleared');
  };

  const handleTabChange = (tab: 'solve' | 'visualize' | 'history') => {
    setState(prev => ({
      ...prev,
      activeTab: tab,
    }));
  };

  const currentResult = state.results[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
      
      <Header onThemeToggle={() => {}} />
      
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Panel */}
          <div className="lg:col-span-1 space-y-6">
            <MathInput
              input={state.currentInput}
              onInputChange={handleInputChange}
              onCalculate={handleCalculate}
              onGenerateExplanation={handleGenerateExplanation}
              isLoading={state.isLoading}
            />
            
            <StatusBar
              apiStatus={apiStatus}
              lastCalculation={currentResult?.timestamp}
            />
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg border border-gray-200">
              {/* Tab Navigation */}
              <div className="border-b border-gray-200 px-6">
                <nav className="flex space-x-8">
                  {[
                    { id: 'solve', label: 'Solution', icon: '🧮' },
                    { id: 'visualize', label: 'Visualization', icon: '📊' },
                    { id: 'history', label: 'History', icon: '📚' },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => handleTabChange(tab.id as any)}
                      className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                        state.activeTab === tab.id
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      <span className="mr-2">{tab.icon}</span>
                      {tab.label}
                    </button>
                  ))}
                </nav>
              </div>

              {/* Tab Content */}
              <div className="p-6 min-h-[500px]">
                {state.activeTab === 'solve' && (
                  <ResultsPanel
                    result={currentResult}
                    isLoading={state.isLoading}
                  />
                )}
                
                {state.activeTab === 'visualize' && (
                  <VisualizationPanel
                    result={currentResult}
                    isLoading={state.isLoading}
                  />
                )}
                
                {state.activeTab === 'history' && (
                  <HistoryPanel
                    results={state.results}
                    onClear={handleClearHistory}
                    onSelectResult={(result: CalculationResult) => {
                      setState(prev => ({
                        ...prev,
                        currentInput: result.input,
                        activeTab: 'solve',
                      }));
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
