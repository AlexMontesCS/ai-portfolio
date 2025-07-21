import React from 'react';
import { CheckCircle, XCircle, Clock, AlertCircle } from 'lucide-react';
import { CalculationResult } from '../types';

interface ResultsPanelProps {
  result?: CalculationResult;
  isLoading: boolean;
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <div className="spinner w-8 h-8 mb-4"></div>
        <p className="text-gray-600">Calculating your mathematical expression...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <AlertCircle className="w-12 h-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No calculations yet</h3>
        <p className="text-gray-600">Enter a mathematical expression to get started.</p>
      </div>
    );
  }

  const { parseResult, solveResult } = result;

  return (
    <div className="space-y-6">
      {/* Parse Results */}
      <div className="border rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-3">
          {parseResult?.success ? (
            <CheckCircle className="w-5 h-5 text-green-500" />
          ) : (
            <XCircle className="w-5 h-5 text-red-500" />
          )}
          <h3 className="font-semibold text-gray-900">Expression Parsing</h3>
        </div>
        
        {parseResult?.success ? (
          <div className="space-y-3">
            <div className="bg-gray-50 rounded-lg p-3 font-mono text-sm">
              {parseResult.parsed_expression}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Variables:</span>
                <div className="font-mono text-blue-600">
                  {parseResult.variables.length > 0 
                    ? parseResult.variables.join(', ') 
                    : 'None'
                  }
                </div>
              </div>
              <div>
                <span className="text-gray-600">Functions:</span>
                <div className="font-mono text-purple-600">
                  {parseResult.functions.length > 0 
                    ? parseResult.functions.join(', ') 
                    : 'None'
                  }
                </div>
              </div>
              <div>
                <span className="text-gray-600">Constants:</span>
                <div className="font-mono text-green-600">
                  {parseResult.constants.length > 0 
                    ? parseResult.constants.join(', ') 
                    : 'None'
                  }
                </div>
              </div>
              <div>
                <span className="text-gray-600">Complexity:</span>
                <div className="font-mono text-orange-600">
                  {parseResult.complexity_score || 'N/A'}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-red-600 bg-red-50 p-3 rounded-lg">
            {parseResult?.error_message || 'Parsing failed'}
          </div>
        )}
      </div>

      {/* Solve Results */}
      <div className="border rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-3">
          {solveResult?.success ? (
            <CheckCircle className="w-5 h-5 text-green-500" />
          ) : (
            <XCircle className="w-5 h-5 text-red-500" />
          )}
          <h3 className="font-semibold text-gray-900">Solution</h3>
          {(() => {
            const executionTime = solveResult?.execution_time_ms || 
              (typeof solveResult?.result === 'object' && solveResult?.result !== null 
                ? (solveResult.result as any).execution_time_ms 
                : null);
            
            return executionTime && (
              <div className="flex items-center space-x-1 text-sm text-gray-500">
                <Clock className="w-4 h-4" />
                <span>{executionTime}ms</span>
              </div>
            );
          })()}
        </div>

        {solveResult?.success ? (
          <div className="space-y-4">
            {/* Result */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Result:
              </label>
              <div className="bg-green-50 border border-green-200 rounded-lg p-3 font-mono text-lg">
                {(() => {
                  const result = solveResult.result;
                  if (typeof result === 'object' && result !== null) {
                    // Handle new API response structure
                    if ('final_result' in result) {
                      const finalResult = (result as any).final_result;
                      return Array.isArray(finalResult) 
                        ? finalResult.join(', ')
                        : String(finalResult || 'No solution');
                    }
                    // Handle object result
                    return JSON.stringify(result, null, 2);
                  }
                  // Handle simple string/array result
                  return Array.isArray(result) 
                    ? result.join(', ')
                    : String(result || 'No solution');
                })()}
              </div>
            </div>

            {/* Method Used */}
            {(() => {
              const methodUsed = solveResult.method_used || 
                (typeof solveResult.result === 'object' && solveResult.result !== null 
                  ? (solveResult.result as any).method_used 
                  : null);
              
              return methodUsed && (
                <div className="text-sm text-gray-600">
                  Method: <span className="font-mono">{methodUsed}</span>
                </div>
              );
            })()}

            {/* Step-by-step Solution */}
            {(() => {
              let steps = solveResult.steps;
              // Check if steps are nested in result object
              if (!steps && typeof solveResult.result === 'object' && solveResult.result !== null) {
                steps = (solveResult.result as any).steps;
              }
              
              return steps && Array.isArray(steps) && steps.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Step-by-step Solution:
                  </label>
                  <div className="space-y-2">
                    {steps.map((step: any, index: number) => (
                      <div
                        key={index}
                        className="border-l-2 border-blue-200 pl-4 py-2 bg-blue-50"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="text-sm font-medium text-blue-900 mb-1">
                              {step.step || index + 1}. {step.description || step.rule || 'Step'}
                            </div>
                            <div className="font-mono text-sm text-gray-700 bg-white px-2 py-1 rounded">
                              {step.expression || step.result || JSON.stringify(step)}
                            </div>
                            {step.explanation && (
                              <div className="text-xs text-blue-700 mt-1">
                                {step.explanation}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* AI Explanations */}
            {result.aiExplanation && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold text-purple-900 mb-3 flex items-center">
                  <span className="mr-2">🤖</span>
                  AI Explanation
                </h4>

                {/* //Overall Explanation
                {result.aiExplanation.overall_explanation && (
                  <div className="mb-4">
                    <h5 className="text-sm font-medium text-purple-800 mb-2">Overall Explanation:</h5>
                    <div 
                      className="bg-white p-3 rounded border-l-2 border-purple-300 text-sm text-purple-700"
                      dangerouslySetInnerHTML={{ __html: result.aiExplanation.overall_explanation }}
                    />
                  </div>
                )} */}

                {/* Key Concepts */}
                {result.aiExplanation.key_concepts && result.aiExplanation.key_concepts.length > 0 && (
                  <div className="mb-4">
                    <h5 className="text-sm font-medium text-purple-800 mb-2">Key Concepts:</h5>
                    <div className="flex flex-wrap gap-2">
                      {result.aiExplanation.key_concepts.map((concept: string, index: number) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full border border-purple-300"
                        >
                          {concept}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Step Explanations */}
                {result.aiExplanation.step_explanations && result.aiExplanation.step_explanations.length > 0 && (
                  <div>
                    <h5 className="text-sm font-medium text-purple-800 mb-2">Step-by-Step Explanations:</h5>
                    <div className="space-y-2">
                      {result.aiExplanation.step_explanations.map((explanation: string, index: number) => (
                        <div key={index} className="bg-white p-3 rounded border-l-2 border-purple-300">
                          <div className="text-xs font-medium text-purple-800 mb-1">
                            Step {index + 1} Explanation:
                          </div>
                          <div 
                            className="text-sm text-purple-700"
                            dangerouslySetInnerHTML={{ __html: explanation }}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="text-red-600 bg-red-50 p-3 rounded-lg">
            {solveResult?.error_message || 'Solving failed'}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsPanel;
