import React from 'react';
import { CalculationResult } from '../types';

interface HistoryPanelProps {
  results: CalculationResult[];
  onClear: () => void;
  onSelectResult: (result: CalculationResult) => void;
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({ 
  results, 
  onClear, 
  onSelectResult 
}) => {
  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <div className="w-12 h-12 bg-gray-200 rounded-lg mb-4 flex items-center justify-center">
          <span className="text-2xl">📚</span>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No calculation history</h3>
        <p className="text-gray-600">Your calculation history will appear here.</p>
      </div>
    );
  }

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Calculation History</h3>
        <button
          onClick={onClear}
          className="text-sm text-red-600 hover:text-red-700 hover:underline"
        >
          Clear History
        </button>
      </div>

      <div className="space-y-3">
        {results.map((result) => (
          <div
            key={result.timestamp}
            className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => onSelectResult(result)}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <div className="font-mono text-sm text-gray-900 mb-1">
                  {result.input.expression}
                </div>
                <div className="text-xs text-gray-500">
                  {formatTimestamp(result.timestamp)} • {result.input.format}
                </div>
              </div>
              <div className="flex space-x-2 ml-4">
                {result.parseResult?.success && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                    Parsed
                  </span>
                )}
                {result.solveResult?.success && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
                    Solved
                  </span>
                )}
                {result.visualizationResult?.success && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-purple-100 text-purple-800">
                    Visualized
                  </span>
                )}
              </div>
            </div>

            {result.solveResult?.success && (
              <div className="text-sm text-gray-700 bg-gray-50 p-2 rounded">
                <strong>Result:</strong>{' '}
                <span className="font-mono">
                  {Array.isArray(result.solveResult.result)
                    ? result.solveResult.result.join(', ')
                    : result.solveResult.result || 'No solution'
                  }
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistoryPanel;
