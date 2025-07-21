import React from 'react';
import { CalculationResult } from '../types';

interface VisualizationPanelProps {
  result?: CalculationResult;
  isLoading: boolean;
}

const VisualizationPanel: React.FC<VisualizationPanelProps> = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <div className="spinner w-8 h-8 mb-4"></div>
        <p className="text-gray-600">Generating visualization...</p>
      </div>
    );
  }

  if (!result?.visualizationResult) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <div className="w-12 h-12 bg-gray-200 rounded-lg mb-4 flex items-center justify-center">
          <span className="text-2xl">📊</span>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No visualization available</h3>
        <p className="text-gray-600">Calculate an expression to see its visual representation.</p>
      </div>
    );
  }

  const { visualizationResult } = result;

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Expression Tree Visualization</h3>
      
      {visualizationResult.success ? (
        <div className="space-y-4">
          <div className="bg-gray-50 border rounded-lg p-4">
            <h4 className="font-medium text-gray-900 mb-2">Tree Structure</h4>
            <div className="font-mono text-sm text-gray-700">
              Type: {visualizationResult.visualization_data?.type || 'Unknown'}
            </div>
            {visualizationResult.visualization_data?.metadata && (
              <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
                <div>
                  <span className="text-gray-600">Nodes:</span>
                  <div className="font-mono">{visualizationResult.visualization_data.metadata.node_count}</div>
                </div>
                <div>
                  <span className="text-gray-600">Depth:</span>
                  <div className="font-mono">{visualizationResult.visualization_data.metadata.depth}</div>
                </div>
                <div>
                  <span className="text-gray-600">Complexity:</span>
                  <div className="font-mono">{visualizationResult.visualization_data.metadata.complexity}</div>
                </div>
              </div>
            )}
          </div>
          
          <div className="border rounded-lg p-4 bg-white">
            <h4 className="font-medium text-gray-900 mb-2">Visualization Data</h4>
            <div className="bg-gray-100 p-3 rounded font-mono text-xs overflow-auto max-h-64">
              <pre>{JSON.stringify(visualizationResult.d3_config, null, 2)}</pre>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-red-600 bg-red-50 p-4 rounded-lg">
          {visualizationResult.error_message || 'Visualization generation failed'}
        </div>
      )}
    </div>
  );
};

export default VisualizationPanel;
