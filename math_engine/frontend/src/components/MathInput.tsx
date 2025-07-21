import React, { useState } from 'react';
import { Send, Calculator, Brain } from 'lucide-react';
import { MathInput as MathInputType } from '../types';

interface MathInputProps {
  input: MathInputType;
  onInputChange: (input: MathInputType) => void;
  onCalculate: () => void;
  onGenerateExplanation?: () => void;
  isLoading: boolean;
}

const MathInput: React.FC<MathInputProps> = ({
  input,
  onInputChange,
  onCalculate,
  onGenerateExplanation,
  isLoading,
}) => {
  const [examples] = useState([
    'x^2 + 2*x + 1 = 0',
    'sin(x) + cos(x)',
    'log(x) + sqrt(y^2 + 1)',
    '(x+1)^2',
    'x^3 - x^2 - 6*x',
  ]);

  const handleExpressionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const expression = e.target.value;
    onInputChange({
      ...input,
      expression,
      isValid: expression.trim().length > 0,
    });
  };

  const handleFormatChange = (format: 'text' | 'latex') => {
    onInputChange({
      ...input,
      format,
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      onCalculate();
    }
  };

  const handleExampleClick = (example: string) => {
    onInputChange({
      ...input,
      expression: example,
      isValid: true,
    });
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Calculator className="w-5 h-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-gray-900">Mathematical Expression</h2>
      </div>

      {/* Format Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Input Format
        </label>
        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="radio"
              name="format"
              value="text"
              checked={input.format === 'text'}
              onChange={() => handleFormatChange('text')}
              className="mr-2 text-blue-600"
            />
            <span className="text-sm text-gray-700">Text</span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              name="format"
              value="latex"
              checked={input.format === 'latex'}
              onChange={() => handleFormatChange('latex')}
              className="mr-2 text-blue-600"
            />
            <span className="text-sm text-gray-700">LaTeX</span>
          </label>
        </div>
      </div>

      {/* Expression Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Expression
        </label>
        <textarea
          value={input.expression}
          onChange={handleExpressionChange}
          onKeyPress={handleKeyPress}
          placeholder={
            input.format === 'latex'
              ? 'Enter LaTeX: \\frac{x^2 + 1}{x - 1}'
              : 'Enter expression: x^2 + 2*x + 1'
          }
          className="w-full h-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none font-mono text-sm"
          disabled={isLoading}
        />
        <p className="text-xs text-gray-500 mt-1">
          Press Ctrl+Enter to calculate
        </p>
      </div>

      {/* Calculate Buttons */}
      <div className="space-y-3">
        <button
          onClick={onCalculate}
          disabled={!input.isValid || isLoading}
          className="w-full flex items-center justify-center space-x-2 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <>
              <div className="spinner"></div>
              <span>Calculating...</span>
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              <span>Calculate</span>
            </>
          )}
        </button>

        {onGenerateExplanation && (
          <button
            onClick={onGenerateExplanation}
            disabled={!input.isValid || isLoading}
            className="w-full flex items-center justify-center space-x-2 bg-purple-600 text-white py-3 px-4 rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <>
                <div className="spinner"></div>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Brain className="w-4 h-4" />
                <span>AI Step-by-Step Solution</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Examples */}
      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Examples:</h3>
        <div className="space-y-2">
          {examples.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              className="w-full text-left px-3 py-2 text-sm text-gray-600 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors font-mono"
              disabled={isLoading}
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MathInput;
