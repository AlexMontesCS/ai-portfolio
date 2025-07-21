import React from 'react';

interface StatusBarProps {
  apiStatus: 'healthy' | 'degraded' | 'unhealthy';
  lastCalculation?: number;
}

const StatusBar: React.FC<StatusBarProps> = ({ apiStatus, lastCalculation }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'unhealthy':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const formatLastCalculation = (timestamp?: number) => {
    if (!timestamp) return 'Never';
    
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    
    if (minutes > 0) {
      return `${minutes}m ${seconds}s ago`;
    }
    return `${seconds}s ago`;
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-3">System Status</h3>
      
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">API Status:</span>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${getStatusColor(apiStatus)}`}></div>
            <span className="text-sm capitalize">{apiStatus}</span>
          </div>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Last Calculation:</span>
          <span className="text-sm text-gray-900">
            {formatLastCalculation(lastCalculation)}
          </span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">API Endpoint:</span>
          <span className="text-sm text-gray-900 font-mono">localhost:8000</span>
        </div>
      </div>
    </div>
  );
};

export default StatusBar;
