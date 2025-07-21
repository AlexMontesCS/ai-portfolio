import React from 'react';
import { Calculator, Github, ExternalLink } from 'lucide-react';

interface HeaderProps {
  onThemeToggle: () => void;
}

const Header: React.FC<HeaderProps> = ({ onThemeToggle }) => {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 py-4 max-w-7xl">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-blue-600 rounded-lg">
              <Calculator className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Math Engine</h1>
              <p className="text-sm text-gray-500">AI-Powered Mathematical Solver</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-6">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-1 text-gray-600 hover:text-blue-600 transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              <span>API Docs</span>
            </a>
            <a
              href="http://localhost:8000/redoc"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-1 text-gray-600 hover:text-blue-600 transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              <span>ReDoc</span>
            </a>
            <a
              href="#"
              className="flex items-center space-x-1 text-gray-600 hover:text-blue-600 transition-colors"
              onClick={onThemeToggle}
            >
              <Github className="w-4 h-4" />
              <span>GitHub</span>
            </a>
          </nav>

          {/* Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600">Online</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
