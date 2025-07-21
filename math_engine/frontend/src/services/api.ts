import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  ParseRequest,
  ParseResponse,
  SolveRequest,
  SolveResponse,
  VisualizationRequest,
  VisualizationResponse,
  HealthResponse,
} from '../types';

class MathEngineAPI {
  private api: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.api = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for debugging
    this.api.interceptors.request.use(
      (config) => {
        console.log('API Request:', config.method?.toUpperCase(), config.url, config.data);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Add response interceptor for debugging
    this.api.interceptors.response.use(
      (response) => {
        console.log('API Response:', response.status, response.data);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.status, error.response?.data);
        return Promise.reject(error);
      }
    );
  }

  // Health Check
  async checkHealth(): Promise<HealthResponse> {
    const response: AxiosResponse<HealthResponse> = await this.api.get('/health');
    return response.data;
  }

  // Expression Parsing
  async parseExpression(request: ParseRequest): Promise<ParseResponse> {
    const response: AxiosResponse<ParseResponse> = await this.api.post('/api/v1/parse', request);
    return response.data;
  }

  // Equation Solving
  async solveExpression(request: SolveRequest): Promise<SolveResponse> {
    const response: AxiosResponse<SolveResponse> = await this.api.post('/api/v1/solve', request);
    return response.data;
  }

  // Complete Solution with AI Explanations
  async generateCompleteSolution(request: SolveRequest): Promise<any> {
    const response: AxiosResponse<any> = await this.api.post('/api/v1/solve/complete', request);
    return response.data;
  }

  // Detailed Explanation
  async generateExplanation(request: SolveRequest): Promise<any> {
    const response: AxiosResponse<any> = await this.api.post('/api/v1/solve/explain', request);
    return response.data;
  }

  // Visualization
  async generateVisualization(request: VisualizationRequest): Promise<VisualizationResponse> {
    const response: AxiosResponse<VisualizationResponse> = await this.api.post(
      '/api/v1/visualize/tree',
      request
    );
    return response.data;
  }

  // Batch operations
  async solveBatch(expressions: string[]): Promise<SolveResponse[]> {
    const requests = expressions.map(expression => ({ expression, steps: true }));
    const promises = requests.map(request => this.solveExpression(request));
    return Promise.all(promises);
  }

  // Utility methods
  async validateExpression(expression: string, format: 'text' | 'latex' = 'text'): Promise<boolean> {
    try {
      const result = await this.parseExpression({ expression, format });
      return result.success;
    } catch (error) {
      return false;
    }
  }

  async getComplexity(expression: string): Promise<number | null> {
    try {
      const result = await this.parseExpression({ expression });
      return result.complexity_score;
    } catch (error) {
      return null;
    }
  }
}

// Create and export API instance
export const mathEngineAPI = new MathEngineAPI();
export default mathEngineAPI;
