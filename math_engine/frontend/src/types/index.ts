// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error_message?: string;
}

// Expression Parsing Types
export interface ParseRequest {
  expression: string;
  format?: 'text' | 'latex' | 'mathml';
}

export interface ParseResponse {
  success: boolean;
  parsed_expression: string | null;
  ast_tree: ASTNode | null;
  variables: string[];
  functions: string[];
  constants: string[];
  complexity_score: number | null;
  error_message: string | null;
}

export interface ASTNode {
  root: ASTNodeData;
  expression: string;
  latex: string;
  variables: string[];
  functions: string[];
  complexity: number;
}

export interface ASTNodeData {
  type: 'atom' | 'operation';
  value?: string;
  operator?: string;
  args?: ASTNodeData[];
  sympy_type: string;
  is_number?: boolean;
  is_symbol?: boolean;
}

// Equation Solving Types
export interface SolveRequest {
  expression: string;
  variable?: string;
  steps?: boolean;
  domain?: 'real' | 'complex';
  method?: 'algebraic' | 'numerical';
  approximation?: boolean;
  explanation?: boolean;
  precision?: number;
}

export interface SolveResponse {
  success: boolean;
  result: string | string[] | null;
  solutions: Solution[];
  steps: SolutionStep[];
  execution_time_ms: number;
  method_used: string;
  error_message: string | null;
  explanations?: string[];
}

// AI Explanation Types
export interface AIExplanation {
  overall_explanation: string;
  key_concepts: string[];
  step_explanations?: string[];
}

export interface CompleteSolutionResponse {
  success: boolean;
  original_expression: string;
  final_result: string;
  solution_steps: SolutionStep[];
  overall_explanation: string;
  step_explanations: string[];
  key_concepts: string[];
  execution_time_ms: number;
  is_equation: boolean;
  variable_solved?: string;
  error_message?: string;
  warning?: string;
}

export interface Solution {
  value: string;
  type: 'exact' | 'numerical';
  domain: 'real' | 'complex';
}

export interface SolutionStep {
  step: number;
  description: string;
  expression: string;
  rule_applied?: string;
  explanation?: string;
}

// Visualization Types
export interface VisualizationRequest {
  expression: string;
  type?: 'tree' | 'graph' | 'plot';
  options?: VisualizationOptions;
}

export interface VisualizationOptions {
  width?: number;
  height?: number;
  interactive?: boolean;
  theme?: 'light' | 'dark';
}

export interface VisualizationResponse {
  success: boolean;
  visualization_data: VisualizationData | null;
  svg_output: string | null;
  d3_config: D3Config | null;
  error_message: string | null;
}

export interface VisualizationData {
  type: string;
  tree_structure: TreeNode;
  metadata: {
    node_count: number;
    depth: number;
    complexity: number;
  };
}

export interface TreeNode {
  id: string;
  label: string;
  type: string;
  children?: TreeNode[];
  properties: {
    operator?: string;
    value?: string;
    sympy_type: string;
  };
}

export interface D3Config {
  nodes: D3Node[];
  links: D3Link[];
  layout: {
    type: 'tree' | 'force';
    width: number;
    height: number;
  };
}

export interface D3Node {
  id: string;
  label: string;
  type: string;
  level?: number;
  x?: number;
  y?: number;
}

export interface D3Link {
  source: string;
  target: string;
  type?: string;
}

// Health Check Types
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  timestamp: string;
  services: {
    llm_service: 'enabled' | 'disabled';
    math_engine: 'operational' | 'degraded' | 'down';
    cache: 'enabled' | 'disabled';
  };
}

// UI State Types
export interface MathInput {
  expression: string;
  format: 'text' | 'latex';
  isValid: boolean;
}

export interface CalculationResult {
  input: MathInput;
  parseResult?: ParseResponse;
  solveResult?: SolveResponse;
  visualizationResult?: VisualizationResponse;
  aiExplanation?: AIExplanation;
  timestamp: number;
}

export interface AppState {
  currentInput: MathInput;
  results: CalculationResult[];
  isLoading: boolean;
  activeTab: 'solve' | 'visualize' | 'history';
  theme: 'light' | 'dark';
}

// Error Types
export interface MathEngineError {
  type: 'parse' | 'solve' | 'visualize' | 'network';
  message: string;
  details?: string;
  timestamp: number;
}
