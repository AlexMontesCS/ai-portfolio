"""
Visualization service for generating mathematical expression visualizations.
"""
import json
from typing import Dict, Any, Optional, List
import sympy as sp

from app.core.exceptions import VisualizationError
from app.services.parser import MathParser


class VisualizationService:
    """Service for generating mathematical visualizations."""
    
    def __init__(self):
        """Initialize the visualization service."""
        self.parser = MathParser()
    
    def generate_expression_tree(
        self,
        expression: str,
        width: int = 800,
        height: int = 600,
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Generate visualization data for expression tree.
        
        Args:
            expression: Mathematical expression to visualize
            width: Visualization width
            height: Visualization height
            interactive: Whether to generate interactive visualization
            
        Returns:
            Dictionary containing visualization data
            
        Raises:
            VisualizationError: If visualization generation fails
        """
        try:
            # Parse expression
            expr, ast_tree = self.parser.parse_expression(expression)
            
            # Generate tree structure for D3.js
            tree_data = self._generate_d3_tree(expr)
            
            # Calculate layout parameters
            layout_params = self._calculate_layout(tree_data, width, height)
            
            # Generate D3.js configuration
            d3_config = self._generate_d3_config(
                tree_data, layout_params, interactive
            )
            
            # Generate SVG for static visualization
            svg_content = self._generate_svg(tree_data, layout_params) if not interactive else None
            
            return {
                "tree_data": tree_data,
                "d3_config": d3_config,
                "svg_content": svg_content,
                "layout_params": layout_params,
                "metadata": {
                    "node_count": self._count_nodes(tree_data),
                    "max_depth": self._calculate_max_depth(tree_data),
                    "expression": expression,
                    "latex": sp.latex(expr),
                }
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate visualization: {str(e)}")
    
    def generate_graph_visualization(
        self,
        expression: str,
        x_range: tuple = (-10, 10),
        y_range: tuple = (-10, 10),
        points: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate graph visualization data for plotting functions.
        
        Args:
            expression: Mathematical expression to plot
            x_range: Range of x values (min, max)
            y_range: Range of y values (min, max)
            points: Number of points to generate
            
        Returns:
            Dictionary containing plot data
        """
        try:
            expr, _ = self.parser.parse_expression(expression)
            
            # Check if expression can be plotted
            variables = expr.free_symbols
            if len(variables) != 1:
                raise VisualizationError("Expression must have exactly one variable for plotting")
            
            var = list(variables)[0]
            
            # Generate plot points
            x_vals = []
            y_vals = []
            
            x_min, x_max = x_range
            step = (x_max - x_min) / points
            
            for i in range(points + 1):
                x_val = x_min + i * step
                try:
                    y_val = complex(expr.subs(var, x_val).evalf())
                    if abs(y_val.imag) < 1e-10:  # Essentially real
                        y_val = y_val.real
                        if y_range[0] <= y_val <= y_range[1]:
                            x_vals.append(x_val)
                            y_vals.append(y_val)
                except:
                    continue  # Skip points where function is undefined
            
            return {
                "plot_data": {
                    "x": x_vals,
                    "y": y_vals,
                },
                "expression": expression,
                "variable": str(var),
                "latex": sp.latex(expr),
                "domain": x_range,
                "range": y_range,
                "metadata": {
                    "points_generated": len(x_vals),
                    "total_points_requested": points + 1,
                }
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate graph visualization: {str(e)}")
    
    def _generate_d3_tree(self, expr: sp.Basic) -> Dict[str, Any]:
        """Generate tree structure for D3.js visualization."""
        node_id_counter = [0]  # Use list for mutable counter
        
        def _build_node(node):
            node_id_counter[0] += 1
            node_id = f"node_{node_id_counter[0]}"
            
            if node.is_Atom:
                return {
                    "id": node_id,
                    "name": str(node),
                    "type": "atom",
                    "value": str(node),
                    "latex": sp.latex(node),
                    "children": [],
                    "is_leaf": True,
                    "node_type": self._classify_atom(node),
                }
            else:
                children = [_build_node(arg) for arg in node.args]
                return {
                    "id": node_id,
                    "name": type(node).__name__,
                    "type": "operation",
                    "operator": type(node).__name__,
                    "latex": sp.latex(node),
                    "children": children,
                    "is_leaf": False,
                    "arity": len(node.args),
                    "node_type": "operation",
                }
        
        return _build_node(expr)
    
    def _classify_atom(self, atom: sp.Basic) -> str:
        """Classify an atomic expression."""
        if atom.is_Symbol:
            return "variable"
        elif atom.is_Number:
            if atom.is_Integer:
                return "integer"
            elif atom.is_Rational:
                return "rational"
            elif atom.is_Float:
                return "float"
            else:
                return "number"
        elif atom == sp.pi:
            return "constant_pi"
        elif atom == sp.E:
            return "constant_e"
        elif atom == sp.I:
            return "constant_i"
        elif atom == sp.oo:
            return "infinity"
        else:
            return "unknown"
    
    def _calculate_layout(self, tree_data: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
        """Calculate layout parameters for the tree visualization."""
        node_count = self._count_nodes(tree_data)
        max_depth = self._calculate_max_depth(tree_data)
        
        # Calculate spacing
        horizontal_spacing = width // max(1, max_depth)
        vertical_spacing = height // max(1, node_count // max_depth + 1)
        
        return {
            "width": width,
            "height": height,
            "horizontal_spacing": horizontal_spacing,
            "vertical_spacing": vertical_spacing,
            "node_radius": min(20, max(8, horizontal_spacing // 4)),
            "font_size": min(14, max(10, vertical_spacing // 3)),
            "margin": {"top": 20, "right": 20, "bottom": 20, "left": 20},
        }
    
    def _generate_d3_config(
        self,
        tree_data: Dict[str, Any],
        layout_params: Dict[str, Any],
        interactive: bool
    ) -> Dict[str, Any]:
        """Generate D3.js configuration for interactive visualization."""
        config = {
            "data": tree_data,
            "layout": {
                "type": "tree",
                "orientation": "vertical",
                **layout_params,
            },
            "styles": {
                "nodes": {
                    "atom": {
                        "fill": "#4CAF50",
                        "stroke": "#2E7D32",
                        "stroke_width": 2,
                    },
                    "operation": {
                        "fill": "#2196F3",
                        "stroke": "#1565C0",
                        "stroke_width": 2,
                    },
                    "variable": {
                        "fill": "#FF9800",
                        "stroke": "#F57C00",
                        "stroke_width": 2,
                    },
                    "number": {
                        "fill": "#9C27B0",
                        "stroke": "#6A1B9A",
                        "stroke_width": 2,
                    },
                },
                "links": {
                    "stroke": "#757575",
                    "stroke_width": 2,
                    "fill": "none",
                },
                "text": {
                    "font_family": "Arial, sans-serif",
                    "font_size": layout_params["font_size"],
                    "fill": "#212121",
                    "text_anchor": "middle",
                    "dominant_baseline": "central",
                },
            },
            "interactions": {
                "hover": interactive,
                "click": interactive,
                "zoom": interactive,
                "pan": interactive,
            },
            "animations": {
                "enabled": interactive,
                "duration": 500,
                "easing": "cubic-in-out",
            },
        }
        
        return config
    
    def _generate_svg(self, tree_data: Dict[str, Any], layout_params: Dict[str, Any]) -> str:
        """Generate static SVG visualization."""
        width = layout_params["width"]
        height = layout_params["height"]
        
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '<style>',
            '.node-atom { fill: #4CAF50; stroke: #2E7D32; stroke-width: 2; }',
            '.node-operation { fill: #2196F3; stroke: #1565C0; stroke-width: 2; }',
            '.link { stroke: #757575; stroke-width: 2; fill: none; }',
            f'.text {{ font-family: Arial, sans-serif; font-size: {layout_params["font_size"]}px; fill: #212121; text-anchor: middle; dominant-baseline: central; }}',
            '</style>',
            '</defs>',
        ]
        
        # This is a simplified SVG generation - in practice, you'd want
        # a more sophisticated layout algorithm
        positions = self._calculate_positions(tree_data, layout_params)
        
        # Draw links first
        links = self._generate_links(tree_data, positions)
        svg_parts.extend(links)
        
        # Draw nodes
        nodes = self._generate_nodes(tree_data, positions, layout_params)
        svg_parts.extend(nodes)
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _calculate_positions(self, tree_data: Dict[str, Any], layout_params: Dict[str, Any]) -> Dict[str, tuple]:
        """Calculate positions for nodes in the tree."""
        positions = {}
        
        def _position_nodes(node, x, y, level_width):
            positions[node["id"]] = (x, y)
            
            if node["children"]:
                child_count = len(node["children"])
                child_spacing = level_width / (child_count + 1)
                start_x = x - level_width / 2 + child_spacing
                
                for i, child in enumerate(node["children"]):
                    child_x = start_x + i * child_spacing
                    child_y = y + layout_params["vertical_spacing"]
                    _position_nodes(child, child_x, child_y, level_width * 0.8)
        
        _position_nodes(tree_data, layout_params["width"] // 2, 50, layout_params["width"] * 0.8)
        return positions
    
    def _generate_links(self, tree_data: Dict[str, Any], positions: Dict[str, tuple]) -> List[str]:
        """Generate SVG links between nodes."""
        links = []
        
        def _add_links(node):
            if node["children"]:
                parent_pos = positions[node["id"]]
                for child in node["children"]:
                    child_pos = positions[child["id"]]
                    links.append(
                        f'<line x1="{parent_pos[0]}" y1="{parent_pos[1]}" '
                        f'x2="{child_pos[0]}" y2="{child_pos[1]}" class="link" />'
                    )
                    _add_links(child)
        
        _add_links(tree_data)
        return links
    
    def _generate_nodes(self, tree_data: Dict[str, Any], positions: Dict[str, tuple], layout_params: Dict[str, Any]) -> List[str]:
        """Generate SVG nodes."""
        nodes = []
        
        def _add_nodes(node):
            pos = positions[node["id"]]
            radius = layout_params["node_radius"]
            
            # Node circle
            node_class = f"node-{node['type']}"
            nodes.append(
                f'<circle cx="{pos[0]}" cy="{pos[1]}" r="{radius}" class="{node_class}" />'
            )
            
            # Node text
            nodes.append(
                f'<text x="{pos[0]}" y="{pos[1]}" class="text">{node["name"]}</text>'
            )
            
            for child in node["children"]:
                _add_nodes(child)
        
        _add_nodes(tree_data)
        return nodes
    
    def _count_nodes(self, tree_data: Dict[str, Any]) -> int:
        """Count total nodes in the tree."""
        count = 1
        for child in tree_data.get("children", []):
            count += self._count_nodes(child)
        return count
    
    def _calculate_max_depth(self, tree_data: Dict[str, Any]) -> int:
        """Calculate maximum depth of the tree."""
        if not tree_data.get("children"):
            return 1
        return 1 + max(self._calculate_max_depth(child) for child in tree_data["children"])
