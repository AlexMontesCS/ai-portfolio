"""
Visualization endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, Response

from app.models.schemas import VisualizationRequest, VisualizationResponse
from app.services.visualization import VisualizationService
from app.core.exceptions import VisualizationError

router = APIRouter()
viz_service = VisualizationService()


@router.post("/tree", response_model=VisualizationResponse)
async def visualize_expression_tree(request: VisualizationRequest) -> VisualizationResponse:
    """
    Generate expression tree visualization data.
    
    This endpoint creates visualization data for rendering mathematical
    expression trees using D3.js or similar libraries.
    
    **Example request:**
    ```json
    {
        "expression": "x^2 + 2*x*y + y^2",
        "visualization_type": "tree",
        "format": "text",
        "width": 800,
        "height": 600,
        "interactive": true
    }
    ```
    
    **Response includes:**
    - D3.js configuration for interactive trees
    - SVG content for static visualization
    - Tree structure data
    - Layout parameters
    """
    try:
        viz_data = viz_service.generate_expression_tree(
            expression=request.expression,
            width=request.width or 800,
            height=request.height or 600,
            interactive=request.interactive
        )
        
        return VisualizationResponse(
            success=True,
            visualization_data=viz_data["tree_data"],
            svg_content=viz_data.get("svg_content"),
            d3_config=viz_data.get("d3_config")
        )
        
    except VisualizationError as e:
        return VisualizationResponse(
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during visualization: {str(e)}"
        )


@router.post("/graph")
async def visualize_function_graph(request: VisualizationRequest) -> JSONResponse:
    """
    Generate function graph visualization data for plotting.
    
    This endpoint generates coordinate data for plotting mathematical
    functions as graphs.
    
    **Example request:**
    ```json
    {
        "expression": "sin(x) + cos(x)",
        "visualization_type": "graph",
        "format": "text",
        "width": 800,
        "height": 600
    }
    ```
    
    **Query parameters:**
    - x_min: Minimum x value (default: -10)
    - x_max: Maximum x value (default: 10)
    - y_min: Minimum y value (default: -10)
    - y_max: Maximum y value (default: 10)
    - points: Number of points to generate (default: 1000)
    """
    try:
        # You can extend this to accept range parameters
        x_range = (-10, 10)  # Could be from query parameters
        y_range = (-10, 10)  # Could be from query parameters
        points = 1000       # Could be from query parameters
        
        plot_data = viz_service.generate_graph_visualization(
            expression=request.expression,
            x_range=x_range,
            y_range=y_range,
            points=points
        )
        
        return JSONResponse(content={
            "success": True,
            "plot_data": plot_data["plot_data"],
            "expression": plot_data["expression"],
            "variable": plot_data["variable"],
            "latex": plot_data["latex"],
            "domain": plot_data["domain"],
            "range": plot_data["range"],
            "metadata": plot_data["metadata"]
        })
        
    except VisualizationError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during graph generation: {str(e)}"
        )


@router.get("/tree/{expression}/svg")
async def get_expression_tree_svg(expression: str) -> Response:
    """
    Get SVG representation of an expression tree.
    
    Returns the SVG content directly for embedding or download.
    
    **Example:** `/api/v1/visualize/tree/x^2+2*x+1/svg`
    """
    try:
        viz_data = viz_service.generate_expression_tree(
            expression=expression,
            width=800,
            height=600,
            interactive=False
        )
        
        svg_content = viz_data.get("svg_content")
        if not svg_content:
            raise VisualizationError("SVG generation failed")
        
        return Response(
            content=svg_content,
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": f"inline; filename=expression_tree_{expression[:20]}.svg"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SVG: {str(e)}"
        )


@router.post("/d3-config")
async def get_d3_configuration(request: VisualizationRequest) -> JSONResponse:
    """
    Get D3.js configuration for custom visualization implementations.
    
    This endpoint returns the complete D3.js configuration object
    that can be used to render interactive visualizations.
    
    **Example request:**
    ```json
    {
        "expression": "log(x) + sqrt(y)",
        "visualization_type": "tree",
        "width": 1000,
        "height": 800,
        "interactive": true
    }
    ```
    """
    try:
        viz_data = viz_service.generate_expression_tree(
            expression=request.expression,
            width=request.width or 800,
            height=request.height or 600,
            interactive=request.interactive
        )
        
        d3_config = viz_data.get("d3_config")
        if not d3_config:
            raise VisualizationError("D3 configuration generation failed")
        
        return JSONResponse(content={
            "success": True,
            "d3_config": d3_config,
            "metadata": viz_data.get("metadata", {}),
            "layout_params": viz_data.get("layout_params", {})
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate D3 configuration: {str(e)}"
        )
