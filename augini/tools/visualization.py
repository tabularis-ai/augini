from typing import Optional, Dict, Any, ClassVar, List, Union, Literal
import pandas as pd
import numpy as np
from langchain.callbacks.manager import CallbackManagerForToolRun
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pathlib import Path
import json

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .base import AuginiBaseTool, ToolResult
from ..config.base import ToolConfig


class VisualizationTool(AuginiBaseTool):
    """Tool for generating visualizations from DataFrame data."""
    
    # Class variables for tool metadata
    name: ClassVar[str] = "visualization_tool"
    description: ClassVar[str] = """Generates visualizations from DataFrame data.
    Supports various plot types and multiple backends (Matplotlib, Seaborn, Plotly)."""
    
    # Define instance variables
    _default_backend: str
    _default_style: str
    _default_figsize: tuple
    _export_dir: str
    
    def __init__(self, config: ToolConfig):
        """Initialize the visualization tool."""
        super().__init__(config=config)
        # Safely access parameters with fallbacks
        parameters = getattr(config, 'parameters', {}) or {}
        self._default_backend = parameters.get("default_backend", "matplotlib")
        self._default_style = parameters.get("default_style", "default")
        self._default_figsize = parameters.get("default_figsize", (10, 6))
        self._export_dir = parameters.get("export_dir", "./visualizations")
        
        # Create export directory if it doesn't exist
        if self._export_dir:
            Path(self._export_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def default_backend(self) -> str:
        return self._default_backend
        
    @property
    def default_style(self) -> str:
        return self._default_style
        
    @property
    def default_figsize(self) -> tuple:
        return self._default_figsize
        
    @property
    def export_dir(self) -> str:
        return self._export_dir
    
    def _run(
        self, 
        df: pd.DataFrame,
        plot_type: str,
        x: Optional[str] = None,
        y: Optional[Union[str, List[str]]] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        backend: Optional[str] = None,
        style: Optional[str] = None,
        figsize: Optional[tuple] = None,
        export_format: Optional[str] = None,
        export_filename: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> ToolResult:
        """
        Generate a visualization based on the specified parameters.
        
        Args:
            df: The DataFrame to visualize
            plot_type: Type of plot (histogram, scatter, bar, line, box, heatmap, etc.)
            x: Column name for x-axis
            y: Column name(s) for y-axis
            hue: Column name for color grouping
            title: Plot title
            backend: Plotting backend ('matplotlib', 'seaborn', 'plotly')
            style: Visual style for the plot
            figsize: Figure size as (width, height) in inches
            export_format: Format to export the plot ('png', 'jpg', 'svg', 'html', 'json')
            export_filename: Filename for the exported plot
            additional_params: Additional parameters for the specific plot type
            
        Returns:
            ToolResult containing the visualization data and metadata
        """
        # Set defaults
        backend = backend or self.default_backend
        style = style or self.default_style
        figsize = figsize or self.default_figsize
        additional_params = additional_params or {}
        
        # Validate inputs
        if plot_type not in self._get_supported_plot_types(backend):
            return ToolResult(
                success=False,
                result=None,
                error=f"Plot type '{plot_type}' not supported with backend '{backend}'",
                metadata={"supported_types": self._get_supported_plot_types(backend)}
            )
        
        # Generate the visualization
        try:
            if backend == "plotly" and not PLOTLY_AVAILABLE:
                return ToolResult(
                    success=False,
                    result=None,
                    error="Plotly is not installed. Install with 'pip install plotly'.",
                    metadata={}
                )
            
            # Create the visualization
            fig, plot_data = self._create_visualization(
                df=df,
                plot_type=plot_type,
                x=x,
                y=y,
                hue=hue,
                title=title,
                backend=backend,
                style=style,
                figsize=figsize,
                additional_params=additional_params
            )
            
            # Export if requested
            export_path = None
            if export_format and export_filename:
                export_path = self._export_visualization(
                    fig=fig,
                    backend=backend,
                    export_format=export_format,
                    export_filename=export_filename
                )
            
            # Generate code for reproducing the visualization
            code = self._generate_code(
                df_name="df",
                plot_type=plot_type,
                x=x,
                y=y,
                hue=hue,
                title=title,
                backend=backend,
                style=style,
                figsize=figsize,
                additional_params=additional_params
            )
            
            return ToolResult(
                success=True,
                result={
                    "plot_type": plot_type,
                    "backend": backend,
                    "plot_data": plot_data,
                    "export_path": str(export_path) if export_path else None,
                    "code": code
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Error generating visualization: {str(e)}",
                metadata={}
            )
    
    def _create_visualization(
        self,
        df: pd.DataFrame,
        plot_type: str,
        x: Optional[str],
        y: Optional[Union[str, List[str]]],
        hue: Optional[str],
        title: Optional[str],
        backend: str,
        style: str,
        figsize: tuple,
        additional_params: Dict[str, Any]
    ) -> tuple:
        """Create the visualization using the specified backend."""
        if backend == "matplotlib":
            return self._create_matplotlib_visualization(
                df, plot_type, x, y, hue, title, style, figsize, additional_params
            )
        elif backend == "seaborn":
            return self._create_seaborn_visualization(
                df, plot_type, x, y, hue, title, style, figsize, additional_params
            )
        elif backend == "plotly":
            return self._create_plotly_visualization(
                df, plot_type, x, y, hue, title, style, figsize, additional_params
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _create_matplotlib_visualization(
        self,
        df: pd.DataFrame,
        plot_type: str,
        x: Optional[str],
        y: Optional[Union[str, List[str]]],
        hue: Optional[str],
        title: Optional[str],
        style: str,
        figsize: tuple,
        additional_params: Dict[str, Any]
    ) -> tuple:
        """Create visualization using Matplotlib."""
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == "histogram":
            if x:
                ax.hist(df[x], **additional_params)
                ax.set_xlabel(x)
            else:
                raise ValueError("x parameter is required for histogram")
                
        elif plot_type == "scatter":
            if x and y and isinstance(y, str):
                if hue:
                    for category in df[hue].unique():
                        subset = df[df[hue] == category]
                        ax.scatter(subset[x], subset[y], label=category, **additional_params)
                    ax.legend()
                else:
                    ax.scatter(df[x], df[y], **additional_params)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            else:
                raise ValueError("x and y parameters are required for scatter plot")
                
        elif plot_type == "line":
            if x and y and isinstance(y, str):
                if hue:
                    for category in df[hue].unique():
                        subset = df[df[hue] == category]
                        ax.plot(subset[x], subset[y], label=category, **additional_params)
                    ax.legend()
                else:
                    ax.plot(df[x], df[y], **additional_params)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            else:
                raise ValueError("x and y parameters are required for line plot")
                
        elif plot_type == "bar":
            if x and y and isinstance(y, str):
                if hue:
                    grouped = df.groupby([x, hue])[y].mean().unstack()
                    grouped.plot(kind='bar', ax=ax, **additional_params)
                else:
                    df.groupby(x)[y].mean().plot(kind='bar', ax=ax, **additional_params)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
            else:
                raise ValueError("x and y parameters are required for bar plot")
                
        elif plot_type == "box":
            if x:
                if y and isinstance(y, str):
                    ax.boxplot(df[y], labels=[y], **additional_params)
                    ax.set_ylabel(y)
                else:
                    ax.boxplot(df[x], labels=[x], **additional_params)
                    ax.set_ylabel(x)
            else:
                raise ValueError("x parameter is required for box plot")
                
        elif plot_type == "heatmap":
            if not (x and y and isinstance(y, list) and len(y) > 0):
                # If specific columns aren't provided, use correlation matrix
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                im = ax.imshow(corr_matrix, **additional_params)
                ax.set_xticks(np.arange(len(corr_matrix.columns)))
                ax.set_yticks(np.arange(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns)
                ax.set_yticklabels(corr_matrix.columns)
                plt.colorbar(im, ax=ax)
            else:
                raise ValueError("Heatmap requires numerical data")
        
        if title:
            ax.set_title(title)
            
        plt.tight_layout()
        
        # Convert plot to base64 for return
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        return fig, plot_data
    
    def _create_seaborn_visualization(
        self,
        df: pd.DataFrame,
        plot_type: str,
        x: Optional[str],
        y: Optional[Union[str, List[str]]],
        hue: Optional[str],
        title: Optional[str],
        style: str,
        figsize: tuple,
        additional_params: Dict[str, Any]
    ) -> tuple:
        """Create visualization using Seaborn."""
        sns.set_style(style)
        plt.figure(figsize=figsize)
        
        if plot_type == "histogram":
            if x:
                ax = sns.histplot(data=df, x=x, hue=hue, **additional_params)
            else:
                raise ValueError("x parameter is required for histogram")
                
        elif plot_type == "scatter":
            if x and y and isinstance(y, str):
                ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, **additional_params)
            else:
                raise ValueError("x and y parameters are required for scatter plot")
                
        elif plot_type == "line":
            if x and y and isinstance(y, str):
                ax = sns.lineplot(data=df, x=x, y=y, hue=hue, **additional_params)
            else:
                raise ValueError("x and y parameters are required for line plot")
                
        elif plot_type == "bar":
            if x and y and isinstance(y, str):
                ax = sns.barplot(data=df, x=x, y=y, hue=hue, **additional_params)
            else:
                raise ValueError("x and y parameters are required for bar plot")
                
        elif plot_type == "box":
            if x:
                if y and isinstance(y, str):
                    ax = sns.boxplot(data=df, x=x, y=y, hue=hue, **additional_params)
                else:
                    ax = sns.boxplot(data=df, x=x, hue=hue, **additional_params)
            else:
                raise ValueError("x parameter is required for box plot")
                
        elif plot_type == "heatmap":
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            ax = sns.heatmap(corr_matrix, annot=True, **additional_params)
                
        elif plot_type == "pairplot":
            if hue:
                g = sns.pairplot(df, hue=hue, **additional_params)
            else:
                g = sns.pairplot(df, **additional_params)
            ax = g.fig.axes[0]
        
        if title and plot_type != "pairplot":
            plt.title(title)
            
        plt.tight_layout()
        
        # Convert plot to base64 for return
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        fig = plt.gcf()
        return fig, plot_data
    
    def _create_plotly_visualization(
        self,
        df: pd.DataFrame,
        plot_type: str,
        x: Optional[str],
        y: Optional[Union[str, List[str]]],
        hue: Optional[str],
        title: Optional[str],
        style: str,
        figsize: tuple,
        additional_params: Dict[str, Any]
    ) -> tuple:
        """Create visualization using Plotly."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not installed. Install with 'pip install plotly'.")
        
        width, height = figsize[0] * 100, figsize[1] * 100  # Convert to pixels
        
        if plot_type == "histogram":
            if x:
                fig = px.histogram(df, x=x, color=hue, title=title, 
                                  width=width, height=height, **additional_params)
            else:
                raise ValueError("x parameter is required for histogram")
                
        elif plot_type == "scatter":
            if x and y and isinstance(y, str):
                fig = px.scatter(df, x=x, y=y, color=hue, title=title,
                                width=width, height=height, **additional_params)
            else:
                raise ValueError("x and y parameters are required for scatter plot")
                
        elif plot_type == "line":
            if x and y and isinstance(y, str):
                fig = px.line(df, x=x, y=y, color=hue, title=title,
                             width=width, height=height, **additional_params)
            else:
                raise ValueError("x and y parameters are required for line plot")
                
        elif plot_type == "bar":
            if x and y and isinstance(y, str):
                fig = px.bar(df, x=x, y=y, color=hue, title=title,
                            width=width, height=height, **additional_params)
            else:
                raise ValueError("x and y parameters are required for bar plot")
                
        elif plot_type == "box":
            if x:
                if y and isinstance(y, str):
                    fig = px.box(df, x=x, y=y, color=hue, title=title,
                                width=width, height=height, **additional_params)
                else:
                    fig = px.box(df, x=x, color=hue, title=title,
                                width=width, height=height, **additional_params)
            else:
                raise ValueError("x parameter is required for box plot")
                
        elif plot_type == "heatmap":
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr_matrix, title=title,
                           width=width, height=height, **additional_params)
                
        else:
            raise ValueError(f"Plot type '{plot_type}' not supported with Plotly")
        
        # Convert to JSON for return
        plot_data = fig.to_json()
        
        return fig, plot_data
    
    def _export_visualization(
        self,
        fig,
        backend: str,
        export_format: str,
        export_filename: str
    ) -> Path:
        """Export the visualization to a file."""
        export_path = Path(self.export_dir) / f"{export_filename}.{export_format}"
        
        if backend in ["matplotlib", "seaborn"]:
            fig.savefig(export_path, format=export_format, bbox_inches='tight')
        elif backend == "plotly":
            if export_format == "html":
                fig.write_html(export_path)
            elif export_format == "json":
                with open(export_path, 'w') as f:
                    f.write(fig.to_json())
            else:
                fig.write_image(export_path)
        
        return export_path
    
    def _generate_code(
        self,
        df_name: str,
        plot_type: str,
        x: Optional[str],
        y: Optional[Union[str, List[str]]],
        hue: Optional[str],
        title: Optional[str],
        backend: str,
        style: str,
        figsize: tuple,
        additional_params: Dict[str, Any]
    ) -> str:
        """Generate code for reproducing the visualization."""
        code_lines = ["import pandas as pd"]
        
        if backend == "matplotlib":
            code_lines.extend([
                "import matplotlib.pyplot as plt",
                f"plt.style.use('{style}')",
                f"fig, ax = plt.subplots(figsize={figsize})"
            ])
            
            if plot_type == "histogram":
                params = ", ".join([f"{k}={repr(v)}" for k, v in additional_params.items()])
                code_lines.append(f"ax.hist({df_name}['{x}'], {params})")
                code_lines.append(f"ax.set_xlabel('{x}')")
                
            elif plot_type == "scatter":
                params = ", ".join([f"{k}={repr(v)}" for k, v in additional_params.items()])
                if hue:
                    code_lines.extend([
                        f"for category in {df_name}['{hue}'].unique():",
                        f"    subset = {df_name}[{df_name}['{hue}'] == category]",
                        f"    ax.scatter(subset['{x}'], subset['{y}'], label=category, {params})",
                        "ax.legend()"
                    ])
                else:
                    code_lines.append(f"ax.scatter({df_name}['{x}'], {df_name}['{y}'], {params})")
                code_lines.extend([
                    f"ax.set_xlabel('{x}')",
                    f"ax.set_ylabel('{y}')"
                ])
                
            # Add more plot types as needed...
            
        elif backend == "seaborn":
            code_lines.extend([
                "import seaborn as sns",
                f"sns.set_style('{style}')",
                f"plt.figure(figsize={figsize})"
            ])
            
            params = ", ".join([f"{k}={repr(v)}" for k, v in additional_params.items()])
            
            if plot_type == "histogram":
                hue_param = f", hue='{hue}'" if hue else ""
                code_lines.append(f"ax = sns.histplot(data={df_name}, x='{x}'{hue_param}, {params})")
                
            elif plot_type == "scatter":
                hue_param = f", hue='{hue}'" if hue else ""
                code_lines.append(f"ax = sns.scatterplot(data={df_name}, x='{x}', y='{y}'{hue_param}, {params})")
                
            # Add more plot types as needed...
            
        elif backend == "plotly":
            code_lines.extend([
                "import plotly.express as px",
                "import plotly.graph_objects as go"
            ])
            
            width, height = figsize[0] * 100, figsize[1] * 100
            params = ", ".join([f"{k}={repr(v)}" for k, v in additional_params.items()])
            
            if plot_type == "histogram":
                color_param = f", color='{hue}'" if hue else ""
                title_param = f", title='{title}'" if title else ""
                code_lines.append(f"fig = px.histogram({df_name}, x='{x}'{color_param}{title_param}, width={width}, height={height}, {params})")
                
            elif plot_type == "scatter":
                color_param = f", color='{hue}'" if hue else ""
                title_param = f", title='{title}'" if title else ""
                code_lines.append(f"fig = px.scatter({df_name}, x='{x}', y='{y}'{color_param}{title_param}, width={width}, height={height}, {params})")
                
            # Add more plot types as needed...
            
        if title and backend != "plotly":
            code_lines.append(f"plt.title('{title}')")
            
        if backend in ["matplotlib", "seaborn"]:
            code_lines.append("plt.tight_layout()")
            code_lines.append("plt.show()")
        elif backend == "plotly":
            code_lines.append("fig.show()")
            
        return "\n".join(code_lines)
    
    def _get_supported_plot_types(self, backend: str) -> List[str]:
        """Get the list of supported plot types for a given backend."""
        common_types = ["histogram", "scatter", "line", "bar", "box", "heatmap"]
        
        if backend == "matplotlib":
            return common_types
        elif backend == "seaborn":
            return common_types + ["pairplot", "violin", "kde"]
        elif backend == "plotly":
            return common_types + ["bubble", "sunburst", "treemap"]
        else:
            return [] 