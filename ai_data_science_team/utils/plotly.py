

import json
import plotly.io as pio

def plotly_from_dict(plotly_graph_dict: dict):
    """
    Convert a Plotly graph dictionary to a Plotly graph object.
    
    Parameters:
    -----------
    plotly_graph_dict: dict
        A Plotly graph dictionary.
        
    Returns:
    --------
    plotly_graph: plotly.graph_objs.graph_objs.Figure
        A Plotly graph object.
    """
    
    if plotly_graph_dict is None:
        return None

    try:
        return pio.from_json(json.dumps(plotly_graph_dict))
    except Exception:
        return None
