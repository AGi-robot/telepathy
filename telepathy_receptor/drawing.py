"""Thin shim: the drawing language lives in ~/AGi/capability/canvas (pip: grace-canvas).

Raster painting ('image' actions) is registered by grace-imagegen from
~/AGi/capability/imaging.
Re-exports keep `telepathy_receptor.node` and the existing tests working unchanged.
"""
from canvas.engine import (DRAW_TOOL, DRAW_INSTRUCTIONS, validate_actions,
                           canvas_context, drawing_history, drawing_summary,
                           completed_stream_actions, repair_model_actions,
                           KINDS, ACTION_SCHEMA, register_action_kind,
                           build_draw_tool, build_draw_schema)

try:
    from imagegen import PAINT_INSTRUCTIONS, register_image_kind
    register_image_kind()
    DRAW_TOOL = build_draw_tool()
except ImportError:  # raster painting not installed - vector-only canvas
    PAINT_INSTRUCTIONS = ''

__all__ = ['DRAW_TOOL', 'DRAW_INSTRUCTIONS', 'PAINT_INSTRUCTIONS',
           'validate_actions', 'canvas_context', 'drawing_history',
           'drawing_summary', 'completed_stream_actions',
           'repair_model_actions', 'KINDS', 'ACTION_SCHEMA',
           'register_action_kind', 'build_draw_tool', 'build_draw_schema']
