"""
Backward-compatibility shim.

All classes have been moved to the individual window_*.py modules.
This file re-exports everything so that existing imports continue to work
without changes.
"""

from .window_shared import (  # noqa: F401
    ClickableLabel,
    NumericTableWidgetItem,
    BarDelegate,
    CenteredBarDelegate,
    CollapsibleBox,
)
from .window_msms import (  # noqa: F401
    MSMSPopupWindow,
    InteractiveMSMSChartView,
    MSMSViewerWindow,
    EnhancedMirrorPlotWindow,
)
from .window_ms1 import (  # noqa: F401
    MS1ViewerWindow,
    InteractiveMS1ChartView,
    InteractiveMS1SingleChartView,
    MS1SingleSpectrumWindow,
)
from .window_eic import (  # noqa: F401
    InteractiveChartView,
    EICExtractionWorker,
    EICWindow,
    EmbeddedScatterPlotView,
    Interactive2DScatterChartView,
)
from .window_multi_adduct import MultiAdductWindow  # noqa: F401

__all__ = [
    "ClickableLabel",
    "NumericTableWidgetItem",
    "BarDelegate",
    "CenteredBarDelegate",
    "CollapsibleBox",
    "MSMSPopupWindow",
    "InteractiveMSMSChartView",
    "MSMSViewerWindow",
    "EnhancedMirrorPlotWindow",
    "MS1ViewerWindow",
    "InteractiveMS1ChartView",
    "InteractiveMS1SingleChartView",
    "MS1SingleSpectrumWindow",
    "InteractiveChartView",
    "EICExtractionWorker",
    "EICWindow",
    "EmbeddedScatterPlotView",
    "Interactive2DScatterChartView",
]
