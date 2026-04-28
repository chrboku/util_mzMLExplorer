"""
Backward-compatibility shim.

All classes have been moved to the individual window_*.py modules.
This file re-exports everything so that existing imports continue to work
without changes.
"""

from .window_eic import (  # noqa: F401
    EICExtractionWorker,
    EICWindow,
    EmbeddedScatterPlotView,
    Interactive2DScatterChartView,
    InteractiveChartView,
)
from .window_ms1 import (  # noqa: F401
    InteractiveMS1ChartView,
    InteractiveMS1SingleChartView,
    MS1SingleSpectrumWindow,
    MS1ViewerWindow,
)
from .window_msms import (  # noqa: F401
    EnhancedMirrorPlotWindow,
    InteractiveMSMSChartView,
    MSMSPopupWindow,
    MSMSViewerWindow,
)
from .window_multi_adduct import MultiAdductWindow  # noqa: F401
from .window_shared import (  # noqa: F401
    BarDelegate,
    CenteredBarDelegate,
    ClickableLabel,
    CollapsibleBox,
    NumericTableWidgetItem,
)

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
