"""
Shared UI helper widgets used across the various viewer windows.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTableWidgetItem,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor


class ClickableLabel(QLabel):
    """QLabel that emits a *clicked* signal when left-clicked."""

    clicked = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically using the UserRole value."""

    def __lt__(self, other):
        try:
            v1 = self.data(Qt.ItemDataRole.UserRole)
            v2 = other.data(Qt.ItemDataRole.UserRole)
            if v1 is not None and v2 is not None:
                return float(v1) < float(v2)
        except (TypeError, ValueError):
            pass
        return super().__lt__(other)


class BarDelegate(QStyledItemDelegate):
    """
    Item delegate that paints a horizontal background bar proportional to the
    cell's numeric value relative to the column maximum.

    The bar is drawn in the group colour (passed per-row via UserRole + 1) at
    alpha 0.5, and the text is then drawn on top.
    """

    # Qt.ItemDataRole values we piggy-back on
    BAR_FRAC_ROLE = Qt.ItemDataRole.UserRole + 2  # float  0.0-1.0
    BAR_COLOR_ROLE = Qt.ItemDataRole.UserRole + 3  # QColor

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        # Let the base class draw the standard background (selection highlight etc.)
        super().paint(painter, option, index)

        frac = index.data(self.BAR_FRAC_ROLE)
        color = index.data(self.BAR_COLOR_ROLE)

        if frac is None or color is None or frac <= 0:
            return

        bar_width = int(option.rect.width() * min(frac, 1.0))
        bar_rect = option.rect.adjusted(0, 1, 0, -1)
        bar_rect.setWidth(bar_width)

        if isinstance(color, QColor):
            bar_color = QColor(color)
        else:
            bar_color = QColor(color)
        bar_color.setAlphaF(0.5)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(bar_rect, bar_color)
        painter.restore()

        # Re-draw the text on top so it stays readable
        painter.save()
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text and False:
            painter.drawText(
                option.rect.adjusted(4, 0, -4, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                str(text),
            )
        painter.restore()


class CenteredBarDelegate(QStyledItemDelegate):
    """
    Item delegate that paints a centered horizontal bar indicating ppm deviation
    from the theoretical m/z value.

    The bar extends from the cell centre toward the left (negative deviation) or
    right (positive deviation).  The half-width of the cell represents the m/z
    tolerance in ppm, so the leftmost edge = -tolerance and the rightmost = +tolerance.
    The bar is drawn in the group colour at alpha 0.5 so the cell text stays readable.
    """

    PPM_DEVIATION_ROLE = Qt.ItemDataRole.UserRole + 4  # float  (ppm deviation)
    PPM_RANGE_ROLE = Qt.ItemDataRole.UserRole + 5  # float  (tolerance ± ppm)
    PPM_BAR_COLOR_ROLE = Qt.ItemDataRole.UserRole + 6  # QColor

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        super().paint(painter, option, index)

        ppm_dev = index.data(self.PPM_DEVIATION_ROLE)
        ppm_range = index.data(self.PPM_RANGE_ROLE)
        color = index.data(self.PPM_BAR_COLOR_ROLE)

        if ppm_dev is None or ppm_range is None or ppm_range <= 0:
            return

        frac = max(-1.0, min(1.0, ppm_dev / ppm_range))  # -1.0 … +1.0
        if abs(frac) < 1e-6:
            return

        rect = option.rect
        center_x = rect.left() + rect.width() / 2
        bar_w = max(1, int(abs(frac) * rect.width() / 2))
        bar_top = rect.top() + 2
        bar_h = max(1, rect.height() - 4)

        if frac > 0:
            bar_rect = QRect(int(center_x), bar_top, bar_w, bar_h)
        else:
            bar_rect = QRect(int(center_x) - bar_w, bar_top, bar_w, bar_h)

        # Negative offset → dodgerblue, positive → firebrick (direction is always clear)
        bar_color = QColor("dodgerblue") if frac < 0 else QColor("firebrick")
        bar_color.setAlphaF(0.5)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(bar_rect, bar_color)
        painter.restore()


class CollapsibleBox(QWidget):
    """A collapsible widget with a clickable header"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px;
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_content)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_area.setVisible(False)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

    def toggle_content(self):
        """Toggle visibility of content area"""
        is_visible = self.content_area.isVisible()
        self.content_area.setVisible(not is_visible)

    def add_widget(self, widget):
        """Add a widget to the content area"""
        self.content_layout.addWidget(widget)

    def set_expanded(self, expanded):
        """Set the expanded state"""
        if expanded != self.content_area.isVisible():
            self.toggle_content()
