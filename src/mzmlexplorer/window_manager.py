"""
Window Manager — tracks every top-level window that is opened, keeps their
parent–child relationships in a tree-view side panel, and maintains a history
of recently-closed windows so the user can restore them.

Public API
----------
get_window_manager()        → WindowManager | None
set_window_manager(wm)      → None

WindowManager.register_window(window, parent_window, title, wtype) → int (wid)
WindowManager.bring_to_front(wid)
WindowManager.close_window(wid)
WindowManager.restore_window(wid)
WindowManager.purge_from_history(wid)
WindowManager.has_open_children() → bool
WindowManager.update_title(window, new_title)
WindowManager.close_all_children_and_exit()

WindowManagerPanel(window_manager, parent)   — QWidget side panel
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Set

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QCloseEvent, QColor, QFont
from PyQt6.QtWidgets import (
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HISTORY = 15

_WTYPE_ICONS: Dict[str, str] = {
    "EIC": "📈",
    "MS1": "📊",
    "MSMS": "🔬",
    "FileExplorer": "🗂",
    "MultiAdduct": "🧪",
    "Comparator": "⚖",
    "Game": "🎮",
    "Main": "🏠",
    "Other": "🪟",
}

_HISTORY_BRUSH = QBrush(QColor(130, 130, 130))

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[WindowManager] = None  # forward-declared; actual class below


def get_window_manager() -> Optional[WindowManager]:
    """Return the active WindowManager singleton, or None if not yet created."""
    return _instance


def set_window_manager(wm: WindowManager) -> None:
    """Install *wm* as the module-level singleton."""
    global _instance
    _instance = wm


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------


class _WindowRecord:
    _counter: int = 0

    def __init__(
        self,
        window: QWidget,
        title: str,
        wtype: str,
        parent_id: Optional[int],
    ) -> None:
        _WindowRecord._counter += 1
        self.wid: int = _WindowRecord._counter
        self.window: QWidget = window
        self.title: str = title
        self.wtype: str = wtype
        self.parent_id: Optional[int] = parent_id
        self.children_ids: Set[int] = set()
        self.is_open: bool = True


# ---------------------------------------------------------------------------
# Window Manager
# ---------------------------------------------------------------------------


class WindowManager(QObject):
    """
    Manages all open windows, their parent–child relationships, and a bounded
    history of recently-closed windows.

    Lifecycle
    ---------
    1. Call ``register_window`` whenever a new window is shown.
    2. The manager intercepts the Qt Close event for every registered window
       (except the main window) by acting as an event-filter.
    3. Instead of closing the widget, the manager hides it and appends it to
       the history deque (max ``MAX_HISTORY`` entries).
    4. When the history overflows the oldest entry is truly destroyed via
       ``deleteLater``.
    5. The main window's close event is *not* consumed here; the application
       should handle it in ``MzMLExplorerMainWindow.closeEvent``.
    """

    tree_changed = pyqtSignal()

    def __init__(self, main_window: QWidget) -> None:
        super().__init__()
        self._main_window = main_window
        self._records: Dict[int, _WindowRecord] = {}
        self._window_to_wid: Dict[int, int] = {}  # id(window) → wid
        self._history: deque = deque()  # wids (oldest first)
        self._moving_wids: Set[int] = set()  # re-entrancy guard

        # Register the main window itself
        rec = _WindowRecord(main_window, "mzML Explorer", "Main", None)
        self._records[rec.wid] = rec
        self._window_to_wid[id(main_window)] = rec.wid
        self._main_wid: int = rec.wid
        main_window.installEventFilter(self)

    # ------------------------------------------------------------------ public

    def register_window(
        self,
        window: QWidget,
        parent_window: Optional[QWidget] = None,
        title: str = "",
        wtype: str = "Other",
    ) -> int:
        """
        Register *window* with the manager and return its assigned wid.
        If *window* was already registered, return its existing wid.

        Parameters
        ----------
        window:        The top-level widget to track.
        parent_window: The window from which this one was opened (determines
                       tree position).  Defaults to the main window.
        title:         Human-readable label shown in the panel.
        wtype:         Category string used to select an icon.
        """
        existing_wid = self._window_to_wid.get(id(window))
        if existing_wid is not None:
            return existing_wid

        # Resolve parent id
        parent_id: Optional[int] = None
        if parent_window is not None:
            parent_id = self._window_to_wid.get(id(parent_window))
        if parent_id is None:
            parent_id = self._main_wid

        if not title:
            try:
                title = window.windowTitle() or wtype
            except RuntimeError:
                title = wtype

        rec = _WindowRecord(window, title, wtype, parent_id)
        self._records[rec.wid] = rec
        self._window_to_wid[id(window)] = rec.wid

        parent_rec = self._records.get(parent_id)
        if parent_rec is not None:
            parent_rec.children_ids.add(rec.wid)

        # Prevent Qt from deleting the widget on close; we manage lifetime.
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        window.installEventFilter(self)

        wid = rec.wid
        window.destroyed.connect(lambda _=None, w=wid: self._on_window_destroyed(w))

        self.tree_changed.emit()
        return rec.wid

    def bring_to_front(self, wid: int) -> None:
        """Raise and activate the window identified by *wid*."""
        rec = self._records.get(wid)
        if rec is None:
            return
        if not rec.is_open:
            self.restore_window(wid)
            return
        w = rec.window
        try:
            w.setWindowState(w.windowState() & ~Qt.WindowState.WindowMinimized)
            w.show()
            w.raise_()
            w.activateWindow()
        except RuntimeError:
            pass

    def close_window(self, wid: int) -> None:
        """
        Programmatically close *wid* (hides it and adds to history).
        For the main window, delegates to the Qt close machinery.
        """
        if wid == self._main_wid:
            self._main_window.close()
            return
        rec = self._records.get(wid)
        if rec is None or not rec.is_open:
            return
        self._move_to_history(wid)

    def restore_window(self, wid: int) -> None:
        """Restore a previously closed window from the history back to open."""
        rec = self._records.get(wid)
        if rec is None or rec.is_open:
            return
        try:
            if rec.window is None:
                return
        except RuntimeError:
            return

        rec.is_open = True
        if wid in self._history:
            self._history.remove(wid)

        # Re-attach to parent (or main if parent is gone / in history)
        parent_rec = self._records.get(rec.parent_id)
        if parent_rec is not None and parent_rec.is_open:
            parent_rec.children_ids.add(wid)
        else:
            rec.parent_id = self._main_wid
            main_rec = self._records.get(self._main_wid)
            if main_rec is not None:
                main_rec.children_ids.add(wid)

        try:
            rec.window.show()
            rec.window.raise_()
            rec.window.activateWindow()
        except RuntimeError:
            pass

        self.tree_changed.emit()

    def purge_from_history(self, wid: int) -> None:
        """Permanently destroy a window that is sitting in the history."""
        rec = self._records.get(wid)
        if rec is None or rec.is_open:
            return

        if wid in self._history:
            self._history.remove(wid)

        parent_rec = self._records.get(rec.parent_id)
        if parent_rec is not None:
            parent_rec.children_ids.discard(wid)

        # Re-parent any still-open children to main
        for child_wid in list(rec.children_ids):
            child_rec = self._records.get(child_wid)
            if child_rec is not None and child_rec.is_open:
                child_rec.parent_id = self._main_wid
                main_rec = self._records.get(self._main_wid)
                if main_rec is not None:
                    main_rec.children_ids.add(child_wid)

        del self._records[wid]
        if rec.window is not None:
            self._window_to_wid.pop(id(rec.window), None)
            try:
                rec.window.removeEventFilter(self)
                rec.window.deleteLater()
            except RuntimeError:
                pass

        self.tree_changed.emit()

    def has_open_children(self) -> bool:
        """Return True if any non-main windows are currently open (visible)."""
        return any(r.is_open for wid, r in self._records.items() if wid != self._main_wid)

    def update_title(self, window: QWidget, new_title: str) -> None:
        """Update the display label for a registered window."""
        wid = self._window_to_wid.get(id(window))
        if wid is not None and wid in self._records:
            self._records[wid].title = new_title
            self.tree_changed.emit()

    def close_all_children_and_exit(self) -> None:
        """
        Run cleanup on every open child window, hide them, schedule their
        deletion, and then close the main window.
        Called when the user confirms "close all and exit".
        """
        to_close = [rec for wid, rec in list(self._records.items()) if wid != self._main_wid]
        for rec in to_close:
            if rec.window is not None:
                try:
                    rec.window.removeEventFilter(self)
                    self._run_window_cleanup(rec.window)
                    rec.window.hide()
                    rec.window.deleteLater()
                except RuntimeError:
                    pass
            self._records.pop(rec.wid, None)
            if rec.window is not None:
                self._window_to_wid.pop(id(rec.window), None)

        self._history.clear()
        self.tree_changed.emit()

        # Remove our event filter so the main window closes normally
        self._main_window.removeEventFilter(self)
        self._main_window.close()

    # ----------------------------------------------------------------- private

    def _move_to_history(self, wid: int) -> None:
        """Hide window, update hierarchy, and append to history."""
        rec = self._records.get(wid)
        if rec is None or not rec.is_open:
            return
        if wid in self._moving_wids:
            return

        self._moving_wids.add(wid)
        try:
            rec.is_open = False

            # Detach from parent
            parent_rec = self._records.get(rec.parent_id)
            if parent_rec is not None:
                parent_rec.children_ids.discard(wid)

            # Re-parent open children to main
            main_rec = self._records.get(self._main_wid)
            for child_wid in list(rec.children_ids):
                child_rec = self._records.get(child_wid)
                if child_rec is not None and child_rec.is_open:
                    child_rec.parent_id = self._main_wid
                    if main_rec is not None:
                        main_rec.children_ids.add(child_wid)
            rec.children_ids.clear()

            # Run window-level cleanup (stops background threads, etc.)
            self._run_window_cleanup(rec.window)

            try:
                rec.window.hide()
            except RuntimeError:
                pass

            self._history.append(wid)
            self._evict_oldest_if_needed()

        finally:
            self._moving_wids.discard(wid)

        self.tree_changed.emit()

    def _evict_oldest_if_needed(self) -> None:
        """Trim the history deque to MAX_HISTORY by deleting the oldest entry."""
        while len(self._history) > MAX_HISTORY:
            oldest_wid = self._history.popleft()
            oldest_rec = self._records.pop(oldest_wid, None)
            if oldest_rec is not None and oldest_rec.window is not None:
                try:
                    oldest_rec.window.removeEventFilter(self)
                    self._window_to_wid.pop(id(oldest_rec.window), None)
                    oldest_rec.window.deleteLater()
                except RuntimeError:
                    pass

    @staticmethod
    def _run_window_cleanup(window: QWidget) -> None:
        """
        Invoke the widget's closeEvent directly so any cleanup code (e.g.
        stopping background threads) runs before the window is hidden.
        We do NOT go through close() to avoid triggering the event filter.
        """
        try:
            ce = QCloseEvent()
            window.closeEvent(ce)
        except (RuntimeError, AttributeError, Exception):
            pass

    def _on_window_destroyed(self, wid: int) -> None:
        """
        Fallback: the C++ object was destroyed externally (e.g. the window
        somehow still had WA_DeleteOnClose set True elsewhere).
        Clean up internal state gracefully.
        """
        rec = self._records.get(wid)
        if rec is None:
            return  # Already cleaned up

        parent_rec = self._records.get(rec.parent_id)
        if parent_rec is not None:
            parent_rec.children_ids.discard(wid)

        if wid in self._history:
            self._history.remove(wid)

        for child_wid in list(rec.children_ids):
            child_rec = self._records.get(child_wid)
            if child_rec is not None:
                child_rec.parent_id = self._main_wid
                main_rec = self._records.get(self._main_wid)
                if main_rec is not None:
                    main_rec.children_ids.add(child_wid)

        self._records.pop(wid, None)
        # Do NOT access rec.window — the C++ object is already gone.
        self.tree_changed.emit()

    # --------------------------------------------------- QObject event filter

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Close:
            wid = self._window_to_wid.get(id(obj))
            if wid is not None:
                if wid == self._main_wid:
                    return False  # Let MzMLExplorerMainWindow.closeEvent handle it
                rec = self._records.get(wid)
                if rec is not None and rec.is_open and wid not in self._moving_wids:
                    self._move_to_history(wid)
                    return True  # Consume the Close event
        return False


# ---------------------------------------------------------------------------
# Side-panel widget
# ---------------------------------------------------------------------------


class WindowManagerPanel(QWidget):
    """
    A QWidget that displays the window tree and history in a QTreeWidget.

    - Open windows are shown as a hierarchy rooted at 'mzML Explorer'.
    - Recently-closed windows appear in a 'Recently Closed' section.

    Left-click on an open entry → bring to front.
    Left-click on a history entry → restore.
    Right-click → context menu (Bring to Front / Close  or  Restore / Remove).
    """

    def __init__(self, window_manager: WindowManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._wm = window_manager
        self._setup_ui()
        window_manager.tree_changed.connect(self._rebuild_tree)
        self._rebuild_tree()

    # ---------------------------------------------------------------------- UI

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setUniformRowHeights(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

    # ----------------------------------------------------------------- rebuild

    def _collect_expanded_wids(self) -> Set[int]:
        """Walk the current tree and collect wids of expanded items."""
        expanded: Set[int] = set()

        def _walk(item: QTreeWidgetItem) -> None:
            if item.isExpanded():
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if data is not None:
                    expanded.add(data[1])
            for i in range(item.childCount()):
                _walk(item.child(i))

        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            _walk(root.child(i))
        return expanded

    def _rebuild_tree(self) -> None:
        expanded_wids = self._collect_expanded_wids()

        self._tree.clear()
        records = self._wm._records

        # ── Open windows tree ──────────────────────────────────────────────
        main_rec = records.get(self._wm._main_wid)
        if main_rec is not None:
            root_item = QTreeWidgetItem(self._tree)
            root_item.setText(0, f"{_WTYPE_ICONS['Main']} mzML Explorer")
            bold = QFont()
            bold.setBold(True)
            root_item.setFont(0, bold)
            root_item.setData(0, Qt.ItemDataRole.UserRole, ("open", self._wm._main_wid))

            def _add_open_children(parent_wid: int, parent_item: QTreeWidgetItem) -> None:
                parent_rec = records.get(parent_wid)
                if parent_rec is None:
                    return
                for child_wid in sorted(parent_rec.children_ids):
                    child_rec = records.get(child_wid)
                    if child_rec is None or not child_rec.is_open:
                        continue
                    icon = _WTYPE_ICONS.get(child_rec.wtype, "🪟")
                    child_item = QTreeWidgetItem(parent_item)
                    child_item.setText(0, f"{icon} {child_rec.title}")
                    child_item.setData(0, Qt.ItemDataRole.UserRole, ("open", child_wid))
                    # Expand by default or restore saved state
                    should_expand = child_wid in expanded_wids or child_wid not in expanded_wids
                    child_item.setExpanded(True)
                    _add_open_children(child_wid, child_item)

            _add_open_children(self._wm._main_wid, root_item)

            # Root is always expanded
            root_item.setExpanded(True)

        # ── History section ────────────────────────────────────────────────
        history_records = [records[wid] for wid in self._wm._history if wid in records]
        if history_records:
            hist_root = QTreeWidgetItem(self._tree)
            hist_root.setText(0, "🕐 Recently Closed")
            bold = QFont()
            bold.setBold(True)
            hist_root.setFont(0, bold)
            hist_root.setData(0, Qt.ItemDataRole.UserRole, ("section", -1))

            # Most-recent first
            for rec in reversed(history_records):
                icon = _WTYPE_ICONS.get(rec.wtype, "🪟")
                h_item = QTreeWidgetItem(hist_root)
                h_item.setText(0, f"{icon} {rec.title}")
                h_item.setForeground(0, _HISTORY_BRUSH)
                h_item.setData(0, Qt.ItemDataRole.UserRole, ("history", rec.wid))

            # Restore expansion of history root
            if -1 in expanded_wids or hist_root.text(0) not in ("",):
                hist_root.setExpanded(True)

    # ----------------------------------------------------------------- events

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        kind, wid = data
        if kind == "open":
            self._wm.bring_to_front(wid)
        elif kind == "history":
            self._wm.restore_window(wid)

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        kind, wid = data
        if kind == "section":
            return

        menu = QMenu(self)
        if kind == "open":
            if wid != self._wm._main_wid:
                act_front = menu.addAction("Bring to Front")
                act_front.triggered.connect(lambda: self._wm.bring_to_front(wid))
                act_close = menu.addAction("Close")
                act_close.triggered.connect(lambda: self._wm.close_window(wid))
            else:
                act_front = menu.addAction("Bring to Front")
                act_front.triggered.connect(lambda: self._wm.bring_to_front(wid))
        elif kind == "history":
            act_restore = menu.addAction("Restore")
            act_restore.triggered.connect(lambda: self._wm.restore_window(wid))
            menu.addSeparator()
            act_remove = menu.addAction("Remove from History")
            act_remove.triggered.connect(lambda: self._wm.purge_from_history(wid))

        if menu.actions():
            menu.exec(self._tree.mapToGlobal(pos))
