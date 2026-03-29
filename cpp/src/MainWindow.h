#pragma once
#include <QMainWindow>
#include <QTableWidget>
#include <QTreeWidget>
#include <QSplitter>
#include <QLabel>
#include <QProgressBar>
#include <QMenu>
#include <QAction>
#include <memory>
#include "FileManager.h"
#include "CompoundManager.h"

class EICWindow;
class FileExplorerWindow;

/**
 * MzMLExplorerMainWindow - Main application window.
 *
 * Provides:
 * - File list management (load, clear, drag & drop)
 * - Compound tree management
 * - EIC extraction launch
 * - Context menus for files and compounds
 */
class MzMLExplorerMainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MzMLExplorerMainWindow(QWidget* parent = nullptr);
    ~MzMLExplorerMainWindow() override;

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private slots:
    void loadFiles();
    void loadCompounds();
    void clearCompounds();
    void clearFiles();
    void generateTemplates();
    void showAboutDialog();
    void showFilesContextMenu(const QPoint& pos);
    void showCompoundsContextMenu(const QPoint& pos);
    void onTreeItemDoubleClicked(QTreeWidgetItem* item, int column);
    void extractEICs();
    void openFileExplorer(const QString& filepath);

private:
    void initUI();
    void createMenuBar();
    void updateFilesTable();
    void updateCompoundsTree();
    void loadFilesFromPath(const QString& path);
    void loadCompoundsFromPath(const QString& path);
    void loadCompoundsFromTSV(const QString& path);
    void loadStylesheet();
    void loadSettings();
    void saveSettings();

    // Data managers
    FileManager fileManager;
    CompoundManager compoundManager;

    // UI components
    QTableWidget* filesTable = nullptr;
    QTreeWidget* compoundsTree = nullptr;
    QLabel* statusLabel = nullptr;
    QProgressBar* progressBar = nullptr;

    // Kept-alive child windows
    std::vector<QWidget*> eicWindows;
    std::vector<QWidget*> fileExplorerWindows;
};
