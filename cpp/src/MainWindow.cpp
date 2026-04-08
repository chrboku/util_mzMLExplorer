#include "MainWindow.h"
#include "EICWindow.h"
#include "FileExplorerWindow.h"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QMimeData>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QSplitter>
#include <QTreeWidgetItem>
#include <QSettings>
#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QTimer>
#include <QStatusBar>
#include <QPushButton>
#include <QAbstractItemView>
#include <QTextStream>
#include <QFileInfo>
#include <fstream>
#include <sstream>
#include <iostream>

MzMLExplorerMainWindow::MzMLExplorerMainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("mzML Explorer");
    setMinimumSize(1024, 600);
    resize(1400, 800);
    setAcceptDrops(true);

    initUI();
    loadStylesheet();
    loadSettings();
}

MzMLExplorerMainWindow::~MzMLExplorerMainWindow() {
    saveSettings();
}

void MzMLExplorerMainWindow::createMenuBar() {
    auto* menuBar = this->menuBar();

    // File menu
    auto* fileMenu = menuBar->addMenu("&File");
    auto* loadFilesAction = fileMenu->addAction("&Load File List...");
    connect(loadFilesAction, &QAction::triggered, this, &MzMLExplorerMainWindow::loadFiles);

    auto* clearFilesAction = fileMenu->addAction("Clear &Files");
    connect(clearFilesAction, &QAction::triggered, this, &MzMLExplorerMainWindow::clearFiles);

    fileMenu->addSeparator();

    auto* loadCompoundsAction = fileMenu->addAction("&Load Compounds...");
    connect(loadCompoundsAction, &QAction::triggered, this, &MzMLExplorerMainWindow::loadCompounds);

    auto* clearCompoundsAction = fileMenu->addAction("Clear &Compounds");
    connect(clearCompoundsAction, &QAction::triggered, this, &MzMLExplorerMainWindow::clearCompounds);

    fileMenu->addSeparator();

    auto* genTemplatesAction = fileMenu->addAction("Generate &Templates...");
    connect(genTemplatesAction, &QAction::triggered, this, &MzMLExplorerMainWindow::generateTemplates);

    fileMenu->addSeparator();

    auto* exitAction = fileMenu->addAction("E&xit");
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);

    // Analysis menu
    auto* analysisMenu = menuBar->addMenu("&Analysis");
    auto* extractEICAction = analysisMenu->addAction("&Extract EICs");
    connect(extractEICAction, &QAction::triggered, this, &MzMLExplorerMainWindow::extractEICs);

    // Help menu
    auto* helpMenu = menuBar->addMenu("&Help");
    auto* aboutAction = helpMenu->addAction("&About");
    connect(aboutAction, &QAction::triggered, this, &MzMLExplorerMainWindow::showAboutDialog);
}

void MzMLExplorerMainWindow::initUI() {
    createMenuBar();

    auto* central = new QWidget(this);
    setCentralWidget(central);

    auto* mainLayout = new QVBoxLayout(central);

    auto* splitter = new QSplitter(Qt::Horizontal, central);

    // Left panel: Files table
    auto* leftPanel = new QGroupBox("Loaded Files", splitter);
    leftPanel->setAcceptDrops(true);
    auto* leftLayout = new QVBoxLayout(leftPanel);

    filesTable = new QTableWidget(this);
    filesTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    filesTable->setAcceptDrops(true);
    filesTable->setContextMenuPolicy(Qt::CustomContextMenu);
    filesTable->verticalHeader()->setDefaultSectionSize(20);
    filesTable->verticalHeader()->setMinimumSectionSize(16);
    connect(filesTable, &QTableWidget::customContextMenuRequested,
            this, &MzMLExplorerMainWindow::showFilesContextMenu);
    leftLayout->addWidget(filesTable);

    // File action buttons
    auto* fileBtnLayout = new QHBoxLayout();
    auto* loadFilesBtn = new QPushButton("Load File List");
    connect(loadFilesBtn, &QPushButton::clicked, this, &MzMLExplorerMainWindow::loadFiles);
    auto* clearFilesBtn = new QPushButton("Clear Files");
    connect(clearFilesBtn, &QPushButton::clicked, this, &MzMLExplorerMainWindow::clearFiles);
    fileBtnLayout->addWidget(loadFilesBtn);
    fileBtnLayout->addWidget(clearFilesBtn);
    leftLayout->addLayout(fileBtnLayout);

    // Right panel: Compounds tree
    auto* rightPanel = new QGroupBox("Compounds", splitter);
    auto* rightLayout = new QVBoxLayout(rightPanel);

    compoundsTree = new QTreeWidget(this);
    compoundsTree->setHeaderLabels({"Compound / Adduct", "m/z", "RT (min)"});
    compoundsTree->setContextMenuPolicy(Qt::CustomContextMenu);
    compoundsTree->setSelectionMode(QAbstractItemView::ExtendedSelection);
    compoundsTree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    connect(compoundsTree, &QTreeWidget::customContextMenuRequested,
            this, &MzMLExplorerMainWindow::showCompoundsContextMenu);
    connect(compoundsTree, &QTreeWidget::itemDoubleClicked,
            this, &MzMLExplorerMainWindow::onTreeItemDoubleClicked);
    rightLayout->addWidget(compoundsTree);

    // Compound action buttons
    auto* compBtnLayout = new QHBoxLayout();
    auto* loadCompBtn = new QPushButton("Load Compounds");
    connect(loadCompBtn, &QPushButton::clicked, this, &MzMLExplorerMainWindow::loadCompounds);
    auto* clearCompBtn = new QPushButton("Clear Compounds");
    connect(clearCompBtn, &QPushButton::clicked, this, &MzMLExplorerMainWindow::clearCompounds);
    auto* extractBtn = new QPushButton("Extract EICs");
    extractBtn->setObjectName("primaryButton");
    connect(extractBtn, &QPushButton::clicked, this, &MzMLExplorerMainWindow::extractEICs);
    compBtnLayout->addWidget(loadCompBtn);
    compBtnLayout->addWidget(clearCompBtn);
    compBtnLayout->addWidget(extractBtn);
    rightLayout->addLayout(compBtnLayout);

    splitter->addWidget(leftPanel);
    splitter->addWidget(rightPanel);
    splitter->setSizes({600, 400});

    mainLayout->addWidget(splitter);

    // Status bar
    statusLabel = new QLabel("Ready", this);
    statusBar()->addWidget(statusLabel);

    progressBar = new QProgressBar(this);
    progressBar->setVisible(false);
    progressBar->setMaximumWidth(200);
    statusBar()->addPermanentWidget(progressBar);
}

void MzMLExplorerMainWindow::loadStylesheet() {
    // Try to load style.css from the application directory
    QStringList paths = {
        QApplication::applicationDirPath() + "/style.css",
        QDir::currentPath() + "/style.css",
        QString::fromStdString(
            std::string(QStandardPaths::writableLocation(
                QStandardPaths::AppDataLocation).toStdString()) + "/style.css"),
    };

    for (const auto& p : paths) {
        QFile f(p);
        if (f.open(QFile::ReadOnly)) {
            QString css = f.readAll();
            qApp->setStyleSheet(css);
            return;
        }
    }
}

void MzMLExplorerMainWindow::loadSettings() {
    QSettings settings("mzMLExplorer", "mzMLExplorer");
    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("windowState").toByteArray());
}

void MzMLExplorerMainWindow::saveSettings() {
    QSettings settings("mzMLExplorer", "mzMLExplorer");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("windowState", saveState());
}

void MzMLExplorerMainWindow::updateFilesTable() {
    const auto& files = fileManager.getFiles();

    filesTable->setColumnCount(5);
    filesTable->setHorizontalHeaderLabels({"Filename", "Group", "Sample Type", "Sample Name", "Filepath"});
    filesTable->setRowCount((int)files.size());
    filesTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    filesTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);

    for (int i = 0; i < (int)files.size(); ++i) {
        const auto& f = files[i];
        QString color = QString::fromStdString(fileManager.getGroupColor(f.group));

        auto makeItem = [&](const QString& text) -> QTableWidgetItem* {
            auto* item = new QTableWidgetItem(text);
            item->setFlags(item->flags() & ~Qt::ItemIsEditable);
            item->setForeground(QColor(color));
            return item;
        };

        filesTable->setItem(i, 0, makeItem(QString::fromStdString(f.filename)));
        filesTable->setItem(i, 1, makeItem(QString::fromStdString(f.group)));
        filesTable->setItem(i, 2, makeItem(QString::fromStdString(f.sampleType)));
        filesTable->setItem(i, 3, makeItem(QString::fromStdString(f.sampleName)));
        filesTable->setItem(i, 4, makeItem(QString::fromStdString(f.filepath)));
    }
}

void MzMLExplorerMainWindow::updateCompoundsTree() {
    compoundsTree->clear();
    const auto& compounds = compoundManager.getCompounds();

    for (const auto& c : compounds) {
        auto* compItem = new QTreeWidgetItem(compoundsTree);
        compItem->setText(0, QString::fromStdString(c.name));

        QString rtText = QString("%1 [%2-%3]")
            .arg(c.rtMin, 0, 'f', 2)
            .arg(c.rtStartMin, 0, 'f', 2)
            .arg(c.rtEndMin, 0, 'f', 2);
        compItem->setText(2, rtText);

        // Add adduct children
        auto adducts = compoundManager.getCompoundAdducts(c.name);
        for (const auto& adduct : adducts) {
            auto* adductItem = new QTreeWidgetItem(compItem);
            adductItem->setText(0, QString::fromStdString(adduct));

            auto mz = compoundManager.calculateCompoundMz(c.name, adduct);
            if (mz.has_value()) {
                adductItem->setText(1, QString::number(mz.value(), 'f', 4));
            }
        }
    }

    compoundsTree->expandAll();
}

void MzMLExplorerMainWindow::loadFiles() {
    QString filePath = QFileDialog::getOpenFileName(
        this, "Load File List", "",
        "Excel files (*.xlsx);;TSV files (*.tsv);;CSV files (*.csv);;All Files (*)");

    if (!filePath.isEmpty()) {
        loadFilesFromPath(filePath);
    }
}

void MzMLExplorerMainWindow::loadFilesFromPath(const QString& path) {
    try {
        fileManager.loadFilesFromTSV(path.toStdString());
        updateFilesTable();
        statusLabel->setText(QString("Loaded %1 files").arg(fileManager.getFiles().size()));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error",
            QString("Failed to load file list:\n%1").arg(e.what()));
    }
}

void MzMLExplorerMainWindow::loadCompounds() {
    QString filePath = QFileDialog::getOpenFileName(
        this, "Load Compounds", "",
        "Excel files (*.xlsx);;TSV files (*.tsv);;CSV files (*.csv);;All Files (*)");

    if (!filePath.isEmpty()) {
        loadCompoundsFromPath(filePath);
    }
}

void MzMLExplorerMainWindow::loadCompoundsFromPath(const QString& path) {
    try {
        loadCompoundsFromTSV(path);
        updateCompoundsTree();
        statusLabel->setText(QString("Loaded %1 compounds")
                             .arg(compoundManager.getCompounds().size()));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error",
            QString("Failed to load compounds:\n%1").arg(e.what()));
    }
}

void MzMLExplorerMainWindow::loadCompoundsFromTSV(const QString& path) {
    std::ifstream ifs(path.toStdString());
    if (!ifs) throw std::runtime_error("Cannot open: " + path.toStdString());

    std::string line;
    std::getline(ifs, line);

    // Detect delimiter
    char delim = '\t';
    if (line.find(',') != std::string::npos && line.find('\t') == std::string::npos)
        delim = ',';

    auto splitLine = [&](const std::string& l) {
        std::vector<std::string> fields;
        std::istringstream ss(l);
        std::string f;
        while (std::getline(ss, f, delim)) {
            while (!f.empty() && (f.back() == '\r' || f.back() == '\n')) f.pop_back();
            fields.push_back(f);
        }
        return fields;
    };

    auto headers = splitLine(line);
    // Normalize headers to lowercase
    for (auto& h : headers) {
        std::transform(h.begin(), h.end(), h.begin(), ::tolower);
    }

    std::vector<std::vector<std::string>> rows;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        rows.push_back(splitLine(line));
    }

    compoundManager.loadCompounds(headers, rows);
}

void MzMLExplorerMainWindow::clearCompounds() {
    compoundManager.clearCompounds();
    compoundsTree->clear();
    statusLabel->setText("Compounds cleared");
}

void MzMLExplorerMainWindow::clearFiles() {
    fileManager.clearFiles();
    filesTable->setRowCount(0);
    statusLabel->setText("Files cleared");
}

void MzMLExplorerMainWindow::generateTemplates() {
    QString dir = QFileDialog::getExistingDirectory(this, "Select Output Directory");
    if (dir.isEmpty()) return;

    // Generate file list template
    {
        QFile f(dir + "/file_list_template.tsv");
        if (f.open(QFile::WriteOnly | QFile::Text)) {
            QTextStream out(&f);
            out << "Filepath\tFilename\tGroup\tSampleType\tSampleName\tComment\n";
            out << "/path/to/sample1.mzML\tsample1.mzML\tControl\tSample\tSample 1\t\n";
            out << "/path/to/sample2.mzML\tsample2.mzML\tTreatment\tSample\tSample 2\t\n";
        }
    }

    // Generate compounds template
    {
        QFile f(dir + "/compounds_template.tsv");
        if (f.open(QFile::WriteOnly | QFile::Text)) {
            QTextStream out(&f);
            out << "Name\tChemicalFormula\tMass\tRT_min\tRT_start_min\tRT_end_min\t"
                   "Common_adducts\tGroup\tSMILES\n";
            out << "Glucose\tC6H12O6\t\t5.0\t4.0\t6.0\t[M+H]+,[M-H]-\t\t\n";
            out << "Sucrose\tC12H22O11\t\t8.0\t7.0\t9.0\t[M+Na]+\t\t\n";
        }
    }

    // Generate adducts template
    {
        QFile f(dir + "/adducts_template.tsv");
        if (f.open(QFile::WriteOnly | QFile::Text)) {
            QTextStream out(&f);
            out << "Adduct\tCharge\tMultiplier\tElementsAdded\tElementsLost\n";
            out << "[M+H]+\t1\t1\tH\t\n";
            out << "[M-H]-\t-1\t1\t\tH\n";
            out << "[M+Na]+\t1\t1\tNa\t\n";
            out << "[M+K]+\t1\t1\tK\t\n";
            out << "[M+NH4]+\t1\t1\tNH4\t\n";
        }
    }

    QMessageBox::information(this, "Templates Generated",
        QString("Template files created in:\n%1").arg(dir));
}

void MzMLExplorerMainWindow::showAboutDialog() {
    QMessageBox::about(this, "About mzML Explorer",
        "mzML Explorer v1.0.0\n\n"
        "A tool for visualizing LC-HRMS data from mzML files.\n\n"
        "Features:\n"
        "• Load mzML files via TSV/CSV templates\n"
        "• Extract ion chromatograms (EICs)\n"
        "• Interactive plotting with zoom and pan\n"
        "• MS1 and MS/MS spectra visualization\n"
        "• Group-based color coding\n\n"
        "C++ rewrite using Qt6.\n\n"
        "(c) 2025 Plant-Microbe Metabolomics, BOKU University");
}

void MzMLExplorerMainWindow::showFilesContextMenu(const QPoint& pos) {
    QModelIndex idx = filesTable->indexAt(pos);
    if (!idx.isValid()) return;

    int row = idx.row();
    const auto& files = fileManager.getFiles();
    if (row >= (int)files.size()) return;

    QString filepath = QString::fromStdString(files[row].filepath);

    QMenu menu(this);
    auto* openExplorerAction = menu.addAction("Open File Explorer");
    connect(openExplorerAction, &QAction::triggered, [this, filepath]() {
        openFileExplorer(filepath);
    });

    auto* removeAction = menu.addAction("Remove File");
    connect(removeAction, &QAction::triggered, [this, row]() {
        // Remove file from manager (simplified)
        updateFilesTable();
    });

    menu.exec(filesTable->viewport()->mapToGlobal(pos));
}

void MzMLExplorerMainWindow::showCompoundsContextMenu(const QPoint& pos) {
    QTreeWidgetItem* item = compoundsTree->itemAt(pos);
    if (!item) return;

    QMenu menu(this);

    auto* expandAction = menu.addAction("Expand All");
    connect(expandAction, &QAction::triggered, compoundsTree, &QTreeWidget::expandAll);
    auto* collapseAction = menu.addAction("Collapse All");
    connect(collapseAction, &QAction::triggered, compoundsTree, &QTreeWidget::collapseAll);

    menu.exec(compoundsTree->viewport()->mapToGlobal(pos));
}

void MzMLExplorerMainWindow::onTreeItemDoubleClicked(QTreeWidgetItem* item, int /*column*/) {
    if (!item) return;
    // If parent is null, this is a compound node. If has parent, it's an adduct node.
    if (item->parent()) {
        // It's an adduct - could trigger EIC extraction for this specific compound+adduct
        QString compoundName = item->parent()->text(0);
        QString adduct = item->text(0);
        statusLabel->setText(QString("Selected: %1 / %2").arg(compoundName, adduct));
    }
}

void MzMLExplorerMainWindow::extractEICs() {
    if (fileManager.getFiles().empty()) {
        QMessageBox::warning(this, "No Files", "Please load mzML files first.");
        return;
    }
    if (compoundManager.isEmpty()) {
        QMessageBox::warning(this, "No Compounds", "Please load compounds first.");
        return;
    }

    auto* eicWin = new EICWindow(&fileManager, &compoundManager, this);
    eicWin->setAttribute(Qt::WA_DeleteOnClose);
    eicWin->show();
    eicWindows.push_back(eicWin);

    connect(eicWin, &QWidget::destroyed, [this, eicWin]() {
        eicWindows.erase(
            std::remove(eicWindows.begin(), eicWindows.end(), eicWin),
            eicWindows.end());
    });
}

void MzMLExplorerMainWindow::openFileExplorer(const QString& filepath) {
    auto* win = new FileExplorerWindow(filepath, &fileManager, nullptr);
    win->setAttribute(Qt::WA_DeleteOnClose);
    win->show();
    fileExplorerWindows.push_back(win);

    connect(win, &QWidget::destroyed, [this, win]() {
        fileExplorerWindows.erase(
            std::remove(fileExplorerWindows.begin(), fileExplorerWindows.end(), win),
            fileExplorerWindows.end());
    });
}

void MzMLExplorerMainWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
    }
}

void MzMLExplorerMainWindow::dropEvent(QDropEvent* event) {
    for (const auto& url : event->mimeData()->urls()) {
        QString path = url.toLocalFile();
        if (path.endsWith(".tsv", Qt::CaseInsensitive) ||
            path.endsWith(".csv", Qt::CaseInsensitive)) {
            // Try to determine if it's files or compounds based on content
            loadFilesFromPath(path);
        } else if (path.endsWith(".mzML", Qt::CaseInsensitive) ||
                   path.endsWith(".mzml.gz", Qt::CaseInsensitive)) {
            // Direct mzML file drop - add as single file
            FileEntry entry;
            entry.filepath = path.toStdString();
            entry.filename = QFileInfo(path).fileName().toStdString();
            entry.group = "Default";
            fileManager.addFile(entry);
            fileManager.regenerateGroupColors();
            updateFilesTable();
        }
    }
}

void MzMLExplorerMainWindow::closeEvent(QCloseEvent* event) {
    saveSettings();
    QMainWindow::closeEvent(event);
}
