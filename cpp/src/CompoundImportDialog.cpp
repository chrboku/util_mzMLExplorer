#include "CompoundImportDialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QMessageBox>
#include <QSplitter>
#include <fstream>
#include <sstream>
#include <algorithm>

CompoundImportDialog::CompoundImportDialog(const QString& fp, QWidget* parent)
    : QDialog(parent), filePath(fp)
{
    setWindowTitle("Import Compounds");
    setModal(true);
    resize(900, 600);
    initUI();
    loadFile(';');
    updatePreview();
}

std::vector<std::string> CompoundImportDialog::splitLine(const std::string& line, char delim) const {
    std::vector<std::string> fields;
    std::istringstream ss(line);
    std::string f;
    while (std::getline(ss, f, delim)) {
        while (!f.empty() && (f.back() == '\r' || f.back() == '\n')) f.pop_back();
        fields.push_back(f);
    }
    return fields;
}

void CompoundImportDialog::loadFile(char delim) {
    std::ifstream ifs(filePath.toStdString());
    if (!ifs) return;

    rawHeaders.clear();
    rawRows.clear();

    std::string line;
    std::getline(ifs, line);
    rawHeaders = splitLine(line, delim);

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        rawRows.push_back(splitLine(line, delim));
    }

    // Populate column combos
    QStringList cols;
    cols << "(None)";
    for (const auto& h : rawHeaders) cols << QString::fromStdString(h);

    auto populateCombo = [&](QComboBox* combo, const QString& preferred = "") {
        combo->clear();
        combo->addItems(cols);
        if (!preferred.isEmpty()) {
            int idx = combo->findText(preferred, Qt::MatchFixedString | Qt::MatchCaseSensitive);
            if (idx >= 0) combo->setCurrentIndex(idx);
        }
    };

    populateCombo(nameColCombo,    "Name");
    populateCombo(mzColCombo,      "mz");
    populateCombo(formulaColCombo, "ChemicalFormula");
    populateCombo(rtMinColCombo,   "RT_min");
    populateCombo(rtStartColCombo, "RT_start_min");
    populateCombo(rtEndColCombo,   "RT_end_min");
    populateCombo(adductsColCombo, "Common_adducts");
    populateCombo(groupColCombo,   "Group");

    updatePreview();
}

void CompoundImportDialog::updatePreview() {
    int maxCols = (int)rawHeaders.size();
    int maxRows = std::min((int)rawRows.size(), 20);

    previewTable->setColumnCount(maxCols);
    QStringList hs;
    for (const auto& h : rawHeaders) hs << QString::fromStdString(h);
    previewTable->setHorizontalHeaderLabels(hs);
    previewTable->setRowCount(maxRows);

    for (int r = 0; r < maxRows; ++r) {
        for (int c = 0; c < maxCols && c < (int)rawRows[r].size(); ++c) {
            previewTable->setItem(r, c, new QTableWidgetItem(
                QString::fromStdString(rawRows[r][c])));
        }
    }

    statusLabel->setText(QString("Preview: %1 rows, %2 columns")
                         .arg(rawRows.size()).arg(rawHeaders.size()));
}

void CompoundImportDialog::onDelimiterChanged() {
    QString d = delimCombo->currentText();
    char delim = ';';
    if (d == ",") delim = ',';
    else if (d == "\\t (Tab)") delim = '\t';
    else if (d == "|") delim = '|';
    else if (d == ":") delim = ':';
    loadFile(delim);
}

void CompoundImportDialog::onImportClicked() {
    // Remap headers to canonical names based on combo selections
    importedHeaders = rawHeaders;
    importedRows    = rawRows;

    // Apply prefix to name column if specified
    std::string prefix = namePrefixEdit->text().toStdString();
    if (!prefix.empty()) {
        std::string nameCol = nameColCombo->currentText().toStdString();
        int nameIdx = -1;
        for (int i = 0; i < (int)importedHeaders.size(); ++i) {
            if (importedHeaders[i] == nameCol) { nameIdx = i; break; }
        }
        if (nameIdx >= 0) {
            for (auto& row : importedRows) {
                if (nameIdx < (int)row.size()) {
                    row[nameIdx] = prefix + row[nameIdx];
                }
            }
        }
    }

    accept();
}

void CompoundImportDialog::initUI() {
    auto* layout = new QVBoxLayout(this);

    // File info
    auto* fileLabel = new QLabel(QString("Importing from: %1").arg(filePath), this);
    fileLabel->setWordWrap(true);
    fileLabel->setStyleSheet("QLabel { font-weight: bold; padding: 5px; "
                              "background-color: #f0f0f0; border: 1px solid #ccc; }");
    layout->addWidget(fileLabel);

    auto* splitter = new QSplitter(Qt::Vertical, this);

    // Parameters section
    auto* paramsGroup = new QGroupBox("Import Parameters", splitter);
    auto* paramsLayout = new QFormLayout(paramsGroup);

    delimCombo = new QComboBox(this);
    delimCombo->addItems({";", ",", "\\t (Tab)", "|", ":"});
    connect(delimCombo, &QComboBox::currentTextChanged,
            this, &CompoundImportDialog::onDelimiterChanged);
    paramsLayout->addRow("Delimiter:", delimCombo);

    namePrefixEdit = new QLineEdit(this);
    namePrefixEdit->setPlaceholderText("Optional prefix for compound names");
    paramsLayout->addRow("Name Prefix:", namePrefixEdit);

    nameColCombo    = new QComboBox(this);
    mzColCombo      = new QComboBox(this);
    formulaColCombo = new QComboBox(this);
    rtMinColCombo   = new QComboBox(this);
    rtStartColCombo = new QComboBox(this);
    rtEndColCombo   = new QComboBox(this);
    adductsColCombo = new QComboBox(this);
    groupColCombo   = new QComboBox(this);

    paramsLayout->addRow("Name Column:",          nameColCombo);
    paramsLayout->addRow("m/z Column:",           mzColCombo);
    paramsLayout->addRow("Formula Column:",       formulaColCombo);
    paramsLayout->addRow("RT Center Column:",     rtMinColCombo);
    paramsLayout->addRow("RT Start Column:",      rtStartColCombo);
    paramsLayout->addRow("RT End Column:",        rtEndColCombo);
    paramsLayout->addRow("Adducts Column:",       adductsColCombo);
    paramsLayout->addRow("Group Column:",         groupColCombo);

    splitter->addWidget(paramsGroup);

    // Preview section
    auto* previewGroup = new QGroupBox("Preview", splitter);
    auto* previewLayout = new QVBoxLayout(previewGroup);
    previewTable = new QTableWidget(this);
    previewTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    previewLayout->addWidget(previewTable);
    splitter->addWidget(previewGroup);

    splitter->setSizes({250, 300});
    layout->addWidget(splitter);

    statusLabel = new QLabel("", this);
    layout->addWidget(statusLabel);

    // Buttons
    auto* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();

    auto* cancelBtn = new QPushButton("Cancel", this);
    connect(cancelBtn, &QPushButton::clicked, this, &CompoundImportDialog::reject);
    btnLayout->addWidget(cancelBtn);

    auto* importBtn = new QPushButton("Import", this);
    importBtn->setDefault(true);
    connect(importBtn, &QPushButton::clicked, this, &CompoundImportDialog::onImportClicked);
    btnLayout->addWidget(importBtn);

    layout->addLayout(btnLayout);
}
