#include "MultiAdductWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <algorithm>
#include <cmath>

MultiAdductWindow::MultiAdductWindow(const std::string& cmpName,
                                      FileManager* fm,
                                      CompoundManager* cm,
                                      QWidget* parent)
    : QWidget(parent), compoundName(cmpName), fileManager(fm), compoundManager(cm)
{
    setWindowFlags(Qt::Window | Qt::WindowCloseButtonHint |
                   Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(QString("Multi-Adduct Analysis: %1")
                   .arg(QString::fromStdString(compoundName)));
    resize(1200, 800);
    initUI();
}

void MultiAdductWindow::initUI() {
    auto* layout = new QVBoxLayout(this);

    // Compound info
    const auto* c = compoundManager->getCompoundByName(compoundName);
    QString infoText;
    if (c) {
        infoText = QString("<b>%1</b>")
            .arg(QString::fromStdString(c->name));
        if (!c->chemicalFormula.empty()) {
            infoText += QString(" | Formula: %1").arg(QString::fromStdString(c->chemicalFormula));
        }
        if (c->mass > 0) {
            infoText += QString(" | Mass: %1 Da").arg(c->mass, 0, 'f', 4);
        }
        infoText += QString(" | RT: %1 [%2-%3] min")
            .arg(c->rtMin, 0, 'f', 2)
            .arg(c->rtStartMin, 0, 'f', 2)
            .arg(c->rtEndMin, 0, 'f', 2);
    }

    compoundInfoLabel = new QLabel(infoText, this);
    compoundInfoLabel->setStyleSheet("QLabel { padding: 5px; font-size: 12px; "
                                      "background-color: #e8f0fe; border: 1px solid #dadce0; }");
    layout->addWidget(compoundInfoLabel);

    // Controls
    auto* controlsGroup = new QGroupBox("Settings", this);
    auto* controlsLayout = new QFormLayout(controlsGroup);

    mzToleranceSpin = new QDoubleSpinBox(this);
    mzToleranceSpin->setRange(0.1, 50.0);
    mzToleranceSpin->setValue(5.0);
    mzToleranceSpin->setSuffix(" ppm");
    controlsLayout->addRow("m/z Tolerance:", mzToleranceSpin);
    layout->addWidget(controlsGroup);

    auto* extractBtn = new QPushButton("Extract All Adducts", this);
    connect(extractBtn, &QPushButton::clicked, this, &MultiAdductWindow::onExtractClicked);
    layout->addWidget(extractBtn);

    // Splitter
    auto* splitter = new QSplitter(Qt::Vertical, this);

    // Chart
    chart = new QChart();
    chart->setTitle(QString("Multi-Adduct EICs: %1")
                    .arg(QString::fromStdString(compoundName)));
    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignRight);

    auto* xAxis = new QValueAxis(chart);
    xAxis->setTitleText("Retention Time (min)");
    auto* yAxis = new QValueAxis(chart);
    yAxis->setTitleText("Intensity");
    chart->addAxis(xAxis, Qt::AlignBottom);
    chart->addAxis(yAxis, Qt::AlignLeft);

    chartView = new QChartView(chart, this);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->setMinimumHeight(350);
    splitter->addWidget(chartView);

    // Results table
    resultsTable = new QTableWidget(this);
    resultsTable->setColumnCount(6);
    resultsTable->setHorizontalHeaderLabels({
        "File", "Group", "Adduct", "m/z", "Peak Area", "Max Intensity"
    });
    resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    resultsTable->setSortingEnabled(true);
    splitter->addWidget(resultsTable);

    splitter->setSizes({400, 200});
    layout->addWidget(splitter);

    // Close button
    auto* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    auto* closeBtn = new QPushButton("Close", this);
    connect(closeBtn, &QPushButton::clicked, this, &MultiAdductWindow::close);
    btnLayout->addWidget(closeBtn);
    layout->addLayout(btnLayout);
}

void MultiAdductWindow::onExtractClicked() {
    if (!compoundManager || !fileManager) return;

    chart->removeAllSeries();
    resultsTable->setRowCount(0);

    const auto* c = compoundManager->getCompoundByName(compoundName);
    if (!c) return;

    auto adducts = compoundManager->getAllAvailableAdducts();
    double tol = mzToleranceSpin->value();

    // Color palette for adducts
    auto palette = Utils::generateColorPalette((int)adducts.size());

    auto* xAxis = qobject_cast<QValueAxis*>(chart->axes(Qt::Horizontal).isEmpty() ?
                  nullptr : chart->axes(Qt::Horizontal).first());
    auto* yAxis = qobject_cast<QValueAxis*>(chart->axes(Qt::Vertical).isEmpty() ?
                  nullptr : chart->axes(Qt::Vertical).first());

    int adductIdx = 0;
    for (const auto& adduct : adducts) {
        auto mzOpt = compoundManager->calculateCompoundMz(compoundName, adduct);
        if (!mzOpt.has_value()) { adductIdx++; continue; }

        double targetMz = mzOpt.value();
        auto [mzMin, mzMax] = Utils::getMassToleranceWindow(targetMz, tol);

        QString adductColor = QString::fromStdString(palette[adductIdx % palette.size()]);

        for (const auto& f : fileManager->getFiles()) {
            LoadedFileData data;
            try {
                data = fileManager->loadSingleFile(f.filepath);
            } catch (...) { continue; }

            auto* series = new QLineSeries(chart);
            series->setName(QString("%1 / %2").arg(QString::fromStdString(f.filename))
                                              .arg(QString::fromStdString(adduct)));
            QPen pen{QColor(adductColor)};
            pen.setWidth(1);
            series->setPen(pen);

            double maxInt = 0.0;
            for (const auto& sp : data.ms1) {
                if (sp.scanTime < c->rtStartMin || sp.scanTime > c->rtEndMin) continue;
                double maxAtScan = 0.0;
                for (size_t i = 0; i < sp.mz.size(); ++i) {
                    if (sp.mz[i] >= mzMin && sp.mz[i] <= mzMax)
                        maxAtScan = std::max(maxAtScan, sp.intensity[i]);
                }
                series->append(sp.scanTime, maxAtScan);
                maxInt = std::max(maxInt, maxAtScan);
            }

            if (series->count() > 0) {
                chart->addSeries(series);
                if (xAxis) series->attachAxis(xAxis);
                if (yAxis) series->attachAxis(yAxis);

                int row = resultsTable->rowCount();
                resultsTable->insertRow(row);
                resultsTable->setItem(row, 0, new QTableWidgetItem(QString::fromStdString(f.filename)));
                resultsTable->setItem(row, 1, new QTableWidgetItem(QString::fromStdString(f.group)));
                resultsTable->setItem(row, 2, new QTableWidgetItem(QString::fromStdString(adduct)));
                resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(targetMz, 'f', 4)));
                resultsTable->setItem(row, 4, new QTableWidgetItem("-"));
                resultsTable->setItem(row, 5, new QTableWidgetItem(QString::number(maxInt, 'e', 3)));
            } else {
                delete series;
            }
        }

        adductIdx++;
    }

    // Auto-range axes
    if (xAxis) xAxis->setRange(c->rtStartMin, c->rtEndMin);
    chart->update();
}

void MultiAdductWindow::updateCompoundInfo() {
    // Could refresh compound info label if needed
}
