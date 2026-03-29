#include "FileExplorerWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QMessageBox>
#include <QSplitter>
#include <QMouseEvent>
#include <QFileInfo>
#include <algorithm>
#include <cmath>
#include <iostream>

FileExplorerWindow::FileExplorerWindow(const QString& fp, FileManager* fm, QWidget* parent)
    : QWidget(parent), filepath(fp), fileManager(fm)
{
    setWindowFlags(Qt::Window | Qt::WindowCloseButtonHint |
                   Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(QString("File Explorer: %1").arg(QFileInfo(fp).fileName()));
    resize(1400, 900);
    initUI();
    loadAndPlotTIC();
}

void FileExplorerWindow::initUI() {
    auto* layout = new QVBoxLayout(this);

    // Controls
    auto* controlsGroup = new QGroupBox("Navigation", this);
    auto* controlsLayout = new QFormLayout(controlsGroup);

    rtSpin = new QDoubleSpinBox(this);
    rtSpin->setRange(0.0, 100.0);
    rtSpin->setValue(5.0);
    rtSpin->setSuffix(" min");
    rtSpin->setSingleStep(0.1);
    rtSpin->setDecimals(3);
    controlsLayout->addRow("Retention Time:", rtSpin);

    rtWindowSpin = new QDoubleSpinBox(this);
    rtWindowSpin->setRange(0.001, 5.0);
    rtWindowSpin->setValue(0.05);
    rtWindowSpin->setSuffix(" min");
    rtWindowSpin->setSingleStep(0.01);
    rtWindowSpin->setDecimals(3);
    controlsLayout->addRow("RT Window (±):", rtWindowSpin);

    msLevelCombo = new QComboBox(this);
    msLevelCombo->addItems({"MS1", "MS2"});
    controlsLayout->addRow("MS Level:", msLevelCombo);

    auto* goBtn = new QPushButton("Go", this);
    connect(goBtn, &QPushButton::clicked, this, &FileExplorerWindow::onRtSpinChanged);
    controlsLayout->addRow("", goBtn);

    layout->addWidget(controlsGroup);

    statusLabel = new QLabel("Loading...", this);
    layout->addWidget(statusLabel);

    // Main splitter: TIC on left, spectrum on right
    auto* mainSplitter = new QSplitter(Qt::Horizontal, this);

    // TIC chart
    ticChart = new QChart();
    ticChart->setTitle("Total Ion Chromatogram");
    ticChart->legend()->hide();

    ticXAxis = new QValueAxis(ticChart);
    ticXAxis->setTitleText("RT (min)");
    ticYAxis = new QValueAxis(ticChart);
    ticYAxis->setTitleText("Intensity");
    ticChart->addAxis(ticXAxis, Qt::AlignBottom);
    ticChart->addAxis(ticYAxis, Qt::AlignLeft);

    ticChartView = new QChartView(ticChart, this);
    ticChartView->setRenderHint(QPainter::Antialiasing);
    ticChartView->setMinimumWidth(500);
    mainSplitter->addWidget(ticChartView);

    // Right side: spectrum chart + table
    auto* rightSplitter = new QSplitter(Qt::Vertical, this);

    spectrumChart = new QChart();
    spectrumChart->setTitle("Spectrum");
    spectrumChart->legend()->hide();

    specXAxis = new QValueAxis(spectrumChart);
    specXAxis->setTitleText("m/z");
    specYAxis = new QValueAxis(spectrumChart);
    specYAxis->setTitleText("Intensity");
    spectrumChart->addAxis(specXAxis, Qt::AlignBottom);
    spectrumChart->addAxis(specYAxis, Qt::AlignLeft);

    spectrumChartView = new QChartView(spectrumChart, this);
    spectrumChartView->setRenderHint(QPainter::Antialiasing);
    spectrumChartView->setMinimumHeight(300);
    rightSplitter->addWidget(spectrumChartView);

    spectrumTable = new QTableWidget(this);
    spectrumTable->setColumnCount(2);
    spectrumTable->setHorizontalHeaderLabels({"m/z", "Intensity"});
    spectrumTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    spectrumTable->setSortingEnabled(true);
    rightSplitter->addWidget(spectrumTable);

    rightSplitter->setSizes({350, 200});
    mainSplitter->addWidget(rightSplitter);
    mainSplitter->setSizes({600, 700});

    layout->addWidget(mainSplitter);

    // Close button
    auto* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    auto* closeBtn = new QPushButton("Close", this);
    connect(closeBtn, &QPushButton::clicked, this, &FileExplorerWindow::close);
    btnLayout->addWidget(closeBtn);
    layout->addLayout(btnLayout);
}

void FileExplorerWindow::loadAndPlotTIC() {
    try {
        fileData = fileManager->loadSingleFile(filepath.toStdString());
        dataLoaded = true;
        statusLabel->setText(QString("Loaded: %1 MS1 + %2 MS2 spectra")
                             .arg(fileData.ms1.size())
                             .arg(fileData.ms2.size()));
    } catch (const std::exception& e) {
        statusLabel->setText(QString("Error: %1").arg(e.what()));
        return;
    }

    // Build TIC from MS1
    if (fileData.ms1.empty()) return;

    auto* ticSeries = new QLineSeries(ticChart);
    QPen pen(QColor("#1f77b4"));
    pen.setWidth(1);
    ticSeries->setPen(pen);

    double maxTIC = 0.0;
    double minRT = fileData.ms1.front().scanTime;
    double maxRT = fileData.ms1.back().scanTime;

    for (const auto& sp : fileData.ms1) {
        double tic = 0.0;
        for (double v : sp.intensity) tic += v;
        ticSeries->append(sp.scanTime, tic);
        maxTIC = std::max(maxTIC, tic);
        minRT = std::min(minRT, sp.scanTime);
        maxRT = std::max(maxRT, sp.scanTime);
    }

    ticChart->addSeries(ticSeries);
    ticSeries->attachAxis(ticXAxis);
    ticSeries->attachAxis(ticYAxis);

    ticXAxis->setRange(minRT, maxRT);
    ticYAxis->setRange(0, maxTIC * 1.1);

    // Set RT spin range
    rtSpin->setRange(minRT, maxRT);
    rtSpin->setValue((minRT + maxRT) / 2.0);

    // Show spectrum at center RT
    double centerRT = (minRT + maxRT) / 2.0;
    updateMS1Spectrum(centerRT - 0.05, centerRT + 0.05);
}

void FileExplorerWindow::onTICPointSelected(double rtValue) {
    rtSpin->setValue(rtValue);
    onRtSpinChanged();
}

void FileExplorerWindow::onRtSpinChanged() {
    double rt = rtSpin->value();
    double window = rtWindowSpin->value();
    if (msLevelCombo->currentText() == "MS1") {
        updateMS1Spectrum(rt - window, rt + window);
    } else {
        updateMS2Spectrum(rt - window, rt + window);
    }
}

void FileExplorerWindow::onRtWindowChanged() {
    onRtSpinChanged();
}

void FileExplorerWindow::updateMS1Spectrum(double rtMin, double rtMax) {
    if (!dataLoaded || fileData.ms1.empty()) return;

    // Find the closest MS1 spectrum to the center RT
    double centerRT = (rtMin + rtMax) / 2.0;
    const MS1Spectrum* closest = nullptr;
    double minDist = 1e9;

    for (const auto& sp : fileData.ms1) {
        double dist = std::abs(sp.scanTime - centerRT);
        if (dist < minDist) {
            minDist = dist;
            closest = &sp;
        }
    }

    if (!closest) return;

    // Update spectrum chart
    spectrumChart->removeAllSeries();
    auto* series = new QLineSeries(spectrumChart);
    QPen pen(QColor("#1f77b4"));
    pen.setWidth(1);
    series->setPen(pen);

    double maxMz = 0, minMz = 1e9;
    double maxInt = 0;

    for (size_t i = 0; i < closest->mz.size(); ++i) {
        series->append(closest->mz[i], 0.0);
        series->append(closest->mz[i], closest->intensity[i]);
        series->append(closest->mz[i], 0.0);
        maxMz = std::max(maxMz, closest->mz[i]);
        minMz = std::min(minMz, closest->mz[i]);
        maxInt = std::max(maxInt, closest->intensity[i]);
    }

    spectrumChart->addSeries(series);
    series->attachAxis(specXAxis);
    series->attachAxis(specYAxis);

    if (!closest->mz.empty()) {
        specXAxis->setRange(minMz - 1, maxMz + 1);
        specYAxis->setRange(0, maxInt * 1.1);
    }

    spectrumChart->setTitle(QString("MS1 Spectrum @ RT: %1 min")
                             .arg(closest->scanTime, 0, 'f', 3));

    // Update table
    spectrumTable->setSortingEnabled(false);
    spectrumTable->setRowCount((int)closest->mz.size());
    for (int i = 0; i < (int)closest->mz.size(); ++i) {
        spectrumTable->setItem(i, 0, new QTableWidgetItem(
            QString::number(closest->mz[i], 'f', 4)));
        spectrumTable->setItem(i, 1, new QTableWidgetItem(
            QString::number(closest->intensity[i], 'e', 3)));
    }
    spectrumTable->setSortingEnabled(true);

    statusLabel->setText(QString("Showing MS1 spectrum at RT: %1 min (%2 peaks)")
                         .arg(closest->scanTime, 0, 'f', 3)
                         .arg(closest->mz.size()));
}

void FileExplorerWindow::updateMS2Spectrum(double rtMin, double rtMax) {
    if (!dataLoaded || fileData.ms2.empty()) return;

    double centerRT = (rtMin + rtMax) / 2.0;
    const MS2Spectrum* closest = nullptr;
    double minDist = 1e9;

    for (const auto& sp : fileData.ms2) {
        double dist = std::abs(sp.scanTime - centerRT);
        if (dist < minDist) {
            minDist = dist;
            closest = &sp;
        }
    }

    if (!closest) return;

    spectrumChart->removeAllSeries();
    auto* series = new QLineSeries(spectrumChart);
    QPen pen(QColor("#ff7f0e"));
    pen.setWidth(1);
    series->setPen(pen);

    double maxMz = 0, minMz = 1e9, maxInt = 0;
    for (size_t i = 0; i < closest->mz.size(); ++i) {
        series->append(closest->mz[i], 0.0);
        series->append(closest->mz[i], closest->intensity[i]);
        series->append(closest->mz[i], 0.0);
        maxMz = std::max(maxMz, closest->mz[i]);
        minMz = std::min(minMz, closest->mz[i]);
        maxInt = std::max(maxInt, closest->intensity[i]);
    }

    spectrumChart->addSeries(series);
    series->attachAxis(specXAxis);
    series->attachAxis(specYAxis);

    if (!closest->mz.empty()) {
        specXAxis->setRange(minMz - 1, maxMz + 1);
        specYAxis->setRange(0, maxInt * 1.1);
    }

    spectrumChart->setTitle(
        QString("MS2 Spectrum @ RT: %1 min | Precursor: %2")
        .arg(closest->scanTime, 0, 'f', 3)
        .arg(closest->precursorMz, 0, 'f', 4));

    // Table
    spectrumTable->setSortingEnabled(false);
    spectrumTable->setRowCount((int)closest->mz.size());
    for (int i = 0; i < (int)closest->mz.size(); ++i) {
        spectrumTable->setItem(i, 0, new QTableWidgetItem(
            QString::number(closest->mz[i], 'f', 4)));
        spectrumTable->setItem(i, 1, new QTableWidgetItem(
            QString::number(closest->intensity[i], 'e', 3)));
    }
    spectrumTable->setSortingEnabled(true);

    statusLabel->setText(
        QString("Showing MS2 spectrum at RT: %1 min (precursor m/z: %2, %3 peaks)")
        .arg(closest->scanTime, 0, 'f', 3)
        .arg(closest->precursorMz, 0, 'f', 4)
        .arg(closest->mz.size()));
}
