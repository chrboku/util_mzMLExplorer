#include "EICWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QMessageBox>
#include <QSplitter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QValueAxis>
#include <QtMath>
#include <algorithm>
#include <numeric>
#include <iostream>

// ---------------------------------------------------------------------------
// EICExtractionWorker
// ---------------------------------------------------------------------------
EICExtractionWorker::EICExtractionWorker(FileManager* fm, CompoundManager* cm,
                                          double tol, QObject* parent)
    : QThread(parent), fileManager(fm), compoundManager(cm), mzTolerance(tol)
{}

void EICExtractionWorker::run() {
    const auto& files = fileManager->getFiles();
    const auto& compounds = compoundManager->getCompounds();
    int total = (int)(files.size() * compounds.size());
    int current = 0;

    for (const auto& f : files) {
        LoadedFileData data;
        try {
            data = fileManager->loadSingleFile(f.filepath);
        } catch (const std::exception& e) {
            emit error(QString("Error loading %1: %2")
                       .arg(QString::fromStdString(f.filename))
                       .arg(e.what()));
            continue;
        }

        emit progress(current, total, QString::fromStdString(f.filename));

        for (const auto& c : compounds) {
            auto adducts = compoundManager->getCompoundAdducts(c.name);
            for (const auto& adduct : adducts) {
                auto mzOpt = compoundManager->calculateCompoundMz(c.name, adduct);
                if (!mzOpt.has_value()) continue;

                double targetMz = mzOpt.value();
                double rtStart = c.rtStartMin;
                double rtEnd   = c.rtEndMin;

                auto [mzMin, mzMax] = Utils::getMassToleranceWindow(targetMz, mzTolerance);

                EICResult result;
                result.filepath  = f.filepath;
                result.filename  = f.filename;
                result.group     = f.group;
                result.color     = fileManager->getGroupColor(f.group);
                result.compoundName = c.name;
                result.adduct    = adduct;
                result.targetMz  = targetMz;
                result.rtCenter  = c.rtMin;
                result.rtStart   = rtStart;
                result.rtEnd     = rtEnd;

                // Extract EIC from MS1 data
                for (const auto& sp : data.ms1) {
                    if (sp.scanTime < rtStart || sp.scanTime > rtEnd) continue;

                    double maxInt = 0.0;
                    for (size_t i = 0; i < sp.mz.size(); ++i) {
                        if (sp.mz[i] >= mzMin && sp.mz[i] <= mzMax) {
                            maxInt = std::max(maxInt, sp.intensity[i]);
                        }
                    }

                    result.rtValues.push_back(sp.scanTime);
                    result.intensityValues.push_back(maxInt);
                }

                // Calculate peak metrics
                if (!result.intensityValues.empty()) {
                    auto maxIt = std::max_element(result.intensityValues.begin(),
                                                  result.intensityValues.end());
                    result.peakMaxIntensity = *maxIt;
                    int maxIdx = (int)(maxIt - result.intensityValues.begin());
                    result.peakRt = result.rtValues[maxIdx];

                    // Simple trapezoid integration
                    double area = 0.0;
                    for (size_t i = 1; i < result.rtValues.size(); ++i) {
                        double dt = (result.rtValues[i] - result.rtValues[i-1]) * 60.0; // s
                        area += 0.5 * (result.intensityValues[i] + result.intensityValues[i-1]) * dt;
                    }
                    result.peakArea = area;
                }

                emit resultReady(result);
                current++;
            }
        }
    }

    emit finished(QString("Extraction complete. Processed %1 spectra.").arg(current));
}

// ---------------------------------------------------------------------------
// InteractiveChartView
// ---------------------------------------------------------------------------
InteractiveChartView::InteractiveChartView(QChart* chart, QWidget* parent)
    : QChartView(chart, parent)
{
    setRenderHint(QPainter::Antialiasing);
    setRubberBand(QChartView::NoRubberBand);
    setMouseTracking(true);
}

void InteractiveChartView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = true;
        panStartPos = event->position();
        auto* xAx = chart()->axes(Qt::Horizontal).isEmpty() ?
                    nullptr : qobject_cast<QValueAxis*>(chart()->axes(Qt::Horizontal).first());
        auto* yAx = chart()->axes(Qt::Vertical).isEmpty() ?
                    nullptr : qobject_cast<QValueAxis*>(chart()->axes(Qt::Vertical).first());
        if (xAx) rangeX = std::make_pair(xAx->min(), xAx->max());
        if (yAx) rangeY = std::make_pair(yAx->min(), yAx->max());
        setCursor(Qt::ClosedHandCursor);
    }
    QChartView::mousePressEvent(event);
}

void InteractiveChartView::mouseMoveEvent(QMouseEvent* event) {
    if (isPanning) {
        QPointF delta = event->position() - panStartPos;
        auto* xAx = chart()->axes(Qt::Horizontal).isEmpty() ?
                    nullptr : qobject_cast<QValueAxis*>(chart()->axes(Qt::Horizontal).first());
        auto* yAx = chart()->axes(Qt::Vertical).isEmpty() ?
                    nullptr : qobject_cast<QValueAxis*>(chart()->axes(Qt::Vertical).first());

        if (xAx) {
            QRectF plotArea = chart()->plotArea();
            double xScale = (rangeX.second - rangeX.first) / plotArea.width();
            double xShift = -delta.x() * xScale;
            xAx->setRange(rangeX.first + xShift, rangeX.second + xShift);
        }
        if (yAx) {
            QRectF plotArea = chart()->plotArea();
            double yScale = (rangeY.second - rangeY.first) / plotArea.height();
            double yShift = delta.y() * yScale;
            yAx->setRange(rangeY.first + yShift, rangeY.second + yShift);
        }
    }
    QChartView::mouseMoveEvent(event);
}

void InteractiveChartView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = false;
        setCursor(Qt::ArrowCursor);
    }
    QChartView::mouseReleaseEvent(event);
}

void InteractiveChartView::wheelEvent(QWheelEvent* event) {
    auto* xAx = chart()->axes(Qt::Horizontal).isEmpty() ?
                nullptr : qobject_cast<QValueAxis*>(chart()->axes(Qt::Horizontal).first());
    auto* yAx = chart()->axes(Qt::Vertical).isEmpty() ?
                nullptr : qobject_cast<QValueAxis*>(chart()->axes(Qt::Vertical).first());

    if (!xAx) { QChartView::wheelEvent(event); return; }

    double factor = event->angleDelta().y() > 0 ? 0.8 : 1.25;
    QPointF mousePos = event->position();
    QPointF chartVal = chart()->mapToValue(mousePos);

    double xMin = xAx->min(), xMax = xAx->max();
    double xRange = xMax - xMin;
    double xFrac = (chartVal.x() - xMin) / xRange;

    double newRange = xRange * factor;
    xAx->setRange(chartVal.x() - xFrac * newRange,
                  chartVal.x() + (1.0 - xFrac) * newRange);

    if (yAx) {
        double yMin = yAx->min(), yMax = yAx->max();
        double yRange = yMax - yMin;
        double yFrac = (chartVal.y() - yMin) / yRange;
        double newYRange = yRange * factor;
        yAx->setRange(chartVal.y() - yFrac * newYRange,
                      chartVal.y() + (1.0 - yFrac) * newYRange);
    }

    event->accept();
}

// ---------------------------------------------------------------------------
// EICWindow
// ---------------------------------------------------------------------------
EICWindow::EICWindow(FileManager* fm, CompoundManager* cm, QWidget* parent)
    : QWidget(parent), fileManager(fm), compoundManager(cm)
{
    setWindowFlags(Qt::Window | Qt::WindowCloseButtonHint |
                   Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle("EIC Extraction");
    setMinimumSize(1000, 700);
    resize(1400, 800);
    initUI();
}

void EICWindow::initUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // Top controls
    auto* controlsGroup = new QGroupBox("Extraction Settings", this);
    auto* controlsLayout = new QFormLayout(controlsGroup);

    mzToleranceSpin = new QDoubleSpinBox(this);
    mzToleranceSpin->setRange(0.1, 50.0);
    mzToleranceSpin->setValue(5.0);
    mzToleranceSpin->setSuffix(" ppm");
    mzToleranceSpin->setDecimals(1);
    controlsLayout->addRow("m/z Tolerance:", mzToleranceSpin);

    auto* extractBtn = new QPushButton("Extract EICs", this);
    extractBtn->setObjectName("primaryButton");
    connect(extractBtn, &QPushButton::clicked, this, &EICWindow::onExtractClicked);
    controlsLayout->addRow("", extractBtn);

    mainLayout->addWidget(controlsGroup);

    // Status / progress
    statusLabel = new QLabel("Ready", this);
    progressBar = new QProgressBar(this);
    progressBar->setVisible(false);
    mainLayout->addWidget(statusLabel);
    mainLayout->addWidget(progressBar);

    // Splitter: chart + table
    auto* splitter = new QSplitter(Qt::Vertical, this);

    // Chart
    chart = new QChart();
    chart->setTitle("Extracted Ion Chromatograms");
    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignRight);

    xAxis = new QValueAxis(chart);
    xAxis->setTitleText("Retention Time (min)");
    chart->addAxis(xAxis, Qt::AlignBottom);

    yAxis = new QValueAxis(chart);
    yAxis->setTitleText("Intensity");
    chart->addAxis(yAxis, Qt::AlignLeft);

    chartView = new InteractiveChartView(chart, this);
    chartView->setMinimumHeight(350);
    splitter->addWidget(chartView);

    // Results table
    resultsTable = new QTableWidget(this);
    resultsTable->setColumnCount(7);
    resultsTable->setHorizontalHeaderLabels({
        "Filename", "Group", "Compound", "Adduct",
        "m/z", "Peak Area", "Max Intensity"
    });
    resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    resultsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    resultsTable->setSortingEnabled(true);
    splitter->addWidget(resultsTable);

    splitter->setSizes({450, 250});
    mainLayout->addWidget(splitter);

    // Close button
    auto* closeBtnLayout = new QHBoxLayout();
    closeBtnLayout->addStretch();
    auto* closeBtn = new QPushButton("Close", this);
    connect(closeBtn, &QPushButton::clicked, this, &EICWindow::close);
    closeBtnLayout->addWidget(closeBtn);
    mainLayout->addLayout(closeBtnLayout);
}

void EICWindow::onExtractClicked() {
    if (worker && worker->isRunning()) {
        QMessageBox::information(this, "Running",
            "Extraction already in progress.");
        return;
    }

    clearResults();

    double tol = mzToleranceSpin->value();

    worker = new EICExtractionWorker(fileManager, compoundManager, tol, this);
    connect(worker, &EICExtractionWorker::resultReady,
            this, &EICWindow::onResultReady);
    connect(worker, &EICExtractionWorker::finished,
            this, &EICWindow::onExtractionFinished);
    connect(worker, &EICExtractionWorker::progress,
            this, &EICWindow::onProgressUpdate);
    connect(worker, &QThread::finished, worker, &QObject::deleteLater);

    progressBar->setRange(0, 0);
    progressBar->setVisible(true);
    statusLabel->setText("Extracting EICs...");

    worker->start();
}

void EICWindow::onResultReady(const EICResult& result) {
    results.push_back(result);

    int row = resultsTable->rowCount();
    resultsTable->insertRow(row);
    addResultToTable(result, row);
    addSeriesToChart(result);
}

void EICWindow::addResultToTable(const EICResult& result, int row) {
    QColor color(QString::fromStdString(result.color));

    auto makeItem = [&](const QString& text) -> QTableWidgetItem* {
        auto* item = new QTableWidgetItem(text);
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        item->setForeground(color);
        return item;
    };

    resultsTable->setItem(row, 0, makeItem(QString::fromStdString(result.filename)));
    resultsTable->setItem(row, 1, makeItem(QString::fromStdString(result.group)));
    resultsTable->setItem(row, 2, makeItem(QString::fromStdString(result.compoundName)));
    resultsTable->setItem(row, 3, makeItem(QString::fromStdString(result.adduct)));
    resultsTable->setItem(row, 4, makeItem(QString::number(result.targetMz, 'f', 4)));
    resultsTable->setItem(row, 5, makeItem(QString::number(result.peakArea, 'e', 3)));
    resultsTable->setItem(row, 6, makeItem(QString::number(result.peakMaxIntensity, 'e', 3)));
}

void EICWindow::addSeriesToChart(const EICResult& result) {
    if (result.rtValues.empty()) return;

    auto* series = new QLineSeries(chart);
    series->setName(QString::fromStdString(result.filename));

    QColor color(QString::fromStdString(result.color));
    QPen pen(color);
    pen.setWidth(1);
    series->setPen(pen);

    for (size_t i = 0; i < result.rtValues.size(); ++i) {
        series->append(result.rtValues[i], result.intensityValues[i]);
    }

    chart->addSeries(series);
    series->attachAxis(xAxis);
    series->attachAxis(yAxis);

    // Update axis ranges
    if (chart->series().size() == 1) {
        xAxis->setRange(result.rtStart, result.rtEnd);
        yAxis->setRange(0, result.peakMaxIntensity * 1.1);
    } else {
        double curMax = yAxis->max();
        if (result.peakMaxIntensity * 1.1 > curMax) {
            yAxis->setRange(0, result.peakMaxIntensity * 1.1);
        }
    }
}

void EICWindow::onExtractionFinished(const QString& message) {
    progressBar->setVisible(false);
    statusLabel->setText(message);
}

void EICWindow::onProgressUpdate(int current, int total, const QString& filename) {
    if (total > 0) {
        progressBar->setRange(0, total);
        progressBar->setValue(current);
    }
    statusLabel->setText(QString("Processing: %1 (%2/%3)").arg(filename).arg(current).arg(total));
}

void EICWindow::clearResults() {
    results.clear();
    resultsTable->setRowCount(0);
    chart->removeAllSeries();
}

void EICWindow::updatePlot() {
    // Triggered when user changes compound/adduct filter
}
