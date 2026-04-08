#include "MSMSWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QPushButton>
#include <QHeaderView>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QValueAxis>
#include <algorithm>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// InteractiveMSMSChartView
// ---------------------------------------------------------------------------
InteractiveMSMSChartView::InteractiveMSMSChartView(QChart* chart, QWidget* parent)
    : QChartView(chart, parent)
{
    setRenderHint(QPainter::Antialiasing);
    setRubberBand(QChartView::NoRubberBand);
}

void InteractiveMSMSChartView::wheelEvent(QWheelEvent* event) {
    auto axes_x = chart()->axes(Qt::Horizontal);
    if (axes_x.isEmpty()) { QChartView::wheelEvent(event); return; }
    auto* xAx = qobject_cast<QValueAxis*>(axes_x.first());
    if (!xAx) { QChartView::wheelEvent(event); return; }
    double factor = event->angleDelta().y() > 0 ? 0.8 : 1.25;
    QPointF val = chart()->mapToValue(event->position());
    double xMin = xAx->min(), xMax = xAx->max();
    double frac = (val.x() - xMin) / (xMax - xMin);
    double newRange = (xMax - xMin) * factor;
    xAx->setRange(val.x() - frac * newRange, val.x() + (1.0 - frac) * newRange);
    event->accept();
}

void InteractiveMSMSChartView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = true;
        panStart = event->position();
        auto axX = chart()->axes(Qt::Horizontal);
        auto axY = chart()->axes(Qt::Vertical);
        if (!axX.isEmpty()) {
            auto* ax = qobject_cast<QValueAxis*>(axX.first());
            if (ax) { xMin = ax->min(); xMax = ax->max(); }
        }
        if (!axY.isEmpty()) {
            auto* ay = qobject_cast<QValueAxis*>(axY.first());
            if (ay) { yMin = ay->min(); yMax = ay->max(); }
        }
        setCursor(Qt::ClosedHandCursor);
    }
    QChartView::mousePressEvent(event);
}

void InteractiveMSMSChartView::mouseMoveEvent(QMouseEvent* event) {
    if (isPanning) {
        QPointF delta = event->position() - panStart;
        auto axX = chart()->axes(Qt::Horizontal);
        auto axY = chart()->axes(Qt::Vertical);
        QRectF pa = chart()->plotArea();
        if (!axX.isEmpty()) {
            auto* ax = qobject_cast<QValueAxis*>(axX.first());
            if (ax) {
                double s = (xMax - xMin) / pa.width();
                ax->setRange(xMin - delta.x() * s, xMax - delta.x() * s);
            }
        }
        if (!axY.isEmpty()) {
            auto* ay = qobject_cast<QValueAxis*>(axY.first());
            if (ay) {
                double s = (yMax - yMin) / pa.height();
                ay->setRange(yMin + delta.y() * s, yMax + delta.y() * s);
            }
        }
    }
    QChartView::mouseMoveEvent(event);
}

void InteractiveMSMSChartView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = false;
        setCursor(Qt::ArrowCursor);
    }
    QChartView::mouseReleaseEvent(event);
}

// ---------------------------------------------------------------------------
// MSMSViewerWindow
// ---------------------------------------------------------------------------
MSMSViewerWindow::MSMSViewerWindow(
    const std::vector<MSMSSpectrum>& sp,
    double tgtMz, double rtCtr,
    const std::string& cmpName, const std::string& addt,
    QWidget* parent)
    : QWidget(parent), spectra(sp), targetMz(tgtMz), rtCenter(rtCtr),
      compoundName(cmpName), adduct(addt)
{
    setWindowFlags(Qt::Window | Qt::WindowCloseButtonHint |
                   Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(QString("MS/MS Spectra: %1 - %2")
                   .arg(QString::fromStdString(compoundName))
                   .arg(QString::fromStdString(adduct)));
    resize(1200, 800);
    initUI();
}

void MSMSViewerWindow::initUI() {
    auto* layout = new QVBoxLayout(this);

    // Header
    auto* header = new QLabel(
        QString("<b>%1 (%2)</b> | Precursor m/z: %3 | RT: %4 min | Spectra: %5")
        .arg(QString::fromStdString(compoundName))
        .arg(QString::fromStdString(adduct))
        .arg(targetMz, 0, 'f', 4)
        .arg(rtCenter, 0, 'f', 2)
        .arg((int)spectra.size()),
        this);
    header->setStyleSheet("QLabel { padding: 5px; font-size: 12px; }");
    layout->addWidget(header);

    // Similarity table
    similarityTable = new QTableWidget(this);
    similarityTable->setMaximumHeight(200);
    similarityTable->setColumnCount(4);
    similarityTable->setHorizontalHeaderLabels({
        "File 1", "File 2", "Cosine Similarity", "Matched Peaks"
    });
    similarityTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    populateSimilarityTable();
    layout->addWidget(new QLabel("Pairwise Cosine Similarities:", this));
    layout->addWidget(similarityTable);

    // Individual spectrum charts
    auto* scrollArea = new QScrollArea(this);
    auto* scrollWidget = new QWidget(scrollArea);
    scrollArea->setWidget(scrollWidget);
    scrollArea->setWidgetResizable(true);

    auto* scrollLayout = new QVBoxLayout(scrollWidget);

    for (const auto& sp : spectra) {
        // Per-spectrum chart
        auto* chart = new QChart();
        chart->setTitle(QString("%1 | RT: %2 min | Precursor m/z: %3")
                        .arg(QString::fromStdString(sp.filename))
                        .arg(sp.scanTime, 0, 'f', 2)
                        .arg(sp.precursorMz, 0, 'f', 4));
        chart->legend()->hide();

        auto* series = new QLineSeries(chart);
        QPen pen(QColor(QString::fromStdString(sp.color)));
        pen.setWidth(1);
        series->setPen(pen);

        double maxInt = 0.0;
        for (size_t i = 0; i < sp.mz.size(); ++i) {
            series->append(sp.mz[i], 0.0);
            series->append(sp.mz[i], sp.intensity[i]);
            series->append(sp.mz[i], 0.0);
            maxInt = std::max(maxInt, sp.intensity[i]);
        }
        chart->addSeries(series);

        auto* xAxis = new QValueAxis(chart);
        xAxis->setTitleText("m/z");
        auto* yAxis = new QValueAxis(chart);
        yAxis->setTitleText("Intensity");
        if (!sp.mz.empty()) {
            xAxis->setRange(*std::min_element(sp.mz.begin(), sp.mz.end()) - 5,
                            *std::max_element(sp.mz.begin(), sp.mz.end()) + 5);
        }
        yAxis->setRange(0, maxInt * 1.1);

        chart->addAxis(xAxis, Qt::AlignBottom);
        chart->addAxis(yAxis, Qt::AlignLeft);
        series->attachAxis(xAxis);
        series->attachAxis(yAxis);

        auto* chartView = new InteractiveMSMSChartView(chart);
        chartView->setMinimumHeight(300);
        scrollLayout->addWidget(chartView);
        spectraPlots.push_back(chartView);
    }

    layout->addWidget(scrollArea);

    // Close button
    auto* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    auto* closeBtn = new QPushButton("Close", this);
    connect(closeBtn, &QPushButton::clicked, this, &MSMSViewerWindow::close);
    btnLayout->addWidget(closeBtn);
    layout->addLayout(btnLayout);
}

void MSMSViewerWindow::populateSimilarityTable() {
    int n = (int)spectra.size();
    int pairs = n * (n - 1) / 2;
    similarityTable->setRowCount(pairs);

    int row = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            SpectrumData s1, s2;
            s1.mz = spectra[i].mz;
            s1.intensity = spectra[i].intensity;
            s1.precursorMz = spectra[i].precursorMz;
            s2.mz = spectra[j].mz;
            s2.intensity = spectra[j].intensity;
            s2.precursorMz = spectra[j].precursorMz;

            double score = Utils::calculateCosineSimilarity(s1, s2, 0.02);

            similarityTable->setItem(row, 0, new QTableWidgetItem(
                QString::fromStdString(spectra[i].filename)));
            similarityTable->setItem(row, 1, new QTableWidgetItem(
                QString::fromStdString(spectra[j].filename)));
            similarityTable->setItem(row, 2, new QTableWidgetItem(
                QString::number(score, 'f', 4)));
            similarityTable->setItem(row, 3, new QTableWidgetItem("-"));
            row++;
        }
    }
}

// ---------------------------------------------------------------------------
// EnhancedMirrorPlotWindow
// ---------------------------------------------------------------------------
EnhancedMirrorPlotWindow::EnhancedMirrorPlotWindow(
    const MSMSViewerWindow::MSMSSpectrum& s1,
    const MSMSViewerWindow::MSMSSpectrum& s2,
    double tol, QWidget* parent)
    : QWidget(parent), spec1(s1), spec2(s2), mzTolerance(tol)
{
    setWindowFlags(Qt::Window | Qt::WindowCloseButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle("MS/MS Mirror Plot");
    resize(900, 600);
    initUI();
}

void EnhancedMirrorPlotWindow::initUI() {
    auto* layout = new QVBoxLayout(this);

    auto* chart = new QChart();
    chart->setTitle(QString("MS/MS Mirror: %1 vs %2")
                    .arg(QString::fromStdString(spec1.filename))
                    .arg(QString::fromStdString(spec2.filename)));
    chart->legend()->hide();

    // Upper spectrum (spec1)
    auto* series1 = new QLineSeries(chart);
    QPen pen1(QColor(QString::fromStdString(spec1.color)));
    pen1.setWidth(1);
    series1->setPen(pen1);

    double maxInt1 = 0.0, maxInt2 = 0.0;
    for (double v : spec1.intensity) maxInt1 = std::max(maxInt1, v);
    for (double v : spec2.intensity) maxInt2 = std::max(maxInt2, v);
    if (maxInt1 == 0.0) maxInt1 = 1.0;
    if (maxInt2 == 0.0) maxInt2 = 1.0;

    double mzMin = 0.0, mzMax = 0.0;
    auto allMz = spec1.mz;
    allMz.insert(allMz.end(), spec2.mz.begin(), spec2.mz.end());
    if (!allMz.empty()) {
        mzMin = *std::min_element(allMz.begin(), allMz.end()) - 5;
        mzMax = *std::max_element(allMz.begin(), allMz.end()) + 5;
    }

    for (size_t i = 0; i < spec1.mz.size(); ++i) {
        double norm = spec1.intensity[i] / maxInt1;
        series1->append(spec1.mz[i], 0.0);
        series1->append(spec1.mz[i], norm);
        series1->append(spec1.mz[i], 0.0);
    }
    chart->addSeries(series1);

    // Lower spectrum (spec2, mirrored)
    auto* series2 = new QLineSeries(chart);
    QPen pen2(QColor(QString::fromStdString(spec2.color)));
    pen2.setWidth(1);
    series2->setPen(pen2);

    for (size_t i = 0; i < spec2.mz.size(); ++i) {
        double norm = -(spec2.intensity[i] / maxInt2);
        series2->append(spec2.mz[i], 0.0);
        series2->append(spec2.mz[i], norm);
        series2->append(spec2.mz[i], 0.0);
    }
    chart->addSeries(series2);

    auto* xAxis = new QValueAxis(chart);
    xAxis->setTitleText("m/z");
    xAxis->setRange(mzMin, mzMax);

    auto* yAxis = new QValueAxis(chart);
    yAxis->setTitleText("Relative Intensity");
    yAxis->setRange(-1.1, 1.1);

    chart->addAxis(xAxis, Qt::AlignBottom);
    chart->addAxis(yAxis, Qt::AlignLeft);
    series1->attachAxis(xAxis);
    series1->attachAxis(yAxis);
    series2->attachAxis(xAxis);
    series2->attachAxis(yAxis);

    auto* chartView = new InteractiveMSMSChartView(chart);
    layout->addWidget(chartView);

    auto* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    auto* closeBtn = new QPushButton("Close", this);
    connect(closeBtn, &QPushButton::clicked, this, &EnhancedMirrorPlotWindow::close);
    btnLayout->addWidget(closeBtn);
    layout->addLayout(btnLayout);
}
