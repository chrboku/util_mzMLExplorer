#include "MS1Window.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QLabel>
#include <QPushButton>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QValueAxis>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// InteractiveMS1ChartView
// ---------------------------------------------------------------------------
InteractiveMS1ChartView::InteractiveMS1ChartView(QChart* chart, QWidget* parent)
    : QChartView(chart, parent)
{
    setRenderHint(QPainter::Antialiasing);
    setRubberBand(QChartView::NoRubberBand);
    setMouseTracking(true);
}

void InteractiveMS1ChartView::wheelEvent(QWheelEvent* event) {
    auto axes_x = chart()->axes(Qt::Horizontal);
    auto axes_y = chart()->axes(Qt::Vertical);
    if (axes_x.isEmpty() || axes_y.isEmpty()) {
        QChartView::wheelEvent(event);
        return;
    }

    auto* xAx = qobject_cast<QValueAxis*>(axes_x.first());
    if (!xAx) { QChartView::wheelEvent(event); return; }

    double factor = event->angleDelta().y() > 0 ? 0.8 : 1.25;
    QPointF chartVal = chart()->mapToValue(event->position());

    double xMin = xAx->min(), xMax = xAx->max();
    double xFrac = (chartVal.x() - xMin) / (xMax - xMin);
    double newXRange = (xMax - xMin) * factor;
    xAx->setRange(chartVal.x() - xFrac * newXRange,
                  chartVal.x() + (1.0 - xFrac) * newXRange);

    event->accept();
}

void InteractiveMS1ChartView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = true;
        panStart = event->position();
        auto axes_x = chart()->axes(Qt::Horizontal);
        auto axes_y = chart()->axes(Qt::Vertical);
        if (!axes_x.isEmpty()) {
            auto* ax = qobject_cast<QValueAxis*>(axes_x.first());
            if (ax) { rangeXMin = ax->min(); rangeXMax = ax->max(); }
        }
        if (!axes_y.isEmpty()) {
            auto* ay = qobject_cast<QValueAxis*>(axes_y.first());
            if (ay) { rangeYMin = ay->min(); rangeYMax = ay->max(); }
        }
        setCursor(Qt::ClosedHandCursor);
    }
    QChartView::mousePressEvent(event);
}

void InteractiveMS1ChartView::mouseMoveEvent(QMouseEvent* event) {
    if (isPanning) {
        QPointF delta = event->position() - panStart;
        auto axes_x = chart()->axes(Qt::Horizontal);
        auto axes_y = chart()->axes(Qt::Vertical);
        QRectF plotArea = chart()->plotArea();

        if (!axes_x.isEmpty()) {
            auto* ax = qobject_cast<QValueAxis*>(axes_x.first());
            if (ax) {
                double xScale = (rangeXMax - rangeXMin) / plotArea.width();
                double xShift = -delta.x() * xScale;
                ax->setRange(rangeXMin + xShift, rangeXMax + xShift);
            }
        }
        if (!axes_y.isEmpty()) {
            auto* ay = qobject_cast<QValueAxis*>(axes_y.first());
            if (ay) {
                double yScale = (rangeYMax - rangeYMin) / plotArea.height();
                double yShift = delta.y() * yScale;
                ay->setRange(rangeYMin + yShift, rangeYMax + yShift);
            }
        }
    }
    QChartView::mouseMoveEvent(event);
}

void InteractiveMS1ChartView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = false;
        setCursor(Qt::ArrowCursor);
    }
    QChartView::mouseReleaseEvent(event);
}

// ---------------------------------------------------------------------------
// MS1ViewerWindow
// ---------------------------------------------------------------------------
MS1ViewerWindow::MS1ViewerWindow(
    const std::map<std::string, FileSpectrumData>& spectra,
    double tgtMz, double rtCtr,
    const std::string& cmpName, const std::string& addt,
    double tol, const std::string& frml,
    QWidget* parent)
    : QWidget(parent),
      ms1Spectra(spectra), targetMz(tgtMz), rtCenter(rtCtr),
      compoundName(cmpName), adduct(addt), mzTolerance(tol), formula(frml)
{
    setWindowFlags(Qt::Window | Qt::WindowCloseButtonHint |
                   Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(QString("MS1 Spectra: %1 - %2 (RT: %3 min)")
                   .arg(QString::fromStdString(compoundName))
                   .arg(QString::fromStdString(adduct))
                   .arg(rtCenter, 0, 'f', 2));
    resize(1400, 900);
    initUI();
}

void MS1ViewerWindow::initUI() {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(5, 5, 5, 5);
    layout->setSpacing(5);

    // Header
    QString headerText = QString("<b>%1 (%2)</b> | m/z: %3 | RT: %4 min | Files: %5")
        .arg(QString::fromStdString(compoundName))
        .arg(QString::fromStdString(adduct))
        .arg(targetMz, 0, 'f', 4)
        .arg(rtCenter, 0, 'f', 2)
        .arg((int)ms1Spectra.size());

    if (!formula.empty()) {
        headerText += QString(" | Formula: %1").arg(QString::fromStdString(formula));
    }

    auto* headerLabel = new QLabel(headerText, this);
    headerLabel->setStyleSheet("QLabel { margin: 2px; padding: 5px; font-size: 12px; }");
    headerLabel->setFixedHeight(30);
    layout->addWidget(headerLabel);

    // Scroll area for spectra grid
    auto* scrollArea = new QScrollArea(this);
    auto* scrollWidget = new QWidget(scrollArea);
    scrollArea->setWidget(scrollWidget);
    scrollArea->setWidgetResizable(true);

    gridLayout = new QGridLayout(scrollWidget);
    gridLayout->setSpacing(5);
    gridLayout->setContentsMargins(5, 5, 5, 5);

    // Create charts for each file
    int row = 0, col = 0;
    const int maxCols = 2;

    for (const auto& [filepath, fileData] : ms1Spectra) {
        // File header label
        QString displayName = QString::fromStdString(fileData.filename);
        int dotIdx = displayName.lastIndexOf('.');
        if (dotIdx > 0) displayName = displayName.left(dotIdx);

        QString headerStr = QString("<b>%1</b> | Group: %2 | RT: %3 min")
            .arg(displayName)
            .arg(QString::fromStdString(fileData.group))
            .arg(fileData.spectrum.rt, 0, 'f', 4);

        auto* fileLabel = new QLabel(headerStr, scrollWidget);
        fileLabel->setStyleSheet(
            "QLabel { background-color: #e8f0fe; padding: 4px; "
            "margin: 1px; border: 1px solid #dadce0; border-radius: 2px; }");
        fileLabel->setMaximumHeight(25);
        gridLayout->addWidget(fileLabel, row * 2, col);

        // Create chart
        auto* chartView = createMS1Chart(fileData.spectrum, fileData.filename, fileData.color);
        chartView->setMinimumSize(700, 350);
        chartViews[filepath] = chartView;
        gridLayout->addWidget(chartView, row * 2 + 1, col);

        col++;
        if (col >= maxCols) { col = 0; row++; }
    }

    layout->addWidget(scrollArea);

    // Close button
    auto* btnLayout = new QHBoxLayout();
    btnLayout->addStretch();
    auto* closeBtn = new QPushButton("Close", this);
    connect(closeBtn, &QPushButton::clicked, this, &MS1ViewerWindow::close);
    btnLayout->addWidget(closeBtn);
    layout->addLayout(btnLayout);
}

QChartView* MS1ViewerWindow::createMS1Chart(const SpectrumData& spectrum,
                                              const std::string& filename,
                                              const std::string& color) {
    auto* chart = new QChart();
    chart->setTitle(QString("MS1 Spectrum - %1").arg(QString::fromStdString(filename)));
    chart->legend()->hide();

    // Main spectrum series (stick-style)
    auto* series = new QLineSeries(chart);
    QPen pen(QColor(QString::fromStdString(color)));
    pen.setWidth(1);
    series->setPen(pen);

    double zoomWindow = 10.0;
    double zoomMin = std::max(0.0, targetMz - zoomWindow);
    double zoomMax = targetMz + zoomWindow;

    double maxIntensity = 0.0;
    for (size_t i = 0; i < spectrum.mz.size(); ++i) {
        double mz = spectrum.mz[i];
        double intensity = spectrum.intensity[i];
        series->append(mz, 0.0);
        series->append(mz, intensity);
        series->append(mz, 0.0);
        maxIntensity = std::max(maxIntensity, intensity);
    }
    chart->addSeries(series);

    // Highlight series for peaks within EIC window
    auto* highlightSeries = new QLineSeries(chart);
    QPen highlightPen(QColor("#F18F01"));
    highlightPen.setWidth(3);
    highlightSeries->setPen(highlightPen);

    for (size_t i = 0; i < spectrum.mz.size(); ++i) {
        if (std::abs(spectrum.mz[i] - targetMz) <= mzTolerance) {
            highlightSeries->append(spectrum.mz[i], 0.0);
            highlightSeries->append(spectrum.mz[i], spectrum.intensity[i]);
            highlightSeries->append(spectrum.mz[i], 0.0);
        }
    }
    chart->addSeries(highlightSeries);

    // Axes
    auto* xAxis = new QValueAxis(chart);
    xAxis->setTitleText("m/z");
    xAxis->setRange(zoomMin, zoomMax);

    auto* yAxis = new QValueAxis(chart);
    yAxis->setTitleText("Intensity");
    yAxis->setRange(0, maxIntensity * 1.1);

    chart->addAxis(xAxis, Qt::AlignBottom);
    chart->addAxis(yAxis, Qt::AlignLeft);
    series->attachAxis(xAxis);
    series->attachAxis(yAxis);
    highlightSeries->attachAxis(xAxis);
    highlightSeries->attachAxis(yAxis);

    auto* chartView = new InteractiveMS1ChartView(chart);
    return chartView;
}
