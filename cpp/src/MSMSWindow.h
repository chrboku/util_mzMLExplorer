#pragma once
#include <QWidget>
#include <QChartView>
#include <QChart>
#include <QLineSeries>
#include <QValueAxis>
#include <QTableWidget>
#include <QLabel>
#include <vector>
#include <map>
#include <string>
#include "Utils.h"

/**
 * InteractiveMSMSChartView - Custom chart view for MS/MS spectra.
 */
class InteractiveMSMSChartView : public QChartView {
    Q_OBJECT
public:
    explicit InteractiveMSMSChartView(QChart* chart, QWidget* parent = nullptr);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    bool isPanning = false;
    QPointF panStart;
    double xMin, xMax, yMin, yMax;
};

/**
 * MSMSViewerWindow - Window for MS/MS spectrum comparison.
 */
class MSMSViewerWindow : public QWidget {
    Q_OBJECT
public:
    struct MSMSSpectrum {
        std::string filename;
        std::string group;
        std::string color;
        double precursorMz;
        double scanTime;
        std::vector<double> mz;
        std::vector<double> intensity;
    };

    MSMSViewerWindow(const std::vector<MSMSSpectrum>& spectra,
                     double targetMz,
                     double rtCenter,
                     const std::string& compoundName,
                     const std::string& adduct,
                     QWidget* parent = nullptr);

private:
    void initUI();
    QChartView* createMirrorPlot(const MSMSSpectrum& ref, const MSMSSpectrum& query);
    void populateSimilarityTable();

    std::vector<MSMSSpectrum> spectra;
    double targetMz;
    double rtCenter;
    std::string compoundName;
    std::string adduct;

    QTableWidget* similarityTable = nullptr;
    std::vector<QChartView*> spectraPlots;
};

/**
 * EnhancedMirrorPlotWindow - Side-by-side MS/MS comparison.
 */
class EnhancedMirrorPlotWindow : public QWidget {
    Q_OBJECT
public:
    EnhancedMirrorPlotWindow(const MSMSViewerWindow::MSMSSpectrum& spec1,
                              const MSMSViewerWindow::MSMSSpectrum& spec2,
                              double mzTolerance = 0.02,
                              QWidget* parent = nullptr);

private:
    void initUI();
    void addMatchedPeaks();

    MSMSViewerWindow::MSMSSpectrum spec1, spec2;
    double mzTolerance;
};
