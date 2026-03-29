#pragma once
#include <QWidget>
#include <QTableWidget>
#include <QChartView>
#include <QChart>
#include <QLineSeries>
#include <QValueAxis>
#include <QLabel>
#include <QProgressBar>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QThread>
#include <vector>
#include <map>
#include <string>
#include "FileManager.h"
#include "CompoundManager.h"

struct EICResult {
    std::string filepath;
    std::string filename;
    std::string group;
    std::string color;
    std::string compoundName;
    std::string adduct;
    double targetMz;
    double rtCenter;
    double rtStart;
    double rtEnd;
    std::vector<double> rtValues;
    std::vector<double> intensityValues;
    double peakArea = 0.0;
    double peakMaxIntensity = 0.0;
    double peakRt = 0.0;
};

/**
 * EICExtractionWorker - Thread worker for EIC extraction.
 */
class EICExtractionWorker : public QThread {
    Q_OBJECT
public:
    EICExtractionWorker(FileManager* fm, CompoundManager* cm,
                         double mzTolerance, QObject* parent = nullptr);

    void run() override;

signals:
    void progress(int current, int total, const QString& filename);
    void resultReady(const EICResult& result);
    void finished(const QString& message);
    void error(const QString& message);

private:
    FileManager* fileManager;
    CompoundManager* compoundManager;
    double mzTolerance;
};

/**
 * InteractiveChartView - Custom QChartView with pan/zoom support.
 */
class InteractiveChartView : public QChartView {
    Q_OBJECT
public:
    explicit InteractiveChartView(QChart* chart, QWidget* parent = nullptr);

signals:
    void contextMenuRequested(double rtValue, QPointF mousePos);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    bool isPanning = false;
    QPointF panStartPos;
    std::pair<double,double> rangeX, rangeY;
};

/**
 * EICWindow - Window for displaying EIC plots and results.
 */
class EICWindow : public QWidget {
    Q_OBJECT
public:
    EICWindow(FileManager* fileManager, CompoundManager* compoundManager,
              QWidget* parent = nullptr);

private slots:
    void onExtractClicked();
    void onResultReady(const EICResult& result);
    void onExtractionFinished(const QString& message);
    void onProgressUpdate(int current, int total, const QString& filename);
    void updatePlot();

private:
    void initUI();
    void setupChart();
    void clearResults();
    void addResultToTable(const EICResult& result, int row);
    void addSeriesToChart(const EICResult& result);

    FileManager* fileManager;
    CompoundManager* compoundManager;

    // UI components
    QDoubleSpinBox* mzToleranceSpin = nullptr;
    QComboBox* compoundCombo = nullptr;
    QComboBox* adductCombo = nullptr;
    QTableWidget* resultsTable = nullptr;
    InteractiveChartView* chartView = nullptr;
    QChart* chart = nullptr;
    QValueAxis* xAxis = nullptr;
    QValueAxis* yAxis = nullptr;
    QLabel* statusLabel = nullptr;
    QProgressBar* progressBar = nullptr;

    std::vector<EICResult> results;
    EICExtractionWorker* worker = nullptr;
};
