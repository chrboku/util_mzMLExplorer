#pragma once
#include <QWidget>
#include <QChartView>
#include <QChart>
#include <QLineSeries>
#include <QValueAxis>
#include <QTableWidget>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <string>
#include "FileManager.h"

/**
 * FileExplorerWindow - Browse the raw mzML data for a single file.
 *
 * Provides:
 * - TIC (Total Ion Chromatogram) view
 * - MS1 spectrum view at selected RT
 * - MS/MS spectrum view at selected RT
 */
class FileExplorerWindow : public QWidget {
    Q_OBJECT
public:
    FileExplorerWindow(const QString& filepath,
                       FileManager* fileManager,
                       QWidget* parent = nullptr);

private slots:
    void onTICPointSelected(double rtValue);
    void updateMS1Spectrum(double rtMin, double rtMax);
    void updateMS2Spectrum(double rtMin, double rtMax);
    void onRtSpinChanged();
    void onRtWindowChanged();

private:
    void initUI();
    void loadAndPlotTIC();
    QChartView* createTICChartView();
    QChartView* createSpectrumChartView(const QString& title);

    QString filepath;
    FileManager* fileManager;
    LoadedFileData fileData;
    bool dataLoaded = false;

    QLabel* statusLabel = nullptr;
    QDoubleSpinBox* rtSpin = nullptr;
    QDoubleSpinBox* rtWindowSpin = nullptr;
    QComboBox* msLevelCombo = nullptr;

    QChartView* ticChartView = nullptr;
    QChart* ticChart = nullptr;
    QValueAxis* ticXAxis = nullptr;
    QValueAxis* ticYAxis = nullptr;

    QChartView* spectrumChartView = nullptr;
    QChart* spectrumChart = nullptr;
    QValueAxis* specXAxis = nullptr;
    QValueAxis* specYAxis = nullptr;

    QTableWidget* spectrumTable = nullptr;
};
