#pragma once
#include <QWidget>
#include <QChartView>
#include <QChart>
#include <QLineSeries>
#include <QValueAxis>
#include <QLabel>
#include <QScrollArea>
#include <QGridLayout>
#include <vector>
#include <map>
#include <string>
#include "MzMLReader.h"

/**
 * InteractiveMS1ChartView - Custom chart view for MS1 spectra.
 */
class InteractiveMS1ChartView : public QChartView {
    Q_OBJECT
public:
    explicit InteractiveMS1ChartView(QChart* chart, QWidget* parent = nullptr);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    bool isPanning = false;
    QPointF panStart;
    double rangeXMin, rangeXMax, rangeYMin, rangeYMax;
};

/**
 * MS1ViewerWindow - Window for displaying MS1 spectra.
 */
class MS1ViewerWindow : public QWidget {
    Q_OBJECT
public:
    struct SpectrumData {
        double rt;
        std::vector<double> mz;
        std::vector<double> intensity;
        std::string scanId;
        std::string filterString;
    };

    struct FileSpectrumData {
        std::string filename;
        std::string group;
        std::string color;
        SpectrumData spectrum;
    };

    MS1ViewerWindow(const std::map<std::string, FileSpectrumData>& ms1Spectra,
                    double targetMz,
                    double rtCenter,
                    const std::string& compoundName,
                    const std::string& adduct,
                    double mzTolerance,
                    const std::string& formula = "",
                    QWidget* parent = nullptr);

private:
    void initUI();
    QChartView* createMS1Chart(const SpectrumData& spectrum,
                                const std::string& filename,
                                const std::string& color);

    std::map<std::string, FileSpectrumData> ms1Spectra;
    double targetMz;
    double rtCenter;
    std::string compoundName;
    std::string adduct;
    double mzTolerance;
    std::string formula;

    QGridLayout* gridLayout = nullptr;
    std::map<std::string, QChartView*> chartViews;
};
