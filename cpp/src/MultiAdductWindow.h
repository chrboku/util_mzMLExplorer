#pragma once
#include <QWidget>
#include <QTableWidget>
#include <QChartView>
#include <QChart>
#include <QLineSeries>
#include <QValueAxis>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <vector>
#include <map>
#include <string>
#include "CompoundManager.h"
#include "FileManager.h"

/**
 * MultiAdductWindow - Shows a compound across multiple adducts simultaneously.
 */
class MultiAdductWindow : public QWidget {
    Q_OBJECT
public:
    MultiAdductWindow(const std::string& compoundName,
                      FileManager* fileManager,
                      CompoundManager* compoundManager,
                      QWidget* parent = nullptr);

private slots:
    void onExtractClicked();
    void updateCompoundInfo();

private:
    void initUI();

    std::string compoundName;
    FileManager* fileManager;
    CompoundManager* compoundManager;

    QLabel* compoundInfoLabel = nullptr;
    QDoubleSpinBox* mzToleranceSpin = nullptr;
    QTableWidget* resultsTable = nullptr;
    QChartView* chartView = nullptr;
    QChart* chart = nullptr;
};
