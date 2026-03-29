#pragma once
#include <QDialog>
#include <QTableWidget>
#include <QComboBox>
#include <QLineEdit>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QTextEdit>
#include <vector>
#include <string>
#include <map>

/**
 * CompoundImportDialog - Dialog for importing compounds from CSV/TSV files.
 */
class CompoundImportDialog : public QDialog {
    Q_OBJECT
public:
    struct ImportParams {
        char delimiter;
        std::string namePrefix;
        std::string nameColumn;
        std::string mzColumn;
        std::string formulaColumn;
        std::string rtMinColumn;
        std::string rtStartColumn;
        std::string rtEndColumn;
        std::string adductsColumn;
        std::string groupColumn;
        std::string smilesColumn;
        int headerRow = 0;
    };

    explicit CompoundImportDialog(const QString& filePath, QWidget* parent = nullptr);

    // Get import result
    std::vector<std::string> getHeaders() const { return importedHeaders; }
    std::vector<std::vector<std::string>> getRows() const { return importedRows; }

private slots:
    void onDelimiterChanged();
    void onImportClicked();
    void updatePreview();

private:
    void initUI();
    void loadFile(char delim);
    std::vector<std::string> splitLine(const std::string& line, char delim) const;

    QString filePath;
    std::vector<std::string> rawHeaders;
    std::vector<std::vector<std::string>> rawRows;
    std::vector<std::string> importedHeaders;
    std::vector<std::vector<std::string>> importedRows;

    QComboBox* delimCombo = nullptr;
    QLineEdit* namePrefixEdit = nullptr;
    QComboBox* nameColCombo = nullptr;
    QComboBox* mzColCombo = nullptr;
    QComboBox* formulaColCombo = nullptr;
    QComboBox* rtMinColCombo = nullptr;
    QComboBox* rtStartColCombo = nullptr;
    QComboBox* rtEndColCombo = nullptr;
    QComboBox* adductsColCombo = nullptr;
    QComboBox* groupColCombo = nullptr;
    QTableWidget* previewTable = nullptr;
    QLabel* statusLabel = nullptr;
};
