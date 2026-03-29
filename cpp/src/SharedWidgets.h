#pragma once
#include <QLabel>
#include <QPushButton>
#include <QWidget>
#include <QVBoxLayout>
#include <QTableWidgetItem>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QPainter>
#include <QColor>
#include <QRect>
#include <vector>
#include <string>

// Colour presets for annotation labels
struct AnnotationColorPreset {
    QString name;
    QString color;
};

extern const std::vector<AnnotationColorPreset> ANNOTATION_COLOR_PRESETS;

/**
 * ClickableLabel - A QLabel that emits a clicked() signal when left-clicked.
 */
class ClickableLabel : public QLabel {
    Q_OBJECT
public:
    explicit ClickableLabel(const QString& text = "", QWidget* parent = nullptr);

signals:
    void clicked();

protected:
    void mousePressEvent(QMouseEvent* event) override;
};

/**
 * NumericTableWidgetItem - QTableWidgetItem that sorts numerically.
 */
class NumericTableWidgetItem : public QTableWidgetItem {
public:
    explicit NumericTableWidgetItem(const QString& text = "");
    bool operator<(const QTableWidgetItem& other) const override;
};

/**
 * BarDelegate - Paints a proportional horizontal bar in table cells.
 */
class BarDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    static constexpr int BAR_FRAC_ROLE  = Qt::UserRole + 2;
    static constexpr int BAR_COLOR_ROLE = Qt::UserRole + 3;

    explicit BarDelegate(QObject* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override;
};

/**
 * CenteredBarDelegate - Paints a centered bar showing ppm deviation.
 */
class CenteredBarDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    static constexpr int PPM_DEVIATION_ROLE  = Qt::UserRole + 4;
    static constexpr int PPM_RANGE_ROLE      = Qt::UserRole + 5;
    static constexpr int PPM_BAR_COLOR_ROLE  = Qt::UserRole + 6;

    explicit CenteredBarDelegate(QObject* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override;
};

/**
 * CollapsibleBox - A widget with a collapsible content area.
 */
class CollapsibleBox : public QWidget {
    Q_OBJECT
public:
    explicit CollapsibleBox(const QString& title = "", QWidget* parent = nullptr);

    void addWidget(QWidget* widget);
    void setExpanded(bool expanded);
    bool isExpanded() const;

private slots:
    void toggleContent();

private:
    QString baseTitle;
    QPushButton* toggleButton;
    QWidget* contentArea;
    QVBoxLayout* contentLayout;
};
