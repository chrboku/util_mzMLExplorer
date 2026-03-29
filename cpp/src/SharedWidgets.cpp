#include "SharedWidgets.h"
#include <QMouseEvent>
#include <QVBoxLayout>
#include <algorithm>

const std::vector<AnnotationColorPreset> ANNOTATION_COLOR_PRESETS = {
    {"Blue",   "#1a73e8"},
    {"Red",    "#b22222"},
    {"Green",  "#2ca02c"},
    {"Orange", "#e65c00"},
    {"Yellow", "#c6a800"},
    {"Black",  "#202124"},
    {"Purple", "#7b1fa2"},
};

// ---------------------------------------------------------------------------
// ClickableLabel
// ---------------------------------------------------------------------------
ClickableLabel::ClickableLabel(const QString& text, QWidget* parent)
    : QLabel(text, parent)
{
    setCursor(Qt::PointingHandCursor);
}

void ClickableLabel::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit clicked();
    }
    QLabel::mousePressEvent(event);
}

// ---------------------------------------------------------------------------
// NumericTableWidgetItem
// ---------------------------------------------------------------------------
NumericTableWidgetItem::NumericTableWidgetItem(const QString& text)
    : QTableWidgetItem(text)
{}

bool NumericTableWidgetItem::operator<(const QTableWidgetItem& other) const {
    QVariant v1 = data(Qt::UserRole);
    QVariant v2 = other.data(Qt::UserRole);
    if (v1.isValid() && v2.isValid()) {
        bool ok1, ok2;
        double d1 = v1.toDouble(&ok1);
        double d2 = v2.toDouble(&ok2);
        if (ok1 && ok2) return d1 < d2;
    }
    return QTableWidgetItem::operator<(other);
}

// ---------------------------------------------------------------------------
// BarDelegate
// ---------------------------------------------------------------------------
BarDelegate::BarDelegate(QObject* parent)
    : QStyledItemDelegate(parent)
{}

void BarDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option,
                         const QModelIndex& index) const
{
    QStyledItemDelegate::paint(painter, option, index);

    QVariant fracVar  = index.data(BAR_FRAC_ROLE);
    QVariant colorVar = index.data(BAR_COLOR_ROLE);
    if (!fracVar.isValid() || !colorVar.isValid()) return;

    double frac = fracVar.toDouble();
    if (frac <= 0.0) return;

    int barWidth = (int)(option.rect.width() * std::min(frac, 1.0));
    QRect barRect = option.rect.adjusted(0, 1, 0, -1);
    barRect.setWidth(barWidth);

    QColor color = colorVar.value<QColor>();
    color.setAlphaF(0.5);

    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, false);
    painter->fillRect(barRect, color);
    painter->restore();
}

// ---------------------------------------------------------------------------
// CenteredBarDelegate
// ---------------------------------------------------------------------------
CenteredBarDelegate::CenteredBarDelegate(QObject* parent)
    : QStyledItemDelegate(parent)
{}

void CenteredBarDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option,
                                  const QModelIndex& index) const
{
    QStyledItemDelegate::paint(painter, option, index);

    QVariant ppmDevVar   = index.data(PPM_DEVIATION_ROLE);
    QVariant ppmRangeVar = index.data(PPM_RANGE_ROLE);
    if (!ppmDevVar.isValid() || !ppmRangeVar.isValid()) return;

    double ppmDev   = ppmDevVar.toDouble();
    double ppmRange = ppmRangeVar.toDouble();
    if (ppmRange <= 0.0) return;

    double frac = std::max(-1.0, std::min(1.0, ppmDev / ppmRange));
    if (std::abs(frac) < 1e-6) return;

    QRect rect = option.rect;
    int centerX = rect.left() + rect.width() / 2;
    int barW = std::max(1, (int)(std::abs(frac) * rect.width() / 2));
    int barTop = rect.top() + 2;
    int barH = std::max(1, rect.height() - 4);

    QRect barRect;
    if (frac > 0) {
        barRect = QRect(centerX, barTop, barW, barH);
    } else {
        barRect = QRect(centerX - barW, barTop, barW, barH);
    }

    QColor color = (frac < 0) ? QColor("dodgerblue") : QColor("firebrick");
    color.setAlphaF(0.5);

    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, false);
    painter->fillRect(barRect, color);
    painter->restore();
}

// ---------------------------------------------------------------------------
// CollapsibleBox
// ---------------------------------------------------------------------------
CollapsibleBox::CollapsibleBox(const QString& title, QWidget* parent)
    : QWidget(parent), baseTitle(title)
{
    toggleButton = new QPushButton("▶  " + title, this);
    toggleButton->setCheckable(true);
    toggleButton->setChecked(false);
    toggleButton->setProperty("role", "collapsibleHeader");
    connect(toggleButton, &QPushButton::clicked, this, &CollapsibleBox::toggleContent);

    contentArea = new QWidget(this);
    contentLayout = new QVBoxLayout(contentArea);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentArea->setVisible(false);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(0);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(toggleButton);
    mainLayout->addWidget(contentArea);
}

void CollapsibleBox::toggleContent() {
    bool vis = contentArea->isVisible();
    contentArea->setVisible(!vis);
    QString prefix = vis ? "▶  " : "▼  ";
    toggleButton->setText(prefix + baseTitle);
}

void CollapsibleBox::addWidget(QWidget* widget) {
    contentLayout->addWidget(widget);
}

void CollapsibleBox::setExpanded(bool expanded) {
    if (expanded != contentArea->isVisible()) toggleContent();
}

bool CollapsibleBox::isExpanded() const {
    return contentArea->isVisible();
}
