#include <QApplication>
#include <QIcon>
#include <QDir>
#include <QStyleFactory>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    app.setApplicationName("mzML Explorer");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("BOKU University");
    app.setOrganizationDomain("boku.ac.at");

    // Set fusion style for a clean, modern look
    app.setStyle(QStyleFactory::create("Fusion"));

    MzMLExplorerMainWindow window;
    window.show();

    return app.exec();
}
