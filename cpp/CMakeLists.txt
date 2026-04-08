cmake_minimum_required(VERSION 3.16)
project(mzMLExplorer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Find Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Charts Concurrent)

# Find zlib (for compressed mzML)
find_package(ZLIB REQUIRED)

# Find OpenSSL (for SHA-256 hashing in cache)
find_package(OpenSSL REQUIRED)

# Source files
set(SOURCES
    src/main.cpp
    src/FormulaTools.cpp
    src/Utils.cpp
    src/MzMLReader.cpp
    src/FileManager.cpp
    src/CompoundManager.cpp
    src/SharedWidgets.cpp
    src/MainWindow.cpp
    src/EICWindow.cpp
    src/MS1Window.cpp
    src/MSMSWindow.cpp
    src/MultiAdductWindow.cpp
    src/FileExplorerWindow.cpp
    src/CompoundImportDialog.cpp
    third_party/pugixml/pugixml.cpp
)

set(HEADERS
    src/FormulaTools.h
    src/Utils.h
    src/MzMLReader.h
    src/FileManager.h
    src/CompoundManager.h
    src/SharedWidgets.h
    src/MainWindow.h
    src/EICWindow.h
    src/MS1Window.h
    src/MSMSWindow.h
    src/MultiAdductWindow.h
    src/FileExplorerWindow.h
    src/CompoundImportDialog.h
)

add_executable(mzmlexplorer ${SOURCES} ${HEADERS})

target_include_directories(mzmlexplorer PRIVATE
    src/
    third_party/pugixml/
)

target_link_libraries(mzmlexplorer
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Charts
    Qt6::Concurrent
    ZLIB::ZLIB
    OpenSSL::SSL
    OpenSSL::Crypto
)

# Copy style.css to build directory
configure_file(${CMAKE_SOURCE_DIR}/../src/mzmlexplorer/style.css
    ${CMAKE_BINARY_DIR}/style.css COPYONLY)

if(WIN32)
    set_target_properties(mzmlexplorer PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
endif()

install(TARGETS mzmlexplorer
    RUNTIME DESTINATION bin
)
