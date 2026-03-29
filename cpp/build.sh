#!/bin/bash
# Build script for mzML Explorer (C++)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== mzML Explorer C++ Build ==="
echo "Build type: ${BUILD_TYPE:-Release}"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
    "$@"

# Build
cmake --build . --parallel "$(nproc)"

echo ""
echo "=== Build Complete ==="
echo "Binary: $BUILD_DIR/mzmlexplorer"
echo ""
echo "To run: $BUILD_DIR/mzmlexplorer"
