#!/bin/bash
# Builds mlx + mlx-c as static libraries for CGo linking.
#
# Prerequisites: Xcode, CMake (3.24+), C++17 compiler
# Output: mlx/internal/bridge/deps/{include,lib}
#
# Usage:
#   ./scripts/build_mlx.sh              # Build with default version
#   MLX_C_VERSION=v0.5.0 ./build_mlx.sh # Build specific version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MLX_C_VERSION="${MLX_C_VERSION:-v0.5.0}"

# Check prerequisites
if ! xcrun --find metal &>/dev/null; then
    echo "ERROR: 'metal' shader compiler not found."
    echo "Full Xcode is required (not just Command Line Tools)."
    echo ""
    echo "After installing Xcode, run:"
    echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    exit 1
fi
echo "=== Metal compiler: $(xcrun --find metal) ==="
BUILD_DIR="/tmp/mlx-c-build-$$"
DEST_DIR="$REPO_ROOT/mlx/internal/bridge/deps"

echo "=== Building mlx-c ${MLX_C_VERSION} ==="
echo "Build dir: $BUILD_DIR"
echo "Install dir: $DEST_DIR"

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Clone mlx-c
echo "=== Cloning mlx-c ${MLX_C_VERSION} ==="
git clone --depth 1 --branch "$MLX_C_VERSION" https://github.com/ml-explore/mlx-c.git "$BUILD_DIR/mlx-c"

# Build
echo "=== Configuring ==="
cd "$BUILD_DIR/mlx-c"
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX="$DEST_DIR" \
    -DMLX_C_BUILD_EXAMPLES=OFF \
    -DMLX_C_BUILD_TESTS=OFF \
    -DMLX_METAL_PATH="$DEST_DIR/lib"

echo "=== Building (this may take a few minutes) ==="
cmake --build build --parallel "$(sysctl -n hw.ncpu)"

echo "=== Installing ==="
cmake --install build

echo "=== Cleaning up ==="
rm -rf "$BUILD_DIR"

echo "=== Done ==="
echo "Headers: $DEST_DIR/include/"
echo "Libraries: $DEST_DIR/lib/"
ls -la "$DEST_DIR/lib/"*.a 2>/dev/null || echo "(no .a files found - check build output)"
