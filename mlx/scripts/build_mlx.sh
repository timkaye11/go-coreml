#!/bin/bash
# Builds mlx + mlx-c as static libraries for CGo linking.
#
# Prerequisites:
#   - macOS on Apple Silicon
#   - full Xcode installation (not only Command Line Tools)
#   - CMake 3.24+
#   - git
#
# Output:
#   internal/bridge/deps/{include,lib}
#
# Usage:
#   ./scripts/build_mlx.sh
#   MLX_C_VERSION=v0.5.0 ./scripts/build_mlx.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MLX_C_VERSION="${MLX_C_VERSION:-v0.5.0}"
BUILD_DIR="${BUILD_DIR:-/tmp/mlx-c-build-$$}"
DEST_DIR="$REPO_ROOT/internal/bridge/deps"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: MLX bootstrap is only supported on macOS."
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: MLX bootstrap expects Apple Silicon (arm64)."
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake is required."
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required."
    exit 1
fi

if ! xcrun --find metal >/dev/null 2>&1; then
    echo "ERROR: 'metal' shader compiler not found."
    echo "Full Xcode is required. After installing Xcode, run:"
    echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    exit 1
fi

echo "=== MLX bootstrap ==="
echo "mlx-c version: $MLX_C_VERSION"
echo "build dir: $BUILD_DIR"
echo "install dir: $DEST_DIR"
echo "metal: $(xcrun --find metal)"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$DEST_DIR"

echo "=== Cloning mlx-c ${MLX_C_VERSION} ==="
git clone --depth 1 --branch "$MLX_C_VERSION" https://github.com/ml-explore/mlx-c.git "$BUILD_DIR/mlx-c"

cd "$BUILD_DIR/mlx-c"

echo "=== Configuring ==="
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX="$DEST_DIR" \
    -DMLX_C_BUILD_EXAMPLES=OFF \
    -DMLX_C_BUILD_TESTS=OFF \
    -DMLX_METAL_PATH="$DEST_DIR/lib"

echo "=== Building ==="
cmake --build build --parallel "$(sysctl -n hw.ncpu)"

echo "=== Installing ==="
cmake --install build

echo "=== Cleaning up ==="
rm -rf "$BUILD_DIR"

echo "=== Done ==="
echo "Headers installed under: $DEST_DIR/include"
echo "Libraries installed under: $DEST_DIR/lib"
ls -la "$DEST_DIR/lib/"*.a 2>/dev/null || true
