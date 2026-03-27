# go-coreml-mlx

This module provides the MLX backend bindings used by `gopeft`.

## Important

This package is **not** buildable from a bare module download alone unless the
MLX bridge dependencies have been bootstrapped first.

The CGo bridge expects generated MLX headers and static libraries under:

```text
internal/bridge/deps/include
internal/bridge/deps/lib
```

Those artifacts are intentionally not checked into git. They must be created
locally on the target machine.

## Bootstrap

From the module root:

```bash
./scripts/build_mlx.sh
```

This will:

- clone `mlx-c`
- build the MLX static bridge
- install headers and libraries under `internal/bridge/deps/`

## Requirements

- macOS on Apple Silicon
- full Xcode install
- `cmake`
- `git`

If the build fails with an error like:

```text
fatal error: 'mlx/c/mlx.h' file not found
```

then the bootstrap step has not been run yet, or the generated deps directory is
missing/incomplete.

## Intended Setup For Another Machine

1. Clone the `go-coreml` fork/branch that contains this module.
2. Run:

```bash
cd mlx
./scripts/build_mlx.sh
```

3. Then build the depending `gopeft` repo with its `replace` pointing at this
local checkout.
