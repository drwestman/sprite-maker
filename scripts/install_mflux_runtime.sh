#!/usr/bin/env bash
set -euo pipefail

root="${1:-}"
if [[ -z "$root" ]]; then
  echo "usage: install_mflux_runtime.sh <managed-mflux-root>" >&2
  exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MFLUX requires Apple Silicon macOS" >&2
  exit 3
fi

python_bin="$(command -v python3.11 || true)"
if [[ -z "$python_bin" ]]; then
  echo "Python 3.11 is required to install the managed MFLUX runtime" >&2
  exit 4
fi

runtime="$root/runtime"
mkdir -p "$root/checkpoints"
if [[ -x "$runtime/bin/python" ]]; then
  runtime_python_version="$("$runtime/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  if [[ "$runtime_python_version" != "3.11" ]]; then
    rm -rf "$runtime"
  fi
fi
if [[ ! -x "$runtime/bin/python" ]]; then
  rm -rf "$runtime"
  "$python_bin" -m venv "$runtime"
fi

printf '%s\n' 'Installing pinned MFLUX dependencies' >&2
"$runtime/bin/python" -m pip install --disable-pip-version-check --no-input \
  "mflux==0.19.1" \
  "mlx==0.32.0"

printf '%s\n' 'Verifying the managed MFLUX interpreter' >&2
"$runtime/bin/python" - <<'PY'
import importlib.metadata
import platform
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit("managed runtime is not Python 3.11")
if platform.system() != "Darwin" or platform.machine() != "arm64":
    raise SystemExit("managed runtime is not Apple Silicon macOS")
for package, expected in (("mflux", "0.19.1"), ("mlx", "0.32.0")):
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise SystemExit(f"{package} is {actual}, expected {expected}")
PY

printf '%s\n' 'MFLUX runtime installation complete' >&2
