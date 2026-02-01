# Build for macOS and Linux
maturin build --release --sdist --compatibility pypi --target universal2-apple-darwin --out dist && \
docker run --rm -v "$(pwd)":/io ghcr.io/pyo3/maturin build --release --compatibility pypi --manylinux 2014 --out dist 