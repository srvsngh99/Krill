# KrillLM Build System
BINARY_NAME = krillm
BUILD_DIR = .build/release
PREFIX ?= /usr/local
VERSION = $(shell cat VERSION 2>/dev/null || echo "0.2.0")

.PHONY: build release install uninstall clean test bench

# Debug build (default)
build: metallib
	swift build

# Compile MLX Metal shaders into metallib (required for GPU inference)
metallib:
	@echo "Compiling Metal shaders..."
	@mkdir -p .build/debug .build/release
	@METAL_DIR=.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal; \
	INCLUDE_DIR=.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels; \
	AIR_FILES=""; \
	for f in $$METAL_DIR/*.metal $$METAL_DIR/**/*.metal; do \
		[ -f "$$f" ] || continue; \
		AIR=$$(basename $$f .metal).air; \
		xcrun -sdk macosx metal -c -I"$$INCLUDE_DIR" "$$f" -o "/tmp/$$AIR" 2>/dev/null && \
		AIR_FILES="$$AIR_FILES /tmp/$$AIR"; \
	done; \
	if [ -n "$$AIR_FILES" ]; then \
		xcrun -sdk macosx metallib $$AIR_FILES -o .build/debug/default.metallib 2>/dev/null; \
		cp .build/debug/default.metallib .build/release/default.metallib 2>/dev/null; \
		echo "Metal shaders compiled."; \
	else \
		echo "WARNING: Metal Toolchain not installed. Run: xcodebuild -downloadComponent MetalToolchain"; \
	fi

# Optimized release build
release:
	swift build -c release --arch arm64
	@echo "Binary at $(BUILD_DIR)/$(BINARY_NAME)"
	@ls -lh $(BUILD_DIR)/$(BINARY_NAME)

# Install to PREFIX/bin
install: release
	install -d $(PREFIX)/bin
	install -m 755 $(BUILD_DIR)/$(BINARY_NAME) $(PREFIX)/bin/$(BINARY_NAME)
	@echo "Installed $(BINARY_NAME) to $(PREFIX)/bin/"

# Remove from PREFIX/bin
uninstall:
	rm -f $(PREFIX)/bin/$(BINARY_NAME)
	@echo "Removed $(BINARY_NAME) from $(PREFIX)/bin/"

# Run tests
test:
	swift test

# Quick benchmark (requires a model)
bench:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make bench MODEL=llama-3.2-3b"; \
		exit 1; \
	fi
	$(BUILD_DIR)/$(BINARY_NAME) bench $(MODEL) --runs 3

# Clean build artifacts
clean:
	swift package clean
	rm -rf .build

# Build release tarball for distribution
dist: release
	mkdir -p dist
	tar -czf dist/krillm-$(VERSION)-arm64-apple-macos.tar.gz \
		-C $(BUILD_DIR) $(BINARY_NAME)
	@echo "Tarball at dist/krillm-$(VERSION)-arm64-apple-macos.tar.gz"
	shasum -a 256 dist/krillm-$(VERSION)-arm64-apple-macos.tar.gz

# Print version
version:
	@echo $(VERSION)
