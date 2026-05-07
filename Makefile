# KrillLM Build System
BINARY_NAME = krillm
BUILD_DIR = .build/release
PREFIX ?= /usr/local
VERSION = $(shell cat VERSION 2>/dev/null || echo "0.2.0")
CONFIGURATION ?= debug
SWIFT_BUILD_FLAGS_debug =
SWIFT_BUILD_FLAGS_release = -c release --arch arm64
SWIFT_ENV = CLANG_MODULE_CACHE_PATH="$(CURDIR)/.build/clang-module-cache"
KRILL_MODEL ?= llama-3.2-1b
OLLAMA_MODEL ?= llama3.2:1b
BENCH_PROMPT ?= Explain quantum computing in simple terms.
BENCH_MAX_TOKENS ?= 32
BENCH_RUNS ?= 5
BENCH_WARMUP ?= 2
BENCH_OUTPUT ?= .build/benchmarks/krillm-vs-ollama.json

.PHONY: build release install uninstall clean test bench bench-compare metallib dist version

# Debug build (default)
build:
	@mkdir -p .build/clang-module-cache
	$(SWIFT_ENV) swift build
	$(MAKE) metallib CONFIGURATION=debug

# Compile MLX Metal shaders into metallib (required for GPU inference)
metallib:
	@echo "Compiling Metal shaders..."
	@set -eu; \
	mkdir -p .build/clang-module-cache; \
	BUILD_PATH=$$($(SWIFT_ENV) swift build --show-bin-path $(SWIFT_BUILD_FLAGS_$(CONFIGURATION))); \
	METAL_DIR=.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal; \
	INCLUDE_DIR=.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels; \
	if [ ! -d "$$METAL_DIR" ]; then \
		echo "ERROR: MLX checkout is missing. Run swift build first."; \
		exit $${REQUIRE_METALLIB:-0}; \
	fi; \
	if ! xcrun -sdk macosx -find metal >/dev/null 2>&1 || ! xcrun -sdk macosx -find metallib >/dev/null 2>&1; then \
		echo "ERROR: Metal Toolchain not installed. Run: xcodebuild -downloadComponent MetalToolchain"; \
		exit $${REQUIRE_METALLIB:-0}; \
	fi; \
	AIR_DIR=$$(mktemp -d "$${TMPDIR:-/tmp}/krillm-metallib.XXXXXX"); \
	trap 'rm -rf "$$AIR_DIR"' EXIT; \
	find "$$METAL_DIR" -type f -name '*.metal' -print | while IFS= read -r f; do \
		REL=$${f#$$METAL_DIR/}; \
		AIR=$$(printf '%s' "$$REL" | tr '/.' '__').air; \
		xcrun -sdk macosx metal -c -fmodules-cache-path="$$AIR_DIR/module-cache" -I"$$METAL_DIR" -I"$$INCLUDE_DIR" "$$f" -o "$$AIR_DIR/$$AIR"; \
		printf '%s\n' "$$AIR_DIR/$$AIR" >> "$$AIR_DIR/air-files"; \
	done; \
	if [ ! -s "$$AIR_DIR/air-files" ]; then \
		echo "ERROR: no MLX Metal shader sources found in $$METAL_DIR"; \
		exit $${REQUIRE_METALLIB:-0}; \
	fi; \
	xcrun -sdk macosx metallib $$(cat "$$AIR_DIR/air-files") -o "$$BUILD_PATH/mlx.metallib"; \
	mkdir -p "$$BUILD_PATH/Resources" "$$BUILD_PATH/mlx-swift_Cmlx.bundle"; \
	cp "$$BUILD_PATH/mlx.metallib" "$$BUILD_PATH/Resources/mlx.metallib"; \
	cp "$$BUILD_PATH/mlx.metallib" "$$BUILD_PATH/Resources/default.metallib"; \
	cp "$$BUILD_PATH/mlx.metallib" "$$BUILD_PATH/mlx-swift_Cmlx.bundle/default.metallib"; \
	find "$$BUILD_PATH" -path '*.xctest/Contents/MacOS' -type d -print | while IFS= read -r test_macos_dir; do \
		test_bundle=$${test_macos_dir%/Contents/MacOS}; \
		mkdir -p "$$test_macos_dir/Resources" "$$test_bundle/Contents/Resources/mlx-swift_Cmlx.bundle"; \
		cp "$$BUILD_PATH/mlx.metallib" "$$test_macos_dir/mlx.metallib"; \
		cp "$$BUILD_PATH/mlx.metallib" "$$test_macos_dir/Resources/mlx.metallib"; \
		cp "$$BUILD_PATH/mlx.metallib" "$$test_bundle/Contents/Resources/mlx-swift_Cmlx.bundle/default.metallib"; \
	done; \
	echo "Metal shaders compiled to $$BUILD_PATH/mlx.metallib"

# Optimized release build
release:
	swift build -c release --arch arm64
	$(MAKE) metallib CONFIGURATION=release REQUIRE_METALLIB=1
	@echo "Binary at $(BUILD_DIR)/$(BINARY_NAME)"
	@ls -lh $(BUILD_DIR)/$(BINARY_NAME)

# Install to PREFIX/bin
install: release
	install -d $(PREFIX)/bin
	install -m 755 $(BUILD_DIR)/$(BINARY_NAME) $(PREFIX)/bin/$(BINARY_NAME)
	install -m 644 $(BUILD_DIR)/mlx.metallib $(PREFIX)/bin/mlx.metallib
	install -d $(PREFIX)/bin/mlx-swift_Cmlx.bundle
	install -m 644 $(BUILD_DIR)/mlx-swift_Cmlx.bundle/default.metallib $(PREFIX)/bin/mlx-swift_Cmlx.bundle/default.metallib
	@echo "Installed $(BINARY_NAME) to $(PREFIX)/bin/"

# Remove from PREFIX/bin
uninstall:
	rm -f $(PREFIX)/bin/$(BINARY_NAME)
	rm -f $(PREFIX)/bin/mlx.metallib
	rm -rf $(PREFIX)/bin/mlx-swift_Cmlx.bundle
	@echo "Removed $(BINARY_NAME) from $(PREFIX)/bin/"

# Run tests
test:
	@mkdir -p .build/clang-module-cache
	$(SWIFT_ENV) swift build --build-tests
	$(MAKE) metallib CONFIGURATION=debug
	$(SWIFT_ENV) swift test --skip-build

# Quick benchmark (requires a model)
bench:
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make bench MODEL=llama-3.2-3b"; \
		exit 1; \
	fi
	$(BUILD_DIR)/$(BINARY_NAME) bench $(MODEL) --runs 3

# Reproducible local comparison against Ollama. Exits 77 when prerequisites are missing.
bench-compare:
	python3 tools/krillm_vs_ollama_benchmark.py \
		--krillm-bin $(BUILD_DIR)/$(BINARY_NAME) \
		--krill-model "$(KRILL_MODEL)" \
		--ollama-model "$(OLLAMA_MODEL)" \
		--prompt "$(BENCH_PROMPT)" \
		--max-tokens $(BENCH_MAX_TOKENS) \
		--runs $(BENCH_RUNS) \
		--warmup $(BENCH_WARMUP) \
		--output "$(BENCH_OUTPUT)"

# Clean build artifacts
clean:
	swift package clean
	rm -rf .build

# Build release tarball for distribution
dist: release
	mkdir -p dist
	tar -czf dist/krillm-$(VERSION)-arm64-apple-macos.tar.gz \
		-C $(BUILD_DIR) $(BINARY_NAME) mlx.metallib mlx-swift_Cmlx.bundle
	@echo "Tarball at dist/krillm-$(VERSION)-arm64-apple-macos.tar.gz"
	shasum -a 256 dist/krillm-$(VERSION)-arm64-apple-macos.tar.gz

# Print version
version:
	@echo $(VERSION)
