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
GEMMA4_BENCH_OUTPUT ?= .build/benchmarks/gemma4-e2b-multimodal-4bit.json
GEMMA4_KRILL_MODEL ?= $(HOME)/.krillm/models/blobs/gemma-4-e2b
GEMMA4_OLLAMA_MODEL ?= gemma4:e2b
GEMMA4_BENCH_RUNS ?= 3
GEMMA4_BENCH_WARMUP ?= 1
KRILLM_PYTHON ?= $(HOME)/.krillm/venv/bin/python3
KRILLM_VENV_PYTHON ?= python3

.PHONY: build release install uninstall clean test bench bench-compare bench-concurrent bench-gemma4-multimodal bench-release-gate parity-gate metallib dist dist-app app-bundle version

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

KRILLM_URL ?=

# Reproducible local comparison against Ollama. Exits 77 when prerequisites are missing.
# Use KRILLM_URL to benchmark against a running KrillLM server (warm-server mode).
bench-compare:
	@if [ -n "$(KRILLM_URL)" ]; then \
		python3 tools/krillm_vs_ollama_benchmark.py \
			--krillm-url "$(KRILLM_URL)" \
			--krill-model "$(KRILL_MODEL)" \
			--ollama-model "$(OLLAMA_MODEL)" \
			--prompt "$(BENCH_PROMPT)" \
			--max-tokens $(BENCH_MAX_TOKENS) \
			--runs $(BENCH_RUNS) \
			--warmup $(BENCH_WARMUP) \
			--output "$(BENCH_OUTPUT)"; \
	else \
		python3 tools/krillm_vs_ollama_benchmark.py \
			--krillm-bin $(BUILD_DIR)/$(BINARY_NAME) \
			--krill-model "$(KRILL_MODEL)" \
			--ollama-model "$(OLLAMA_MODEL)" \
			--prompt "$(BENCH_PROMPT)" \
			--max-tokens $(BENCH_MAX_TOKENS) \
			--runs $(BENCH_RUNS) \
			--warmup $(BENCH_WARMUP) \
			--output "$(BENCH_OUTPUT)"; \
	fi

# Concurrent-throughput sweep (aggregate tok/s under N simultaneous streams),
# the axis where KrillLM's continuous batcher beats Ollama. Server-mode only:
# point at a running KrillLM server (launch with KRILL_NUM_PARALLEL>=16, and
# KRILL_NGRAM_SPEC=1 for the low-concurrency n-gram win) and/or an Ollama daemon.
# A serial-vs-batched A/B (find the crossover N*) is two runs with the KrillLM
# server launched at KRILL_NUM_PARALLEL=1 then =16, passing SERVER_ARM=serial/batched.
CONCURRENCY_SWEEP ?= 1,2,4,8,16
BENCH_CONCURRENT_OUTPUT ?= .build/benchmarks/concurrent-throughput.json
SERVER_ARM ?= unspecified
bench-concurrent:
	@python3 tools/krillm_concurrent_benchmark.py \
		$(if $(KRILLM_URL),--krillm-url "$(KRILLM_URL)" --krill-model "$(KRILL_MODEL)",) \
		$(if $(OLLAMA_HOST),--ollama-host "$(OLLAMA_HOST)" --ollama-model "$(OLLAMA_MODEL)",) \
		--concurrency-sweep "$(CONCURRENCY_SWEEP)" \
		--max-tokens $(BENCH_MAX_TOKENS) \
		--runs $(BENCH_RUNS) --warmup $(BENCH_WARMUP) \
		--server-arm "$(SERVER_ARM)" \
		--output "$(BENCH_CONCURRENT_OUTPUT)"

# Gemma 4 text/image/audio comparison against Ollama. All KrillLM paths are
# native (the mlx-vlm bridge was removed in WS6 Step 4): native_cli runs the
# release binary per request — no server or Python deps. Requires `make
# release` first (uses .build/release/krillm). For server-mode numbers, run
# the script directly with --krillm-url + --krillm-image-mode native_server.
bench-gemma4-multimodal: release
	$(KRILLM_PYTHON) tools/gemma4_multimodal_benchmark.py \
		--krill-model "$(GEMMA4_KRILL_MODEL)" \
		--ollama-model "$(GEMMA4_OLLAMA_MODEL)" \
		--krillm-image-mode native_cli \
		--runs $(GEMMA4_BENCH_RUNS) \
		--warmup $(GEMMA4_BENCH_WARMUP) \
		--output "$(GEMMA4_BENCH_OUTPUT)"

GATE_REPORT ?= .build/benchmarks/release-gate.json
GATE_ALLOW_FLAGS ?=

# Release benchmark gate. Evaluates benchmark reports against performance thresholds.
# Usage:
#   make bench-release-gate                                  # uses default multimodal report
#   make bench-release-gate GATE_INPUT=path/to/report.json   # custom combined report
#   make bench-release-gate GATE_KRILLM=k.json GATE_OLLAMA=o.json  # sequential comparison
#   make bench-release-gate GATE_ALLOW_FLAGS="--allow-dtype-mismatch"
bench-release-gate:
	@if [ -n "$(GATE_KRILLM)" ] && [ -n "$(GATE_OLLAMA)" ]; then \
		python3 tools/release_gate.py \
			--krillm-report "$(GATE_KRILLM)" \
			--ollama-report "$(GATE_OLLAMA)" \
			--output "$(GATE_REPORT)" \
			$(GATE_ALLOW_FLAGS); \
	else \
		python3 tools/release_gate.py \
			"$${GATE_INPUT:-$(GEMMA4_BENCH_OUTPUT)}" \
			--output "$(GATE_REPORT)" \
			$(GATE_ALLOW_FLAGS); \
	fi

# macOS Ollama parity gate. Boots `krillm serve` and asserts Ollama-client
# response-shape parity per docs/OLLAMA_MAC_PARITY_PLAN.md.
# Usage:
#   make parity-gate                          # mac_parity profile
#   make parity-gate PARITY_PROFILE=strict_parity
#   make parity-gate PARITY_ARGS="--base-url http://127.0.0.1:57455"
PARITY_PROFILE ?= mac_parity
PARITY_ARGS ?=
parity-gate: build
	python3 tools/parity_gate.py --profile $(PARITY_PROFILE) $(PARITY_ARGS)

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

# macOS .app bundle. Live microphone capture (`/mic` in interactive chat) needs
# the OS to attribute mic access to KrillLM rather than the parent terminal,
# which requires running inside a code-signed bundle that declares
# NSMicrophoneUsageDescription. This target produces that bundle: the release
# binary + adjacent metallib (the MLX loader searches the executable directory
# first) + Info.plist, ad-hoc signed for local TCC.
APP_DIR ?= dist/krillm.app
BUNDLE_ID ?= io.github.srvsngh99.krillm
CODESIGN_ID ?= -
MLX_BUNDLE ?= mlx-swift_Cmlx.bundle

app-bundle: release
	@echo "Packaging $(APP_DIR)..."
	@rm -rf "$(APP_DIR)"
	@mkdir -p "$(APP_DIR)/Contents/MacOS" "$(APP_DIR)/Contents/Resources/$(MLX_BUNDLE)"
	@cp "$(BUILD_DIR)/$(BINARY_NAME)" "$(APP_DIR)/Contents/MacOS/$(BINARY_NAME)"
	# The metallib must live in Contents/Resources (codesign rejects non-code
	# files in Contents/MacOS). The KLMRuntime loader searches the SPM resource
	# bundle Contents/Resources/$(MLX_BUNDLE)/default.metallib, so reproduce that
	# bundle here - WITH its own Info.plist so `codesign --deep` accepts it as a
	# nested bundle (a bare *.bundle folder is rejected as "unsuitable format").
	@cp "$(BUILD_DIR)/mlx.metallib" "$(APP_DIR)/Contents/Resources/$(MLX_BUNDLE)/default.metallib"
	@printf '%s\n' \
		'<?xml version="1.0" encoding="UTF-8"?>' \
		'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
		'<plist version="1.0"><dict>' \
		'  <key>CFBundleIdentifier</key><string>$(BUNDLE_ID).mlx</string>' \
		'  <key>CFBundleName</key><string>$(MLX_BUNDLE)</string>' \
		'  <key>CFBundlePackageType</key><string>BNDL</string>' \
		'  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>' \
		'</dict></plist>' > "$(APP_DIR)/Contents/Resources/$(MLX_BUNDLE)/Info.plist"
	@printf '%s\n' \
		'<?xml version="1.0" encoding="UTF-8"?>' \
		'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
		'<plist version="1.0">' \
		'<dict>' \
		'  <key>CFBundleIdentifier</key><string>$(BUNDLE_ID)</string>' \
		'  <key>CFBundleName</key><string>KrillLM</string>' \
		'  <key>CFBundleExecutable</key><string>$(BINARY_NAME)</string>' \
		'  <key>CFBundlePackageType</key><string>APPL</string>' \
		'  <key>CFBundleShortVersionString</key><string>$(VERSION)</string>' \
		'  <key>CFBundleVersion</key><string>$(VERSION)</string>' \
		'  <key>LSMinimumSystemVersion</key><string>14.0</string>' \
		'  <key>NSMicrophoneUsageDescription</key><string>KrillLM records microphone audio for local voice input to on-device speech and multimodal models.</string>' \
		'</dict>' \
		'</plist>' > "$(APP_DIR)/Contents/Info.plist"
	@codesign --force --deep --sign "$(CODESIGN_ID)" --identifier "$(BUNDLE_ID)" "$(APP_DIR)" \
		&& echo "Signed $(APP_DIR) (identity=$(CODESIGN_ID))" \
		|| echo "WARNING: codesign failed - /mic permission may attribute to the terminal instead of KrillLM."
	@echo "App bundle at $(APP_DIR)"
	@echo "Run interactive chat (mic enabled):"
	@echo "  $(APP_DIR)/Contents/MacOS/$(BINARY_NAME) run gemma-4-e2b"

# Zip the .app bundle for distribution.
dist-app: app-bundle
	@mkdir -p dist
	@cd dist && zip -qry "krillm-$(VERSION)-arm64-apple-macos-app.zip" "krillm.app"
	@echo "Tarball at dist/krillm-$(VERSION)-arm64-apple-macos-app.zip"
	@shasum -a 256 "dist/krillm-$(VERSION)-arm64-apple-macos-app.zip"

# Print version
version:
	@echo $(VERSION)
