# KrillLM Build System
BINARY_NAME = krillm
BUILD_DIR = .build/release
PREFIX ?= /usr/local
VERSION = $(shell cat VERSION 2>/dev/null || echo "0.2.0")

.PHONY: build release install uninstall clean test bench

# Debug build (default)
build:
	swift build

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
