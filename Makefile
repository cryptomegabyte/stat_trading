# Crypto Trading Bot Demo - Makefile
# A comprehensi# Running the bot
run-paper:
	@echo "📈 Starting paper trading simulation..."
	cargo run -- --mode paper --model gas_vg

run-live:
	@echo "🚀 Starting live trading (ensure API keys are set)..."
	@echo "⚠️  WARNING: This will execute real trades!"
	cargo run -- --mode liveand development automation tool

.PHONY: help build release test clippy clean run-paper run-live run-backtest format check-all install-deps docs

# Default target
help:
	@echo "🚀 Crypto Trading Bot Demo - Available Commands:"
	@echo ""
	@echo "Build & Test:"
	@echo "  make build          - Build the project in debug mode"
	@echo "  make release        - Build the project in release mode (optimized)"
	@echo "  make test           - Run all tests"
	@echo "  make clippy         - Run clippy linter"
	@echo "  make check-all      - Run all checks (build, test, clippy)"
	@echo ""
	@echo "Running the Bot:"
	@echo "  make run-paper      - Run paper trading simulation (100k trades)"
	@echo "  make run-live       - Run live trading (10k trades, requires API keys)"
	@echo "  make run-backtest   - Run backtesting mode"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format code with rustfmt"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make docs           - Generate documentation"
	@echo "  make install-deps   - Install/update dependencies"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make dev            - Full development cycle (format, check-all)"
	@echo "  make paper          - Quick paper trading (format, build, run-paper)"

# Build targets
build:
	@echo "🔨 Building project in debug mode..."
	cargo build

release:
	@echo "⚡ Building project in release mode..."
	cargo build --release

# Testing and quality
test:
	@echo "🧪 Running tests..."
	cargo test

clippy:
	@echo "🔍 Running clippy linter..."
	cargo clippy

check-all: build test clippy
	@echo "✅ All checks passed!"

# Running the bot
run-paper:
	@echo "📈 Starting paper trading simulation..."
	cargo run -- --paper-trading --model gas_vg --max-trades 100000

run-live:
	@echo "🚀 Starting live trading (ensure API keys are set)..."
	@echo "⚠️  WARNING: This will execute real trades!"
	cargo run -- --mode live --model gas_vg --max-trades 10000

run-backtest:
	@echo "📊 Starting backtesting..."
	cargo run -- --backtest --model gas_vg

# Development tools
format:
	@echo "🎨 Formatting code..."
	cargo fmt

clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean

docs:
	@echo "📚 Generating documentation..."
	cargo doc --open --no-deps

install-deps:
	@echo "📦 Installing/updating dependencies..."
	cargo update

# Combined workflows
dev: format check-all
	@echo "🎯 Development cycle complete!"

paper: format build run-paper
	@echo "📈 Paper trading session started!"

# Model comparison (useful for development)
compare-models:
	@echo "🔬 Running model comparison backtest..."
	cargo run -- --backtest --model gas_vg
	cargo run -- --backtest --model gas_ghd
	cargo run -- --backtest --model gas_nig
	cargo run -- --backtest --model gas_gld
	cargo run -- --backtest --model hybrid_egarch_lstm

# Performance monitoring
bench:
	@echo "⚡ Running benchmarks..."
	cargo bench

# Docker support (if needed in future)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t crypto-trading-bot .

docker-run:
	@echo "🐳 Running in Docker..."
	docker run --rm crypto-trading-bot

# Environment setup
setup:
	@echo "🔧 Setting up development environment..."
	@echo "Make sure you have Rust installed: https://rustup.rs/"
	@echo "Setting up git hooks..."
	@echo "#!/bin/bash" > .git/hooks/pre-commit
	@echo "make format" >> .git/hooks/pre-commit
	@echo "make clippy" >> .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "✅ Development environment ready!"

# Quick status
status:
	@echo "📊 Project Status:"
	@echo "  Rust version: $$(rustc --version)"
	@echo "  Cargo version: $$(cargo --version)"
	@echo "  Target: $$(rustc -Vv | grep host | cut -d' ' -f2)"
	@echo "  Dependencies: $$(cargo tree | wc -l) packages"
	@echo ""
	@echo "  Build status:"
	@cargo check --quiet && echo "  ✅ Code compiles" || echo "  ❌ Code has compilation errors"
	@cargo test --quiet && echo "  ✅ Tests pass" || echo "  ❌ Tests failing"
	@cargo clippy --quiet && echo "  ✅ Clippy clean" || echo "  ❌ Clippy warnings"