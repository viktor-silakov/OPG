.PHONY: install test test-e2e test-fast cleanup help

# 🎯 Default target
help: ## Show this help message
	@echo "Fish Speech E2E Test Management"
	@echo "=============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# 📦 Dependencies
install: ## Install all dependencies including test dependencies
	poetry install --with test

install-prod: ## Install only production dependencies
	poetry install --without test

# 🧪 Testing
test: ## Run full E2E test suite
	poetry run python tests/test_e2e_training.py

test-pytest: ## Run tests via pytest
	poetry run pytest tests/test_e2e_training.py -v

test-fast: ## Run tests excluding slow E2E tests
	poetry run pytest -m "not slow" -v

test-verbose: ## Run tests with verbose output
	poetry run pytest tests/test_e2e_training.py -v -s

test-html: ## Run tests and generate HTML report
	poetry run pytest tests/test_e2e_training.py -v --html=tests/report.html --self-contained-html --json-report --json-report-file=tests/report.json

# 🧹 Cleanup
cleanup: ## Clean up test data and checkpoints
	poetry run python tests/cleanup.py

cleanup-all: cleanup ## Same as cleanup (alias)

# 🔍 System checks
check-system: ## Check system requirements for training
	poetry run python -c "from finetune_tts import check_requirements; print('✅ System OK' if check_requirements() else '❌ System Issues')"

check-gpu: ## Check GPU/MPS availability
	poetry run python -c "import torch; print(f'🔥 MPS: {torch.backends.mps.is_available()}'); print(f'🚀 CUDA: {torch.cuda.is_available()}')"

# 📊 Reports
show-report: ## Show latest test report
	@if [ -f tests/test_report.json ]; then cat tests/test_report.json | python -m json.tool; else echo "❌ No test report found. Run 'make test' first."; fi

# 🛠️ Development
update-deps: ## Update dependencies and lock file
	poetry update

lock: ## Update lock file only
	poetry lock --no-update

# 🚀 Quick commands
quick-test: install test ## Install dependencies and run full test
test-with-report: install test-html open-report ## Run full test with HTML report and open it
open-report: ## Open HTML test report in browser
	@if [ -f tests/report.html ]; then open tests/report.html; else echo "❌ No HTML report found. Run 'make test-html' first."; fi
dev-setup: install check-system ## Set up development environment 