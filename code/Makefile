.PHONY: setup install test test-data test-prompts test-evaluation lint clean

# 项目根目录
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# 创建 conda 环境
setup:
	conda create -n society-rl python=3.10 -y
	@echo "Run: conda activate society-rl && pip install -e '.[dev]'"

# 安装项目
install:
	pip install -e '.[dev]'

# 运行所有测试
test:
	python -m pytest tests/ -v

# 运行特定模块测试
test-data:
	python -m pytest tests/test_data.py -v

test-prompts:
	python -m pytest tests/test_paired_prompts.py -v

test-evaluation:
	python -m pytest tests/test_evaluation.py -v

# 代码检查
lint:
	python -m ruff check src/ scripts/ tests/

# 清理
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .ruff_cache/
