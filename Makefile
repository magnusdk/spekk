.PHONY: install-dev test clean

# Install development dependencies and setup array-api-tests
install-dev:
	uv sync --group dev
	@if [ ! -d "array-api-tests" ]; then \
		git submodule add https://github.com/data-apis/array-api-tests.git array-api-tests; \
	fi
	git submodule update --init --recursive
	uv pip install -r array-api-tests/requirements.txt

# Run all tests (spekk tests + Array API compliance)
test:
	pytest tests/
	ARRAY_API_TESTS_MODULE=spekk.ops ARRAY_API_TESTS_VERSION="2023.12" SPEKK_BACKEND=numpy pytest array-api-tests/

# Clean build artifacts and cache
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete