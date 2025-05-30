# Dia TTS Server Test Suite

This directory contains comprehensive unit tests for the Dia TTS FastAPI server.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and shared fixtures
├── test_worker_pool.py      # GPU management and worker pool tests
├── test_audio_processing.py # Audio generation and processing tests
├── test_api_endpoints.py    # API endpoint tests with mocking
└── test_job_queue.py        # Job queue system tests
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e ".[dev]"
# or
pip install pytest pytest-asyncio httpx
```

### Run All Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api_endpoints.py -v

# Run specific test class
pytest tests/test_worker_pool.py::TestGPUManagement -v

# Run specific test
pytest tests/test_worker_pool.py::TestGPUManagement::test_can_use_torch_compile_with_ampere -v
```

### Test Markers

Tests are marked with categories:
- `unit`: Unit tests (fast, mocked)
- `integration`: Integration tests (requires server)
- `slow`: Slow tests
- `gpu`: Tests requiring GPU

Run tests by marker:
```bash
# Only unit tests
pytest -m unit

# Skip GPU tests
pytest -m "not gpu"
```

## Test Coverage

The test suite covers:

### 1. Worker Pool & GPU Management (`test_worker_pool.py`)
- GPU detection and capability checking
- Single/multi-GPU model loading
- Worker pool initialization and management
- GPU assignment and round-robin scheduling
- Worker cleanup and shutdown

### 2. Audio Processing (`test_audio_processing.py`)
- Text preprocessing with speaker tags
- Voice mapping and role-based selection
- Audio generation with various parameters
- Speed adjustment and resampling
- Audio prompt handling
- File saving and cleanup
- Job processing workflow

### 3. API Endpoints (`test_api_endpoints.py`)
- Health and status endpoints
- Model and voice listing
- Synchronous/asynchronous generation
- Job status and results
- Voice mapping CRUD operations
- Audio prompt management
- Configuration endpoints
- OpenAI compatibility layer

### 4. Job Queue System (`test_job_queue.py`)
- Job model and status transitions
- Queue operations (add, remove, query)
- Job retention and cleanup
- Concurrent job processing
- Thread safety

## Writing New Tests

### Use Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_my_feature(mock_dia_model, sample_request_data):
    # mock_dia_model is pre-configured
    # sample_request_data has test data
    pass
```

### Mock External Dependencies

Always mock:
- Dia model loading/generation
- File I/O operations
- GPU/CUDA operations
- Network requests

```python
@patch('server.model')
@patch('torch.cuda.is_available', return_value=False)
def test_cpu_mode(mock_cuda, mock_model):
    # Test CPU mode behavior
    pass
```

### Test Edge Cases

Consider:
- Empty/invalid inputs
- Missing dependencies
- Hardware limitations
- Concurrent operations
- Error conditions

## Continuous Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -e ".[dev]"
    pytest tests/ -v --tb=short
```

## Troubleshooting

### Import Errors
Ensure the `src` directory is in the Python path. The `conftest.py` handles this automatically.

### Mock Issues
If mocks aren't working, check:
1. Import order (mock before importing server)
2. Patch targets (use full module paths)
3. Mock reset between tests

### Async Tests
Use `pytest-asyncio` for async endpoint tests:
```python
@pytest.mark.asyncio
async def test_async_endpoint():
    # Test async code
    pass
```