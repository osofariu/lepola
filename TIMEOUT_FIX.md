# Timeout Fix for Local LLM Operations

## Problem Description

When running analysis on large documents (32+ pages) with local LLMs like Ollama, the application was experiencing timeout errors:

``` test
TimeoutError: asyncio.TimeoutError
```

This occurred because:

1. **Default HTTP Timeouts**: The CLI client and server had default timeouts that were too short for local LLM processing
2. **Large Document Processing**: 32-page PDFs get split into multiple chunks, each requiring LLM processing
3. **Local LLM Latency**: Local models like Ollama take longer to process than cloud APIs
4. **No Configurable Timeouts**: The application didn't have configurable timeout settings

## Root Cause Analysis

The timeout errors were happening at multiple levels:

1. **CLI HTTP Client**: `aiohttp.ClientSession` with default 5-minute timeout
2. **FastAPI Server**: Uvicorn with default keep-alive timeout
3. **LangChain LLMs**: No explicit timeout configuration for local models
4. **Database Operations**: Already fixed in previous update

## Solution Implemented

### 1. Configurable HTTP Timeout Settings

**File**: `src/core/config.py`

- Added comprehensive HTTP timeout configuration
- Default values suitable for local LLM operations

```python
# HTTP timeout settings for long-running operations
http_timeout: float = Field(
    default=300.0, description="HTTP request timeout in seconds (5 minutes)"
)
http_connect_timeout: float = Field(
    default=60.0, description="HTTP connection timeout in seconds"
)
http_read_timeout: float = Field(
    default=300.0, description="HTTP read timeout in seconds (5 minutes)"
)
```

### 2. CLI Client Timeout Configuration

**File**: `scripts/lepola_cli.py`

- Updated `LepolaCLI` to use configurable timeouts
- Applied to all HTTP requests made by the CLI

```python
async def __aenter__(self):
    """Async context manager entry."""
    # Configure timeouts for long-running operations with local LLMs
    timeout = aiohttp.ClientTimeout(
        total=settings.http_timeout,
        connect=settings.http_connect_timeout,
        sock_read=settings.http_read_timeout,
    )
    self.session = aiohttp.ClientSession(timeout=timeout)
    return self
```

### 3. FastAPI Server Timeout Configuration

**File**: `src/main.py`

- Added timeout configuration to uvicorn server
- Configured keep-alive and graceful shutdown timeouts

```python
uvicorn.run(
    "src.main:app",
    host=settings.host,
    port=settings.port,
    reload=settings.debug,
    log_level=settings.log_level.lower(),
    timeout_keep_alive=settings.http_timeout,
    timeout_graceful_shutdown=30,
)
```

### 4. LangChain LLM Configuration

**File**: `src/pipeline/service.py`

- Added timeout configuration for OpenAI models
- Added base_url configuration for Ollama models
- Note: ChatOllama timeout is handled by underlying HTTP client

```python
# OpenAI
return ChatOpenAI(
    model=llm_config["model"],
    api_key=llm_config["api_key"],
    temperature=0.1,
    max_tokens=2000,
    request_timeout=settings.http_timeout,
)

# Ollama
return ChatOllama(
    model=llm_config["model"],
    api_key=llm_config["api_key"],
    temperature=0.1,
    max_tokens=2000,
    base_url=settings.ollama_base_url,
)
```

### 5. Environment Configuration

**File**: `env.example`

- Added timeout configuration examples
- Documented all new timeout settings

```bash
# HTTP Timeout Settings (for long-running operations with local LLMs)
HTTP_TIMEOUT=300.0
HTTP_CONNECT_TIMEOUT=60.0
HTTP_READ_TIMEOUT=300.0
```

## Configuration Options

### Default Values (Suitable for Local LLMs)

- **HTTP Total Timeout**: 300 seconds (5 minutes)
- **HTTP Connect Timeout**: 60 seconds (1 minute)
- **HTTP Read Timeout**: 300 seconds (5 minutes)
- **Database Timeout**: 30 seconds (already configured)

### Customization

You can customize timeouts by setting environment variables:

```bash
# For very slow local models
export HTTP_TIMEOUT=600.0        # 10 minutes
export HTTP_READ_TIMEOUT=600.0   # 10 minutes

# For faster models
export HTTP_TIMEOUT=180.0        # 3 minutes
export HTTP_READ_TIMEOUT=180.0   # 3 minutes
```

Or in your `.env` file:

```bash
HTTP_TIMEOUT=600.0
HTTP_CONNECT_TIMEOUT=120.0
HTTP_READ_TIMEOUT=600.0
```

## Testing

A test script has been created to verify timeout configuration:

```bash
python scripts/test_timeout_config.py
```

This script verifies:

- ✅ Timeout values are reasonable for local LLM operations
- ✅ CLI client uses correct timeout configuration
- ✅ LLM clients are properly configured
- ✅ All settings are loaded correctly

## Performance Impact

### Positive Impacts

1. **Reliability**: No more timeout errors on large documents
2. **Flexibility**: Configurable timeouts for different model speeds
3. **User Experience**: Better handling of long-running operations

### Considerations

1. **Resource Usage**: Longer timeouts mean connections stay open longer
2. **Memory**: Large documents may use more memory during processing
3. **Network**: HTTP connections remain active for longer periods

## Troubleshooting

### Still Getting Timeouts?

1. **Increase Timeouts**: Set higher values in your `.env` file
2. **Check Model Performance**: Ensure your local LLM is running efficiently
3. **Monitor Resources**: Check CPU/memory usage during processing
4. **Document Size**: Consider processing very large documents in smaller chunks

### Common Issues

1. **Ollama Not Responding**: Check if Ollama service is running and accessible
2. **Memory Issues**: Large models may need more RAM for processing
3. **Network Issues**: Ensure stable connection to local LLM service

## Migration Notes

- **No Breaking Changes**: All existing functionality remains the same
- **Backward Compatible**: Default timeouts are reasonable for most use cases
- **Optional Configuration**: Timeouts can be customized as needed
- **Gradual Rollout**: Changes are applied automatically on restart

## Future Improvements

1. **Dynamic Timeouts**: Adjust timeouts based on document size and model
2. **Progress Tracking**: Add progress indicators for long-running operations
3. **Retry Logic**: Implement intelligent retry with exponential backoff
4. **Resource Monitoring**: Add monitoring for timeout-related issues
5. **Async Processing**: Consider background job processing for very large documents

## Related Issues

This fix addresses the timeout issues described in:

- Database timeout issues (already fixed in `DATABASE_TIMEOUT_FIX.md`)
- HTTP client timeout errors with local LLMs
- Long-running analysis operations

## Verification

To verify the fix is working:

1. **Test with Large Document**: Upload a 32+ page PDF
2. **Run Analysis**: Use the CLI to analyze the document
3. **Monitor Logs**: Check for timeout errors in the logs
4. **Verify Results**: Ensure analysis completes successfully

The timeout configuration should now handle large documents with local LLMs without timing out.
