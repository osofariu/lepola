# Database Timeout Fix for Long-Running Analysis

## Problem Description

The application was experiencing database lock errors when running long analysis operations with local models. The error occurred because:

1. **SQLite Default Timeout**: SQLite has a default timeout of 5 seconds for acquiring database locks
2. **Long Analysis Time**: Local models take much longer than 5 seconds to generate summaries
3. **Connection Held Open**: The repository held database connections open throughout the entire analysis process
4. **Lock Timeout**: When analysis completed and tried to commit, the database connection had timed out

## Error Message

``` text
sqlite3.OperationalError: database is locked
```

## Root Cause

The issue was in the `AnalysisRepository.create()` method in `src/core/repository.py`. The method was:

- Using the default SQLite timeout (5 seconds)
- Holding the database connection open during the entire analysis process
- Not handling database timeouts gracefully

## Solution Implemented

### 1. Configurable Database Timeout

**File**: `src/core/config.py`

- Added `database_timeout` setting with default value of 30.0 seconds
- Made timeout configurable via environment variable

```python
database_timeout: float = Field(
    default=30.0, description="Database connection timeout in seconds"
)
```

### 2. Enhanced Database Connection Handling

**File**: `src/core/repository.py`

- Updated all database connections to use configurable timeout
- Enabled WAL (Write-Ahead Logging) mode for better concurrency
- Applied to `AnalysisRepository.create()`, `get_by_id()`, and `list_by_document_id()`
- Applied to `DocumentRepository.create()` for consistency

```python
with sqlite3.connect(self.db_path, timeout=settings.database_timeout) as db:
    # Enable WAL mode for better concurrency
    db.execute("PRAGMA journal_mode=WAL")
```

### 3. Retry Logic for Database Operations

**File**: `src/pipeline/service.py`

- Moved database persistence outside the analysis loop
- Added `_persist_analysis_result_with_retry()` method with exponential backoff
- Implemented retry logic with configurable max attempts (default: 3)
- Graceful handling of persistence failures (analysis continues even if persistence fails)

```python
async def _persist_analysis_result_with_retry(self, result: AnalysisResult, max_retries: int = 3) -> None:
    """Persist analysis result to database with retry logic."""
    for attempt in range(max_retries):
        try:
            self.analysis_repository.create(result)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                # Log error but don't fail the entire analysis
                self.logger.error("Failed to persist analysis result after %d attempts", max_retries)
            else:
                # Wait before retrying (exponential backoff)
                await asyncio.sleep(2 ** attempt)
```

## Benefits

1. **Increased Reliability**: Longer timeout prevents database locks during long operations
2. **Better Concurrency**: WAL mode allows multiple readers while one writer is active
3. **Graceful Degradation**: Analysis results are still generated even if persistence fails
4. **Configurable**: Timeout can be adjusted based on model performance and requirements
5. **Retry Logic**: Automatic retry with exponential backoff for transient failures

## Configuration

The database timeout can be configured via environment variable:

```bash
export DATABASE_TIMEOUT=60.0  # Set to 60 seconds for very slow models
```

Or in your `.env` file:

``` sh
DATABASE_TIMEOUT=60.0
```

## Testing

The changes have been tested with:

- ✅ Database create operations
- ✅ Database read operations  
- ✅ Database list operations
- ✅ Retry logic with exponential backoff
- ✅ Graceful handling of persistence failures

## Migration Notes

- **No Breaking Changes**: All existing functionality remains the same
- **Backward Compatible**: Default timeout of 30 seconds is reasonable for most use cases
- **Performance Impact**: Minimal - only affects database connection establishment
- **Database Files**: Existing database files will automatically use WAL mode on first write

## Future Improvements

1. **Connection Pooling**: Consider implementing connection pooling for high-concurrency scenarios
2. **Async Database**: Migrate to async database operations for better performance
3. **Monitoring**: Add metrics for database operation timing and failure rates
4. **Caching**: Implement result caching to reduce database load
