# Troubleshooting

## 🚨 Common Issues and Solutions

### Backend does not start
```bash
# Check port
lsof -i :3000

# Kill process if busy
kill -9 <PID>

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Frontend does not connect to API
1. Make sure the backend is running
2. Check the `frontend/.env` file (should contain `VITE_API_URL=http://localhost:3000`)
3. Check CORS settings in server.ts
4. Open http://localhost:3000/status to check

### WebSocket does not connect
```bash
# Check server status
curl http://localhost:3000/status

# Check WebSocket
curl -I http://localhost:3000/status
```

### System prompt does not load
1. Make sure `frontend/public/system-prompt.md` exists
2. Check that the frontend server is running
3. Open http://localhost:5173/system-prompt.md
4. Copy `system-prompt.md` from the root to `frontend/public/`

### Errors during generation
1. Check API keys in `.env`
2. Make sure the Fish Speech server is running on port 7860
3. Check backend logs in the console
4. Make sure the JSON format is correct

### Audio issues
1. Install FFmpeg: `brew install ffmpeg` (macOS)
2. Check permissions for `output/` and `generated-scripts/` folders
3. Make sure there is enough free space
4. Check that the Fish Speech API is available

### Progress does not update
1. Check browser console for WebSocket errors
2. Make sure the `ws` package is installed in the backend
3. Check that ports are not blocked by a firewall
4. Make sure jobId is passed correctly

## Semaphore Warnings

### Problem
```
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

### Cause
This warning appears due to improper cleanup of Python multiprocessing in fs-python.

### Solutions

#### 1. Reduce the number of parallel processes

Create/edit the `.env` file:
```env
CONCURRENT_REQUESTS=1
```

Or for faster processing (with possible warnings):
```env
CONCURRENT_REQUESTS=2
```

#### 2. Restart on critical errors

If you see many warnings, restart the process:
```bash
# Stop the current process (Ctrl+C)
# Then start again
npm run process
```

#### 3. Clear fs-python cache

```bash
cd fs-python
poetry run python cli_tts.py --clear-cache
cd ..
```

### Prevention

1. **Do not run** multiple instances simultaneously
2. **Wait** for the previous generation to finish
3. **Use** `CONCURRENT_REQUESTS=1` for maximum stability

## Poetry Errors

### Problem
```
Poetry not found or commands do not work
```

### Solution
```bash
# Reinstall Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Restart terminal
# Then check
cd fs-python
poetry check
```

## Memory Errors

### Problem
```
Killed: 9 or OOM (Out of Memory)
```

### Solution
```bash
# Reduce parallelism
echo "CONCURRENT_REQUESTS=1" > .env

# Free up memory
sudo purge  # on macOS
```

## MPS Errors (Apple Silicon)

### Problem
```
MPS unavailable or GPU errors
```

### Solution
```bash
# Check MPS availability
cd fs-python
poetry run python -c "import torch; print(torch.backends.mps.is_available())"

# If MPS is unavailable, use CPU:
poetry run python cli_tts.py "test" --device cpu
```

## Slow Generation

### Causes and Solutions

1. **First run** - models are loading (normal)
2. **Many parallel processes** - reduce `CONCURRENT_REQUESTS`
3. **No caching** - wait, subsequent runs will be faster
4. **CPU instead of MPS** - check Apple Silicon GPU availability

## File Errors

### Problem
```
Voice file not found or Audio file not created
```

### Solution
```bash
# Check voices
cd fs-python
poetry run python cli_tts.py --list-voices

# Check structure
ls -la voices/
ls -la ../output/
```

## Port Issues

### Problem
```
Port 3000 already in use
```

### Solution
```bash
# Find process on port 3000
lsof -ti:3000

# Kill process
kill -9 $(lsof -ti:3000)

# Or change port in .env
echo "PORT=3001" >> .env
```

## Dependency Issues

### Node.js dependencies
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Same for frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Python dependencies (Fish Speech)
```bash
cd fs-python
# Reinstall poetry environment
poetry env remove python
poetry install
```

## Filesystem Issues

### Permissions
```bash
# Check folder permissions
ls -la output/
ls -la generated-scripts/

# Fix permissions if needed
chmod 755 output/ generated-scripts/
```

### Disk space
```bash
# Check free space
df -h

# Clear temporary files
rm -rf fs-python/cache/*
rm -rf fs-python/output/*.wav
```

## General Recommendations

1. **One generation at a time** - do not run in parallel
2. **Be patient on first run** - models are loading
3. **Monitor resources** - watch memory and CPU
4. **Regular cleanup** - remove old files from `output/`
5. **Check logs** - always check console and logs
6. **Restart services** - sometimes a simple restart helps

## System Diagnostics

### Quick check
```bash
# Check all services
curl http://localhost:3000/status
curl http://localhost:5173  # frontend
curl http://localhost:7860  # Fish Speech (if running)

# Check processes
ps aux | grep node
ps aux | grep python
```

### Full diagnostics
```bash
# Check versions
node --version
npm --version
python3 --version

# Check installed packages
npm list --depth=0
cd frontend && npm list --depth=0

# Check Python environment
cd fs-python
poetry show
```

## Getting Help

If the problem is not resolved:

1. Check terminal logs
2. Make sure all dependencies are installed
3. Try generating a single test file
4. Create an Issue in the repository with a problem description
5. Attach logs and system information
6. Restart the system if nothing helps 