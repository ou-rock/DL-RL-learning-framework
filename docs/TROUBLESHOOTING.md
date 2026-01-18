# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### "command not found: lf"

**Problem:** The `lf` command is not recognized.

**Solution:**
1. Ensure you installed with `pip install -e .`
2. Check your PATH includes pip scripts:
   ```bash
   python -m site --user-base
   # Add the 'bin' subdirectory to PATH
   ```
3. Try running directly: `python -m learning_framework.cli`

### "ModuleNotFoundError"

**Problem:** Python can't find learning_framework.

**Solution:**
1. Verify installation: `pip show learning-framework`
2. Reinstall: `pip install -e .`
3. Check you're using the correct Python environment

## Configuration Issues

### "Config file not found"

**Problem:** Configuration file is missing.

**Solution:**
```bash
cp user_data/config-example.yaml user_data/config.yaml
```

### "materials_directories is empty"

**Problem:** No learning materials configured.

**Solution:**
Edit `user_data/config.yaml` and add your materials path:
```yaml
materials_directories:
  - "/path/to/your/DeepLearning"
```

## Quiz Issues

### "No quiz questions available"

**Problem:** Concept has no quiz questions.

**Causes:**
1. Quiz file not found for concept
2. JSON format error in quiz file

**Solution:**
1. Check `data/quizzes/<concept>.json` exists
2. Validate JSON: `python -c "import json; json.load(open('data/quizzes/<concept>.json'))"`
3. Run `lf index` to refresh

### "Concept not found"

**Problem:** Specified concept doesn't exist.

**Solution:**
1. List available concepts: `lf learn` (shows menu)
2. Check spelling (it's case-sensitive)
3. Run `lf index` if you added new materials

## Challenge Issues

### "Challenge template not found"

**Problem:** Challenge file missing.

**Solution:**
1. List available challenges: `lf challenge --list`
2. Verify file exists: `ls data/challenges/`
3. Check challenge name spelling

### "Test failed: module not found"

**Problem:** Your implementation can't be imported.

**Causes:**
1. Syntax error in your code
2. File saved to wrong location

**Solution:**
1. Check for Python syntax: `python -m py_compile your_file.py`
2. Verify location: files should be in `user_data/implementations/`
3. Check the expected function names match

### "Gradient check failed"

**Problem:** Your gradients don't match numerical gradients.

**Causes:**
1. Bug in backward pass
2. Numerical precision issues
3. Missing gradient terms

**Solution:**
1. Review the chain rule application
2. Check for off-by-one errors in indices
3. Verify all intermediate gradients are computed
4. Use smaller epsilon (1e-5) in gradient checking

## Visualization Issues

### "Port already in use"

**Problem:** Visualization server can't start.

**Solution:**
```bash
# Kill existing process on port
lsof -ti:8765 | xargs kill -9

# Or use a different port
lf viz --port 8766
```

### "Visualization not loading"

**Problem:** Browser shows blank page or error.

**Solution:**
1. Check browser console for errors (F12)
2. Ensure JavaScript is enabled
3. Try a different browser
4. Check firewall settings

## GPU Backend Issues

### "API key not set"

**Problem:** Vast.ai API key not configured.

**Solution:**
1. Get API key from https://vast.ai
2. Add to config:
   ```yaml
   vastai_api_key: "your-key-here"
   ```

### "Budget exceeded"

**Problem:** Job cost exceeds budget limits.

**Solution:**
1. Wait for daily budget reset
2. Reduce epochs or batch size
3. Use `--estimate` flag to check cost first
4. Adjust budget limits in config

### "No GPU instances available"

**Problem:** Can't find suitable GPU.

**Causes:**
1. All instances rented
2. Requirements too strict

**Solution:**
1. Try again later
2. Reduce GPU RAM requirements
3. Use `lf scale --list-gpus` to see availability

## C++ Build Issues

### "CMake not found"

**Problem:** CMake is not installed.

**Solution:**
- macOS: `brew install cmake`
- Ubuntu: `sudo apt install cmake`
- Windows: Download from https://cmake.org

### "pybind11 not found"

**Problem:** pybind11 package missing.

**Solution:**
```bash
pip install pybind11
```

### "Compiler not found"

**Problem:** C++ compiler not available.

**Solution:**
- macOS: `xcode-select --install`
- Ubuntu: `sudo apt install build-essential`
- Windows: Install Visual Studio Build Tools

### "Build failed with errors"

**Problem:** C++ compilation errors.

**Solution:**
1. Check error message for specific issue
2. Verify C++17 support: `g++ --version` (need 7.0+)
3. Clear build cache: `rm -rf cpp/build && python cpp/build.py`
4. Use Python fallback: the framework works without C++

## Database Issues

### "Database locked"

**Problem:** SQLite database is locked.

**Causes:**
1. Another process using the database
2. Corrupted database file

**Solution:**
1. Close other lf instances
2. If corrupted, backup and recreate:
   ```bash
   mv user_data/progress.db user_data/progress.db.bak
   lf progress  # Creates new database
   ```

## Getting More Help

### Enable Verbose Output

```bash
lf --verbose <command>
```

### Check Logs

Logs are stored in `user_data/logs/` (if enabled in config).

### Report Issues

If you can't solve your issue:
1. Check existing issues on GitHub
2. Include:
   - Python version: `python --version`
   - OS: `uname -a` or system info
   - Full error message
   - Steps to reproduce
