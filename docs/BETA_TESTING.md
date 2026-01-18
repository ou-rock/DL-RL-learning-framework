# Beta Testing Guide

Thank you for being a beta tester! Your feedback helps improve the Learning Framework.

## How to Report Issues

### Using the CLI

```bash
# Report a bug
lf feedback bug --title "Quiz crashes" --description "Details here"

# Request a feature
lf feedback feature --title "Dark mode" --description "Details here"

# Report usability issue
lf feedback usability --title "Confusing UI" --description "Details here"
```

### What to Include

**For Bugs:**
- Clear title describing the issue
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

**For Features:**
- Clear description of desired functionality
- Use case explaining why it's useful
- Any examples from other tools

**For Usability:**
- What was confusing
- What you were trying to do
- How you expected it to work

## Feedback Categories

| Category | When to Use |
|----------|-------------|
| Bug | Something doesn't work correctly |
| Feature | New functionality you'd like |
| Usability | Confusing UI or workflow |
| Performance | Slow operations |
| Documentation | Missing or unclear docs |

## Known Issues

Check [GitHub Issues](https://github.com/learning-framework/issues) before reporting.

## What We're Looking For

During beta testing, we especially want feedback on:

1. **Installation experience** - Was setup straightforward?
2. **Learning flow** - Does the tier progression make sense?
3. **Quiz effectiveness** - Do questions test understanding?
4. **Challenge difficulty** - Are challenges appropriate?
5. **Error messages** - Are errors helpful?
6. **Documentation** - Is anything missing or unclear?

## Beta Tester Checklist

Try these scenarios and report any issues:

- [ ] Fresh installation from git clone
- [ ] Configure with your own materials
- [ ] Complete one full learning session
- [ ] Take at least 3 quizzes
- [ ] Attempt one implementation challenge
- [ ] View visualizations in browser
- [ ] Use daily review feature for a week
- [ ] (Optional) Submit a GPU job

## Viewing Your Feedback

```bash
# List all submitted feedback
lf feedback list

# Filter by type
lf feedback list --type bug

# Export to file
lf feedback export feedback_report.json
```

## Thank You!

Your feedback directly shapes the framework. All beta testers will be acknowledged in the release notes.
