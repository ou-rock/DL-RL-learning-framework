# Interactive DL/RL Mastery Framework - Design Document

**Date:** 2026-01-12
**Author:** LH
**Status:** Draft - Ready for Implementation

---

## Executive Summary

An interactive learning system for mastering deep learning and reinforcement learning fundamentals through progressive hands-on practice. The system enables learning on minimal local resources (CPU-only) and validates understanding by scaling implementations to remote GPU environments. Core algorithms are implemented in both Python (for learning) and C++ (for deep systems understanding).

### Core Objectives

1. **Visualize concepts** - Interactive and static visualizations of DL/RL fundamentals
2. **Integrate existing materials** - Auto-index and organize current learning resources (textbooks, code examples)
3. **Progressive mastery testing** - Three-tier validation: concept quiz → implementation → GPU scaling
4. **Minimal local resources** - Learn effectively on CPU without GPU requirements
5. **Deep systems understanding** - Implement critical algorithms in C++ to understand memory, numerical stability, and performance

---

## System Architecture

### Three-Tier Architecture

**Layer 1: Computation Core**
- Pure Python reference implementations for all algorithms
- C++ performance kernels for student implementations
- Unified abstraction layer supporting local CPU and remote GPU execution
- CMake build system with automatic compilation and Python fallback

**Layer 2: Learning Environment (Local)**
- Terminal-based CLI for navigation, quizzes, and progress tracking
- Lightweight web dashboard for browser-based visualizations
- Material indexer that scans existing DeepLearning/ and RL/ folders
- Progress tracker using local SQLite and JSON files

**Layer 3: Remote Execution Engine**
- Job packaging and submission system
- Multi-backend support (Google Colab, Vast.ai, SSH servers)
- Results validator comparing implementations against baselines
- Cost tracking and budget controls

---

## Detailed Component Design

### Component 1: Material Indexer & Knowledge Graph

**Purpose:** Discover and organize existing learning materials, build prerequisite relationships.

**Auto-Discovery Scanner:**
- Walks directory structure looking for patterns:
  - Sequential chapters: `ch01/`, `ch02/`, etc.
  - README.md files as topic descriptions
  - Python files with ML keywords (`backprop`, `gradient`, `Q-learning`)
- Scans existing projects: `深度学习入门/`, `dezero自制框架/`, `Hands-on-RL/`

**Knowledge Graph Structure:**
```python
{
  "concept": "backpropagation",
  "prerequisites": ["gradients", "chain_rule"],
  "difficulty": "intermediate",
  "materials": [
    {"type": "explanation", "path": "深度学习入门/ch05/README.md"},
    {"type": "code", "path": "深度学习入门/ch05/train_neuralnet.py"},
    {"type": "visualization", "path": "dezero自制框架/step10.py"}
  ],
  "tags": ["neural_networks", "optimization"]
}
```

**Manual Annotation System:**
- Simple comment tags in files: `# @concept: backprop @best-example`
- YAML sidecar files for complex relationships
- CLI commands: `lf tag <file> --concept backprop --difficulty easy`

**Implementation:**
- Convention-based scanning (80% automated)
- Manual tagging for refinement (20% curation)
- Builds directed acyclic graph of concept dependencies

---

### Component 2: Visualization Engine

**Purpose:** Create visual representations of abstract concepts to aid understanding.

**Static Visualizations (Matplotlib):**
- Network architecture diagrams
- Training curves (loss, accuracy over epochs)
- Gradient flow through layers
- Export as PNG, display in terminal or browser

**Interactive Visualizations (Vanilla JS + Canvas):**
- Parameter sliders (learning rate, batch size) → see real-time training effects
- Computational graph explorer (click nodes to inspect tensor values, gradients)
- RL environment viewer (watch agent policy improve over episodes)
- 3D activation landscapes (rotate, zoom)

**Implementation Approach:**
- Python backend generates JSON data: `{"epoch": [1,2,3], "loss": [0.5, 0.3, 0.1]}`
- Simple HTML templates with vanilla JavaScript (< 500 lines total)
- WebSocket connection for real-time updates during training
- No heavy frameworks - runs on any browser

**Technology Stack:**
- Static: matplotlib, seaborn
- Interactive: Python http.server + vanilla JS/Canvas
- Real-time: WebSockets for live training visualization

---

### Component 3: Assessment & Mastery System

**Purpose:** Verify understanding through three progressive tiers of increasing rigor.

#### Tier 1: Concept Understanding (Quiz System)

Extends existing German vocabulary quiz architecture from `skills/tools/`.

**Database Structure:**
```python
{
  "concept_id": "backprop_001",
  "concept": "backpropagation",
  "question_type": "multiple_choice",
  "question": "Why does vanishing gradient happen with sigmoid activation?",
  "options": ["A: Sigmoid outputs 0-1", "B: Derivative max is 0.25", ...],
  "correct": "B",
  "explanation": "Sigmoid derivative max is 0.25...",
  "difficulty": 2,
  "last_reviewed": "2026-01-10",
  "success_count": 3,
  "fail_count": 1
}
```

**Question Types:**
- Multiple choice (concept comprehension)
- True/False with explanations
- Fill-in-the-blank (equations, formulas)
- Ordering steps (arrange algorithm steps correctly)

**Spaced Repetition (SM-2 Algorithm):**
- Correct answer → interval increases (1 day → 3 days → 7 days → 14 days)
- Wrong answer → reset to 1 day
- Daily quiz pulls concepts due for review
- CLI: `lf quiz --daily` or `lf quiz --topic backprop`

#### Tier 2: Implementation Challenges

**Progressive Difficulty Levels:**

**Level A - Fill-in-the-blank (Learning Phase):**
```python
def backprop(x, y, weights):
    # Forward pass (provided)
    hidden = sigmoid(x @ weights['W1'])
    output = softmax(hidden @ weights['W2'])

    # YOUR CODE: Compute output layer gradient
    grad_output = ___________

    # YOUR CODE: Compute hidden layer gradient
    grad_hidden = ___________

    return {'W1': grad_W1, 'W2': grad_W2}
```

**Level B - From-scratch (Understanding Phase):**
```
Challenge: Implement SGD optimizer with momentum

Requirements:
- Input: gradients, parameters, learning_rate, momentum
- Update rule: velocity = momentum * velocity - lr * grad
- Return: updated parameters

Tests verify:
- Convergence on quadratic function
- Momentum acceleration vs vanilla SGD
- Numerical gradient checking
```

**Level C - Debug & Fix (Mastery Phase):**
```python
# This backprop has 3 bugs causing:
# 1. Exploding gradients
# 2. Wrong loss value
# 3. Slow convergence
# Find and fix all bugs

def buggy_backprop(x, y, weights):
    # Implementation with intentional bugs
    ...
```

**Automated Testing System:**
- Unit tests for correctness (pytest-based)
- Numerical gradient checking (analytical vs finite differences)
- Convergence tests (must reach target accuracy in N iterations)
- Performance tests (must run in < T seconds)
- Reference comparison (output matches baseline within epsilon)

#### Tier 3: GPU Scaling Validation

**Proof-of-Mastery Workflow:**

1. **Pass local tests** → System generates GPU challenge:
   ```
   You've mastered backprop on toy problems.
   GPU Challenge: Train ResNet-18 on CIFAR-10 using YOUR implementation
   Expected: >90% accuracy in <30 epochs
   ```

2. **Package implementation** → System creates deployment bundle:
   - Your C++ implementation (compiled .so)
   - Python wrapper
   - Training script with hyperparameters
   - Dataset download script
   - Validation script

3. **Submit job** → System handles remote execution:
   ```bash
   lf scale backprop --backend vastai
   # Uploads code, starts training, monitors progress
   # Shows: Epoch 5/30, Loss: 0.234, ETA: 15 min, Cost: $0.05
   ```

4. **Validate results** → Compare against baseline:
   - Your implementation: 91.2% accuracy, 28 epochs
   - PyTorch baseline: 92.1% accuracy, 25 epochs
   - Status: ✓ PASSED (within 2% of baseline)

**Progress Tracking:**
```
Concept: Backpropagation
├─ Tier 1: Quiz ✓ (5/5 correct, last: 2026-01-10)
├─ Tier 2: Implementation ✓ (passed all tests, 2026-01-11)
└─ Tier 3: GPU Scaling ✓ (91.2% CIFAR-10, 2026-01-12)
Status: MASTERED
Next review: 2026-01-26 (spaced repetition)
```

---

### Component 4: C++ Integration Strategy

**Purpose:** Implement performance-critical kernels in C++ to force deep understanding of memory, numerical stability, and computational efficiency.

**Build System (CMake + Auto-compilation):**

**Project Structure:**
```
cpp/
├── CMakeLists.txt
├── src/
│   ├── matrix.cpp
│   ├── activations.cpp
│   ├── backprop.cpp
│   └── optimizers.cpp
├── include/
│   └── learning_core.h
└── bindings/
    └── pybind11_wrapper.cpp
```

**First-Run Auto-Compilation:**
```python
try:
    import learning_core_cpp
    USE_CPP = True
except ImportError:
    print("C++ backend not found. Compiling...")
    subprocess.run(["cmake", "-B", "build", "-S", "cpp"])
    subprocess.run(["cmake", "--build", "build"])
    try:
        import learning_core_cpp
        USE_CPP = True
    except:
        print("Compilation failed. Falling back to Python.")
        USE_CPP = False
```

**C++ Learning Progression:**

**Stage 1: Basic Matrix Operations (Week 1-2)**
- Implement Matrix class with manual memory management
- Learning goals: row-major layout, cache-friendly iteration, pointer arithmetic

**Stage 2: Activation Functions (Week 3-4)**
- Implement forward/backward pass for ReLU, sigmoid, softmax
- Learning goals: numerical stability, inlining, edge cases

**Stage 3: Backpropagation Engine (Week 5-6)**
- Implement computational graph with reverse-mode autodiff
- Learning goals: dynamic memory, topological sorting, gradient accumulation

**Stage 4: Optimizers (Week 7-8)**
- Implement SGD, momentum, Adam
- Learning goals: numerical stability in updates, memory-efficient state tracking

**Pybind11 Integration:**
```cpp
PYBIND11_MODULE(learning_core_cpp, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<int, int>())
        .def("matmul", &Matrix::matmul)
        .def("to_numpy", [](const Matrix& mat) {
            return py::array_t<float>({mat.rows, mat.cols}, mat.data);
        });
}
```

**Platform Compatibility:**
- Windows: MSVC compiler, `/O2 /fp:fast` flags
- Linux: GCC/Clang, `-O3 -march=native -ffast-math` flags
- CMake handles cross-platform builds automatically

---

### Component 5: Remote GPU Execution System

**Purpose:** Scale validated implementations to real-world datasets on remote GPUs with multi-backend support.

#### Backend Abstraction Layer

**Base Interface:**
```python
class GPUBackend(ABC):
    @abstractmethod
    def submit_job(self, job_config: dict) -> str:
        """Submit job, return job_id"""

    @abstractmethod
    def get_status(self, job_id: str) -> dict:
        """Returns status, progress, metrics"""

    @abstractmethod
    def get_results(self, job_id: str) -> dict:
        """Download results, logs, metrics"""

    @abstractmethod
    def estimate_cost(self, job_config: dict) -> float:
        """Estimate cost before running"""
```

#### Vast.ai Backend Implementation

**Connection Workflow:**
1. Search for available GPU instances via API
2. Rent cheapest suitable instance
3. Wait for SSH availability
4. Upload job bundle via SFTP
5. Create conda environment
6. Start training in tmux session
7. Monitor progress via SSH + log parsing
8. Download results via SCP
9. Destroy instance to stop billing

**Automated Workflow:**
```python
def submit_job(self, job_config):
    # Find and rent GPU
    offer = self.find_cheapest_gpu()
    instance_id = self.rent_instance(offer["id"])

    # SSH setup
    ssh_info = self.wait_for_ready(instance_id)
    ssh = self._connect_ssh(ssh_info)

    # Upload and extract job bundle
    sftp = ssh.open_sftp()
    sftp.put(job_bundle_path, "/root/project/job.tar.gz")
    self._exec_ssh(ssh, "cd ~/project && tar -xzf job.tar.gz")

    # Setup environment
    self._exec_ssh(ssh, """
        conda create -n lf python=3.10 -y
        source activate lf
        pip install -r requirements.txt
    """)

    # Start training in tmux
    self._exec_ssh(ssh, f"""
        tmux new-session -d -s train_{instance_id}
        tmux send-keys 'cd ~/project && python train.py' Enter
    """)

    return f"vast_{instance_id}"
```

**Monitoring:**
```python
def get_status(self, job_id):
    ssh = self._connect_ssh(job_info["ssh"])

    # Check tmux session status
    session_status = self._exec_ssh(ssh,
        f"tmux has-session -t {session} && echo running || echo finished")

    # Parse latest logs for progress
    logs = self._exec_ssh(ssh, "tail -20 ~/project/train.log")
    progress = self._parse_progress(logs)

    return {
        "status": session_status,
        "progress": progress["percent"],
        "current_epoch": progress["epoch"],
        "latest_loss": progress["loss"],
        "latest_accuracy": progress["accuracy"]
    }
```

**Cost Control:**
- Daily budget limit (default: $5.00)
- Per-job cost limit (default: $1.00)
- Auto-cancel if job exceeds 150% of estimated cost
- Real-time cost tracking during execution

#### Other Backend Support

**Google Colab:**
- Programmatic notebook creation
- Code upload via Google Drive API
- Free tier (T4 GPU, 12-hour limit)

**SSH Generic Backend:**
- Direct SSH to any remote server
- Persistent connection for owned workstations
- Full control, no API dependencies

**AWS/GCP (Future):**
- Auto-provision EC2/Compute instances
- Auto-shutdown when complete
- Results stored in S3/Cloud Storage

---

### Component 6: CLI Interface & User Experience

**Purpose:** Provide intuitive command-line interface for all learning activities.

**Main Commands:**

```bash
# Learning & Navigation
lf learn                        # Browse topics and start learning
lf learn --topic backprop       # Learn specific concept
lf visualize backprop           # View visualizations

# Assessment
lf quiz --daily                 # Daily review (spaced repetition)
lf quiz --topic backprop        # Topic-specific quiz
lf challenge backprop --level fill  # Implementation challenge
lf test backprop.py             # Test your implementation

# GPU Scaling
lf scale backprop.py --backend vastai --estimate  # Estimate cost
lf scale backprop.py --backend vastai             # Submit job
lf logs vast_12345 --follow     # Stream training logs
lf status vast_12345            # Check job status
lf results vast_12345           # Download and validate results

# Progress & Management
lf progress                     # Overall progress
lf progress --topic neural_networks  # Topic-specific progress
lf index                        # Re-index learning materials
lf config                       # Configure API keys, budgets

# Utilities
lf tunnel vast_12345 --remote 8888 --local 8080  # SSH tunnel
lf ssh vast_12345               # SSH into remote instance
```

**User Experience Principles:**
- Rich terminal output (tables, progress bars, colors)
- Clear error messages with suggested fixes
- Confirmation prompts before spending money
- Real-time feedback during long operations
- Helpful defaults (minimize required options)

---

### Component 7: Data Flow & File Structure

**Project Directory Structure:**

```
learning-framework/
├── learning_framework/           # Main package
│   ├── cli.py                   # CLI interface
│   ├── core/                    # Python implementations
│   ├── knowledge/               # Indexer, knowledge graph
│   ├── assessment/              # Quiz, challenges, testing
│   ├── visualization/           # Static & interactive viz
│   ├── backends/                # GPU backends
│   ├── remote/                  # Job packaging, cost control
│   └── progress/                # Progress tracking
│
├── cpp/                         # C++ implementations
│   ├── src/                     # Matrix, activations, backprop
│   ├── include/                 # Headers
│   └── bindings/                # Pybind11 wrappers
│
├── data/                        # Concept definitions, quizzes
│   ├── concepts/                # Concept database (JSON)
│   ├── quizzes/                 # Quiz questions
│   ├── challenges/              # Implementation challenges
│   └── baselines/               # Reference implementations
│
├── user_data/                   # User-specific data
│   ├── progress.db              # SQLite database
│   ├── config.yaml              # Configuration
│   ├── implementations/         # User's code
│   └── jobs/                    # GPU job history
│
└── materials/                   # Auto-indexed materials
    ├── index.json               # Generated index
    ├── DeepLearning/            # Symlink to existing folder
    └── RL/                      # Symlink to existing folder
```

**Key Data Flows:**

**Learning Flow:**
```
User selects concept
→ Load from knowledge graph
→ Display explanation + code examples
→ Show visualizations
→ Take quiz
→ Update progress database
→ Offer implementation challenge
```

**Implementation Flow:**
```
User requests challenge
→ Copy template to user_data/
→ User implements
→ Run automated tests
→ Update progress
→ Offer GPU scaling
```

**GPU Scaling Flow:**
```
User submits job
→ Package implementation
→ Rent GPU instance (Vast.ai)
→ SSH upload code
→ Setup environment
→ Start training in tmux
→ Monitor progress
→ Download results
→ Validate vs baseline
→ Destroy instance
→ Update mastery status
```

---

## Technical Specifications

### Dependencies

**Python Core:**
- Python 3.10+
- numpy, matplotlib (core computation & visualization)
- click, rich (CLI interface)
- pytest (testing)
- paramiko (SSH connections)
- requests (API calls)
- pybind11 (C++ bindings)

**C++ Components:**
- CMake 3.15+
- C++17 compatible compiler (MSVC, GCC, Clang)
- pybind11

**Optional:**
- plotly (enhanced visualizations)
- pytorch (baseline comparisons)

### Platform Support

**Development (Windows):**
- Windows 10/11
- Visual Studio 2019+ or MSVC
- VS Code recommended
- Works entirely offline (GPU optional)

**Deployment (Linux):**
- Ubuntu 20.04+ (remote GPU servers)
- GCC 9+ or Clang 10+
- Same Python code, recompiled C++

### Resource Requirements

**Local (Minimal):**
- CPU: Any modern processor (no GPU required)
- RAM: 4GB minimum, 8GB recommended
- Disk: 2GB for framework + datasets
- Internet: Only for GPU job submission

**Remote GPU (Vast.ai):**
- GPU: RTX 3060+ (8GB VRAM minimum)
- Cost: $0.10-0.30/hour
- Typical job: 20-40 minutes, $0.05-0.20

---

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Project scaffolding and build system
- [ ] Basic CLI with main commands
- [ ] Material indexer (auto-discovery)
- [ ] Progress database schema
- [ ] Configuration management

### Phase 2: Learning & Assessment (Weeks 3-4)
- [ ] Knowledge graph implementation
- [ ] Quiz system (extend vocab quiz)
- [ ] Spaced repetition algorithm
- [ ] Static visualizations (matplotlib)
- [ ] Concept database for 5 core topics

### Phase 3: Implementation Challenges (Weeks 5-6)
- [ ] Challenge template system
- [ ] Automated test runner
- [ ] Numerical gradient checking
- [ ] Fill-in-blank, from-scratch, debug levels
- [ ] C++ basic matrix operations

### Phase 4: Visualization & Interaction (Weeks 7-8)
- [ ] Local web server for interactive viz
- [ ] Computational graph viewer
- [ ] Parameter exploration interface
- [ ] Real-time training visualization
- [ ] Integration with learning flow

### Phase 5: GPU Backend (Weeks 9-10)
- [ ] Backend abstraction layer
- [ ] Vast.ai implementation
- [ ] Job packaging system
- [ ] Cost controller
- [ ] Results validation

### Phase 6: C++ Deep Dive (Weeks 11-12)
- [ ] Advanced C++ implementations
- [ ] Backpropagation engine
- [ ] Optimizers (SGD, Adam)
- [ ] Performance benchmarking
- [ ] Cross-platform testing

### Phase 7: Polish & Documentation (Weeks 13-14)
- [ ] User documentation
- [ ] Tutorial content
- [ ] Error handling improvements
- [ ] Performance optimization
- [ ] Beta testing

---

## Success Criteria

### Learning Effectiveness
- User can explain concepts clearly after completing all three tiers
- Implementation passes numerical gradient checks
- GPU-scaled code achieves within 5% of baseline accuracy

### System Performance
- Local training (MNIST) completes in < 10 seconds
- Quiz response time < 100ms
- Visualization loads in < 2 seconds
- GPU job submission < 5 minutes

### Cost Efficiency
- Average concept mastery cost < $0.50 (GPU time)
- No wasted GPU time (auto-shutdown works)
- Local learning completely free

### User Experience
- Intuitive CLI (< 5 commands for common tasks)
- Clear feedback at every step
- Errors provide actionable suggestions
- Progress always visible

---

## Risk Mitigation

### Technical Risks

**Risk:** C++ compilation fails on user's machine
**Mitigation:** Pure Python fallback always available, detailed build troubleshooting guide

**Risk:** Remote GPU instance unavailable
**Mitigation:** Multi-backend support, fallback to next cheapest option

**Risk:** Job exceeds budget
**Mitigation:** Hard cost limits, auto-cancel at 150% threshold, pre-job estimates

### Learning Risks

**Risk:** User gets stuck on concept
**Mitigation:** Prerequisite checking, adaptive difficulty, hints in challenges

**Risk:** Spaced repetition overwhelming
**Mitigation:** Configurable daily limits, ability to postpone reviews

**Risk:** GPU validation too expensive
**Mitigation:** Thorough local testing first, optional GPU tier, batch multiple concepts

---

## Future Enhancements

### Near-term (3-6 months)
- Mobile app for quiz reviews
- Collaborative features (compare progress with peers)
- More RL environments (Atari, MuJoCo)
- Video tutorial integration

### Long-term (6-12 months)
- Multi-user deployment (classroom mode)
- Custom concept creation (user-defined topics)
- Advanced visualizations (3D neural networks)
- Integration with research papers (auto-extract concepts)

---

## Appendix

### Example Concept Definition

```json
{
  "backpropagation": {
    "name": "Backpropagation",
    "topic": "neural_networks",
    "difficulty": "intermediate",
    "prerequisites": ["gradients", "chain_rule"],
    "explanation_file": "materials/DeepLearning/深度学习入门/ch05/README.md",
    "code_examples": [
      "materials/DeepLearning/深度学习入门/ch05/train_neuralnet.py",
      "materials/DeepLearning/dezero自制框架/step10.py"
    ],
    "visualizations": [
      {"type": "computational_graph", "file": "viz/backprop_graph.py"}
    ],
    "quiz_file": "data/quizzes/backprop_quiz.json",
    "challenges": {
      "fill": "data/challenges/backprop_fill.py",
      "scratch": "data/challenges/backprop_scratch.py",
      "debug": "data/challenges/backprop_debug.py"
    },
    "gpu_challenge": {
      "dataset": "cifar10",
      "model": "resnet18",
      "target_accuracy": 90.0,
      "max_epochs": 30
    }
  }
}
```

### Database Schema

```sql
CREATE TABLE concepts (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    topic TEXT,
    difficulty TEXT,
    quiz_passed BOOLEAN DEFAULT 0,
    implementation_passed BOOLEAN DEFAULT 0,
    gpu_validated BOOLEAN DEFAULT 0,
    last_reviewed DATE,
    next_review DATE,
    review_interval INTEGER DEFAULT 1
);

CREATE TABLE quiz_results (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER,
    question_id TEXT,
    correct BOOLEAN,
    timestamp DATETIME,
    FOREIGN KEY (concept_id) REFERENCES concepts(id)
);

CREATE TABLE gpu_jobs (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE,
    concept TEXT,
    backend TEXT,
    submitted_at DATETIME,
    completed_at DATETIME,
    status TEXT,
    cost REAL,
    accuracy REAL,
    baseline_accuracy REAL,
    passed BOOLEAN
);
```

### Configuration Example

```yaml
# user_data/config.yaml

vastai_api_key: "your-api-key"

daily_gpu_budget: 5.0
max_job_cost: 1.0

editor: "code"
auto_open_browser: true

quiz_questions_per_session: 10
spaced_repetition_enabled: true

materials_directories:
  - "D:/ourock-test/ourock-test/DeepLearning"
  - "D:/ourock-test/ourock-test/RL"

auto_compile: true
compiler: "auto"
```

---

## Conclusion

This learning framework provides a comprehensive, resource-efficient path to mastering deep learning and reinforcement learning fundamentals. By combining progressive assessment (quiz → implementation → GPU validation) with hands-on C++ implementations, the system ensures deep understanding rather than surface-level knowledge. The multi-backend GPU support allows flexible scaling from free Colab to cost-effective Vast.ai as learning progresses.

**Next Steps:**
1. Review and approve this design
2. Set up git worktree for isolated development
3. Create detailed implementation plan
4. Begin Phase 1 development
