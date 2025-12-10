# ChunkHound DevOps & Infrastructure Guide

> Comprehensive infrastructure, deployment, and operational guide for DevOps engineers

## Table of Contents

- [Core Concepts in Simple Terms](#core-concepts-in-simple-terms)
- [1. Overall Architecture](#1-overall-architecture)
- [2. Directory Structure](#2-directory-structure)
- [3. Database Layer & Concurrency Model](#3-database-layer--concurrency-model)
- [4. Configuration System](#4-configuration-system)
- [5. Deployment Artifacts](#5-deployment-artifacts)
- [6. API Layers & Endpoints](#6-api-layers--endpoints)
- [7. Performance-Critical Paths](#7-performance-critical-paths)
- [8. Monitoring & Observability](#8-monitoring--observability)
- [9. Common Operational Patterns](#9-common-operational-patterns)
- [10. Troubleshooting Guide](#10-troubleshooting-guide)
- [11. Key DevOps Takeaways](#11-key-devops-takeaways)

---

## Core Concepts in Simple Terms

Before diving into the technical details, let's clarify some fundamental concepts.

### What is ChunkHound?

**ChunkHound is NOT an agent** - it's a **smart search engine for code**. Think of it like Google, but specifically for your codebase.

```
Traditional search (grep/find):
  You: "find function login"
  Result: Only finds exact text "login"

ChunkHound semantic search:
  You: "how does authentication work?"
  Result: Finds login functions, JWT validation, session management, etc.
          (understands the MEANING, not just keywords)
```

### Why Does It Need LLM?

ChunkHound uses LLMs for **ONE specific feature**: the `code_research` tool (deep research).

**Without LLM (basic ChunkHound):**
- Index code → chunks stored in database
- Search with regex → find exact patterns
- Search semantically → find similar code by meaning
- ✅ **Works fine without any LLM**

**With LLM (advanced feature):**
- Multi-hop research: "How does user authentication flow through the system?"
- LLM breaks question into sub-questions
- Searches code multiple times
- Synthesizes answer from findings
- ✅ **Optional feature, most users don't need it**

```bash
# Basic usage (NO LLM needed)
chunkhound index /my/code        # Index code
chunkhound search "function"     # Search code
chunkhound mcp stdio             # Run as MCP server

# Advanced usage (LLM required)
chunkhound research "How does auth work?"  # Deep research with LLM
```

### What is an Embedding?

An **embedding** is how computers understand the **meaning** of text as numbers.

**Simple Analogy - Library Books:**

**Old way (keyword index):**
- Book A: Contains word "dog" → Tag: "dog"
- Book B: Contains word "puppy" → Tag: "puppy"
- Search "dog" → Only finds Book A ❌

**Embedding way (meaning vector):**
- Book A about dogs → Vector: [0.9, 0.1, 0.8, ...] (1536 numbers)
- Book B about puppies → Vector: [0.89, 0.12, 0.79, ...] (similar numbers!)
- Search "dog" → Finds BOTH books ✅ (vectors are similar)

### How ChunkHound Uses Embeddings

```python
# Step 1: Break code into chunks
chunk1 = "def login(username, password): ..."
chunk2 = "def authenticate_user(creds): ..."

# Step 2: Convert to embeddings (via OpenAI/Ollama API)
embedding1 = [0.2, 0.8, 0.1, 0.9, ...]  # 1536 numbers
embedding2 = [0.19, 0.82, 0.09, 0.91, ...]  # Similar! (both about auth)

# Step 3: Store in database
database.store(chunk1, embedding1)
database.store(chunk2, embedding2)

# Step 4: Search by meaning
query = "how to verify user credentials?"
query_embedding = [0.21, 0.81, 0.11, 0.89, ...]

# Find similar embeddings (vector similarity)
results = database.find_similar(query_embedding)
# Returns: chunk1, chunk2 (both about authentication!)
```

### Why Embeddings are Powerful

**Without embeddings (regex only):**
```bash
Search: "authentication"
Finds: Only code with word "authentication"
Misses: login(), verify_user(), check_credentials()
```

**With embeddings (semantic):**
```bash
Search: "authentication"
Finds: ALL related code (login, verify, auth, credentials, sessions, tokens)
Why: Embedding model learned that these concepts are related
```

### Real-World Example

**Scenario:** You're new to a codebase and want to understand authentication.

**Regex search (basic, no embeddings):**
```bash
chunkhound search --regex "def.*login"
# Finds: Only functions with "login" in name
```

**Semantic search (with embeddings):**
```bash
chunkhound search --semantic "user authentication"
# Finds: login(), verify_token(), check_session(),
#        authenticate_user(), validate_credentials()
# Even if they don't contain word "authentication"!
```

**Deep research (with LLM):**
```bash
chunkhound research "Explain the complete authentication flow"
# LLM breaks down question:
#   1. How do users log in? → searches code
#   2. How are tokens validated? → searches code
#   3. How are sessions managed? → searches code
# Then synthesizes full answer from all findings
```

### Visual Summary

```
┌─────────────────────────────────────────────────────────┐
│                    ChunkHound                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Index Code                                          │
│     ├─ Parse files → chunks                            │
│     ├─ Generate embeddings (meanings as numbers)       │
│     └─ Store in database                               │
│                                                         │
│  2. Search Code                                         │
│     ├─ Regex: Exact pattern matching (NO embeddings)   │
│     ├─ Semantic: Similar meaning (uses embeddings)     │
│     └─ Deep Research: Multi-hop with LLM (optional)    │
│                                                         │
│  3. MCP Server                                          │
│     └─ Expose search to AI assistants (Claude, etc.)   │
│                                                         │
└─────────────────────────────────────────────────────────┘

Required Components:
  ✅ Database (DuckDB or LanceDB)
  ✅ Python 3.10+

Optional Components:
  🔧 Embedding provider (OpenAI/Ollama) → enables semantic search
  🔧 LLM provider (Anthropic/OpenAI) → enables deep research
```

### Key Takeaways

1. **ChunkHound = Smart code search engine**, not an AI agent
2. **Embeddings = Convert text meaning to numbers** for semantic search
3. **LLM = Optional** (only needed for deep research feature)
4. **Most useful feature**: Semantic search (find code by meaning, not keywords)
5. **Works without embeddings**: Regex search always available

---

## Deep Dive: How Embedding Models Work

### Embedding Models vs LLMs

There are **TWO different types of AI models** used in ChunkHound:

**Embedding Models** (for vectors):
- Examples: `text-embedding-3-small`, `nomic-embed-text`, `bge-large`
- Purpose: Convert text → numbers (vectors)
- Small, fast, specialized for similarity
- ✅ **Used for semantic search**

**LLMs** (Large Language Models):
- Examples: GPT-4, Claude, Gemini
- Purpose: Understand questions, write answers
- Large, slower, general intelligence
- ✅ **Used only for code_research tool**

**Visual Comparison:**

```
┌─────────────────────────────────────────────────────────┐
│              Embedding Model                            │
├─────────────────────────────────────────────────────────┤
│ Input:  "authentication"                                │
│ Output: [0.2, 0.8, 0.1, 0.9, ...]  (1536 numbers)     │
│                                                         │
│ Input:  "login"                                         │
│ Output: [0.19, 0.81, 0.11, 0.89, ...] (similar!)      │
│                                                         │
│ Input:  "logout"                                        │
│ Output: [0.21, 0.79, 0.09, 0.91, ...] (similar!)      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   LLM                                   │
├─────────────────────────────────────────────────────────┤
│ Input:  "Explain how authentication works"              │
│ Output: "The authentication system uses JWT tokens...   │
│          First, the user calls login() with             │
│          credentials..." (long text explanation)        │
└─────────────────────────────────────────────────────────┘
```

### How Embedding Models Learn Relationships

**Key Point:** Embedding models are **trained AI neural networks** - they learned patterns from massive amounts of text data during training.

**Training Process:**

```
Training Data (billions of examples from Wikipedia, GitHub, books):

"User authentication requires login credentials"
"Authentication system validates login attempts"
"After successful login, authentication token is issued"
"Logout invalidates the authentication session"
"User registration creates new authentication account"

During training (weeks on GPUs), the model learns:
- "login" appears near "authentication" → make vectors similar
- "logout" appears near "authentication" → make vectors similar
- "register" appears near "authentication" → make vectors similar
- "pizza" never appears near "authentication" → make vectors different
```

**Training Techniques:**

**1. Word2Vec (Classic Approach)**
```
Training sentence: "User needs to login to the authentication system"

Model learns:
- "login" appears near "authentication"
- "user" appears near "authentication"
- "system" appears near "authentication"

After seeing MILLIONS of sentences:
→ All auth-related words get similar vectors
```

**2. Sentence Transformers (Modern Approach)**
```
Training with pairs of sentences:

SIMILAR pair:
  "User login with password"
  "Authentication using credentials"
  → Model learns to make these vectors CLOSE

DIFFERENT pair:
  "User login with password"
  "Recipe for chocolate cake"
  → Model learns to make these vectors FAR APART
```

**3. Contrastive Learning (Code-Specific)**
```
Training example:

Anchor:    "def login(username, password):"
Positive:  "def authenticate_user(creds):"     ← Similar! Pull close
Negative:  "def calculate_area(width, height):" ← Different! Push far

Model adjusts vectors to:
- Make similar code chunks close together
- Make unrelated code chunks far apart
```

**Real Training Data Example:**

```python
# From thousands of GitHub repos, the model saw during training:

# Example 1
def login(username, password):
    """Authenticate user credentials"""

# Example 2
class AuthenticationService:
    def verify_login(self, user):

# Example 3
# User logout handler
def handle_logout(session):
    clear_authentication()

# Example 4
async def register_user(email, password):
    # Create authentication record
```

**Model learns:** "login", "logout", "register", "authenticate" all appear in similar contexts → **assign similar vectors**

### Embedding Models vs LLMs Comparison

| Feature | Embedding Model | LLM |
|---------|----------------|-----|
| **Training** | Learn word relationships | Learn language patterns + reasoning |
| **Output** | Fixed-size vector (1536 numbers) | Variable-length text |
| **Size** | Small (~500MB) | Huge (~100GB+) |
| **Speed** | Very fast (milliseconds) | Slower (seconds) |
| **Purpose** | Similarity/search | Text generation/reasoning |
| **Example** | `text-embedding-3-small` | `gpt-4`, `claude-sonnet` |
| **Cost** | Very cheap (fractions of a cent) | Expensive (cents to dollars) |

### ChunkHound's Specific Role

ChunkHound **uses** pre-trained embedding models (doesn't train them). Here's what ChunkHound actually does:

```
┌─────────────────────────────────────────────────────────┐
│                  ChunkHound's Job                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. Parse Code Files                                    │
│    Your code → Break into chunks                       │
│    Example: function definitions, classes, methods     │
│                                                         │
│ 2. Call Pre-Trained Embedding Model (API)             │
│    Send chunk text → Get back vector                   │
│    (Model already trained, you just USE it)            │
│                                                         │
│ 3. Store in Database                                   │
│    Save: [chunk text + vector] together                │
│                                                         │
│ 4. Enable Search                                       │
│    Query → Vector → Find similar vectors → Results     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Step-by-Step Example:**

**Your Code:**
```python
# file: auth.py
def login(username, password):
    """Authenticate user credentials"""
    if verify_password(username, password):
        return create_session(username)
    return None

def logout(session_id):
    """End user session"""
    clear_session(session_id)
```

**What ChunkHound Does:**

**Step 1: Parse & Chunk**
```python
# ChunkHound breaks code into chunks
chunk1 = {
    "file": "auth.py",
    "text": "def login(username, password):\n    if verify_password...",
    "type": "function"
}

chunk2 = {
    "file": "auth.py",
    "text": "def logout(session_id):\n    clear_session...",
    "type": "function"
}
```

**Step 2: Call Embedding Model API**
```python
# ChunkHound sends chunk text to OpenAI/Ollama API
import openai

# For chunk1 (login function)
response1 = openai.embeddings.create(
    model="text-embedding-3-small",  # Pre-trained model
    input=chunk1["text"]
)
vector1 = response1.data[0].embedding  # [0.2, 0.8, 0.1, ...]

# For chunk2 (logout function)
response2 = openai.embeddings.create(
    model="text-embedding-3-small",
    input=chunk2["text"]
)
vector2 = response2.data[0].embedding  # [0.21, 0.79, 0.09, ...]
```

**Step 3: Store in Database**
```python
# ChunkHound saves to DuckDB/LanceDB
database.insert({
    "chunk_id": 1,
    "file_path": "auth.py",
    "text": chunk1["text"],
    "embedding": vector1  # ← The vector!
})

database.insert({
    "chunk_id": 2,
    "file_path": "auth.py",
    "text": chunk2["text"],
    "embedding": vector2  # ← The vector!
})
```

**Step 4: Enable Search**
```python
# When you search
query = "how does authentication work?"

# ChunkHound converts query to vector (same API)
query_vector = openai.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# Find similar vectors in database
results = database.search_similar_vectors(
    query_vector=query_vector,
    limit=10
)

# Returns: Both login() and logout() chunks
# Because their vectors are similar to query vector!
```

### Complete Data Flow

```
Your Codebase                    Pre-Trained Model (API)
     ↓                                    ↓
┌─────────┐                      ┌──────────────┐
│ auth.py │                      │  OpenAI API  │
│ api.py  │  → ChunkHound  →    │   or Ollama  │
│ db.py   │    (orchestrator)    │ (pre-trained)│
└─────────┘                      └──────────────┘
     ↓                                    ↓
Parse into chunks              Send text, get vectors
     ↓                                    ↓
┌─────────────────────────────────────────────┐
│           ChunkHound Database               │
├─────────────────────────────────────────────┤
│ chunk_id │ text           │ vector          │
├──────────┼────────────────┼─────────────────┤
│ 1        │ def login(...) │ [0.2, 0.8, ...] │
│ 2        │ def logout(...)│ [0.21, 0.79,...]│
│ 3        │ class User(...) │ [0.5, 0.3, ...] │
└─────────────────────────────────────────────┘
     ↓
Search by vector similarity
     ↓
Return relevant code chunks
```

### Role Separation

| Task | Who Does It? | How? |
|------|-------------|------|
| **Train embedding model** | OpenAI/Ollama/etc | Train on billions of texts (BEFORE you use it) |
| **Parse code files** | ✅ ChunkHound | Tree-sitter parsers (29 languages) |
| **Break into chunks** | ✅ ChunkHound | Smart chunking logic |
| **Convert text → vector** | Embedding Model (API) | Pre-trained neural network |
| **Store chunks + vectors** | ✅ ChunkHound | DuckDB/LanceDB database |
| **Search by similarity** | ✅ ChunkHound | Vector similarity calculation |
| **Return results** | ✅ ChunkHound | Rank by similarity score |

### Configuration Examples

```bash
# EMBEDDING MODEL (for semantic search)
export CHUNKHOUND_EMBEDDING__PROVIDER=openai
export CHUNKHOUND_EMBEDDING__MODEL=text-embedding-3-small  # ← Embedding model
export CHUNKHOUND_EMBEDDING__API_KEY=sk-proj-...

# Or use local embedding model (free, no API key needed)
export CHUNKHOUND_EMBEDDING__PROVIDER=ollama
export CHUNKHOUND_EMBEDDING__MODEL=nomic-embed-text  # ← Embedding model
export CHUNKHOUND_EMBEDDING__BASE_URL=http://localhost:11434/v1

# LLM (only for deep research, separate from embeddings)
export CHUNKHOUND_LLM__PROVIDER=anthropic
export CHUNKHOUND_LLM__MODEL=claude-sonnet-4-5-20250929  # ← LLM model
export CHUNKHOUND_LLM__API_KEY=sk-ant-...
```

### Simple Analogy

**Think of it like a library:**

- **Embedding Model** = Language expert (already trained, knows word relationships)
- **ChunkHound** = Librarian (organizes books, helps you search)

The librarian (ChunkHound):
1. Takes new books (your code)
2. Asks the language expert (embedding model) to categorize them
3. Stores books with categories (vectors) in shelves (database)
4. When you search, asks expert to categorize your query
5. Finds books with similar categories (vector similarity)

### Key Insights

1. **Embedding models are pre-trained** - You don't train them, you use them via API
2. **Training happened already** - On billions of texts, before you ever use ChunkHound
3. **ChunkHound orchestrates** - Parses, calls APIs, stores, searches
4. **Model knows relationships** - "login" ≈ "authentication" learned during training
5. **You just use it** - Like using Google Translate (you don't train it yourself)

---

## Understanding Tree-sitter: The Code Parser

### What is Tree-sitter?

**Tree-sitter** is a **parser generator** - it reads code and understands its structure, like a grammar checker for programming languages.

**Simple Analogy:**
- **You read English** - You understand sentences, grammar, subjects, verbs
- **Tree-sitter reads code** - It understands functions, classes, variables, blocks

```
English Sentence: "The cat sat on the mat"

Grammar parsing:
├── Subject: "The cat"
├── Verb: "sat"
└── Prepositional phrase: "on the mat"
    ├── Preposition: "on"
    └── Object: "the mat"

---

Python Code: "def login(username, password):"

Tree-sitter parsing:
├── Type: function_definition
├── Name: "login"
└── Parameters:
    ├── "username"
    └── "password"
```

### Why Not Just Use Regex?

**Without Tree-sitter (Regex Approach - BAD):**

```python
# Trying to find functions with regex
pattern = r"def\s+(\w+)\s*\("

Problems:
❌ Misses: async def, lambdas, decorators
❌ Confused by: strings with "def" in them
❌ No context: Can't tell if it's inside a comment
❌ Fragile: Breaks with unusual formatting
```

**With Tree-sitter (Proper Parser - GOOD):**

```python
# Tree-sitter query
query = """
(function_definition
    name: (identifier) @func_name
) @function
"""

Benefits:
✅ Understands: Language syntax perfectly
✅ Ignores: Strings, comments automatically
✅ Context-aware: Knows structure, nesting
✅ Robust: Handles any valid code formatting
```

### Behind the Scenes: Parsing Example

**Your Bash Script:**
```bash
#!/bin/bash

# Deploy application
function deploy_app() {
    echo "Starting deployment..."
    docker-compose up -d
}

# Check health
function check_health() {
    curl http://localhost:8080/health
}

deploy_app
```

**Step 1: Tokenize (break into pieces)**
```
#!/bin/bash        → SHEBANG
# Deploy app...    → COMMENT
function           → KEYWORD
deploy_app         → IDENTIFIER
(                  → LEFT_PAREN
)                  → RIGHT_PAREN
{                  → LEFT_BRACE
echo               → COMMAND
"Starting..."      → STRING
}                  → RIGHT_BRACE
```

**Step 2: Build Abstract Syntax Tree (AST)**
```
(program
  (comment "#!/bin/bash")
  (comment "# Deploy application")
  (function_definition
    name: (word "deploy_app")
    body: (compound_statement
      (command name: (command_name "echo") ...)
      (command name: (command_name "docker-compose") ...)))
  (comment "# Check health")
  (function_definition
    name: (word "check_health")
    body: (compound_statement
      (command name: (command_name "curl") ...)))
  (command name: (command_name "deploy_app")))
```

**Step 3: ChunkHound Queries Tree**
```python
# ChunkHound asks: "Find all function definitions"
results = tree_sitter.query("""
  (function_definition
    name: (word) @name
  ) @definition
""")

# Tree-sitter returns:
[
  {
    "name": "deploy_app",
    "definition": "function deploy_app() { echo \"Starting...\" ... }",
    "start_line": 4,
    "end_line": 7
  },
  {
    "name": "check_health",
    "definition": "function check_health() { curl ... }",
    "start_line": 10,
    "end_line": 12
  }
]
```

**Step 4: ChunkHound Creates Chunks**
```python
chunk1 = {
    "file": "deploy.sh",
    "text": "function deploy_app() {\n    echo \"Starting...\"\n    ...\n}",
    "type": "function",
    "name": "deploy_app",
    "line": 4
}

chunk2 = {
    "file": "deploy.sh",
    "text": "function check_health() {\n    curl ...\n}",
    "type": "function",
    "name": "check_health",
    "line": 10
}
```

### The Complete Flow

```
┌─────────────────────────────────────────────────────┐
│              Your Code File (deploy.sh)             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                  Tree-sitter                        │
├─────────────────────────────────────────────────────┤
│ 1. Loads Bash grammar (knows Bash syntax rules)    │
│ 2. Parses code → Abstract Syntax Tree (AST)        │
│ 3. Validates syntax (catches errors)               │
│ 4. Labels nodes (function, variable, comment, etc) │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                   ChunkHound                        │
├─────────────────────────────────────────────────────┤
│ 1. Queries AST: "Give me all function_definition"  │
│ 2. Extracts matched nodes as chunks                │
│ 3. Gets complete code (start_line to end_line)     │
│ 4. Sends chunks to embedding model → vectors       │
│ 5. Stores chunks + vectors in database             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                    Database                         │
├─────────────────────────────────────────────────────┤
│ chunk_id │ text              │ vector              │
├──────────┼───────────────────┼─────────────────────┤
│ 1        │ function deploy...│ [0.2, 0.8, ...]     │
│ 2        │ function check... │ [0.21, 0.79, ...]   │
└─────────────────────────────────────────────────────┘
```

### Supported Languages (29 Total)

ChunkHound uses Tree-sitter grammars for 29 languages. Each language has its own grammar that understands its specific syntax.

**Location in code:** `chunkhound/parsers/parser_factory.py` (lines 453-514)

**Programming Languages:**
- Python, JavaScript, TypeScript, JSX, TSX
- Java, Kotlin, Groovy
- C, C++, C#
- Go, Rust, Zig
- Haskell, Swift, Dart
- PHP, Objective-C
- MATLAB

**Scripting & Shell:**
- **Bash** (`.sh`, `.bash`, `.zsh`) - parser_factory.py:475, mappings/bash.py

**Configuration & Data:**
- **YAML** (`.yaml`, `.yml`) - parser_factory.py:493, mappings/yaml.py
- JSON (`.json`)
- TOML (`.toml`)
- HCL/Terraform (`.tf`, `.hcl`)

**Markup & Build:**
- Markdown (`.md`)
- Makefile
- Vue (`.vue`)

**Custom Parsers:**
- TEXT (plain text fallback)
- PDF (document parsing)

**File extension mappings:** `parser_factory.py` lines 517-609

### Language-Specific Parsing Examples

**Bash Functions** (`chunkhound/parsers/mappings/bash.py`):
```python
# Tree-sitter query for Bash
query = """
(function_definition
    name: (word) @func_name
) @func_def
"""

# Extracts:
# - Function definitions
# - Variable assignments
# - Loops (for, while)
# - Conditionals (if, case)
# - Comments (including shebang, TODO)
# - Imports (source, . commands)
```

**YAML Blocks** (`chunkhound/parsers/mappings/yaml.py`):
```python
# Tree-sitter query for YAML
query = """
(block_mapping_pair
    key: (flow_node) @key
    value: (_) @value
) @definition
"""

# Extracts:
# - Key-value pairs (perfect for Ansible tasks!)
# - Sequences (lists)
# - Mappings (dictionaries)
# - Documents (multi-doc YAML)
# - Comments
```

### Why This Matters for ChunkHound

#### 1. Accurate Chunking

**Without Tree-sitter (dumb line splitting):**
```python
chunk1 = lines[0:50]    # ❌ Might split a function in half!
chunk2 = lines[51:100]  # ❌ Incomplete code, no context
```

**With Tree-sitter (smart syntax splitting):**
```python
chunk1 = complete_function_1  # ✅ Full function, self-contained
chunk2 = complete_class_1     # ✅ Full class, makes sense
```

#### 2. Intelligent Search

```bash
# User searches: "authentication functions"

Without Tree-sitter:
- Search ALL text (comments, strings, code mixed)
- No context about what's a function vs variable
- Returns messy, irrelevant results

With Tree-sitter:
- Search ONLY function definitions
- Ignore comments, strings (noise)
- Return complete, valid code chunks
```

#### 3. Language-Aware Parsing

```python
# ChunkHound uses the RIGHT grammar for each file
deploy.sh  → Bash grammar → Find bash functions
main.py    → Python grammar → Find python functions
config.yml → YAML grammar → Find yaml blocks (Ansible tasks!)
main.tf    → HCL grammar → Find terraform resources
```

### Comparison: Regex vs Tree-sitter

| Aspect | Regex Approach | Tree-sitter Approach |
|--------|---------------|---------------------|
| **Syntax understanding** | ❌ No understanding | ✅ Full language grammar |
| **Handles decorators** | ❌ Breaks | ✅ Handles perfectly |
| **Multiline code** | ❌ Difficult | ✅ Natural |
| **Nested structures** | ❌ Nearly impossible | ✅ Built-in |
| **Comments in strings** | ❌ Confused | ✅ Distinguishes |
| **Performance** | ⚠️ Fast but fragile | ✅ Fast AND robust |
| **Maintenance** | ❌ Brittle, breaks often | ✅ Stable, reliable |

### Key Benefits for ChunkHound

| Benefit | What It Means |
|---------|---------------|
| **Syntax-aware chunking** | Never cuts functions/classes in half |
| **Language-specific** | Understands 29 languages correctly |
| **Robust parsing** | Handles complex, real-world code |
| **Context preservation** | Chunks are complete, meaningful units |
| **Metadata extraction** | Knows names, types, relationships |
| **Fast** | Written in Rust, highly optimized |
| **Error tolerance** | Handles syntax errors gracefully |

### DevOps Use Cases

**Ansible Playbooks (YAML):**
```yaml
# playbook.yml
- name: Deploy web application
  hosts: webservers
  tasks:
    - name: Install nginx
      apt: name=nginx state=present

# Tree-sitter extracts each task as a separate chunk
# Semantic search: "nginx installation" finds this task!
```

**Bash Scripts:**
```bash
# deploy.sh
function backup_database() {
    mysqldump -u root mydb > backup.sql
}

# Tree-sitter extracts complete function
# Semantic search: "database backup" finds this function!
```

**Terraform (HCL):**
```hcl
# main.tf
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
}

# Tree-sitter extracts resource blocks
# Semantic search: "EC2 instance" finds this resource!
```

### How to Verify Tree-sitter Support

```bash
# Check supported languages in code
cd /home/e196723/views/chunkhound
cat chunkhound/parsers/parser_factory.py | grep "Language\."

# View Bash parser implementation
cat chunkhound/parsers/mappings/bash.py | head -100

# View YAML parser implementation
cat chunkhound/parsers/mappings/yaml.py | head -100

# Check file extension mappings
cat chunkhound/parsers/parser_factory.py | grep -A 1 "\.sh\|\.yaml\|\.tf"
```

### Simple Summary

**Tree-sitter is like a language expert** that:
1. ✅ Reads your code (any of 29 languages)
2. ✅ Understands structure (functions, classes, blocks)
3. ✅ Builds a syntax tree (organized map of code)
4. ✅ Lets ChunkHound query: "Find all functions"
5. ✅ Returns clean, complete code chunks

**Without Tree-sitter:**
- ChunkHound would split code randomly (line 1-100, 101-200)
- Miss function boundaries
- Break code into meaningless pieces
- Search would be inaccurate

**With Tree-sitter:**
- ChunkHound splits code smartly (by functions, classes)
- Respects code structure
- Creates meaningful, complete chunks
- Search is accurate and context-aware

**It's the foundation that makes ChunkHound understand code structure instead of just treating it as plain text!**

---

## 1. Overall Architecture

ChunkHound is a **local-first semantic code search tool** built entirely by AI agents, designed to transform codebases into searchable knowledge bases for AI assistants via the Model Context Protocol (MCP).

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layers                              │
├─────────────────────────────────────────────────────────────┤
│  CLI (main.py)  │  MCP Stdio  │  MCP HTTP (FastMCP)       │
└─────────────┬───────────────┬─────────────────────────────┘
              │               │
┌─────────────▼───────────────▼─────────────────────────────┐
│                    Service Layer                            │
├─────────────────────────────────────────────────────────────┤
│  IndexingCoordinator  │  SearchService  │  EmbeddingService │
│  RealtimeIndexing     │  DeepResearch   │  ClusteringService│
└─────────────┬───────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│                   Provider Layer                            │
├─────────────────────────────────────────────────────────────┤
│  Database: DuckDB/LanceDB  │  Embeddings: OpenAI/Ollama/Voyage│
│  LLM: Anthropic/OpenAI/Gemini  │  Parsers: 29 Languages    │
└─────────────────────────────────────────────────────────────┘
```

### Core Concepts

- **Local-First**: All data stored locally, no cloud dependencies required
- **Provider Abstraction**: Swap databases, embeddings, LLMs via configuration
- **MCP Integration**: Native support for Model Context Protocol (stdio + HTTP)
- **Performance-Tuned**: Mandatory batching, parallel processing, optimized indexing
- **AI-Generated**: 100% codebase written by AI agents (no human-written code)

---

## 2. Directory Structure

```
chunkhound/
├── api/                    # User-facing interfaces
│   └── cli/               # CLI commands (index, search, mcp)
├── mcp_server/            # MCP protocol servers
│   ├── stdio.py           # Stdio transport (JSON-RPC)
│   ├── http_server.py     # HTTP transport (FastMCP)
│   ├── tools.py           # Unified tool registry (SINGLE SOURCE OF TRUTH)
│   └── base.py            # Common server logic
├── services/              # Orchestration layer
│   ├── indexing_coordinator.py  # Main indexing workflow
│   ├── search_service.py        # Search operations
│   ├── embedding_service.py     # Batched embedding generation
│   └── realtime_indexing_service.py  # File watching
├── providers/             # Database & API implementations
│   ├── database/          # DuckDB, LanceDB
│   │   ├── serial_database_provider.py  # Thread-safety wrapper
│   │   └── serial_executor.py           # Single-threaded queue
│   ├── embeddings/        # OpenAI, Ollama, VoyageAI
│   └── llm/               # Anthropic, OpenAI, Gemini, Codex
├── core/                  # Domain models & config
│   ├── config/            # Configuration system
│   ├── models.py          # Data models (File, Chunk, Embedding)
│   └── types/             # Type safety (NewType wrappers)
├── parsers/               # Language-specific parsers
│   └── mappings/          # Tree-sitter grammar mappings
└── utils/                 # Cross-cutting utilities

scripts/                   # DevOps & release automation
├── prepare_release.sh     # Build & lock dependencies
├── build.sh               # PyInstaller binary builds (macOS/Ubuntu)
├── mcp-server.sh          # Easy MCP server startup
└── update_version.py      # Git tag-based versioning

tests/                     # Comprehensive test suite
├── test_smoke.py          # MANDATORY pre-commit checks
├── test_mcp_*.py          # MCP server tests
└── integration/           # End-to-end tests

operations/                # Operational documentation
└── database_concurrency.md  # Concurrency model experiments
```

### Key Files for DevOps

| File | Purpose | DevOps Relevance |
|------|---------|------------------|
| `pyproject.toml` | Build config, flexible dependencies | Build system, version constraints |
| `uv.lock` | Exact dev dependencies | Development environment lock |
| `requirements-lock.txt` | Exact prod dependencies | Production deployment lock |
| `.chunkhound.json` | Per-project config | Configuration management |
| `scripts/prepare_release.sh` | Release automation | CI/CD pipeline |
| `tests/test_smoke.py` | Pre-commit checks | Health checks, deployment validation |

---

## 3. Database Layer & Concurrency Model

### Database Providers

**DuckDB (Primary):**
- **Type**: Embedded analytical database (SQLite-like)
- **Storage**: Single file `.chunkhound/db/chunks.db`
- **Concurrency**: Single-owner process only
- **Locking**: Fails fast on concurrent access (IO Error: Could not set lock)
- **Best for**: CLI usage, single-user workflows, CI/CD environments

**LanceDB (Alternative):**
- **Type**: Embedded vector database with MVCC
- **Storage**: Directory `.chunkhound/db/lancedb.lancedb/`
- **Concurrency**: Multi-process safe (1 writer + N readers tested)
- **MVCC**: Multi-Version Concurrency Control for snapshot isolation
- **Best for**: Concurrent MCP servers, high-performance vector search, multi-user scenarios

### Critical Concurrency Constraints

```python
# SINGLE-THREADED ONLY (enforced by SerialDatabaseProvider)
# - DuckDB: Hard requirement (locking error on concurrent access)
# - LanceDB: ChunkHound design choice (simplifies consistency)

# Thread Safety Architecture:
SerialDatabaseProvider (base class)
├── SerialDatabaseExecutor (single-threaded queue)
│   ├── All DB operations enqueued
│   ├── Executed sequentially in dedicated thread
│   └── Prevents race conditions & corruption
└── Subclasses: DuckDBProvider, LanceDBProvider
```

**Why Single-Threaded?**
- **DuckDB**: Technical limitation (no concurrent access support)
- **LanceDB**: Design simplification (ChunkHound layer serializes, LanceDB handles multi-process internally)
- **Consistency**: Simplifies reasoning about database state
- **Performance**: Not a bottleneck (parsing/embeddings are parallelized)

**Performance Impact:**

| Operation | Threading | Rationale |
|-----------|-----------|-----------|
| File Parsing | ✅ Parallel (CPU cores) | CPU-bound, no DB access |
| Embedding API | ✅ Concurrent batches | Network I/O, rate-limited |
| Database Storage | ⚠️ Single-threaded | DB constraint (enforced) |

### Database Paths & Configuration

```bash
# Default locations
DuckDB:   .chunkhound/db/chunks.db
LanceDB:  .chunkhound/db/lancedb.lancedb/

# Environment override
export CHUNKHOUND_DATABASE__PATH=/custom/path
export CHUNKHOUND_DATABASE__PROVIDER=lancedb

# Storage limits (prevents runaway disk usage)
export CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB=10.0
```

### Multi-Process Safety Matrix

| Scenario | DuckDB | LanceDB |
|----------|--------|---------|
| Single CLI process | ✅ Supported | ✅ Supported |
| Multiple CLI processes | ❌ Lock error | ✅ MVCC safe |
| MCP Stdio + CLI | ❌ Lock error | ✅ MVCC safe |
| Multiple MCP servers | ❌ Lock error | ✅ MVCC safe |
| Concurrent reads | ❌ Lock error | ✅ MVCC snapshots |
| Concurrent writes | ❌ Lock error | ✅ Conflict resolution |

---

## 4. Configuration System

### Precedence Order (Highest to Lowest)

```
1. CLI arguments          --db /path/to/db --provider lancedb
2. Environment variables  CHUNKHOUND_DATABASE__PROVIDER=duckdb
3. Local .chunkhound.json (in target directory)
4. --config file          chunkhound index --config custom.json
5. Default values         (defined in Pydantic models)
```

### Environment Variable Pattern

ChunkHound uses a **double underscore (`__`)** convention for nested configuration:

```bash
# Pattern: CHUNKHOUND_<SECTION>__<SETTING>=value

# Database settings
export CHUNKHOUND_DATABASE__PATH=/data/db
export CHUNKHOUND_DATABASE__PROVIDER=lancedb
export CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB=20.0
export CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD=100

# Embedding settings
export CHUNKHOUND_EMBEDDING__PROVIDER=openai
export CHUNKHOUND_EMBEDDING__API_KEY=sk-proj-...
export CHUNKHOUND_EMBEDDING__MODEL=text-embedding-3-small
export CHUNKHOUND_EMBEDDING__EMBEDDING_BATCH_SIZE=100
export CHUNKHOUND_EMBEDDING__RERANK_URL=http://localhost:8080/rerank
export CHUNKHOUND_EMBEDDING__RERANK_FORMAT=tei
export CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE=32

# LLM settings (for deep research)
export CHUNKHOUND_LLM__PROVIDER=anthropic
export CHUNKHOUND_LLM__API_KEY=$ANTHROPIC_API_KEY
export CHUNKHOUND_LLM__UTILITY_MODEL=claude-haiku-4-5-20251001
export CHUNKHOUND_LLM__SYNTHESIS_MODEL=claude-sonnet-4-5-20250929

# Indexing settings
export CHUNKHOUND_INDEXING__PARALLEL_DISCOVERY=true
export CHUNKHOUND_INDEXING__MAX_DISCOVERY_WORKERS=16
```

### Configuration Files

**Location:** `.chunkhound.json` in project root (or custom path via `--config`)

```json
{
  "database": {
    "provider": "lancedb",
    "path": ".chunkhound/db",
    "max_disk_usage_mb": 10240,
    "lancedb_optimize_fragment_threshold": 100
  },
  "embedding": {
    "provider": "openai",
    "api_key": "sk-proj-...",
    "model": "text-embedding-3-small",
    "embedding_batch_size": 100,
    "max_concurrent_batches": 8,
    "rerank_url": "http://localhost:8080/rerank",
    "rerank_format": "tei",
    "rerank_batch_size": 32
  },
  "llm": {
    "provider": "anthropic",
    "api_key": "sk-ant-api03-...",
    "utility_model": "claude-haiku-4-5-20251001",
    "synthesis_model": "claude-sonnet-4-5-20250929"
  },
  "indexing": {
    "exclude": ["**/dist/**", "**/*.min.js", "**/node_modules/**"],
    "parallel_discovery": true,
    "max_discovery_workers": 16,
    "chunk_batch_size": 1000
  }
}
```

### Validation Rules

**Command-Specific Requirements:**

```bash
# Indexing: Requires embedding provider for semantic search
chunkhound index .
# ❌ Error if no embedding provider configured

chunkhound index . --no-embeddings
# ✅ OK - skips embeddings, regex-only search

# Search: Embedding optional (falls back to regex)
chunkhound search "query"
# ✅ OK - uses regex if no embeddings

chunkhound search "query" --semantic
# ❌ Error if no embeddings exist

# MCP: Embedding optional
chunkhound mcp stdio
# ✅ OK - tools adapt to available features
```

### Configuration Management Best Practices

**Development:**
```bash
# Use local .chunkhound.json
cd /my/project
cat > .chunkhound.json <<EOF
{
  "embedding": {
    "provider": "ollama",
    "base_url": "http://localhost:11434/v1",
    "model": "nomic-embed-text"
  }
}
EOF
```

**Production:**
```bash
# Use environment variables (secrets management)
export CHUNKHOUND_EMBEDDING__API_KEY=$(vault read -field=api_key secret/chunkhound)
export CHUNKHOUND_DATABASE__PATH=/data/shared-db
export CHUNKHOUND_DATABASE__PROVIDER=lancedb
```

**CI/CD:**
```bash
# Override config for ephemeral environments
chunkhound index . \
  --db /tmp/ci-db \
  --database-provider duckdb \
  --no-embeddings  # Skip embeddings in CI
```

---

## 5. Deployment Artifacts

### Package Distribution

**PyPI Package (Primary):**
```bash
# Install from PyPI using uv (recommended)
uv tool install chunkhound

# Or with pip (standard)
pip install chunkhound

# Verify installation
chunkhound --version
```

**Build Artifacts:**
```
dist/
├── chunkhound-X.Y.Z-py3-none-any.whl    # Python wheel (pip install)
├── chunkhound-X.Y.Z.tar.gz              # Source distribution (pip install)
├── chunkhound-macos-universal.tar.gz    # Standalone binary (macOS Intel/ARM)
├── chunkhound-ubuntu16-amd64.tar.gz     # Standalone binary (Ubuntu 16.04+)
└── SHA256SUMS                            # Integrity checksums
```

**Binary Distributions (PyInstaller):**
```bash
# macOS Universal (Intel + Apple Silicon)
wget https://github.com/chunkhound/chunkhound/releases/download/vX.Y.Z/chunkhound-macos-universal.tar.gz
tar xzf chunkhound-macos-universal.tar.gz
./chunkhound-macos-universal/chunkhound-optimized --version

# Ubuntu 16.04+ (glibc 2.23+)
wget https://github.com/chunkhound/chunkhound/releases/download/vX.Y.Z/chunkhound-ubuntu16-amd64.tar.gz
tar xzf chunkhound-ubuntu16-amd64.tar.gz
./chunkhound-ubuntu16-amd64/chunkhound-optimized --version
```

### Version Management (Dynamic)

**Source of Truth:** Git tags via `hatch-vcs`

ChunkHound uses **git tags** as the single source of truth for versioning. No manual version editing required.

```bash
# Create release version
uv run scripts/update_version.py 4.1.0
# → Creates git tag v4.1.0
# → Version auto-derived during build

# Bump versions automatically
uv run scripts/update_version.py --bump major      # v4.1.0 → v5.0.0
uv run scripts/update_version.py --bump minor      # v4.1.0 → v4.2.0
uv run scripts/update_version.py --bump patch      # v4.1.0 → v4.1.1

# Pre-release versions (PEP 440 compliant)
uv run scripts/update_version.py 4.1.0b1          # Beta release
uv run scripts/update_version.py 4.1.0rc1         # Release candidate
uv run scripts/update_version.py --bump minor b1  # v4.1.0 → v4.2.0b1
```

**Version Files:**
```
chunkhound/_version.py      # Auto-generated by hatch-vcs (gitignored)
chunkhound/version.py       # Imports from _version.py or metadata
pyproject.toml              # dynamic = ["version"] (hatch-vcs config)
```

**Important:** NEVER manually edit version strings. Always use git tags.

### Release Process

**Step-by-Step Checklist:**

```bash
# 1. Create version tag
uv run scripts/update_version.py 4.1.0

# 2. Run smoke tests (MANDATORY)
uv run pytest tests/test_smoke.py -v
# Tests: module imports, CLI commands, server startup

# 3. Prepare release (regenerates dependency locks)
./scripts/prepare_release.sh
# - Generates requirements-lock.txt
# - Builds wheel and source dist
# - Creates checksums (SHA256SUMS)

# 4. Test local install
pip install dist/chunkhound-4.1.0-py3-none-any.whl
chunkhound --version
# Expected: chunkhound 4.1.0

# 5. Push tag to trigger release
git push origin v4.1.0

# 6. Publish to PyPI (requires PYPI_TOKEN)
export PYPI_TOKEN=pypi-...
uv publish
```

### Dependency Locking Strategy

ChunkHound uses **three levels of dependency specifications**:

| File | Type | Purpose | Usage |
|------|------|---------|-------|
| `pyproject.toml` | Flexible ranges (`>=`) | Library compatibility | pip install from source |
| `uv.lock` | Exact versions | Dev reproducibility | uv sync |
| `requirements-lock.txt` | Exact versions | Prod deployment | pip install -r |

**Regenerate locks:**
```bash
# Generate production lock file
uv pip compile pyproject.toml --all-extras -o requirements-lock.txt

# Update uv.lock
uv sync
```

**Key Dependencies:**
- Python: 3.10-3.13 (3.14 may work but untested)
- Package Manager: `uv` (MANDATORY for development)
- Database: `duckdb>=1.4.0`, `lancedb>=0.25.3`
- Embeddings: `openai>=1.0.0`, `voyageai>=0.2.0`
- LLM: `anthropic>=0.75.0`, `google-genai>=1.51.0`
- MCP: `mcp>=1.0.0`, `fastmcp>=2.0.0`
- Parsing: `tree-sitter>=0.20.0` + 27 language grammars

---

## 6. API Layers & Endpoints

### CLI Interface

```bash
# Index a codebase
chunkhound index [directory]
  --config FILE              # Custom config file path
  --db PATH                  # Database path (overrides config)
  --database-provider TYPE   # duckdb | lancedb
  --no-embeddings            # Skip embedding generation (regex-only)
  --simulate                 # Dry-run (list files without indexing)
  --profile-startup          # Emit JSON performance diagnostics to stderr
  --exclude PATTERN          # Additional exclusion patterns

# Search indexed code
chunkhound search [query]
  --semantic                 # Vector similarity search (requires embeddings)
  --regex                    # Regular expression pattern matching
  --threshold FLOAT          # Similarity threshold (0.0-1.0, default: 0.7)
  --limit INT                # Max results (default: 10)
  --db PATH                  # Database path

# MCP Server (Stdio transport)
chunkhound mcp stdio
  # Reads JSON-RPC commands from stdin
  # Responds on stdout
  # CRITICAL: NO stdout logging allowed (breaks protocol)

# MCP Server (HTTP transport)
chunkhound mcp http --port 5173
  # FastMCP server on specified port
  # Swagger docs at http://localhost:5173/docs
  # WebSocket support for streaming

# Deep research (CLI wrapper for code_research tool)
chunkhound research "question"
  --token-budget INT         # Context window limit (default: 50000)
  --no-synthesis             # Skip LLM synthesis (cheaper, faster)
  --db PATH                  # Database path
```

### MCP Tools (Unified Registry)

**All tools defined in:** `chunkhound/mcp_server/tools.py` (SINGLE SOURCE OF TRUTH)

```python
# Tool registration pattern
@register_tool(
    description="Find exact code patterns using regular expressions",
    requires_embeddings=False,  # Works without embeddings
    name="search_regex"
)
async def search_regex_impl(
    services: DatabaseServices,
    pattern: str,
    file_pattern: str | None = None,
    page_size: int = 10,
    page: int = 1,
) -> SearchResponse:
    """Implementation"""
```

**Available Tools:**

| Tool | Description | Requires Embeddings | Typical Use Case |
|------|-------------|---------------------|------------------|
| `search_regex` | Regex pattern matching | ❌ No | Find function definitions, imports |
| `search_semantic` | Vector similarity search | ✅ Yes | Conceptual code search |
| `list_files` | Browse indexed files | ❌ No | Explore codebase structure |
| `get_file_content` | Retrieve file contents | ❌ No | Read specific files |
| `code_research` | Multi-hop deep research | ✅ Yes | Answer complex questions |

**Schema Consistency:**
- Both stdio & HTTP servers reference same `TOOL_REGISTRY`
- Schemas auto-generated from function signatures
- HTTP server patches FastMCP schemas for exact parity
- Tests validate consistency: `test_mcp_tool_consistency.py`

### MCP Server Constraints

**Stdio Mode:**
- **Global State**: Required (persistent connection across requests)
- **NO STDOUT LOGS**: Breaks JSON-RPC protocol (only stderr allowed)
- **Logging Disabled**: At module level (lines 56-67 in `stdio.py`)
- **Tree-sitter Warnings**: Suppressed (line 21)
- **Connection**: Single client per server instance

**HTTP Mode:**
- **Lazy Initialization**: Services created per-request (FastMCP pattern)
- **Swagger Docs**: Available at `/docs` endpoint
- **WebSocket**: Supported for streaming responses
- **Concurrent Connections**: Multiple clients supported

### MCP Server Deployment

**Stdio (for IDEs):**
```bash
# Manual startup
./scripts/mcp-server.sh --db /path/to/db

# VS Code config (settings.json)
{
  "mcp.servers": {
    "chunkhound": {
      "command": "/path/to/scripts/mcp-server.sh",
      "args": ["--db", "/path/to/db"]
    }
  }
}

# Cursor/Windsurf config
# Similar pattern with MCP server configuration
```

**HTTP (for web clients):**
```bash
# Start server
chunkhound mcp http --port 5173

# Test endpoint
curl http://localhost:5173/docs

# Docker deployment (optional)
docker run -p 5173:5173 \
  -v /path/to/db:/data/db \
  -e CHUNKHOUND_DATABASE__PATH=/data/db \
  chunkhound mcp http --port 5173
```

---

## 7. Performance-Critical Paths

### Indexing Pipeline Performance

**Tuned Constants** (`indexing_coordinator.py`):

```python
# File count thresholds for worker scaling
SMALL_FILE_COUNT_THRESHOLD = 100     # Use minimal workers (<100 files)
MEDIUM_FILE_COUNT_THRESHOLD = 1000   # Scale up (1000+ files)

# Worker pool sizes
MAX_WORKERS_SMALL_BATCH = 4          # Small workloads
MAX_WORKERS_MEDIUM_BATCH = 8         # Medium workloads
MAX_WORKERS_LARGE_BATCH = 16         # Large monorepos

# Parallel discovery (NEW)
parallel_discovery: bool = True       # Enable parallel file scanning
min_dirs_for_parallel: int = 4        # Auto-activate threshold
max_discovery_workers: int = 16       # Worker process limit
```

**Performance Benchmarks:**

| Operation | Unbatched | Batched | Speedup | Critical? |
|-----------|-----------|---------|---------|-----------|
| **Embeddings** (1000 texts) | 100s | 1s | **100x** | ⚠️ CRITICAL |
| **DB inserts** (5000 chunks) | 250s | 1s | **250x** | ⚠️ CRITICAL |
| **File parsing** (1000 files) | 45s | 8s | **5.6x** | ✅ Important |
| **Directory discovery** (2000 files) | 2.5s | 0.8s | **3.1x** | ✅ Important |

### Embedding Batching (MANDATORY)

**Why Batching is Critical:**
- **Without batching**: 1 API call per chunk = 1000 calls = 100 seconds
- **With batching**: 10 API calls (100 chunks each) = 1 second
- **Rate limits**: OpenAI (500 req/min), VoyageAI (10000 req/min)

**Provider-Specific Batch Sizes:**

```python
# Default batch sizes (tuned per provider)
embedding_batch_size: int = 100        # Chunks per API call

# Max concurrent batches (auto-detected)
OpenAI: 8 batches       # Rate limit: 500 req/min
VoyageAI: 40 batches    # Higher rate limits
Ollama: 4 batches       # Local resource limit (CPU/GPU)
```

**Configuration:**
```bash
# Adjust batch size (lower for rate limits)
export CHUNKHOUND_EMBEDDING__EMBEDDING_BATCH_SIZE=50

# Adjust concurrency (lower for 429 errors)
export CHUNKHOUND_EMBEDDING__MAX_CONCURRENT_BATCHES=4
```

### LanceDB Fragment Optimization

**What are fragments?**
- LanceDB stores data in fragments (separate files per write)
- More fragments = faster writes, slower reads
- Optimization compacts fragments for read performance

**Configuration Trade-offs:**

| Threshold | Write Speed | Read Speed | Use Case |
|-----------|-------------|------------|----------|
| `0` (always) | Slowest | Fastest | Testing/development |
| `50` (aggressive) | Slower | Faster | Read-heavy workloads |
| `100` (balanced) | Good | Good | **Default production** |
| `500` (conservative) | Faster | Slower | Write-heavy workloads |

**Configuration:**
```bash
export CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD=100
```

### Vector Index Optimization

**Drop HNSW indexes for bulk inserts > 50 rows (20x speedup):**

```python
# Automatic optimization in LanceDB provider
if chunk_count > 50:
    await drop_vector_indexes()      # Drop existing HNSW index
    await bulk_insert(chunks)        # Fast bulk insert
    await recreate_vector_indexes()  # Rebuild HNSW index
```

**Why this works:**
- HNSW index updates are expensive (per-row overhead)
- Bulk insert without index = linear write time
- Rebuild index once = logarithmic time
- Net result: 20x speedup for large batches

### Reranking Performance

**Batch Splitting for Reranking:**

```bash
# Model-specific batch limits (auto-detected)
Qwen3-8B:  max_batch_size = 64
Qwen3-4B:  max_batch_size = 96
Qwen3-0.6B: max_batch_size = 128
Others:    max_batch_size = 128 (default)

# User override (bounded by model caps)
export CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE=32

# Example: TEI server with --max-batch-size 32
# ChunkHound automatically splits 100 documents into 4 batches
```

**Multi-Hop Search:**
- 1 initial rerank (top K results)
- N expansion reranks (typically 0-3 hops)
- 5 second time limit (prevents excessive API calls)
- Graceful degradation (falls back to similarity scores)

---

## 8. Monitoring & Observability

### Startup Profiling

**Emit JSON diagnostics for performance analysis:**

```bash
# Enable profiling (outputs to stderr)
CHUNKHOUND_NO_RICH=1 chunkhound index --profile-startup 2>profile.json

# Profile fields:
{
  "startup_profile": {
    "discovery_ms": 154.2,           # File discovery time
    "cleanup_ms": 12.7,              # Database cleanup time
    "change_scan_ms": 3.1,           # Incremental change detection
    "resolved_backend": "git_only",  # Discovery backend used
    "resolved_reasons": ["all_repos"],  # Why this backend
    "git_rows_tracked": 420,         # Files tracked by git
    "git_rows_total": 437,           # Total files found
    "git_pathspecs": 4               # Number of git pathspec filters
  }
}
```

**Use cases:**
- CI/CD performance regression detection
- Profiling indexing bottlenecks
- Debugging slow startups

### Storage Usage Monitoring

**Automatic Enforcement:**

```bash
# Set disk usage limit (prevents runaway storage)
export CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB=10.0

# Behavior:
# - Checked before indexing operations
# - Raises DiskUsageLimitExceededError when exceeded
# - Prevents partial writes (atomic transaction failure)
```

**Manual Monitoring:**

```bash
# Check database size
du -sh .chunkhound/db/

# DuckDB
du -sh .chunkhound/db/chunks.db

# LanceDB
du -sh .chunkhound/db/lancedb.lancedb/
```

### Logging Strategy

**Structured Logging (Loguru):**

```python
from loguru import logger

# Standard log levels
logger.debug("Discovery completed", files=count, duration_ms=elapsed)
logger.info("Indexing started", path=path, provider=provider)
logger.warning("Provider failed, trying fallback", provider=name)
logger.error("Critical failure", error=str(e), traceback=True)
```

**Logging by Interface:**

| Interface | Logging Target | Rationale |
|-----------|----------------|-----------|
| **CLI** | stderr + Rich progress bars | User-friendly output |
| **MCP Stdio** | DISABLED (no stdout/stderr) | JSON-RPC protocol constraint |
| **MCP HTTP** | stderr only | FastMCP server logs |

**Critical:** MCP stdio mode MUST NOT log to stdout (breaks JSON-RPC protocol)

### Health Checks

**Smoke Tests (MANDATORY before commits):**

```bash
# Run smoke tests
uv run pytest tests/test_smoke.py -v

# What it tests:
# 1. Module imports (catches syntax/type annotation errors)
# 2. CLI commands (ensures help text loads)
# 3. Server startup (verifies no import crashes)

# Exit codes:
# 0 = All tests passed (safe to commit)
# 1 = Tests failed (DO NOT commit)
```

**Integration Health Check:**

```bash
# Test full indexing pipeline
chunkhound index /tmp/test-repo --simulate

# Test MCP server startup
timeout 5s chunkhound mcp stdio < /dev/null
# Should exit cleanly after 5 seconds

# Test HTTP server
chunkhound mcp http --port 5173 &
SERVER_PID=$!
sleep 2
curl http://localhost:5173/docs
kill $SERVER_PID
```

---

## 9. Common Operational Patterns

### Development Setup

```bash
# 1. Install uv package manager (REQUIRED)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/chunkhound/chunkhound.git
cd chunkhound

# 3. Install dependencies (development + all extras)
uv sync --dev

# 4. Verify installation with smoke tests
uv run pytest tests/test_smoke.py -v

# 5. Run full test suite (parallel execution)
uv run pytest tests/ -n auto

# 6. Development workflow
uv run ruff format chunkhound  # Format code
uv run ruff check chunkhound   # Lint code
uv run mypy chunkhound         # Type check
```

### Production Deployment

**Option 1: PyPI Install (Recommended)**

```bash
# Using uv (isolated environment)
uv tool install chunkhound

# Using pip (global or virtualenv)
pip install chunkhound

# Verify
chunkhound --version
```

**Option 2: Binary Distribution (No Python Required)**

```bash
# macOS Universal (Intel + Apple Silicon)
wget https://github.com/chunkhound/chunkhound/releases/download/vX.Y.Z/chunkhound-macos-universal.tar.gz
tar xzf chunkhound-macos-universal.tar.gz
./chunkhound-macos-universal/chunkhound-optimized --version

# Ubuntu 16.04+ (glibc 2.23+)
wget https://github.com/chunkhound/chunkhound/releases/download/vX.Y.Z/chunkhound-ubuntu16-amd64.tar.gz
tar xzf chunkhound-ubuntu16-amd64.tar.gz
./chunkhound-ubuntu16-amd64/chunkhound-optimized --version
```

**Option 3: From Locked Requirements**

```bash
# Clone repo
git clone https://github.com/chunkhound/chunkhound.git
cd chunkhound

# Install exact dependencies (reproducible)
pip install -r requirements-lock.txt

# Install chunkhound
pip install .

# Verify
chunkhound --version
```

### MCP Server Deployment

**Stdio Mode (for IDEs):**

```bash
# Using helper script
./scripts/mcp-server.sh --db /path/to/db

# Manual invocation
CHUNKHOUND_DATABASE__PATH=/path/to/db \
CHUNKHOUND_EMBEDDING__API_KEY=sk-... \
chunkhound mcp stdio

# VS Code integration (settings.json)
{
  "mcp.servers": {
    "chunkhound": {
      "command": "/usr/local/bin/chunkhound",
      "args": ["mcp", "stdio"],
      "env": {
        "CHUNKHOUND_DATABASE__PATH": "/path/to/db",
        "CHUNKHOUND_EMBEDDING__API_KEY": "sk-..."
      }
    }
  }
}
```

**HTTP Mode (for web clients):**

```bash
# Start server
CHUNKHOUND_DATABASE__PATH=/data/db \
CHUNKHOUND_EMBEDDING__API_KEY=sk-... \
chunkhound mcp http --port 5173

# Systemd service (production)
cat > /etc/systemd/system/chunkhound-mcp.service <<EOF
[Unit]
Description=ChunkHound MCP Server
After=network.target

[Service]
Type=simple
User=chunkhound
WorkingDirectory=/opt/chunkhound
Environment="CHUNKHOUND_DATABASE__PATH=/data/db"
Environment="CHUNKHOUND_EMBEDDING__API_KEY=sk-..."
ExecStart=/usr/local/bin/chunkhound mcp http --port 5173
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

systemctl enable chunkhound-mcp
systemctl start chunkhound-mcp
```

### Configuration Management

**Per-Project Configuration:**

```bash
# Create local config
cd /my/project
cat > .chunkhound.json <<EOF
{
  "embedding": {
    "provider": "openai",
    "api_key": "sk-proj-..."
  },
  "indexing": {
    "exclude": ["**/dist/**", "**/*.min.js"]
  }
}
EOF

# Index with local config
chunkhound index .
```

**Global Configuration (Environment Variables):**

```bash
# Create shell profile config
cat >> ~/.bashrc <<EOF
export CHUNKHOUND_DATABASE__PATH=/data/global-db
export CHUNKHOUND_DATABASE__PROVIDER=lancedb
export CHUNKHOUND_EMBEDDING__PROVIDER=openai
export CHUNKHOUND_EMBEDDING__API_KEY=sk-...
EOF

source ~/.bashrc

# Index any codebase with global config
chunkhound index /path/to/codebase
```

**CI/CD Configuration (Temporary Overrides):**

```bash
# CI pipeline example (GitHub Actions)
- name: Index codebase
  run: |
    chunkhound index . \
      --db /tmp/ci-db \
      --database-provider duckdb \
      --no-embeddings  # Skip embeddings in CI (faster)
  env:
    CHUNKHOUND_DATABASE__PATH: /tmp/ci-db
```

---

## 10. Troubleshooting Guide

### Database Lock Errors

**Symptom:**
```
Error: IO Error: Could not set lock on file "/path/to/chunks.db"
```

**Cause:**
- Multiple processes trying to access same DuckDB file
- Concurrent CLI invocations
- MCP server + CLI accessing same database

**Solutions:**

```bash
# Option 1: Use LanceDB (multi-process safe)
export CHUNKHOUND_DATABASE__PROVIDER=lancedb
chunkhound index .

# Option 2: Use separate database paths per process
chunkhound index /path/to/codebase --db /tmp/process1-db
chunkhound index /path/to/codebase --db /tmp/process2-db

# Option 3: Use HTTP MCP server (single backend)
# Terminal 1: Start server
chunkhound mcp http --port 5173

# Terminal 2: Use server (no direct DB access)
# Connect via HTTP client
```

### JSON-RPC Protocol Errors (MCP Stdio)

**Symptom:**
```
Error: Invalid JSON-RPC message
Error: Unexpected stdout output
```

**Cause:**
- Stdout logging in MCP stdio mode
- `print()` statements in code
- Library warnings to stdout

**Debug Steps:**

```bash
# 1. Check for print() statements
grep -r "print(" chunkhound/mcp_server/

# 2. Verify logging disabled (stdio.py lines 56-67)
uv run python -c "
from chunkhound.mcp_server import stdio
import logging
assert logging.root.level == logging.CRITICAL
"

# 3. Check for tree-sitter warnings (suppressed at line 21)
grep -A5 "filterwarnings" chunkhound/mcp_server/stdio.py

# 4. Run with stderr redirect to isolate issue
chunkhound mcp stdio 2>/dev/null
```

**Solution:**
- Remove all `print()` statements from `mcp_server/`
- Ensure logging disabled before MCP server start
- Suppress library warnings (tree-sitter, etc.)

### Embedding Rate Limit Errors

**Symptom:**
```
Error: Rate limit exceeded (429)
Error: Too many requests
```

**Cause:**
- Too many concurrent API requests
- Batch size too large
- Provider rate limit reached

**Solutions:**

```bash
# Option 1: Reduce concurrent batches
export CHUNKHOUND_EMBEDDING__MAX_CONCURRENT_BATCHES=2

# Option 2: Reduce batch size
export CHUNKHOUND_EMBEDDING__EMBEDDING_BATCH_SIZE=50

# Option 3: Add retry backoff (automatic in providers)
# Providers automatically retry with exponential backoff

# Option 4: Use local embeddings (no rate limits)
export CHUNKHOUND_EMBEDDING__PROVIDER=ollama
export CHUNKHOUND_EMBEDDING__BASE_URL=http://localhost:11434/v1
export CHUNKHOUND_EMBEDDING__MODEL=nomic-embed-text
```

### Performance Issues

**Slow Indexing:**

```bash
# 1. Enable parallel discovery (NEW)
export CHUNKHOUND_INDEXING__PARALLEL_DISCOVERY=true

# 2. Increase worker count (up to CPU cores)
export CHUNKHOUND_INDEXING__MAX_DISCOVERY_WORKERS=16

# 3. Profile startup to identify bottleneck
CHUNKHOUND_NO_RICH=1 chunkhound index --profile-startup 2>profile.json
cat profile.json | jq '.startup_profile'

# 4. Check for slow embedding provider
# Try local Ollama instead of OpenAI
export CHUNKHOUND_EMBEDDING__PROVIDER=ollama
```

**Large Database Size:**

```bash
# 1. Set storage limit (prevents runaway growth)
export CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB=20.0

# 2. Use LanceDB (more efficient for large datasets)
export CHUNKHOUND_DATABASE__PROVIDER=lancedb

# 3. Optimize LanceDB fragments (reduce read overhead)
export CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD=50

# 4. Exclude large files/directories
cat > .chunkhound.json <<EOF
{
  "indexing": {
    "exclude": [
      "**/node_modules/**",
      "**/dist/**",
      "**/*.min.js",
      "**/*.bundle.js"
    ]
  }
}
EOF
```

**Slow Search Performance:**

```bash
# 1. Use reranking for better results (if available)
export CHUNKHOUND_EMBEDDING__RERANK_URL=http://localhost:8080/rerank
export CHUNKHOUND_EMBEDDING__RERANK_FORMAT=tei

# 2. Optimize LanceDB fragments
export CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD=50

# 3. Use regex search (faster, no embeddings needed)
chunkhound search --regex "pattern"
```

### Reranking Errors

**Server Not Available:**

```bash
# Error: Failed to connect to rerank service
# Solution: Start rerank server
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id BAAI/bge-reranker-base

# Verify server
curl http://localhost:8080/health
```

**Format Mismatch:**

```bash
# Error: Missing required fields in response
# Solution: Set correct format (tei or cohere)
export CHUNKHOUND_EMBEDDING__RERANK_FORMAT=tei  # For TEI servers
export CHUNKHOUND_EMBEDDING__RERANK_FORMAT=cohere  # For vLLM/Cohere
```

**Batch Size Mismatch:**

```bash
# Error: 413 Payload Too Large
# Solution: Match TEI server limit
export CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE=32
```

---

## 11. Key DevOps Takeaways

### Infrastructure Requirements

**Compute:**
- **CPU**: 4-16 cores recommended (parallelism scales with cores)
- **Memory**: ~2GB base + (total_chunks × 0.5KB)
- **Disk**: Database grows ~1KB per code chunk
- **Network**: Embedding API rate limits vary by provider

**Software:**
- **Python**: 3.10-3.13 (3.14 untested)
- **Package Manager**: `uv` (MANDATORY for development)
- **OS**: Linux, macOS (PyInstaller binaries: Ubuntu 16.04+, macOS 10.13+)

### Critical Design Constraints

1. **Database Concurrency**: Single-threaded access (enforced by SerialDatabaseProvider)
2. **Embedding Batching**: MANDATORY (100x performance difference)
3. **MCP Stdio Logging**: NO STDOUT LOGS (breaks JSON-RPC)
4. **Versioning**: Git tag-based (hatch-vcs, no manual editing)
5. **Testing**: Smoke tests MANDATORY before commits

### Deployment Best Practices

**Production Checklist:**

- [ ] Use `requirements-lock.txt` for reproducible installs
- [ ] Run smoke tests before deploying (`test_smoke.py`)
- [ ] Configure storage limits (`max_disk_usage_gb`)
- [ ] Use LanceDB for multi-process scenarios
- [ ] Monitor startup profiling for performance regressions
- [ ] Set up health checks (HTTP `/docs` endpoint)
- [ ] Configure logging to stderr only (MCP stdio)
- [ ] Use environment variables for secrets (not config files)

**Performance Optimization:**

- [ ] Enable parallel discovery (`parallel_discovery: true`)
- [ ] Tune worker count (`max_discovery_workers: 16`)
- [ ] Configure embedding batch sizes per provider
- [ ] Optimize LanceDB fragments (`lancedb_optimize_fragment_threshold: 100`)
- [ ] Use reranking for multi-hop search (optional)
- [ ] Profile startup for bottleneck identification

**Security:**

- [ ] Store API keys in secrets manager (not config files)
- [ ] Use environment variables for sensitive config
- [ ] Restrict filesystem access (database paths)
- [ ] Disable unnecessary network access (local Ollama vs OpenAI)
- [ ] Review exclusion patterns (don't index secrets)

### Contribution Workflow

**Before Committing:**

```bash
# 1. Make changes
# (edit code...)

# 2. Run smoke tests (MANDATORY - catches import/startup errors)
uv run pytest tests/test_smoke.py -v

# 3. Run full test suite
uv run pytest tests/ -n auto

# 4. Format code
uv run ruff format chunkhound

# 5. Lint code
uv run ruff check chunkhound

# 6. Type check
uv run mypy chunkhound

# 7. Commit
git commit -m "feat: Add feature X"
```

**Release Workflow:**

```bash
# 1. Create version tag
uv run scripts/update_version.py 4.1.0

# 2. Run smoke tests
uv run pytest tests/test_smoke.py -v

# 3. Prepare release (regenerate locks)
./scripts/prepare_release.sh

# 4. Test local install
pip install dist/chunkhound-4.1.0-py3-none-any.whl
chunkhound --version

# 5. Push tag
git push origin v4.1.0

# 6. Publish to PyPI
export PYPI_TOKEN=pypi-...
uv publish
```

### Monitoring & Observability

**Key Metrics:**

- **Indexing Speed**: Files/second (profile with `--profile-startup`)
- **Database Size**: MB per 1000 chunks (monitor disk usage)
- **Embedding Latency**: API response time (provider dashboards)
- **Search Latency**: Query response time (application logs)
- **Rate Limit Errors**: 429 errors (provider dashboards)

**Health Checks:**

```bash
# Application health
chunkhound --version

# Database health (DuckDB)
sqlite3 .chunkhound/db/chunks.db "SELECT COUNT(*) FROM chunks;"

# Database health (LanceDB)
ls -lh .chunkhound/db/lancedb.lancedb/

# MCP server health (HTTP)
curl http://localhost:5173/docs

# Embedding provider health
curl -H "Authorization: Bearer $API_KEY" \
  https://api.openai.com/v1/embeddings \
  -d '{"model": "text-embedding-3-small", "input": "test"}'
```

### Common Failure Modes

| Failure | Detection | Recovery | Prevention |
|---------|-----------|----------|------------|
| Database lock | CLI error | Switch to LanceDB | Use single process or LanceDB |
| Rate limit | 429 error | Reduce concurrency | Tune batch sizes |
| Disk full | Storage limit error | Increase limit or clean up | Monitor disk usage |
| JSON-RPC error | MCP client error | Check stdout logs | Disable stdout logging |
| Slow indexing | Profile > 5s/1000 files | Enable parallel discovery | Profile regularly |

---

## Appendix: Quick Reference

### Environment Variables

```bash
# Database
CHUNKHOUND_DATABASE__PATH=/path/to/db
CHUNKHOUND_DATABASE__PROVIDER=lancedb|duckdb
CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB=10.0
CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD=100

# Embedding
CHUNKHOUND_EMBEDDING__PROVIDER=openai|ollama|voyageai
CHUNKHOUND_EMBEDDING__API_KEY=sk-...
CHUNKHOUND_EMBEDDING__MODEL=text-embedding-3-small
CHUNKHOUND_EMBEDDING__EMBEDDING_BATCH_SIZE=100
CHUNKHOUND_EMBEDDING__MAX_CONCURRENT_BATCHES=8
CHUNKHOUND_EMBEDDING__RERANK_URL=http://localhost:8080/rerank
CHUNKHOUND_EMBEDDING__RERANK_FORMAT=tei|cohere|auto
CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE=32

# LLM (for code_research)
CHUNKHOUND_LLM__PROVIDER=anthropic|openai|gemini|ollama
CHUNKHOUND_LLM__API_KEY=sk-...
CHUNKHOUND_LLM__UTILITY_MODEL=claude-haiku-4-5-20251001
CHUNKHOUND_LLM__SYNTHESIS_MODEL=claude-sonnet-4-5-20250929

# Indexing
CHUNKHOUND_INDEXING__PARALLEL_DISCOVERY=true
CHUNKHOUND_INDEXING__MAX_DISCOVERY_WORKERS=16
```

### CLI Commands

```bash
# Index
chunkhound index [dir] [--db PATH] [--no-embeddings] [--simulate] [--profile-startup]

# Search
chunkhound search [query] [--semantic] [--regex] [--threshold 0.7] [--limit 10]

# MCP
chunkhound mcp stdio
chunkhound mcp http --port 5173

# Research
chunkhound research "question" [--token-budget 50000] [--no-synthesis]

# Version
chunkhound --version
```

### File Locations

```bash
# Database (default)
.chunkhound/db/chunks.db          # DuckDB
.chunkhound/db/lancedb.lancedb/   # LanceDB

# Configuration
.chunkhound.json                  # Per-project config

# Logs
stderr                            # All logging output

# Version
chunkhound/_version.py            # Auto-generated (gitignored)
```

### Testing Commands

```bash
# Smoke tests (MANDATORY)
uv run pytest tests/test_smoke.py -v

# Full test suite
uv run pytest tests/ -n auto

# Specific test file
uv run pytest tests/test_mcp_stdio.py -v

# With coverage
uv run pytest tests/ --cov=chunkhound --cov-report=html
```

### Release Commands

```bash
# Version bump
uv run scripts/update_version.py 4.1.0
uv run scripts/update_version.py --bump minor

# Prepare release
./scripts/prepare_release.sh

# Test install
pip install dist/chunkhound-*.whl

# Publish
uv publish
```

---

## Resources

- **GitHub**: https://github.com/chunkhound/chunkhound
- **PyPI**: https://pypi.org/project/chunkhound/
- **Documentation**: See `CLAUDE.md`, `CONTRIBUTING.md`, `MIGRATION_GUIDE.md`
- **Issues**: https://github.com/chunkhound/chunkhound/issues

---

**Last Updated**: 2025-12-05
**Version**: v4.0.1+
**Maintainers**: ChunkHound AI Team
