CREATE TABLE IF NOT EXISTS routing (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    problem_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    difficulty TEXT,
    category TEXT,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS optimizations (
    run_id TEXT PRIMARY KEY,
    label TEXT,
    description TEXT,
    baseline BOOLEAN DEFAULT FALSE,
    caveman BOOLEAN DEFAULT FALSE,
    capabilities_prompt BOOLEAN DEFAULT FALSE,
    quantized_local_lm BOOLEAN DEFAULT FALSE,
    quantized_kv_cache BOOLEAN DEFAULT FALSE,
    web_search_compression BOOLEAN DEFAULT FALSE,
    local_model_solves BOOLEAN DEFAULT FALSE,
    long_context_compression_lemma BOOLEAN DEFAULT FALSE,
    long_context_compression_ai BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS problem_solving (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    problem_id INTEGER NOT NULL,
    attempts INTEGER DEFAULT 0,
    num_tool_calls INTEGER DEFAULT 0,
    tool_invocations TEXT, -- comma separated list of function names for MVP
    model_id TEXT,
    solved BOOLEAN DEFAULT FALSE,
    escalated BOOLEAN DEFAULT FALSE,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_cost FLOAT DEFAULT 0,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_routing_run_id ON routing(run_id);
CREATE INDEX IF NOT EXISTS idx_routing_problem_id ON routing(problem_id);
CREATE INDEX IF NOT EXISTS idx_problem_solving_run_id ON problem_solving(run_id);
CREATE INDEX IF NOT EXISTS idx_problem_solving_problem_id ON problem_solving(problem_id);