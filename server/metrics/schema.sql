CREATE TABLE IF NOT EXISTS routing (
    run_id TEXT PRIMARY KEY,
    problem_id INTEGER,
    model_id TEXT,
    difficulty TEXT,
    reasoning TEXT
);

CREATE TABLE IF NOT EXISTS optimizations (
    run_id TEXT PRIMARY KEY,
    caveman BOOLEAN,
    quantized_local_lm BOOLEAN,
    quantized_kv_cache BOOLEAN,
    web_search_compression BOOLEAN,
    local_model_solves BOOLEAN,
    long_context_compression_lemma BOOLEAN,
    long_context_compression_ai BOOLEAN
);

CREATE TABLE IF NOT EXISTS problem_solving (
    run_id TEXT PRIMARY KEY,
    problem_id INTEGER,
    attempts INTEGER,
    num_tool_calls INTEGER,
    tool_invocations TEXT, -- comma separated list of function names
    model_id TEXT,
    total_cost FLOAT
);