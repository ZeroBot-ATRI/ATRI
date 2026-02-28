-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 短时记忆表：chat_memory
CREATE TABLE IF NOT EXISTS chat_memory (
    id BIGSERIAL PRIMARY KEY,
    message_id VARCHAR(50),
    group_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    user_name VARCHAR(64) DEFAULT '',
    role VARCHAR(20) DEFAULT 'user',
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1024)
);

-- 兼容已有库：为旧表补充 user_name 列
ALTER TABLE chat_memory ADD COLUMN IF NOT EXISTS user_name VARCHAR(64) DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_chat_group_time ON chat_memory (group_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_chat_group_user ON chat_memory (group_id, user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_chat_msg_id ON chat_memory (message_id);
CREATE INDEX IF NOT EXISTS idx_chat_embedding ON chat_memory USING hnsw (embedding vector_cosine_ops);

-- 长时人设表：user_personas
CREATE TABLE IF NOT EXISTS user_personas (
    id BIGSERIAL PRIMARY KEY,
    group_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    persona_text TEXT DEFAULT '该用户暂无特殊画像。',
    last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    new_message_count INTEGER DEFAULT 0,
    is_updating BOOLEAN DEFAULT FALSE,
    lock_acquired_at TIMESTAMP DEFAULT NULL,
    UNIQUE (group_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_persona_group_user ON user_personas (group_id, user_id);
