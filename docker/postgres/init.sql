-- =============================================================================
-- PostgreSQL initialization
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- Full-text search on metadata

-- Indexes created by SQLAlchemy on first run via create_tables()
-- This file handles DB-level setup only.

-- Checkpointer schema for LangGraph is created by langgraph-checkpoint-postgres
-- automatically on first AsyncPostgresSaver.setup() call.
