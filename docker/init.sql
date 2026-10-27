CREATE EXTENSION IF NOT EXISTS vector;
-- Throwaway database for tests/test_db_integration.py (set TEST_DATABASE_URL to use it).
CREATE DATABASE docs_qa_test;
\connect docs_qa_test
CREATE EXTENSION IF NOT EXISTS vector;
