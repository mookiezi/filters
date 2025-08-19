/*
SQL Filter Script for Dataset Cleaning — Batch-Ready (32GB-optimized).

- Defines regex filters for PII, bot/command patterns, and automation noise
- Designed for integration with Python/ETL pipelines
- Matches and removes/rejects risky or unwanted rows
- Optimized for large-scale text datasets
- Compatible with CSV/Parquet ingestion (via Python)
- Extendable with additional patterns

Usage:
    # Run against a loaded table (example: messages)
    psql -d mydb -f filter.sql

    # Or include in a query
    SELECT * FROM messages
    WHERE NOT (content ~* ANY(ARRAY[
        -- patterns defined in filter.sql
    ]));
*/

-- Drop old filter_me and partitions
DROP TABLE IF EXISTS filter_me CASCADE;

-- Create fresh filter table
CREATE TABLE filter_me (
    message_id TEXT PRIMARY KEY,
    message_reference_message_id TEXT,
    guild_id TEXT,
    channel_id TEXT,
    author_id TEXT,
    content TEXT,
    timestamp TIMESTAMPTZ
);

-- Enable parallelism for large insert
SET max_parallel_workers_per_gather = 16;
SET max_parallel_workers = 16;
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET work_mem = '2GB';
SET maintenance_work_mem = '8GB';

-- Clean + insert into filter_me
INSERT INTO filter_me (
    message_id,
    message_reference_message_id,
    guild_id,
    channel_id,
    author_id,
    content,
    timestamp
)
WITH cleaned AS (
    SELECT
        message_id::text,
        message_reference_message_id::text,
        guild_id::text,
        channel_id::text,
        author_id::text,
        REGEXP_REPLACE(
            content,
            E'(^<@!?(\\d+)>\\s*|<@!?(\\d+)>|<@&(\\d+)>|<#(\\d+)>|<a?:\\w+:\\d+>|\\{[^}]+\\}|https?://\\S+|\\[([^\\]]+)\\]\\([^\\)]+\\)|^[\\.\\-\\!\\,\\;\\:]+\\s*\\w*|```|\"|\\\\)',
            '',
            'g'
        ) AS cleaned_content,
        timestamp
    FROM messages
    WHERE author_bot IS NOT TRUE
      AND content IS NOT NULL
      AND content <> ''
      AND to_tsvector('english', content) <> to_tsvector('english', '')
),
deduped AS (
    SELECT message_id, message_reference_message_id, guild_id, channel_id, author_id, cleaned_content, timestamp
    FROM (
        SELECT
            message_id,
            message_reference_message_id,
            guild_id,
            channel_id,
            author_id,
            cleaned_content,
            timestamp,
            ROW_NUMBER() OVER (PARTITION BY message_id ORDER BY timestamp ASC) AS rn
        FROM cleaned
        WHERE TRIM(cleaned_content) <> ''
          AND cleaned_content !~ '^([.\\-]{1,2}|[!,:;])\\s*\\w+'
    ) t
    WHERE rn = 1
)
SELECT
    message_id,
    message_reference_message_id,
    guild_id,
    channel_id,
    author_id,
    cleaned_content,
    timestamp
FROM deduped
WHERE guild_id IS NOT NULL;
