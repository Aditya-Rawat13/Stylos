-- Initialize Stylos production database
-- This script runs when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
-- These will be created by SQLAlchemy, but we can add custom ones here

-- Full-text search configuration for essay content
CREATE TEXT SEARCH CONFIGURATION stylos_search (COPY = english);

-- Custom functions for stylometric analysis
CREATE OR REPLACE FUNCTION calculate_lexical_diversity(text_content TEXT)
RETURNS FLOAT AS $$
DECLARE
    word_count INTEGER;
    unique_words INTEGER;
BEGIN
    -- Simple TTR calculation
    SELECT array_length(string_to_array(lower(text_content), ' '), 1) INTO word_count;
    SELECT array_length(array(SELECT DISTINCT unnest(string_to_array(lower(text_content), ' '))), 1) INTO unique_words;
    
    IF word_count > 0 THEN
        RETURN unique_words::FLOAT / word_count::FLOAT;
    ELSE
        RETURN 0.0;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate average sentence length
CREATE OR REPLACE FUNCTION calculate_avg_sentence_length(text_content TEXT)
RETURNS FLOAT AS $$
DECLARE
    sentence_count INTEGER;
    word_count INTEGER;
BEGIN
    -- Count sentences (rough approximation)
    SELECT array_length(string_to_array(text_content, '.'), 1) - 1 INTO sentence_count;
    SELECT array_length(string_to_array(text_content, ' '), 1) INTO word_count;
    
    IF sentence_count > 0 THEN
        RETURN word_count::FLOAT / sentence_count::FLOAT;
    ELSE
        RETURN 0.0;
    END IF;
END;
$$ LANGUAGE plpgsql;