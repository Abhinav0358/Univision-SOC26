-- Lab 11: Databases and SQL Mastery - Deep Dive Tutorial
-- Syllabus Topics: Joins, ACID, Normalization (1NF to BCNF).

-- =============================================================================
-- 1. DATABASE NORMALIZATION (Designing for Integrity)
-- =============================================================================
-- The goal of normalization is to REMOVE REDUNDANCY (duplication).
-- 1NF (Atomic): No multiple values in one cell. Every column must be simple.
-- 2NF (Partial Dependency): Every non-key column must depend on the WHOLE primary key.
--     e.g., If PK is (StudentID, CourseID), StudentName shouldn't be here (it only needs StudentID).
-- 3NF (Transitive Dependency): No column should depend on another non-key column.
--     e.g., CourseRoom depends on Instructor, which depends on CourseID. Move Instructor to their own table!
-- BCNF: A stricter version of 3NF for tables with multiple overlapping candidate keys.

-- =============================================================================
-- 2. SQL JOINS (Connecting the dots)
-- =============================================================================
-- Joins allow us to combine data from different normalized tables.

-- A. INNER JOIN: The "Intersection".
-- Only returns rows where there is a match in BOTH tables.
-- e.g., Students who have actually enrolled in courses.
SELECT s.Name, e.CourseID
FROM Students s
INNER JOIN Enrollment e ON s.ID = e.StudentID;

-- B. LEFT JOIN: The "Left Table Priority".
-- Returns ALL rows from the left table, and matches from the right.
-- If no match, right columns are NULL.
-- e.g., ALL students, showing their courses (if any).
SELECT s.Name, e.CourseID
FROM Students s
LEFT JOIN Enrollment e ON s.ID = e.StudentID;

-- C. FULL OUTER JOIN: The "Union".
-- Returns all rows from both tables, matching where possible.

-- =============================================================================
-- 3. ACID PROPERTIES (The "Safety Net" for Transactions)
-- =============================================================================
-- A transaction (like sending money) MUST follow these rules:
-- 1. ATOMICITY: "All or Nothing". If the power goes out mid-transfer, nobody loses money.
-- 2. CONSISTENCY: The database remains valid. You can't spend money you don't have.
-- 3. ISOLATION: Concurrent transactions don't mess each other up.
--    (If two people buy the same last ticket at once, only one gets it).
-- 4. DURABILITY: Once the system says "Success", the data is saved to disk forever.

-- Lab 11 Complete. Database integrity achieved!
