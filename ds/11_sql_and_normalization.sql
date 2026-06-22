-- Database Normalization, Joins, and Transactions
-- Focuses on logical database design, relational join operations, and transaction ACID properties.

-- 1. Database Normalization (Eliminating Redundant Data)
-- - First Normal Form (1NF): Requires atomic cell values; columns must hold single values.
-- - Second Normal Form (2NF): Meets 1NF, and removes partial dependencies (non-key attributes
--   must depend on the entire composite primary key).
-- - Third Normal Form (3NF): Meets 2NF, and removes transitive dependencies (non-prime attributes
--   must not depend on other non-prime attributes).
-- - Boyce-Codd Normal Form (BCNF): A stronger version of 3NF where for every functional dependency
--   X -> Y, X must be a superkey.

-- 2. Relational Joins
-- Query patterns demonstrating structural joins across tables:

-- Inner Join: Selects records containing matching primary/foreign keys in both tables.
-- E.g., Retrieve students who have active course enrollments.
SELECT s.Name, e.CourseID
FROM Students s
INNER JOIN Enrollment e ON s.ID = e.StudentID;

-- Left Join: Retrieves all records from the left table and matched records from the right table.
-- Non-matching right-side columns default to NULL.
-- E.g., Retrieve all students, including those not enrolled in any courses.
SELECT s.Name, e.CourseID
FROM Students s
LEFT JOIN Enrollment e ON s.ID = e.StudentID;

-- Full Outer Join: Returns all records from both tables, aligning matching fields and
-- placing NULLs in columns where matches are absent.

-- 3. ACID Properties (Transactional Integrity)
-- Relational databases enforce execution safety using ACID constraints:
-- - Atomicity: Guarantees that all operations within a transaction commit successfully,
--   or the entire unit is aborted and rolled back (all-or-nothing).
-- - Consistency: Validates that database states comply with all defined schema constraints.
-- - Isolation: Ensures that concurrent execution of transactions leaves the database in the
--   same state as if they were executed sequentially.
-- - Durability: Guarantees that completed transactions persist on non-volatile storage
--   even in the event of system crashes.
