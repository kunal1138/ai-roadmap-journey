-- Create a table
CREATE TABLE Students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    marks INTEGER,
    city TEXT
);

-- Insert data
INSERT INTO Students VALUES (1, 'Kunal', 20, 95, 'Nagpur');
INSERT INTO Students VALUES (2, 'Sonu', 21, 80, 'Delhi');
INSERT INTO Students VALUES (3, 'Dimpesh', 22, 45, 'Mumbai');
INSERT INTO Students VALUES (4, 'Om', 16, 60, 'Nagpur');
INSERT INTO Students VALUES (5, 'Abhinav', 20, 88, 'Nagpur');
INSERT INTO Students VALUES (6, 'Raj', 19, 72, 'Pune');
INSERT INTO Students VALUES (7, 'Priya', 21, 55, 'Delhi');
INSERT INTO Students VALUES (8, 'Neha', 18, 90, 'Nagpur');

-- Select all
SELECT * FROM Students;

-- Select specific columns
SELECT name, marks FROM Students;

-- Where condition
SELECT * FROM Students WHERE marks > 70;

-- Order by marks
SELECT * FROM Students ORDER BY marks DESC;

-- Nagpur students
SELECT * FROM Students WHERE city = 'Nagpur';

-- Average marks
SELECT AVG(marks) FROM Students;

-- Count students
SELECT COUNT(*) FROM Students;

-- Group by city
SELECT city, AVG(marks) 
FROM Students 
GROUP BY city;

-- Update marks
UPDATE Students SET marks = 100 WHERE name = 'Kunal';

-- Delete a student
DELETE FROM Students WHERE name = 'Priya';