-- SQL script for 7-max_state.sql
SELECT state, MAX(value) AS max_temp FROM temperatures GROUP BY state ORDER BY state;
