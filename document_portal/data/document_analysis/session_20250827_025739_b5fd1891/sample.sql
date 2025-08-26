-- Sample SQL file for testing multi-format support

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    file_type VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Insert sample data
INSERT INTO users (username, email) VALUES 
    ('john_doe', 'john@example.com'),
    ('jane_smith', 'jane@example.com'),
    ('bob_wilson', 'bob@example.com');

INSERT INTO documents (user_id, title, content, file_type) VALUES 
    (1, 'Sample PDF', 'This is a sample PDF content', 'pdf'),
    (2, 'Sample Excel', 'This is a sample Excel content', 'xlsx'),
    (3, 'Sample CSV', 'This is a sample CSV content', 'csv');
