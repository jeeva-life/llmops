-- Complex SQL file for advanced multi-format testing
-- This file contains multiple tables, views, procedures, and complex queries

-- =====================================================
-- DATABASE SCHEMA FOR E-COMMERCE SYSTEM
-- =====================================================

-- Users table with enhanced features
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(50),
    country VARCHAR(50),
    postal_code VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    profile_image VARCHAR(255),
    bio TEXT,
    preferences JSON
);

-- Categories table
CREATE TABLE categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_id INTEGER,
    slug VARCHAR(100) UNIQUE,
    image_url VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES categories(id)
);

-- Products table
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    sku VARCHAR(50) UNIQUE,
    category_id INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    sale_price DECIMAL(10,2),
    cost_price DECIMAL(10,2),
    stock_quantity INTEGER DEFAULT 0,
    min_stock_level INTEGER DEFAULT 5,
    weight DECIMAL(8,2),
    dimensions VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- Product images table
CREATE TABLE product_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    image_url VARCHAR(255) NOT NULL,
    alt_text VARCHAR(200),
    is_primary BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Orders table
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    order_number VARCHAR(50) UNIQUE NOT NULL,
    status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
    total_amount DECIMAL(10,2) NOT NULL,
    tax_amount DECIMAL(10,2) DEFAULT 0,
    shipping_amount DECIMAL(10,2) DEFAULT 0,
    discount_amount DECIMAL(10,2) DEFAULT 0,
    payment_method VARCHAR(50),
    payment_status ENUM('pending', 'paid', 'failed', 'refunded') DEFAULT 'pending',
    shipping_address TEXT,
    billing_address TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Order items table
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Reviews table
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    title VARCHAR(200),
    comment TEXT,
    is_verified_purchase BOOLEAN DEFAULT FALSE,
    is_approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- =====================================================
-- SAMPLE DATA INSERTION
-- =====================================================

-- Insert sample users
INSERT INTO users (username, email, password_hash, first_name, last_name, phone, city, country) VALUES
('john_doe', 'john@example.com', 'hash123', 'John', 'Doe', '+1-555-0101', 'New York', 'USA'),
('jane_smith', 'jane@example.com', 'hash456', 'Jane', 'Smith', '+1-555-0102', 'San Francisco', 'USA'),
('bob_wilson', 'bob@example.com', 'hash789', 'Bob', 'Wilson', '+1-555-0103', 'Chicago', 'USA'),
('alice_brown', 'alice@example.com', 'hash101', 'Alice', 'Brown', '+1-555-0104', 'Boston', 'USA'),
('charlie_davis', 'charlie@example.com', 'hash202', 'Charlie', 'Davis', '+1-555-0105', 'Seattle', 'USA');

-- Insert sample categories
INSERT INTO categories (name, description, slug, image_url) VALUES
('Electronics', 'Electronic devices and gadgets', 'electronics', '/images/categories/electronics.jpg'),
('Clothing', 'Fashion and apparel', 'clothing', '/images/categories/clothing.jpg'),
('Books', 'Books and publications', 'books', '/images/categories/books.jpg'),
('Home & Garden', 'Home improvement and garden supplies', 'home-garden', '/images/categories/home-garden.jpg'),
('Sports', 'Sports equipment and accessories', 'sports', '/images/categories/sports.jpg');

-- Insert sample products
INSERT INTO products (name, description, sku, category_id, price, sale_price, stock_quantity, weight) VALUES
('iPhone 15 Pro', 'Latest iPhone with advanced features', 'IPH15PRO-001', 1, 999.99, 899.99, 50, 0.187),
('Samsung Galaxy S24', 'Premium Android smartphone', 'SAMS24-001', 1, 899.99, NULL, 30, 0.168),
('Nike Air Max 270', 'Comfortable running shoes', 'NIKE270-001', 2, 129.99, 99.99, 100, 0.85),
('Python Programming Book', 'Learn Python programming', 'BOOKPYTHON-001', 3, 49.99, NULL, 200, 0.5),
('Garden Tool Set', 'Complete garden maintenance kit', 'GARDEN-001', 4, 79.99, 59.99, 25, 2.5);

-- Insert sample product images
INSERT INTO product_images (product_id, image_url, alt_text, is_primary) VALUES
(1, '/images/products/iphone15pro_main.jpg', 'iPhone 15 Pro - Front View', TRUE),
(1, '/images/products/iphone15pro_back.jpg', 'iPhone 15 Pro - Back View', FALSE),
(2, '/images/products/samsung_s24_main.jpg', 'Samsung Galaxy S24 - Front View', TRUE),
(3, '/images/products/nike_airmax_270.jpg', 'Nike Air Max 270 - Side View', TRUE),
(4, '/images/products/python_book.jpg', 'Python Programming Book Cover', TRUE),
(5, '/images/products/garden_tool_set.jpg', 'Garden Tool Set - Complete Kit', TRUE);

-- Insert sample orders
INSERT INTO orders (user_id, order_number, status, total_amount, payment_method, payment_status) VALUES
(1, 'ORD-2024-001', 'delivered', 999.99, 'credit_card', 'paid'),
(2, 'ORD-2024-002', 'shipped', 129.99, 'paypal', 'paid'),
(3, 'ORD-2024-003', 'processing', 899.99, 'credit_card', 'paid'),
(4, 'ORD-2024-004', 'pending', 49.99, 'credit_card', 'pending'),
(5, 'ORD-2024-005', 'delivered', 79.99, 'paypal', 'paid');

-- Insert sample order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
(1, 1, 1, 999.99, 999.99),
(2, 3, 1, 129.99, 129.99),
(3, 2, 1, 899.99, 899.99),
(4, 4, 1, 49.99, 49.99),
(5, 5, 1, 79.99, 79.99);

-- Insert sample reviews
INSERT INTO reviews (product_id, user_id, rating, title, comment, is_verified_purchase, is_approved) VALUES
(1, 1, 5, 'Excellent phone!', 'Great camera and performance', TRUE, TRUE),
(1, 2, 4, 'Good but expensive', 'Good features but pricey', TRUE, TRUE),
(3, 3, 5, 'Very comfortable', 'Perfect for running', TRUE, TRUE),
(4, 4, 4, 'Great learning resource', 'Well written and comprehensive', TRUE, TRUE),
(5, 5, 3, 'Decent quality', 'Tools work but could be better', TRUE, TRUE);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for product details with category and image
CREATE VIEW product_details AS
SELECT 
    p.id,
    p.name,
    p.description,
    p.sku,
    p.price,
    p.sale_price,
    p.stock_quantity,
    c.name as category_name,
    pi.image_url as primary_image,
    AVG(r.rating) as average_rating,
    COUNT(r.id) as review_count
FROM products p
LEFT JOIN categories c ON p.category_id = c.id
LEFT JOIN product_images pi ON p.id = pi.product_id AND pi.is_primary = TRUE
LEFT JOIN reviews r ON p.id = r.product_id AND r.is_approved = TRUE
WHERE p.is_active = TRUE
GROUP BY p.id;

-- View for order summary
CREATE VIEW order_summary AS
SELECT 
    o.id,
    o.order_number,
    o.status,
    o.total_amount,
    o.payment_status,
    u.username,
    u.email,
    COUNT(oi.id) as item_count,
    o.created_at
FROM orders o
JOIN users u ON o.user_id = u.id
LEFT JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.id;

-- =====================================================
-- STORED PROCEDURES
-- =====================================================

-- Procedure to update product stock
DELIMITER //
CREATE PROCEDURE UpdateProductStock(
    IN product_id_param INT,
    IN quantity_change INT
)
BEGIN
    UPDATE products 
    SET stock_quantity = stock_quantity + quantity_change,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = product_id_param;
    
    SELECT 'Stock updated successfully' as message;
END //
DELIMITER ;

-- Procedure to get low stock products
DELIMITER //
CREATE PROCEDURE GetLowStockProducts()
BEGIN
    SELECT 
        id,
        name,
        sku,
        stock_quantity,
        min_stock_level
    FROM products 
    WHERE stock_quantity <= min_stock_level
    AND is_active = TRUE
    ORDER BY stock_quantity ASC;
END //
DELIMITER ;

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Indexes for better query performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_reviews_user ON reviews(user_id);

-- =====================================================
-- TRIGGERS FOR DATA INTEGRITY
-- =====================================================

-- Trigger to update product updated_at timestamp
DELIMITER //
CREATE TRIGGER update_product_timestamp
BEFORE UPDATE ON products
FOR EACH ROW
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END //
DELIMITER ;

-- Trigger to update order updated_at timestamp
DELIMITER //
CREATE TRIGGER update_order_timestamp
BEFORE UPDATE ON orders
FOR EACH ROW
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END //
DELIMITER ;

-- =====================================================
-- COMPLEX QUERIES FOR TESTING
-- =====================================================

-- Query to get top selling products
SELECT 
    p.name,
    p.sku,
    SUM(oi.quantity) as total_sold,
    SUM(oi.total_price) as total_revenue,
    AVG(r.rating) as average_rating
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
LEFT JOIN reviews r ON p.id = r.product_id AND r.is_approved = TRUE
GROUP BY p.id
ORDER BY total_sold DESC
LIMIT 10;

-- Query to get user purchase history
SELECT 
    u.username,
    u.email,
    COUNT(o.id) as total_orders,
    SUM(o.total_amount) as total_spent,
    MAX(o.created_at) as last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.status != 'cancelled'
GROUP BY u.id
ORDER BY total_spent DESC;

-- Query to get category performance
SELECT 
    c.name as category_name,
    COUNT(p.id) as product_count,
    AVG(p.price) as average_price,
    SUM(oi.quantity) as total_sold,
    SUM(oi.total_price) as total_revenue
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY c.id
ORDER BY total_revenue DESC;
