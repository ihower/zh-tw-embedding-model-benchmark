
import sqlite3

def create_tables():
    # 連接到 SQLite 資料庫（如果不存在會自動創建）
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # 創建 paragraphs 表格
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paragraphs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL,
        model TEXT NOT NULL
    )
    ''')

    # 創建 questions 表格
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL,
        model TEXT NOT NULL,
        paragraph_id INTEGER NOT NULL,
        FOREIGN KEY (paragraph_id) REFERENCES paragraphs (id)
    )
    ''')

    # 提交更改並關閉連接
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()
    print("資料庫和表格已成功創建！")