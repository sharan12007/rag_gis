import _sqlite3
import hashlib
def url_valid_check(url: str) -> bool:
    conn= _sqlite3.connect('urls.db')
    cursor = conn.cursor()
    cursor.execute('''
CREATE TABLE IF NOT EXISTS url_index (
    url TEXT PRIMARY KEY,
    content_hash TEXT
)
''')
    conn.commit()
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    result = cursor.execute('''
SELECT content_hash FROM url_index WHERE url = ?
''', (url,)).fetchone()
    if result is None:
        cursor.execute("INSERT INTO url_index (url, content_hash) VALUES (?, ?)", (url, url_hash))
        conn.commit()
        conn.close()
        return True
    else:
        stored_hash = result[0]
        conn.close()
        return False

        