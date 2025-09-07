from utils import *
from data import *

# TiDB 
TIDB_HOST = os.getenv("TIDB_HOST")
TIDB_PORT = int(os.getenv("TIDB_PORT", 4000))
TIDB_USER = os.getenv("TIDB_USER")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
TIDB_DATABASE = os.getenv("TIDB_DATABASE")

# SSL-related env keys (as you provided)
TIDB_SSL_CA = os.getenv("TIDB_SSL_CA")                    # e.g. "/etc/ssl/cert.pem"
SSL_VERIFY_CERT = os.getenv("ssl_verify_cert", "True").lower() == "true"
SSL_VERIFY_IDENTITY = os.getenv("ssl_verify_identity", "True").lower() == "true"

# Gemini (GenAI)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "models/embedding-001"
GEN_MODEL =  "models/gemini-2.5-flash"


# ----------------- Create Gemini client -----------------
# Use the Client pattern (no genai.configure)
client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------- SSL config for pymysql -----------------
# Build ssl dict for pymysql. Keep it conservative but clear.
ssl_config = {}
if TIDB_SSL_CA and os.path.exists(TIDB_SSL_CA):
    # Map to cert_reqs constant
    cert_reqs = ssl.CERT_REQUIRED if SSL_VERIFY_CERT else ssl.CERT_NONE
    # Some drivers accept 'ca' and 'cert_reqs'
    ssl_config = {"ca": TIDB_SSL_CA, "cert_reqs": cert_reqs}
    # note: hostname check control may be driver dependent
else:
    # fallback: pass a minimal ssl flag (TiDB Cloud typically supports)
    ssl_config = {"ssl": True}

# ----------------- Helpers -----------------
def parse_embed_response(resp):
    """
    Accepts various shapes returned by client.models.embed_content and returns
    a list of lists (embeddings).
    """
    # Try common shapes robustly
    try:
        # new-style: resp.embeddings => list of objects with .values or dicts
        if hasattr(resp, "embeddings"):
            out = []
            for item in resp.embeddings:
                if hasattr(item, "values"):
                    out.append(list(item.values))
                elif isinstance(item, dict) and "values" in item:
                    out.append(list(item["values"]))
                else:
                    out.append(list(item))
            return out
        # alternative: resp.data -> [{"embedding": [...]}, ...]
        if isinstance(resp, dict) and "data" in resp:
            out = []
            for d in resp["data"]:
                if "embedding" in d:
                    out.append(list(d["embedding"]))
                else:
                    # fallback: try first value
                    out.append(list(next(iter(d.values()))))
            return out
    except Exception:
        pass
    raise RuntimeError("Couldn't parse embedding response; inspect `resp` structure.")


def get_embeddings_batch(texts, max_retries=3, backoff=1.0):
    """
    Batch embedding call to Gemini via client.models.embed_content.
    Splits requests into <=100 texts per call (Gemini limit).
    Returns list of embeddings.
    """
    if not texts:
        return []

    all_embeds = []
    for start in range(0, len(texts), 100):  # Gemini limit = 100
        sub_texts = texts[start:start + 100]

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=sub_texts
                )
                embeds = parse_embed_response(resp)

                if len(embeds) != len(sub_texts):
                    raise RuntimeError("Embedding count mismatch.")
                all_embeds.extend(embeds)
                break  # success, move to next batch

            except Exception as e:
                print(f"[embed] batch {start}-{start+len(sub_texts)} attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)
                else:
                    print("[embed] falling back to zero vectors for this batch")
                    all_embeds.extend([[0.0] * 768 for _ in sub_texts])

    return all_embeds


def connect_tidb():
    """Open a pymysql connection using ssl_config and basic error handling"""
    try:
        conn = pymysql.connect(
            host=TIDB_HOST,
            port=TIDB_PORT,
            user=TIDB_USER,
            password=TIDB_PASSWORD,
            database=TIDB_DATABASE,
            ssl=ssl_config,
            connect_timeout=20,
            autocommit=False
        )
        return conn
    except pymysql.MySQLError as e:
        raise ConnectionError(f"Could not connect to TiDB: {e}")


# ----------------- Main uploader -----------------
def upload_csv_to_tidb(csv_path, chunksize=2000, insert_batch_size=1000):
    """
    Reads CSV in chunks. For each chunk:
      - prepares texts
      - obtains batch embeddings
      - forms DB rows
      - inserts in sub-batches (insert_batch_size) with executemany
    Embeddings are passed as JSON string and inserted using CAST(%s AS VECTOR).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    conn = None
    try:
        conn = connect_tidb()
        cursor = conn.cursor()
        # Use CAST(%s AS VECTOR) for embedding column to be explicit
        insert_sql = """
        INSERT INTO test.crime_data
        (occ_date, occ_year, occ_month, occ_day, occ_dow, occ_hour,
         mci_category, offence, neighborhood, premises_type, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, CAST(%s AS VECTOR))
        """

        total_inserted = 0
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            # clean & ensure datetime derived fields:
            chunk["OCC_DATE"] = pd.to_datetime(chunk.get("OCC_DATE", None), errors="coerce")
            chunk["OCC_YEAR"] = chunk["OCC_DATE"].dt.year.fillna(chunk.get("OCC_YEAR")).astype('Int64')
            chunk["OCC_MONTH"] = chunk["OCC_DATE"].dt.month.fillna(chunk.get("OCC_MONTH"))
            chunk["OCC_DAY"] = chunk["OCC_DATE"].dt.day.fillna(chunk.get("OCC_DAY")).astype('Int64')
            chunk["OCC_DOW"] = chunk["OCC_DATE"].dt.day_name().fillna(chunk.get("OCC_DOW"))
            # Build texts
            texts = []
            tuples = []  # to preserve row data order
            for row in chunk.itertuples(index=False):
                # Access columns with getattr fallback since CSV column names might vary case
                mci = getattr(row, "MCI_CATEGORY", "") or getattr(row, "mci_category", "") or ""
                off = getattr(row, "OFFENCE", "") or getattr(row, "offence", "") or ""
                hood = getattr(row, "Neighborhood", "") or getattr(row, "neighborhood", "") or ""
                texts.append(f"{mci} {off} {hood}".strip())
                tuples.append(row)

            # Get embeddings in one batch
            embeddings = get_embeddings_batch(texts)

            # Prepare DB rows (as tuples). We'll convert embedding -> JSON string
            db_rows = []
            for r, emb in zip(tuples, embeddings):
                try:
                    occ_date = getattr(r, "OCC_DATE", None) or getattr(r, "occ_date", None)
                    if pd.isna(occ_date):
                        occ_date_val = None
                    else:
                        occ_date_val = pd.to_datetime(occ_date).date()

                    occ_year = getattr(r, "OCC_YEAR", None) or getattr(r, "occ_year", None)
                    occ_month = getattr(r, "OCC_MONTH", None) or getattr(r, "occ_month", None)
                    occ_day = getattr(r, "OCC_DAY", None) or getattr(r, "occ_day", None)
                    occ_dow = getattr(r, "OCC_DOW", None) or getattr(r, "occ_dow", None)
                    occ_hour = getattr(r, "OCC_HOUR", None) or getattr(r, "occ_hour", None)
                    mci_category = getattr(r, "MCI_CATEGORY", None) or getattr(r, "mci_category", None)
                    offence = getattr(r, "OFFENCE", None) or getattr(r, "offence", None)
                    neighborhood = getattr(r, "Neighborhood", None) or getattr(r, "neighborhood", None)
                    premises_type = getattr(r, "PREMISES_TYPE", None) or getattr(r, "premises_type", None)

                    emb_json = json.dumps(list(emb))
                    db_rows.append((
                        occ_date_val,
                        int(occ_year) if pd.notna(occ_year) else None,
                        str(occ_month) if occ_month is not None and not pd.isna(occ_month) else None,
                        int(occ_day) if pd.notna(occ_day) else None,
                        str(occ_dow) if occ_dow is not None and not pd.isna(occ_dow) else None,
                        int(occ_hour) if pd.notna(occ_hour) else None,
                        str(mci_category) if mci_category is not None else None,
                        str(offence) if offence is not None else None,
                        str(neighborhood) if neighborhood is not None else None,
                        str(premises_type) if premises_type is not None else None,
                        emb_json
                    ))

                except Exception as row_err:
                    # skip malformed row but continue
                    print(f"[chunk {chunk_idx}] skipping row due to parse error: {row_err}")

            # Insert db_rows in sub-batches to limit single executemany size
            for i in range(0, len(db_rows), insert_batch_size):
                sub = db_rows[i:i + insert_batch_size]
                try:
                    cursor.executemany(insert_sql, sub)
                    conn.commit()
                    total_inserted += len(sub)
                    print(f"[chunk {chunk_idx}] inserted {len(sub)} rows (total {total_inserted})")
                except Exception as db_err:
                    conn.rollback()
                    print(f"[chunk {chunk_idx}] failed to insert sub-batch: {db_err}. Continuing with next sub-batch.")

        print(f"Upload finished. Total inserted: {total_inserted}")

    except Exception as overall_err:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise RuntimeError(f"Upload failed: {overall_err}") from overall_err
    finally:
        if conn:
            conn.close()


# ----------------- If run directly -----------------
if __name__ == "__main__":
    # Example usage:
    CSV_PATH = "data/Major_Crime_Indicators_Open_Data.csv"
    upload_csv_to_tidb(CSV_PATH, chunksize=2000, insert_batch_size=1000)
