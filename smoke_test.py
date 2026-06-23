from pipeline import embed_query

vec = embed_query("senior backend engineer, Python, Postgres, AWS")
print(f"got embedding of length {len(vec)}")