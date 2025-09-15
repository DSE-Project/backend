import os
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("⚠️ Missing SUPABASE_URL or SUPABASE_KEY in environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
