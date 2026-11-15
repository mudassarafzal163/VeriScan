"""Start the web application:  python run.py  ->  http://127.0.0.1:5000"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from app import create_app  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("FND_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("FND_PORT", 5000)))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    app = create_app()
    print(f"\n  Fake News Detection System running at http://{a.host}:{a.port}\n")
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True, use_reloader=False)
