from pathlib import Path

from flask import Flask, send_from_directory

WEB = Path(__file__).parent / "web"
app = Flask(__name__, static_folder=None)

@app.get("/")
@app.get("/<path:path>")
def serve(path: str = "index.html"):
    return send_from_directory(WEB, path)

if __name__ == "__main__":
    app.run(debug=True)
