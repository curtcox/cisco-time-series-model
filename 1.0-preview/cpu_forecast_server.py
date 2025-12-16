#!/usr/bin/env python3
"""Minimal Flask application exposing CiscoTsmMR CPU forecast plotting as a web service."""

from __future__ import annotations

import base64
import csv
import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

from run_plot_cpu_csv import (
    DEFAULT_SAMPLE,
    _build_forecast_timestamps,
    _create_model,
    _quantile_array,
    _load_cpu_series,
)


app = Flask(__name__)

DEFAULT_HORIZON = 1280
MAX_HORIZON = 4096
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>CiscoTsmMR CPU Forecast Sandbox</title>
    <style>
      :root {
        color: #0f172a;
        background-color: #f8fafc;
        font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }
      body {
        margin: 0 auto;
        max-width: 960px;
        padding: 2.5rem 1.5rem 4rem;
        line-height: 1.5;
      }
      h1 { font-size: 2rem; margin-bottom: 0.5rem; }
      p.lead { margin-top: 0; font-size: 1rem; color: #475569; }
      form {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 20px 35px rgba(15, 23, 42, 0.12);
        margin-bottom: 2rem;
      }
      fieldset {
        border: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 1rem;
      }
      label {
        font-weight: 600;
        color: #0f172a;
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
      }
      input[type="number"],
      textarea {
        border-radius: 0.6rem;
        border: 1px solid #cbd5f5;
        padding: 0.65rem 0.75rem;
        font-size: 1rem;
        font-family: inherit;
        resize: vertical;
      }
      input[type="file"] {
        font-size: 0.95rem;
      }
      button {
        width: fit-content;
        border: none;
        border-radius: 999px;
        padding: 0.85rem 1.6rem;
        font-size: 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #2563eb, #0ea5e9);
        color: white;
        cursor: pointer;
      }
      .error {
        background: #fee2e2;
        border-radius: 0.75rem;
        padding: 0.9rem 1.1rem;
        color: #991b1b;
        margin-bottom: 1rem;
      }
      .result-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 15px 30px rgba(15, 23, 42, 0.1);
      }
      .api-block {
        border-radius: 0.75rem;
        background: #0f172a;
        color: #e2e8f0;
        padding: 1rem 1.25rem;
        font-family: 'JetBrains Mono', 'SFMono-Regular', Menlo, monospace;
        font-size: 0.9rem;
      }
      img.preview {
        width: 100%;
        max-height: 480px;
        object-fit: contain;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
      }
    </style>
  </head>
  <body>
    <h1>CiscoTsmMR CPU Forecast Sandbox</h1>
    <p class="lead">
      Upload a CSV, paste time-series data, or rely on the bundled sample to reproduce
      <code>run_plot_cpu_csv.py</code> directly from your browser.
    </p>
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
    <form method="post" enctype="multipart/form-data">
      <fieldset>
        <label>
          Forecast horizon (minutes)
          <input type="number" name="horizon" min="1" max="{{ max_horizon }}" value="{{ horizon }}" required>
        </label>
        <label>
          CSV upload
          <input type="file" name="csv_file" accept=".csv,text/csv">
          <small>Provide columns named <code>_time</code> and <code>cpu_util</code>.</small>
        </label>
        <label>
          Or paste CSV rows
          <textarea name="csv_text" rows="6">{{ csv_text }}</textarea>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; font-weight: 500;">
          <input type="checkbox" name="use_sample" value="1" {% if use_sample %}checked{% endif %}>
          Use bundled sample when no CSV is supplied
        </label>
        <button type="submit">Generate forecast</button>
      </fieldset>
    </form>

    <section class="result-card">
      <h2>API usage</h2>
      <p>Automate forecasts via <code>/api/forecast</code> (POST JSON, URL-encoded, or multipart):</p>
      <pre class="api-block">curl -o forecast.png -X POST http://localhost:8000/api/forecast \\
  -H "Content-Type: application/json" \\
  -d '{"horizon": {{ horizon }}, "use_sample": true}'</pre>
      {% if image_data %}
        <h2>Preview</h2>
        <img class="preview" src="data:image/png;base64,{{ image_data }}" alt="Forecast preview">
      {% endif %}
    </section>
  </body>
</html>
"""


@dataclass
class SeriesPayload:
    timestamps: list[datetime]
    values: np.ndarray


@lru_cache(maxsize=1)
def _model_singleton():
    return _create_model()


def _parse_timestamp(raw: str) -> datetime:
    """Parse timestamps similarly to run_plot_cpu_csv."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(raw, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    return datetime.fromtimestamp(float(raw), tz=timezone.utc)


def _load_series_from_text(text: str) -> SeriesPayload:
    if not text or not text.strip():
        raise ValueError("CSV text is empty.")
    timestamps: list[datetime] = []
    values: list[float] = []
    reader = csv.DictReader(io.StringIO(text))
    if "_time" not in reader.fieldnames or "cpu_util" not in reader.fieldnames:
        raise ValueError("CSV must contain '_time' and 'cpu_util' columns.")
    for row in reader:
        raw_time = row.get("_time")
        raw_value = row.get("cpu_util")
        if raw_time is None or raw_value is None:
            continue
        ts = _parse_timestamp(raw_time)
        timestamps.append(ts)
        values.append(float(raw_value))
    if not timestamps:
        raise ValueError("No valid rows found in CSV data.")
    return SeriesPayload(timestamps, np.array(values, dtype=np.float32))


def _resolve_series(csv_text: Optional[str], file_storage, use_sample: bool) -> SeriesPayload:
    if file_storage and file_storage.filename:
        data = file_storage.read().decode("utf-8")
        return _load_series_from_text(data)
    if csv_text and csv_text.strip():
        return _load_series_from_text(csv_text)
    if use_sample or not (csv_text or file_storage):
        timestamps, values = _load_cpu_series(DEFAULT_SAMPLE)
        return SeriesPayload(timestamps, values)
    raise ValueError("No CSV data provided and sample usage disabled.")


def _render_plot(timestamps: list[datetime], series: np.ndarray, horizon: int) -> bytes:
    model = _model_singleton()
    forecast = model.forecast(series, horizon_len=horizon)[0]
    mean_forecast = forecast["mean"]
    quantiles = forecast["quantiles"]
    p10 = _quantile_array(quantiles, 0.1)
    p90 = _quantile_array(quantiles, 0.9)

    history_x = mdates.date2num(timestamps)
    forecast_times = list(_build_forecast_timestamps(timestamps[-1], mean_forecast.size))
    forecast_x = mdates.date2num(forecast_times)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(history_x, series, label="CPU utilization history", color="#1f77b4", linewidth=1.6)
    ax.plot(forecast_x, mean_forecast, label="Forecast mean", color="#ff7f0e", linewidth=2.2)
    if p10 is not None and p90 is not None:
        ax.fill_between(forecast_x, p10, p90, color="#ff7f0e", alpha=0.2, label="Forecast 10-90% band")

    boundary = history_x[-1]
    ax.axvline(boundary, color="k", linestyle="--", linewidth=1.0, label="Forecast boundary")
    ax.set_title("CiscoTsmMR Forecast vs. CPU Utilization History")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("CPU Utilization [%]")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\\n%H:%M", tz=timezone.utc))
    fig.autofmt_xdate()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def _extract_params(req) -> Tuple[int, SeriesPayload]:
    horizon_raw = DEFAULT_HORIZON
    csv_text = None
    use_sample = False
    if req.is_json:
        data = req.get_json(silent=True) or {}
        csv_text = data.get("csv_data")
        horizon_raw = data.get("horizon", DEFAULT_HORIZON)
        use_sample = bool(data.get("use_sample", False))
    else:
        form = req.form
        csv_text = form.get("csv_text")
        horizon_raw = form.get("horizon", DEFAULT_HORIZON)
        use_sample = form.get("use_sample") in ("1", "true", "on")
    try:
        horizon = int(horizon_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Horizon must be an integer.") from exc
    if horizon < 1 or horizon > MAX_HORIZON:
        raise ValueError(f"Horizon must be between 1 and {MAX_HORIZON}.")
    series_payload = _resolve_series(csv_text, req.files.get("csv_file"), use_sample)
    return horizon, series_payload


def _sample_preview() -> str:
    sample_lines = DEFAULT_SAMPLE.read_text(encoding="utf-8").splitlines()[:20]
    return "\n".join(sample_lines)


SAMPLE_PREVIEW = _sample_preview()


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template_string(
            HTML_TEMPLATE,
            error=None,
            image_data=None,
            csv_text=SAMPLE_PREVIEW,
            horizon=DEFAULT_HORIZON,
            use_sample=True,
            max_horizon=MAX_HORIZON,
        )
    try:
        horizon, payload = _extract_params(request)
        png_bytes = _render_plot(payload.timestamps, payload.values, horizon)
        img_b64 = base64.b64encode(png_bytes).decode("ascii")
        return render_template_string(
            HTML_TEMPLATE,
            error=None,
            image_data=img_b64,
            csv_text=request.form.get("csv_text", SAMPLE_PREVIEW),
            horizon=horizon,
            use_sample=request.form.get("use_sample") in ("1", "true", "on"),
            max_horizon=MAX_HORIZON,
        )
    except ValueError as exc:
        return render_template_string(
            HTML_TEMPLATE,
            error=str(exc),
            image_data=None,
            csv_text=request.form.get("csv_text", SAMPLE_PREVIEW),
            horizon=request.form.get("horizon", DEFAULT_HORIZON),
            use_sample=request.form.get("use_sample") in ("1", "true", "on"),
            max_horizon=MAX_HORIZON,
        ), 400


@app.post("/api/forecast")
def api_forecast():
    try:
        horizon, payload = _extract_params(request)
        png_bytes = _render_plot(payload.timestamps, payload.values, horizon)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=True,
        download_name="cpu_forecast.png",
    )


def main():
    port = int(os.environ.get("CPU_FORECAST_SERVER_PORT", "8000"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
