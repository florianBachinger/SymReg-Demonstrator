"""
Symbolic Regression Drawing Demonstrator
=========================================
Draw a curve on a fullscreen grid canvas. PyOperon fits a symbolic expression
to the drawn points in a background process. Each new stroke cancels any
in-progress regression and starts a fresh one.

Run:  python app.py
Open: http://localhost:8766/
"""

import asyncio
import json
import multiprocessing as mp
import signal
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Regression subprocess
# ---------------------------------------------------------------------------

def _run_regression(points_json: str, result_queue: mp.Queue):
    """Run symbolic regression in a child process (CPU-bound, killable)."""
    # Ignore SIGINT in worker so only the parent handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from pyoperon.sklearn import SymbolicRegressor
        import sympy as sy

        pts = json.loads(points_json)
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)

        # --- bin by x so we have a proper function (one y per x) -----------
        n_bins = min(200, len(xs))
        x_min, x_max = xs.min(), xs.max()
        if x_max - x_min < 1e-9:
            result_queue.put({"type": "error", "message": "Draw a wider curve"})
            return
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_idx = np.digitize(xs, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        x_binned, y_binned = [], []
        for i in range(n_bins):
            mask = bin_idx == i
            if mask.any():
                x_binned.append(xs[mask].mean())
                y_binned.append(ys[mask].mean())

        X = np.array(x_binned, dtype=np.float64).reshape(-1, 1)
        y = np.array(y_binned, dtype=np.float64)

        if len(X) < 3:
            result_queue.put({"type": "error", "message": "Need more points"})
            return

        # --- fast symbolic regression params --------------------------------
        params = {
            "allowed_symbols": "add,sub,mul,div,constant,variable",
            "population_size": 200,
            "pool_size": 200,
            "generations": 30,
            "female_selector": "tournament",
            "male_selector": "tournament",
            "tournament_size": 3,
            "optimizer_iterations": 10,
            "optimizer": "lm",
            "epsilon": 1e-05,
            "max_evaluations": 100000,
            "max_length": 20,
            "model_selection_criterion": "minimum_description_length",
            "mutation_probability": 0.15,
            "objectives": ["r2", "length"],
            "random_state": None,
            "uncertainty": [0.05],
            "n_threads": 0,
        }

        reg = SymbolicRegressor(**params)
        reg.fit(X, y)

        model = reg.model_
        expr_str = reg.get_model_string(model, 6)
        x_dense = np.linspace(x_min, x_max, 300)

        # --- helper: parse, simplify, get latex and curve ------------------
        def _process_tree(tree_str, precision_label=False):
            """Return (display_str, latex_str, curve_points) for a tree string."""
            try:
                parsed = sy.parse_expr(tree_str.lower())
                simplified = sy.simplify(parsed)
                d = str(simplified)
                l = sy.latex(simplified)
                # evaluate curve via lambdify
                x_sym = sy.Symbol('x0') if 'x0' in tree_str.lower() else (
                    sy.Symbol('x_0') if 'x_0' in tree_str.lower() else
                    list(simplified.free_symbols)[0] if simplified.free_symbols else None
                )
                if x_sym is not None:
                    fn = sy.lambdify(x_sym, simplified, modules='numpy')
                    y_vals = fn(x_dense)
                    y_arr = np.atleast_1d(np.asarray(y_vals, dtype=np.float64))
                    if y_arr.shape == ():
                        y_arr = np.full_like(x_dense, float(y_arr))
                    if y_arr.shape[0] == 1 and x_dense.shape[0] > 1:
                        y_arr = np.full_like(x_dense, y_arr[0])
                    crv = list(zip(x_dense.tolist(), y_arr.tolist()))
                else:
                    # constant expression
                    c = float(simplified)
                    crv = list(zip(x_dense.tolist(), np.full_like(x_dense, c).tolist()))
                return d, l, crv
            except Exception:
                return tree_str, tree_str, []

        # --- best model ----------------------------------------------------
        display_str, latex_str, best_curve = _process_tree(expr_str)

        # fallback: use reg.predict for best model curve (more reliable)
        try:
            y_pred = reg.predict(x_dense.reshape(-1, 1))
            best_curve = list(zip(x_dense.tolist(), y_pred.tolist()))
        except Exception:
            pass

        # --- pareto front ---------------------------------------------------
        pareto = []
        best_idx = 0
        for i, s in enumerate(reg.pareto_front_):
            obj = s["objective_values"]
            tree = s["tree"]
            mdl = s["minimum_description_length"]
            tree_str = reg.get_model_string(tree, 6)
            p_display, p_latex, p_curve = _process_tree(tree_str)
            is_best = (p_display == display_str)
            if is_best:
                best_idx = i
            pareto.append({
                "r2": round(float(obj[0]), 6),
                "length": int(obj[1]),
                "mdl": round(float(mdl), 4),
                "expr": p_display,
                "latex": p_latex,
                "curve": p_curve,
            })

        result_queue.put({
            "type": "result",
            "expression": display_str,
            "latex": latex_str,
            "curve": best_curve,
            "pareto": pareto,
            "bestIdx": best_idx,
        })

    except Exception as e:
        result_queue.put({"type": "error", "message": str(e)})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "templates" / "index.html").read_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    process: mp.Process | None = None
    result_queue: mp.Queue | None = None

    def _kill_process():
        nonlocal process, result_queue
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        process = None
        result_queue = None

    try:
        while True:
            # Check for incoming messages (with a short timeout so we can
            # also poll the result queue)
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                msg = None
            except WebSocketDisconnect:
                break

            if msg is not None:
                if msg["type"] == "cancel":
                    _kill_process()

                elif msg["type"] == "fit":
                    # Cancel any running regression
                    _kill_process()

                    points = msg["points"]
                    result_queue = mp.Queue()
                    process = mp.Process(
                        target=_run_regression,
                        args=(json.dumps(points), result_queue),
                        daemon=True,
                    )
                    process.start()

            # Poll result queue
            if result_queue is not None:
                try:
                    result = result_queue.get_nowait()
                    await websocket.send_text(json.dumps(result))
                    # Clean up finished process
                    if process is not None:
                        process.join(timeout=1)
                    process = None
                    result_queue = None
                except Exception:
                    pass  # queue empty, keep polling

    except WebSocketDisconnect:
        pass
    finally:
        _kill_process()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    uvicorn.run(app, host="0.0.0.0", port=8765)
