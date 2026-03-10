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
import os
import signal
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Regression subprocess
# ---------------------------------------------------------------------------

# Default GP parameters (server-side source of truth)
DEFAULT_PARAMS = {
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

# Keys the client is allowed to override
_CLIENT_KEYS = {
    "allowed_symbols", "population_size", "pool_size", "generations",
    "tournament_size", "optimizer_iterations", "optimizer", "epsilon",
    "max_evaluations", "max_length", "model_selection_criterion",
    "mutation_probability", "random_state", "uncertainty", "n_threads",
}


def _run_regression(points_json: str, params_json: str, result_queue: mp.Queue):
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

        # --- merge client params over defaults ------------------------------
        params = dict(DEFAULT_PARAMS)
        client_params = json.loads(params_json) if params_json else {}
        for k, v in client_params.items():
            if k in _CLIENT_KEYS:
                params[k] = v
        # Wrap uncertainty scalar into list if needed
        if isinstance(params["uncertainty"], (int, float)):
            params["uncertainty"] = [params["uncertainty"]]

        reg = SymbolicRegressor(**params)
        reg.fit(X, y)

        model = reg.model_
        expr_str = reg.get_model_string(model, 3)
        x_dense = np.linspace(x_min, x_max, 300)

        # --- helpers: scientific formatting --------------------------------
        def _sci_fmt(val, decimals=3):
            """Format a float in scientific notation with `decimals` places."""
            if val == 0:
                return "0"
            abs_val = abs(val)
            exp = int(np.floor(np.log10(abs_val)))
            if -2 <= exp <= 2:
                return f"{val:.{decimals}f}"
            mantissa = val / 10 ** exp
            return f"{mantissa:.{decimals}f}e{exp}"

        def _sci_latex_fmt(val, decimals=3):
            """Format a float as LaTeX scientific notation."""
            if val == 0:
                return "0"
            abs_val = abs(val)
            sign = "-" if val < 0 else ""
            exp = int(np.floor(np.log10(abs_val)))
            if -2 <= exp <= 2:
                return f"{val:.{decimals}f}"
            mantissa = abs_val / 10 ** exp
            return sign + f"{mantissa:.{decimals}f}" + r" \times 10^{" + str(exp) + "}"

        def _round_and_sci(expr):
            """Round all Float atoms and return (display_str, latex_str)."""
            # Collect and replace float atoms with rounded values
            for atom in list(expr.atoms(sy.Number)):
                if isinstance(atom, sy.Float):
                    expr = expr.subs(atom, sy.Float(float(atom), 4))
            # Build display string with scientific notation
            d = str(expr)
            for atom in sorted(expr.atoms(sy.Number),
                               key=lambda a: -len(str(a))):
                if isinstance(atom, sy.Float):
                    val = float(atom)
                    d = d.replace(str(atom), _sci_fmt(val))
            # Build LaTeX string with scientific notation
            l = sy.latex(expr)
            for atom in sorted(expr.atoms(sy.Number),
                               key=lambda a: -len(sy.latex(a))):
                if isinstance(atom, sy.Float):
                    val = float(atom)
                    l = l.replace(sy.latex(atom), _sci_latex_fmt(val))
            return d, l

        # --- helper: parse, simplify, get latex and curve ------------------
        def _process_tree(tree_str, precision_label=False):
            """Return (display_str, latex_str, curve_points) for a tree string."""
            try:
                parsed = sy.parse_expr(tree_str.lower())
                simplified = sy.simplify(parsed)
                d, l = _round_and_sci(simplified)
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
            tree_str = reg.get_model_string(tree, 3)
            p_display, p_latex, p_curve = _process_tree(tree_str)
            is_best = (p_display == display_str)
            if is_best:
                best_idx = i
            pareto.append({
                "r2": (-1) * round(float(obj[0]), 6),
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
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (HERE / "templates" / "index.html").read_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    process: mp.Process | None = None
    result_queue: mp.Queue | None = None
    paused: bool = False
    last_points: str | None = None   # JSON string of points for resume
    last_params: str | None = None   # JSON string of params for resume

    def _kill_process():
        nonlocal process, result_queue, paused
        if process is not None and process.is_alive():
            # If paused (SIGSTOP), resume first so terminate can be delivered
            if paused:
                try:
                    os.kill(process.pid, signal.SIGCONT)
                except OSError:
                    pass
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        process = None
        result_queue = None
        paused = False

    def _start_regression(points_json: str, params_json: str):
        nonlocal process, result_queue, last_points, last_params, paused
        _kill_process()
        last_points = points_json
        last_params = params_json
        result_queue = mp.Queue()
        process = mp.Process(
            target=_run_regression,
            args=(points_json, params_json, result_queue),
            daemon=True,
        )
        process.start()
        paused = False

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                msg = None
            except WebSocketDisconnect:
                break

            if msg is not None:
                mtype = msg["type"]

                if mtype == "cancel":
                    _kill_process()

                elif mtype == "fit":
                    points = msg["points"]
                    params = msg.get("params")
                    _start_regression(
                        json.dumps(points),
                        json.dumps(params) if params else "{}",
                    )

                elif mtype == "pause":
                    if process is not None and process.is_alive() and not paused:
                        os.kill(process.pid, signal.SIGSTOP)
                        paused = True
                        await websocket.send_text(json.dumps({"type": "paused"}))

                elif mtype == "resume":
                    if process is not None and process.is_alive() and paused:
                        new_params = json.dumps(msg.get("params")) if msg.get("params") else "{}"
                        if new_params == last_params:
                            # Same params: just SIGCONT the frozen process
                            os.kill(process.pid, signal.SIGCONT)
                            paused = False
                            await websocket.send_text(json.dumps({"type": "resumed"}))
                        else:
                            # Params changed: restart with stored points
                            if last_points is not None:
                                _start_regression(last_points, new_params)
                                await websocket.send_text(json.dumps({"type": "resumed"}))

                elif mtype == "stop":
                    _kill_process()
                    await websocket.send_text(json.dumps({"type": "stopped"}))

            # Poll result queue
            if result_queue is not None and not paused:
                try:
                    result = result_queue.get_nowait()
                    await websocket.send_text(json.dumps(result))
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
