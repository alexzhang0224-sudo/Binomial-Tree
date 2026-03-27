import math
import re
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


DATA_FILE = Path(__file__).with_name("Apple stock Price.xlsx")
TRADING_DAYS = 252


def find_adj_close_column(columns) -> str:
    for col in columns:
        normalized = str(col).strip().lower().replace(" ", "").replace("_", "")
        if normalized in ("adjclose", "adjustedclose", "adjclose*"):
            return col
    for col in columns:
        lowered = str(col).lower()
        if "adj" in lowered and "close" in lowered:
            return col
    raise ValueError("Could not find an 'Adj Close' column in the Excel file.")


def _find_date_column(columns):
    for col in columns:
        if str(col).strip().lower() in ("date", "datetime"):
            return col
    return columns[0]


def _is_blank_cell(val) -> bool:
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s in ("", "-", "—", "NA", "N/A", "n/a")


def _parse_dividend_amount(val) -> float | None:
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    s = str(val).strip().replace(",", "")
    m = re.search(r"([\d.]+)\s*[Dd]ividend", s)
    if m:
        return float(m.group(1))
    m = re.search(r"[Dd]ividend\s*([\d.]+)", s)
    if m:
        return float(m.group(1))
    if re.fullmatch(r"[\d.]+", s):
        return float(s)
    return None


def _row_text_has_dividend(row: pd.Series) -> bool:
    for val in row:
        if pd.isna(val):
            continue
        if "dividend" in str(val).lower():
            return True
    return False


def _column_exact(columns, label: str) -> str | None:
    """Match 'Open','High',... without grabbing 'Adj Close'."""
    label_l = label.lower().replace(" ", "")
    for col in columns:
        cl = str(col).strip().lower().replace(" ", "")
        if cl == label_l:
            return col
    return None


def prepare_yahoo_style_price_table(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Split pasted Yahoo Finance-style dividend rows into column `dividend` by date.

    Dividend-only rows are removed from the OHLC series so daily returns use real prices.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _find_date_column(df.columns)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    open_c = _column_exact(df.columns, "open")
    high_c = _column_exact(df.columns, "high")
    low_c = _column_exact(df.columns, "low")
    close_c = _column_exact(df.columns, "close")
    adj_col = find_adj_close_column(df.columns)

    price_check_cols = [c for c in [open_c, high_c, low_c, close_c] if c is not None]

    div_by_date: dict[pd.Timestamp, float] = {}
    div_row_ix: list[int] = []

    for i, row in df.iterrows():
        is_div_heuristic = False
        if _row_text_has_dividend(row):
            is_div_heuristic = True
        elif price_check_cols and all(_is_blank_cell(row[c]) for c in price_check_cols):
            if _is_blank_cell(row.get(adj_col, np.nan)):
                for c in row.index:
                    if c == date_col:
                        continue
                    amt = _parse_dividend_amount(row[c])
                    if amt is not None and 0 < amt < 500:
                        is_div_heuristic = True
                        break

        if not is_div_heuristic:
            continue

        amount = None
        for c in row.index:
            if c == date_col:
                continue
            if _row_text_has_dividend(pd.Series([row[c]])):
                amount = _parse_dividend_amount(row[c])
                if amount is not None:
                    break
        if amount is None:
            for c in row.index:
                if c == date_col:
                    continue
                amount = _parse_dividend_amount(row[c])
                if amount is not None:
                    break

        if amount is None:
            continue

        d = pd.Timestamp(row[date_col]).normalize()
        div_by_date[d] = div_by_date.get(d, 0.0) + float(amount)
        div_row_ix.append(i)

    df_prices = df.drop(index=div_row_ix).copy()
    dates_norm = df_prices[date_col].dt.normalize()

    div_series = pd.Series(np.nan, index=df_prices.index, dtype=float)
    rolled_to_prior = 0
    lost_dividends = 0
    for d, amt in div_by_date.items():
        mask = dates_norm == d
        if mask.any():
            cur = div_series.loc[mask].fillna(0)
            div_series.loc[mask] = cur + float(amt)
            continue
        prior_dates = df_prices.loc[dates_norm < d, date_col]
        if prior_dates.empty:
            lost_dividends += 1
            continue
        last_prior = pd.Timestamp(prior_dates.max()).normalize()
        mask2 = dates_norm == last_prior
        if mask2.any():
            cur = div_series.loc[mask2].fillna(0)
            div_series.loc[mask2] = cur + float(amt)
            rolled_to_prior += 1

    explicit_div_col = None
    for col in df_prices.columns:
        cl = str(col).strip().lower()
        if cl in ("div", "dividends", "dividend") and col != date_col:
            explicit_div_col = col
            break

    dropped_div_col = None
    if explicit_div_col is not None:
        extra = pd.to_numeric(df_prices[explicit_div_col], errors="coerce")
        if extra.notna().any():
            both_na = div_series.isna() & extra.isna()
            div_series = div_series.fillna(0) + extra.fillna(0)
            div_series = div_series.mask(both_na, np.nan)
            dropped_div_col = (explicit_div_col, True)
        else:
            dropped_div_col = (explicit_div_col, False)
        df_prices = df_prices.drop(columns=[explicit_div_col])

    df_prices["dividend"] = div_series

    removed = len(div_row_ix)
    notes = []
    if removed:
        notes.append(
            f"Removed **{removed}** dividend-only row(s); amounts are in **`dividend`** "
            f"(same date if that date remains; otherwise rolled to the **prior** trading day)."
        )
    else:
        notes.append("No dividend-only rows detected (Yahoo-style split rows).")
    if dropped_div_col:
        name, had_vals = dropped_div_col
        if had_vals:
            notes.append(
                f"Merged values from **`{name}`** into **`dividend`** and dropped the old column."
            )
        else:
            notes.append(f"Dropped unused column **`{name}`** (no numeric dividend entries).")
    if rolled_to_prior > 0:
        notes.append(
            f"**{rolled_to_prior}** dividend event(s) were placed on the **prior** trading day in "
            "`dividend` (ex-div date had no price row after removing the dividend row)."
        )
    if lost_dividends > 0:
        notes.append(
            f"**{lost_dividends}** dividend event(s) could not be placed (no prior trading day in range)."
        )
    return df_prices, " ".join(notes)


def load_volatility_from_returns(
    file_source: Path | BinaryIO, source_name: str
) -> tuple[float, float, int, str, str, str]:
    if isinstance(file_source, Path) and not file_source.exists():
        raise FileNotFoundError(f"Input file not found: {file_source}")

    df_raw = pd.read_excel(file_source)
    df, prep_note = prepare_yahoo_style_price_table(df_raw)
    adj_col = find_adj_close_column(df.columns)
    prices = pd.to_numeric(df[adj_col], errors="coerce").dropna()
    if len(prices) < 2:
        raise ValueError("Need at least 2 valid prices to compute returns.")

    returns = prices.pct_change().dropna()
    vol_daily = returns.std(ddof=1)
    vol_annual = vol_daily * math.sqrt(TRADING_DAYS)
    return vol_daily, vol_annual, len(returns), adj_col, source_name, prep_note


def build_stock_tree(s0: float, u: float, d: float, steps: int) -> list[list[float]]:
    tree: list[list[float]] = []
    for t in range(steps + 1):
        level = []
        for down_moves in range(t + 1):
            up_moves = t - down_moves
            level.append(s0 * (u**up_moves) * (d**down_moves))
        tree.append(level)
    return tree


def build_option_tree(stock_tree: list[list[float]], k: float, r: float, dt: float, p: float) -> list[list[float]]:
    steps = len(stock_tree) - 1
    option_tree: list[list[float]] = [[] for _ in range(steps + 1)]

    # Terminal payoff for European call option.
    option_tree[steps] = [max(price - k, 0.0) for price in stock_tree[steps]]
    discount = math.exp(-r * dt)

    for t in range(steps - 1, -1, -1):
        level_values = []
        for node in range(t + 1):
            value_up = option_tree[t + 1][node]
            value_down = option_tree[t + 1][node + 1]
            level_values.append(discount * (p * value_up + (1 - p) * value_down))
        option_tree[t] = level_values

    return option_tree


def build_tree_graphviz(
    tree: list[list[float]],
    value_prefix: str,
    value_symbol: str,
    *,
    backward_edges: bool = False,
) -> str:
    """Build a Graphviz DOT string for a binomial tree with boxed nodes.

    Stock tree: backward_edges=False (forward in time, parent -> children).
    Option tree: backward_edges=True (backward induction, children -> parent).
    """
    lines = [
        "digraph BinomialTree {",
        'rankdir=LR;',
        'node [shape=box, fontname="Arial"];',
    ]

    for t, level in enumerate(tree):
        lines.append("{ rank=same;")
        for j, value in enumerate(level):
            node_id = f"{value_prefix}_{t}_{j}"
            label = f"{value_symbol}({t},{j})\\n{value:.6f}"
            lines.append(f'{node_id} [label="{label}"];')
        lines.append("}")

    for t in range(len(tree) - 1):
        for j in range(t + 1):
            parent_id = f"{value_prefix}_{t}_{j}"
            up_id = f"{value_prefix}_{t + 1}_{j}"
            down_id = f"{value_prefix}_{t + 1}_{j + 1}"
            if backward_edges:
                lines.append(f"{up_id} -> {parent_id};")
                lines.append(f"{down_id} -> {parent_id};")
            else:
                lines.append(f"{parent_id} -> {up_id};")
                lines.append(f"{parent_id} -> {down_id};")

    lines.append("}")
    return "\n".join(lines)


def build_tree_png_bytes(
    tree: list[list[float]], value_symbol: str, *, backward_edges: bool = False
) -> bytes:
    """Render a tree image with PIL for download (no system Graphviz needed)."""
    steps = len(tree) - 1
    margin = 24
    x_gap = 170
    y_gap = 72
    box_w = 140
    box_h = 48

    width = margin * 2 + steps * x_gap + box_w
    height = margin * 2 + steps * y_gap + box_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def node_xy(t: int, j: int) -> tuple[float, float]:
        x = margin + t * x_gap
        top_offset = (steps - t) * y_gap / 2.0
        y = margin + top_offset + j * y_gap
        return x, y

    # Draw edges first so boxes stay visually on top.
    for t in range(steps):
        for j in range(t + 1):
            x0, y0 = node_xy(t, j)
            x_up, y_up = node_xy(t + 1, j)
            x_dn, y_dn = node_xy(t + 1, j + 1)
            if backward_edges:
                draw.line((x_up, y_up + box_h / 2, x0 + box_w, y0 + box_h / 2), fill="black", width=1)
                draw.line((x_dn, y_dn + box_h / 2, x0 + box_w, y0 + box_h / 2), fill="black", width=1)
            else:
                draw.line((x0 + box_w, y0 + box_h / 2, x_up, y_up + box_h / 2), fill="black", width=1)
                draw.line((x0 + box_w, y0 + box_h / 2, x_dn, y_dn + box_h / 2), fill="black", width=1)

    # Draw nodes and labels.
    for t, level in enumerate(tree):
        for j, value in enumerate(level):
            x, y = node_xy(t, j)
            draw.rectangle((x, y, x + box_w, y + box_h), outline="black", width=2)
            label_1 = f"{value_symbol}({t},{j})"
            label_2 = f"{value:.6f}"
            draw.text((x + 8, y + 8), label_1, fill="black")
            draw.text((x + 8, y + 26), label_2, fill="black")

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def render_textbook_pricing_summary(
    *,
    steps: int,
    dt: float,
    t_maturity: float,
    u: float,
    d: float,
    r: float,
    p: float,
    discount_k: float,
    s0: float,
    k: float,
    call_price: float,
    put_price: float,
    parity_lhs: float,
    parity_rhs: float,
) -> None:
    """Present definitions + numeric substitution in a textbook-style layout."""
    with st.expander("Parameters with formulas", expanded=False):
        st.markdown(
            "Textbook-style definitions and numeric substitution used in this pricing run."
        )

        st.markdown("##### 1. Time to maturity")
        st.latex(r"T \;=\; N\,\Delta t \;=\; \frac{\tau}{m}")
        st.caption("Substitution (your inputs)")
        st.latex(
            rf"T \;=\; {steps} \cdot \Delta t \;=\; {steps} \times {dt:.10f} \;=\; \boxed{{{t_maturity:.10f}\ \mathrm{{years}}}}"
        )

        st.markdown("##### 2. CRR stock price factors")
        st.latex(r"u \;=\; e^{\sigma\sqrt{\Delta t}}, \qquad d \;=\; \frac{1}{u}")
        st.caption("Numerical values")
        # One string only: two rf-... pieces concatenated made "\qquad" + "d" -> invalid "\qquadd".
        st.latex(
            rf"u \;=\; e^{{\sigma\sqrt{{\Delta t}}}} \;=\; \boxed{{{u:.10f}}},"
            rf"\qquad d \;=\; \frac{{1}}{{u}} \;=\; \boxed{{{d:.10f}}}"
        )

        st.markdown("##### 3. Risk-neutral probability")
        st.latex(r"p \;=\; \frac{e^{r\Delta t} - d}{u - d}")
        st.caption("Numerical value")
        st.latex(rf"p \;=\; \boxed{{{p:.10f}}}")

        st.markdown("##### 4. Continuous discount factor on the strike")
        st.latex(r"e^{-rT}")
        st.caption("Substitution")
        st.latex(
            rf"e^{{-rT}} \;=\; \exp\!\bigl(-({r:.10f})({t_maturity:.10f})\bigr) \;=\; \boxed{{{discount_k:.10f}}}"
        )

        st.markdown("##### 5. European call (binomial tree, root node)")
        st.latex(r"C \;=\; V(0,0)")
        st.caption("Tree value at the root node (time step t = 0).")
        st.latex(rf"C \;=\; \boxed{{{call_price:.10f}}}")

        st.markdown("##### 6. Put–call parity")
        st.latex(r"C - P \;=\; S_0 - K\,e^{-rT}")
        st.caption("Parity check (left-hand side vs. right-hand side)")
        st.latex(
            rf"C - P \;=\; {call_price:.10f} - ({put_price:.10f}) \;=\; \boxed{{{parity_lhs:.10f}}}"
        )
        st.latex(
            rf"S_0 - K\,e^{{-rT}} \;=\; {s0:.10f} - {k:.10f}\cdot({discount_k:.10f}) \;=\; \boxed{{{parity_rhs:.10f}}}"
        )

        st.markdown("##### 7. European put (implied by parity)")
        st.latex(r"P \;=\; C - S_0 + K\,e^{-rT}")
        st.caption("Substitution")
        st.latex(
            rf"P \;=\; {call_price:.10f} - {s0:.10f} + {k:.10f}\cdot({discount_k:.10f}) \;=\; \boxed{{{put_price:.10f}}}"
        )

        st.markdown("##### Reference: one-period risk-neutral rollback")
        st.latex(
            r"V(t,j) \;=\; e^{-r\Delta t}\Bigl(p\,V(t+1,j) + (1-p)\,V(t+1,j+1)\Bigr)"
        )


def main() -> None:
    st.set_page_config(
        page_title="Feinstein & Zhang Risk-Neutral Option pricing",
        layout="wide",
    )
    st.title("Feinstein & Zhang Risk-Neutral Option pricing")
    st.write("European call option pricing with a CRR risk-neutral binomial tree.")

    st.subheader("Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        s0 = st.number_input("Stock price (S0)", min_value=0.0001, value=252.89, step=1.0)
        k = st.number_input("Strike price (K)", min_value=0.0001, value=255.0, step=1.0)
    with col2:
        m = st.number_input("Frequency per year (m)", min_value=1, value=12, step=1)
        tau = st.number_input("Tau (number of tree steps)", min_value=1, value=7, step=1)
    with col3:
        r = st.number_input(
            "Risk-free rate r (annual, decimal)",
            value=0.036,
            step=0.005,
            format="%.6f",
        )
    uploaded_xlsx = st.file_uploader(
        "Choose Excel file (.xlsx) for volatility estimation",
        type=["xlsx"],
        help="If not selected, the app uses the default local file.",
    )
    if uploaded_xlsx is None:
        file_source: Path | BinaryIO = DATA_FILE
        source_name = str(DATA_FILE)
    else:
        file_source = uploaded_xlsx
        source_name = uploaded_xlsx.name

    if st.button("Build Tree and Price Option", type="primary"):
        try:
            vol_daily, sigma, n_returns, adj_col, used_file_name, prep_note = (
                load_volatility_from_returns(file_source, source_name)
            )

            # Per your convention: Tau controls tree size, frequency controls time per step.
            steps = int(tau)
            dt = 1.0 / float(m)
            u = math.exp(sigma * math.sqrt(dt))
            d = 1.0 / u
            if abs(u - d) < 1e-15:
                raise ValueError("Invalid tree parameters: u and d are too close.")
            p = (math.exp(r * dt) - d) / (u - d)
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"Risk-neutral probability p={p:.6f} is outside [0, 1]. "
                    "Try a different frequency (m), Tau, or risk-free rate."
                )

            stock_tree = build_stock_tree(s0=s0, u=u, d=d, steps=steps)
            option_tree = build_option_tree(stock_tree=stock_tree, k=k, r=r, dt=dt, p=p)
            call_price = option_tree[0][0]

            # Total calendar time to maturity (years): N steps × Δt = τ/m.
            t_maturity = steps * dt
            discount_k = math.exp(-r * t_maturity)
            # European put from put–call parity: P = C − S0 + K e^(−rT).
            put_price = call_price - s0 + k * discount_k
            parity_lhs = call_price - put_price
            parity_rhs = s0 - k * discount_k

            # Keep all outputs in session state so download clicks do not clear results.
            st.session_state["pricing_results"] = {
                "steps": steps,
                "dt": dt,
                "t_maturity": t_maturity,
                "u": u,
                "d": d,
                "r": r,
                "p": p,
                "discount_k": discount_k,
                "s0": s0,
                "k": k,
                "call_price": call_price,
                "put_price": put_price,
                "parity_lhs": parity_lhs,
                "parity_rhs": parity_rhs,
                "used_file_name": used_file_name,
                "adj_col": adj_col,
                "n_returns": n_returns,
                "vol_daily": vol_daily,
                "sigma": sigma,
                "stock_tree": stock_tree,
                "option_tree": option_tree,
                "prep_note": prep_note,
            }

        except Exception as exc:
            st.error(str(exc))

    if "pricing_results" in st.session_state:
        results = st.session_state["pricing_results"]

        render_textbook_pricing_summary(
            steps=results["steps"],
            dt=results["dt"],
            t_maturity=results["t_maturity"],
            u=results["u"],
            d=results["d"],
            r=results["r"],
            p=results["p"],
            discount_k=results["discount_k"],
            s0=results["s0"],
            k=results["k"],
            call_price=results["call_price"],
            put_price=results["put_price"],
            parity_lhs=results["parity_lhs"],
            parity_rhs=results["parity_rhs"],
        )
        st.success(
            f"**Call C (binomial):** {results['call_price']:.6f}  "
            f"**Put P (put–call parity):** {results['put_price']:.6f}"
        )

        st.subheader("Volatility from Historical Returns")
        prep = results.get("prep_note", "")
        if prep:
            st.info(prep)
        st.write(f"File: `{results['used_file_name']}`")
        st.write(f"Adj Close column used: `{results['adj_col']}`")
        st.write(f"Return observations: `{results['n_returns']}`")
        st.write(f"Daily volatility (std of returns): `{results['vol_daily']:.6%}`")
        st.write(f"Annualized volatility (sigma): `{results['sigma']:.6%}`")

        st.subheader("Tree Parameters")
        params_df = pd.DataFrame(
            {
                "Parameter": ["N (steps)", "dt", "u", "d", "p"],
                "Value": [
                    results["steps"],
                    results["dt"],
                    results["u"],
                    results["d"],
                    results["p"],
                ],
            }
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        stock_tree = results["stock_tree"]
        option_tree = results["option_tree"]
        stock_dot = build_tree_graphviz(stock_tree, value_prefix="S", value_symbol="S")
        option_dot = build_tree_graphviz(
            option_tree, value_prefix="V", value_symbol="V", backward_edges=True
        )

        st.subheader("Stock Price Binomial Tree (Boxes)")
        st.graphviz_chart(stock_dot)
        st.caption("Left to right is step t=0 to t=N. Each node shows S(t,j).")
        stock_png = build_tree_png_bytes(stock_tree, "S", backward_edges=False)
        st.download_button(
            "Download Stock Tree Image (PNG)",
            data=stock_png,
            file_name="stock_tree.png",
            mime="image/png",
        )

        st.subheader("Option Value Binomial Tree (Boxes)")
        st.graphviz_chart(option_dot)
        st.caption(
            "Same layout (t=0 on the left). Arrows point from t+1 toward t for backward induction."
        )
        option_png = build_tree_png_bytes(option_tree, "V", backward_edges=True)
        st.download_button(
            "Download Option Tree Image (PNG)",
            data=option_png,
            file_name="option_tree.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
