# Strategic Asset Allocation & Net-Zero Optimizer 📊

An institutional-grade portfolio optimization and backtesting engine that integrates Bayesian expected returns, convex optimization, and dynamic climate mandates. Built with Python and Streamlit, this tool bridges the gap between traditional quantitative finance and modern ESG constraints.

## 🧠 Mathematical Framework

The core engine (`portfolio_engine.py`) completely bypasses standard mean-variance fragility by implementing robust, institutional methodologies:

*   **Covariance Shrinkage (Ledoit-Wolf):** Mitigates estimation error in sample covariance matrices, ensuring the optimizer does not overweight assets simply due to historical noise.
*   **Bayesian Expected Returns (Black-Litterman):** Reverses-engineers benchmark weights to find implied equilibrium returns, then mathematically blends them with custom absolute and relative market views.
*   **Convex Optimization (`cvxpy`):** 
    *   *Minimum Variance:* Solved via standard Quadratic Programming (OSQP).
    *   *Maximum Sharpe Ratio:* Solved via the **Cornuéjols–Tütüncü transformation**, converting the fractional objective into a strict convex QP.
    *   *Risk Parity:* Solved via the **Spinu (2013) log-barrier formulation**, driving the solution toward equal percentage risk contribution across assets.
*   **Dynamic ESG Constraints:** Applies a hard carbon-intensity cap relative to the user-defined baseline portfolio, accurately mapping the "ESG Penalty" on the efficient frontier.

## ⚙️ Architecture & Features

The project is separated into a strict backend logic engine and a frontend interactive UI:

1.  **`portfolio_engine.py` (The Backend):** Houses the `DataManager`, `RiskEstimator`, `BlackLittermanModel`, `InstitutionalOptimizer`, and the `Backtester`. Data is fetched individually via `yfinance` to prevent MultiIndex formatting drops.
2.  **`app.py` (The Frontend):** A Streamlit dashboard utilizing `Altair` for interactive, reactive charting. 

### Key Dashboard Capabilities
*   **Dynamic Asset Universe:** Users can define custom assets, benchmark weights, and carbon intensities on the fly.
*   **Efficient Frontier Mapping:** Visually charts the exact return sacrificed to meet specific Net-Zero climate mandates.
*   **Risk Attribution:** Euler risk decomposition calculating the Marginal and Percentage Risk Contribution of every asset.
*   **Out-of-Sample Walk-Forward Backtesting:** Tests the strategy using a 3-year rolling lookback and monthly rebalancing. To prevent look-ahead bias, the backtester relies strictly on equilibrium returns (ignoring active views).

## 🚀 Installation & Usage

**Requirement:** Due to the C++ compilation requirements of the underlying quantitative solvers (`osqp`, `scs`), this environment requires a stable Python release (**Python 3.11 or 3.12**).

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Pratham-A-25/institutional-saa-optimizer.git](https://github.com/Pratham-A-25/institutional-saa-optimizer.git)
   cd institutional-saa-optimizer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the dashboard:**
   ```bash
   python -m streamlit run app.py
   ```

## 🛠️ Technology Stack
*   **Language:** Python 3.12
*   **Quantitative Solvers:** `cvxpy`, `scipy`, `osqp`, `scikit-learn`
*   **Data & Analytics:** `pandas`, `numpy`, `yfinance`
*   **Frontend & Visualization:** `streamlit`, `altair`, `matplotlib`

---
*Developed as a demonstration of institutional portfolio construction and quantitative system design.*
