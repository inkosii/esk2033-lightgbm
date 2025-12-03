# app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import os
import io
import traceback

# -----------------------
# Configuration / Theme
# -----------------------
st.set_page_config(page_title="Eskom Demand Forecast (Pro)",
                   page_icon="⚡",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Eskom enterprise colors
ESKOM_PRIMARY = "#003C78"
ESKOM_ACCENT = "#00AEEF"
ESKOM_MUTED = "#6E7F97"
DATA_DIR = Path("data")
ASSETS_DIR = Path("assets")

# Paths expected by the app
DATA_PATH = DATA_DIR / "ESK2033_clean.csv"
MODEL_PKL = DATA_DIR / "electricityDemandModel.pkl"
FEATURES_PKL = DATA_DIR / "featureColumns.pkl"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"

# -----------------------
# Defensive imports / fallbacks
# -----------------------
# branding helpers
try:
    from src.branding import load_branding, inject_css, top_header, glass_card, eskom_header
except Exception:
    def load_branding(path="assets/color_palette.json"):
        return {"primary": ESKOM_PRIMARY, "accent": ESKOM_ACCENT, "muted": ESKOM_MUTED}
    def inject_css(path="assets/custom.css"):
        p = Path(path)
        if p.exists():
            try:
                st.markdown(f"<style>{p.read_text()}</style>", unsafe_allow_html=True)
            except Exception:
                pass
    def top_header(title, subtitle="", logo_path=None, brand=None):
        brand = brand or load_branding()
        left, center, right = st.columns([1, 6, 1])
        with left:
            if logo_path and Path(logo_path).exists():
                st.image(str(logo_path), width=88)
        with center:
            st.markdown(f"<div style='padding-left:4px'><div style='font-size:20px; font-weight:800; color:{brand.get('primary')}'>{title}</div><div style='font-size:13px;color:{brand.get('muted')}'>{subtitle}</div></div>", unsafe_allow_html=True)
        with right:
            st.markdown(f"<div style='text-align:right; color:{brand.get('muted')}; font-size:12px;'>v1.0</div>", unsafe_allow_html=True)

    @st.cache  # small helper no-op
    def glass_card(*args, **kwargs):
        # fallback: simple context manager replacement
        class _C:
            def __enter__(self_inner): pass
            def __exit__(self_inner, exc_type, exc, tb): pass
        return _C()

    def eskom_header(title):
        st.markdown(f"<h2 style='color:{ESKOM_PRIMARY}; margin:0'>{title}</h2>", unsafe_allow_html=True)

# ModelPredictor (tries to import robust class from src.inference; otherwise define fallback)
try:
    from src.inference import ModelPredictor
except Exception:
    import joblib, pickle
    class ModelPredictor:
        """
        Very defensive ModelPredictor fallback: attempts joblib then pickle.
        Provides predict(X) and explain(X) (explain returns None if shap not available).
        """
        def __init__(self, model_path, feature_path=None):
            self.model = None
            self.feature_columns = None
            self.model_path = Path(model_path)
            self.feature_path = Path(feature_path) if feature_path else None
            # try to load model
            if self.model_path.exists():
                try:
                    self.model = joblib.load(self.model_path)
                except Exception:
                    try:
                        with open(self.model_path, "rb") as f:
                            self.model = pickle.load(f)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load model: {e}")
            else:
                raise RuntimeError(f"Model not found at {self.model_path}")
            # load feature columns if present
            if self.feature_path and self.feature_path.exists():
                try:
                    self.feature_columns = joblib.load(self.feature_path)
                except Exception:
                    try:
                        with open(self.feature_path, "rb") as f:
                            self.feature_columns = pickle.load(f)
                    except Exception:
                        self.feature_columns = None
            else:
                self.feature_columns = None

        def predict(self, X):
            # X: pandas DataFrame - ensure shape
            if self.model is None:
                raise RuntimeError("Model not loaded")
            try:
                return self.model.predict(X)
            except Exception:
                # try numpy path
                return np.ravel(self.model.predict(np.asarray(X)))

        def explain(self, X):
            try:
                import shap
            except Exception:
                return None
            try:
                explainer = shap.Explainer(self.model, X)
                return explainer(X)
            except Exception:
                try:
                    explainer = shap.TreeExplainer(self.model)
                    return explainer.shap_values(X)
                except Exception:
                    return None

# DataPrep (try import, else fallback)
try:
    from src.preprocess import DataPrep
except Exception:
    class DataPrep:
        def __init__(self, *args, **kwargs): pass
        def prepare(self, df, feature_columns):
            # simple aligner
            X = pd.DataFrame(index=df.index)
            for f in feature_columns:
                if f in df.columns:
                    X[f] = df[f]
                else:
                    X[f] = 0
            return X

# plots (mini_pred_actual_plot, plot_timeseries, plot_feature_importance, plot_shap_summary)
try:
    from src.plots import mini_pred_actual_plot, plot_timeseries, plot_feature_importance, plot_shap_summary
except Exception:
    import plotly.graph_objects as go
    def mini_pred_actual_plot(df, preds, date_col=None, actual_col=None):
        d = df.copy().reset_index(drop=True)
        d["_pred_"] = preds
        # detect date
        if date_col is None:
            date_col = next((c for c in d.columns if c.lower() in ("date","timestamp","datetime")), None)
        x = pd.to_datetime(d[date_col]) if date_col and date_col in d.columns else d.index
        fig = go.Figure()
        if actual_col and actual_col in d.columns:
            fig.add_trace(go.Scatter(x=x, y=d[actual_col], name="Actual", mode="lines", line=dict(color=ESKOM_PRIMARY)))
        fig.add_trace(go.Scatter(x=x, y=d["_pred_"], name="Predicted", mode="lines", line=dict(color=ESKOM_ACCENT, dash="dash")))
        fig.update_layout(template="plotly_white", height=320, title="Predicted vs Actual (sample)")
        return fig

    def plot_timeseries(df, date_col=None, pred_col="prediction", actual_col=None):
        fig = go.Figure()
        x = pd.to_datetime(df[date_col]) if date_col and date_col in df.columns else df.index
        fig.add_trace(go.Scatter(x=x, y=df[pred_col], name="Predicted", line=dict(color=ESKOM_ACCENT)))
        if actual_col and actual_col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[actual_col], name="Actual", line=dict(color=ESKOM_PRIMARY)))
        fig.update_layout(template="plotly_white", height=520, title="Predicted demand over time")
        return fig

    def plot_feature_importance(model, feature_columns, top_n=20):
        import matplotlib.pyplot as plt, numpy as np
        fig, ax = plt.subplots(figsize=(8,6))
        try:
            importances = model.feature_importances_
        except Exception:
            try:
                importances = np.abs(np.ravel(model.coef_))
            except Exception:
                ax.text(0.5, 0.5, "Feature importance not available", ha="center")
                ax.axis("off")
                return fig
        feat_list = feature_columns if feature_columns else [f"f{i}" for i in range(len(importances))]
        L = min(len(importances), len(feat_list))
        idx = np.argsort(importances)[:L][::-1]
        ax.barh(np.array(feat_list)[idx], importances[idx])
        ax.set_title("Feature importances")
        return fig

    def plot_shap_summary(shap_vals, X):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,6))
        try:
            import shap
            shap.summary_plot(shap_vals, X, show=False)
            return fig
        except Exception:
            ax = fig.add_subplot(111)
            ax.text(0.5,0.5,"SHAP not available", ha="center")
            ax.axis("off")
            return fig

# bias_report (try import, else fallback)
try:
    from src.bias_report import BiasReport
except Exception:
    # Minimal compatibility wrapper expected by app.py
    class BiasReport:
        def __init__(self, training_csv=None):
            self.training_csv = training_csv
        def generate_report(self, df):
            # Simple HTML summary
            if df is None:
                return "<p>No sample provided.</p>"
            n = len(df)
            cols = len(df.columns)
            return f"<div><b>Sample rows:</b> {n:,}<br/><b>Columns:</b> {cols}</div>"

# -----------------------
# Inject CSS (if exists)
# -----------------------
inject_css("assets/custom.css")
BRAND = load_branding("assets/color_palette.json")

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.image(str(ASSETS_DIR / "eskom_logo.png") if (ASSETS_DIR / "eskom_logo.png").exists() else None, width=140)
    st.markdown("## Navigation")
    page = st.radio("", ("Welcome", "Predictions", "Visualizations", "Bias Report"))
    st.markdown("---")
    st.markdown("### Data Controls")
    use_sample = st.checkbox("Use sample dataset (first 500 rows)", value=True)
    upload = st.file_uploader("Upload input CSV (optional)", type=["csv"])
    run_button = st.button("Run / Refresh")
    st.markdown("---")
    st.caption("Eskom • Model Dashboard — Validate before operational use")

# -----------------------
# Load input df
# -----------------------
input_df = None
if upload is not None:
    try:
        input_df = pd.read_csv(upload)
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
elif use_sample and DATA_PATH.exists():
    try:
        input_df = pd.read_csv(DATA_PATH).head(500)
    except Exception:
        input_df = None

# -----------------------
# Predictor loader (cached)
# -----------------------
@st.cache_resource
def get_predictor():
    try:
        return ModelPredictor(model_path=str(MODEL_PKL), feature_path=str(FEATURES_PKL))
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

# Only attempt to instantiate if artifacts exist
predictor = None
try:
    predictor = get_predictor() if MODEL_PKL.exists() and FEATURES_PKL.exists() else None
except Exception:
    predictor = None

prep = DataPrep()

# -----------------------
# Page: Welcome
# -----------------------
if page == "Welcome":
    top_header(title="Eskom — Electricity Demand Forecast",
               subtitle="Explainable • Accountable • Production-ready",
               logo_path=str(ASSETS_DIR / "eskom_logo.png"),
               brand=BRAND)

    col1, col2 = st.columns([3,1], gap="large")
    with col1:
        st.markdown("<div class='welcome-card'><h3>Welcome to the Eskom Demand Forecast Dashboard</h3><p class='muted'>This dashboard provides predictions, explainability using SHAP, and an automated bias & drift check. Use the navigation pane to explore.</p></div>", unsafe_allow_html=True)
        if (ASSETS_DIR / "visual_for_welcome_page.png").exists():
            st.image(str(ASSETS_DIR / "visual_for_welcome_page.png"), use_column_width=True, caption="Predicted vs Actual overview")
        # Mini plot: prefer predictions.csv
        try:
            if PREDICTIONS_CSV.exists():
                df_preds = pd.read_csv(PREDICTIONS_CSV)
                # normalize columns to lowercase mapping for detection
                cols_lower = {c.lower(): c for c in df_preds.columns}
                pred_col = cols_lower.get("predicted") or cols_lower.get("prediction") or cols_lower.get("pred")
                actual_col = cols_lower.get("actual") or cols_lower.get("demand")
                date_col = cols_lower.get("date") or cols_lower.get("timestamp") or None
                preds = df_preds[pred_col] if pred_col in df_preds.columns else df_preds.iloc[:, -1]
                fig = mini_pred_actual_plot(df_preds, preds, date_col=date_col, actual_col=actual_col)
                st.plotly_chart(fig, use_container_width=True)
            elif predictor is not None and DATA_PATH.exists():
                df_full = pd.read_csv(DATA_PATH)
                date_col = next((c for c in df_full.columns if c.lower() == "date"), df_full.columns[0])
                # choose actual target if available
                actual_col = next((c for c in df_full.columns if c.lower() in ("demand","load","actual","value","target")), None)
                sample_df = df_full.sort_values(by=date_col).head(200) if date_col in df_full.columns else df_full.head(200)
                X_sample = prep.prepare(sample_df, predictor.feature_columns) if predictor and getattr(predictor,'feature_columns',None) else sample_df
                preds = predictor.predict(X_sample) if predictor else np.zeros(len(sample_df))
                fig = mini_pred_actual_plot(sample_df, preds, date_col=date_col if date_col in sample_df.columns else None, actual_col=actual_col)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions.csv and no sample/model available to render the mini plot.")
        except Exception as e:
            st.warning(f"Mini plot not available: {e}\n{traceback.format_exc()}")

    with col2:
        st.markdown("#### Quick Stats")
        if DATA_PATH.exists():
            try:
                df_train = pd.read_csv(DATA_PATH)
                st.metric("Training rows", f"{len(df_train):,}")
                st.metric("Feature count", f"{len(df_train.columns)}")
            except Exception:
                st.metric("Training rows", "Unknown")
        else:
            st.metric("Training rows", "No dataset found")
        st.markdown("<div class='card small'><b>Tip:</b> Upload a CSV in the sidebar to run predictions on your own data.</div>", unsafe_allow_html=True)

# -----------------------
# Page: Predictions
# -----------------------
elif page == "Predictions":
    top_header(title="🔮 Predictions", subtitle="Run model on your CSV or sample rows", logo_path=str(ASSETS_DIR / "eskom_logo.png"), brand=BRAND)
    st.info("Provide input data (upload) or use the sample dataset, then press **Run / Refresh** to compute predictions.")

    if input_df is None:
        st.warning("No input data available. Upload a CSV or enable sample data in the sidebar.")
    else:
        st.subheader("Input preview")
        st.dataframe(input_df.head(10))
        if run_button:
            if predictor is None:
                st.error("Model is not loaded — cannot run predictions.")
            else:
                try:
                    X = prep.prepare(input_df, predictor.feature_columns) if getattr(prep,'prepare',None) else input_df
                    preds = predictor.predict(X)
                    results = input_df.copy().reset_index(drop=True)
                    # ensure preds is array-like and length matches
                    if hasattr(preds, "shape") and len(preds) == len(results):
                        results["prediction"] = preds
                    else:
                        results["prediction"] = list(preds)[:len(results)]
                    st.success("Predictions complete")
                    st.dataframe(results.head(20))
                    # Save
                    try:
                        DATA_DIR.mkdir(parents=True, exist_ok=True)
                        results.to_csv(PREDICTIONS_CSV, index=False)
                        st.info(f"Saved predictions.csv to {PREDICTIONS_CSV}")
                    except Exception as e:
                        st.warning(f"Could not save predictions.csv: {e}")
                    # Download
                    st.download_button("Download predictions CSV", results.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
                except Exception as e:
                    st.error(f"Prediction pipeline failed: {e}\n{traceback.format_exc()}")

# -----------------------
# Page: Visualizations
# -----------------------
elif page == "Visualizations":
    top_header(title="📊 Visualizations", subtitle="Explore model outputs and performance", logo_path=str(ASSETS_DIR / "eskom_logo.png"), brand=BRAND)
    st.info("Explore interactive visualizations for insights.")
    viz_choice = st.selectbox("Choose visualization", ["Predicted demand over time", "Feature importance", "SHAP summary"])

    # load predictions if available
    pred_df = None
    if PREDICTIONS_CSV.exists():
        try:
            pred_df = pd.read_csv(PREDICTIONS_CSV)
        except Exception:
            pred_df = None

    if viz_choice == "Predicted demand over time":
        if pred_df is None and input_df is None:
            st.warning("No input or predictions available. Upload or produce predictions first.")
        else:
            dfv = pred_df if pred_df is not None else input_df.copy()
            # try to find suitable date & pred columns
            date_col = next((c for c in dfv.columns if c.lower() in ("date","timestamp","datetime")), None)
            pred_col = next((c for c in dfv.columns if c.lower() in ("prediction","predicted","pred")), None)
            if pred_col is None and "prediction" not in dfv.columns:
                st.warning("No 'prediction' column found in the data.")
            else:
                try:
                    fig = plot_timeseries(dfv, date_col=date_col, pred_col=pred_col or "prediction", actual_col=None)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Plotting failed: {e}\n{traceback.format_exc()}")

    elif viz_choice == "Feature importance":
        if predictor is None:
            st.warning("Model not loaded — feature importance unavailable.")
        else:
            try:
                fig_imp = plot_feature_importance(getattr(predictor, "model", predictor), getattr(predictor, "feature_columns", None))
                st.pyplot(fig_imp)
            except Exception as e:
                st.error(f"Feature importance plot failed: {e}\n{traceback.format_exc()}")

    elif viz_choice == "SHAP summary":
        if predictor is None:
            st.warning("Model not loaded — SHAP unavailable.")
        else:
            # build X
            if pred_df is not None:
                sample_for_shap = pred_df.head(500)
                try:
                    X = prep.prepare(sample_for_shap, predictor.feature_columns) if getattr(prep,'prepare',None) else sample_for_shap
                except Exception:
                    X = sample_for_shap
            elif input_df is not None:
                try:
                    X = prep.prepare(input_df, predictor.feature_columns) if getattr(prep,'prepare',None) else input_df
                except Exception:
                    X = input_df
            else:
                X = None

            if X is None:
                st.warning("No data available for SHAP.")
            else:
                try:
                    shap_vals = predictor.explain(X)
                    st.pyplot(plot_shap_summary(shap_vals, X))
                except Exception as e:
                    st.error(f"SHAP plotting failed: {e}\n{traceback.format_exc()}")

# -----------------------
# Page: Bias Report
# -----------------------
elif page == "Bias Report":
    top_header(title="🧭 Bias & Drift Report", subtitle="Quick audit of data and model errors", logo_path=str(ASSETS_DIR / "eskom_logo.png"), brand=BRAND)
    try:
        br = BiasReport(training_csv=str(DATA_PATH) if DATA_PATH.exists() else None)
    except Exception:
        # fallback: try import from src.bias_engine
        try:
            from src.bias_engine import BiasReport as BE
            br = BE(training_csv=str(DATA_PATH) if DATA_PATH.exists() else None)
        except Exception:
            br = BiasReport(training_csv=str(DATA_PATH) if DATA_PATH.exists() else None)

    # pick sample: prefer predictions.csv then input_df then training snippet
    sample = None
    if PREDICTIONS_CSV.exists():
        try:
            sample = pd.read_csv(PREDICTIONS_CSV)
        except Exception:
            sample = None
    if sample is None and input_df is not None:
        sample = input_df
    if sample is None and DATA_PATH.exists():
        try:
            sample = pd.read_csv(DATA_PATH).head(200)
        except Exception:
            sample = None

    if sample is None:
        st.warning("No sample or uploaded data available to generate a bias report.")
    else:
        # Two possible methods on br: generate_report or generate_html
        if hasattr(br, "generate_report"):
            try:
                st.markdown(br.generate_report(sample), unsafe_allow_html=True)
            except Exception:
                # fallback to generate_html
                if hasattr(br, "generate_html"):
                    st.markdown(br.generate_html(sample), unsafe_allow_html=True)
                else:
                    st.markdown("<p>Bias engine not fully compatible.</p>", unsafe_allow_html=True)
        elif hasattr(br, "generate_html"):
            st.markdown(br.generate_html(sample), unsafe_allow_html=True)
        else:
            st.markdown("<p>Bias engine missing report methods.</p>", unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("<div style='text-align:center; margin-top:20px; color:#667;'>© Eskom • Model Dashboard. By Sinenhlahla Q Nkosi</div>", unsafe_allow_html=True)
