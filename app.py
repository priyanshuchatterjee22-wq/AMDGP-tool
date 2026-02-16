"""
Drug-Gene Interaction Prioritizer - Streamlit UI
=================================================
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import pickle
import time
import warnings
from sklearn.metrics import roc_curve, roc_auc_score
warnings.filterwarnings('ignore')

# ── must be the very first Streamlit call ──────────────────────────────────────
st.set_page_config(
    page_title="Drug-Gene Interaction Prioritizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── import the core engine ─────────────────────────────────────────────
from main_script2 import DrugGeneInteractionPrioritizer, generate_synthetic_data

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }

/* ── header banner ── */
.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d2137 100%);
    border: 1px solid #21d4fd33;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, #21d4fd11 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 { 
    color: #21d4fd; font-size: 2.2rem; font-weight: 800;
    margin: 0 0 .4rem; letter-spacing: -0.5px;
}
.main-header p  { color: #8b949e; margin: 0; font-size: 1rem; }

/* ── metric cards ── */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color .2s;
}
.metric-card:hover { border-color: #21d4fd55; }
.metric-card .val  { font-size: 2rem; font-weight: 700; color: #21d4fd; }
.metric-card .lbl  { font-size: .8rem; color: #8b949e; margin-top: .25rem; }

/* ── section cards ── */
.section-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.section-title {
    color: #e6edf3; font-size: 1.1rem; font-weight: 600;
    margin-bottom: 1rem; display: flex; align-items: center; gap: .5rem;
}

/* ── confidence badges ── */
.badge {
    display: inline-block;
    padding: .2rem .7rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .5px;
}
.badge-vh { background: #1a4731; color: #3fb950; border: 1px solid #3fb95066; }
.badge-h  { background: #1c3a5e; color: #58a6ff; border: 1px solid #58a6ff66; }
.badge-m  { background: #3d2e00; color: #e3b341; border: 1px solid #e3b34166; }
.badge-l  { background: #3d1a1a; color: #f85149; border: 1px solid #f8514966; }

/* ── step pills ── */
.step-pill {
    display: inline-flex; align-items: center; gap: .5rem;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 999px; padding: .35rem 1rem;
    color: #8b949e; font-size: .82rem; margin-bottom: .5rem;
}
.step-pill .num {
    background: #21d4fd22; color: #21d4fd;
    border-radius: 50%; width: 1.4rem; height: 1.4rem;
    display: inline-flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: .75rem;
}

/* ── sidebar labels ── */
.sidebar-section {
    color: #21d4fd; font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 1.2rem 0 .5rem; padding-bottom: .3rem;
    border-bottom: 1px solid #21d4fd33;
}

/* ── dataframe styling ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0d7377, #14a085) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: .55rem 1.5rem !important; width: 100% !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── success / warning ── */
.stSuccess { background: #1a4731 !important; border-color: #3fb950 !important; }
.stWarning { background: #3d2e00 !important; border-color: #e3b341 !important; }
.stError   { background: #3d1a1a !important; border-color: #f85149 !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"]      { color: #8b949e !important; border-radius: 6px !important; }
.stTabs [aria-selected="true"]    { color: #21d4fd !important; background: #21d4fd1a !important; }

/* ── progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #21d4fd, #14a085) !important; }

/* scrollbar */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════
for key, default in {
    "prioritizer":    None,
    "train_results":  None,
    "drug_data":      None,
    "gene_data":      None,
    "X_train":        None,
    "y_train":        None,
    "prioritized":    None,
    "model_trained":  False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_LABELS = {
    "feature_0":  "Drug – Molecular Weight",
    "feature_1":  "Drug – logP (lipophilicity)",
    "feature_2":  "Drug – H-Bond Donors",
    "feature_3":  "Drug – H-Bond Acceptors",
    "feature_4":  "Drug – Rotatable Bonds",
    "feature_5":  "Drug – Polar Surface Area",
    "feature_6":  "Drug – Aromatic Rings",
    "feature_7":  "Drug – Total H-Bonds (derived)",
    "feature_8":  "Drug – MW/Flexibility (derived)",
    "feature_9":  "Gene – Expression Level",
    "feature_10": "Gene – Protein Length",
    "feature_11": "Gene – Binding Domains",
    "feature_12": "Gene – Phosphorylation Sites",
    "feature_13": "Gene – Cellular Location Score",
    "feature_14": "Gene – Pathway Connectivity",
    "feature_15": "Gene – Domain Density (derived)",
    "feature_16": "Gene – Functional Impact (derived)",
    "feature_17": "Interaction – MW × Expression",
    "feature_18": "Interaction – logP × Binding Domains",
}

CONFIDENCE_COLOURS = {
    "Very High": ("badge badge-vh", "#3fb950"),
    "High":      ("badge badge-h",  "#58a6ff"),
    "Medium":    ("badge badge-m",  "#e3b341"),
    "Low":       ("badge badge-l",  "#f85149"),
}

def confidence_badge(level: str) -> str:
    cls, _ = CONFIDENCE_COLOURS.get(level, ("badge badge-l", "#f85149"))
    return f'<span class="{cls}">{level}</span>'

def fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def build_training_data(prioritizer, drug_df, gene_df):
    """Build feature matrix + labels from drug/gene DataFrames."""
    drug_feats = prioritizer.create_drug_features(drug_df)
    gene_feats = prioritizer.create_gene_features(gene_df)
    X, y = [], []
    for i in range(len(drug_df)):
        for j in range(len(gene_df)):
            df_ = drug_feats[i].reshape(1, -1)
            gf_ = gene_feats[j].reshape(1, -1)
            combined = prioritizer.combine_features(df_, gf_)
            X.append(combined[0])
            ds = int(drug_df.iloc[i].get('logP', 0) > 2) + \
                 int(drug_df.iloc[i].get('h_bond_acceptors', 0) > 3)
            gs = int(gene_df.iloc[j].get('binding_domains', 0) > 2) + \
                 int(gene_df.iloc[j].get('expression_level', 0) > 5)
            prob = 0.6 if (ds >= 1 and gs >= 1) else 0.2
            y.append(1 if np.random.random() < prob else 0)
    return np.array(X), np.array(y)

def set_dark_style(fig, *axes):
    fig.patch.set_facecolor('#161b22')
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 DGI Prioritizer")
    st.markdown("<div class='sidebar-section'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio(
        "Go to",
        ["🏠 Home", "📂 Data Input", "🎓 Train Models",
         "🔬 Run Predictions", "📊 Results & Plots", "💾 Model I/O"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-section'>Quick Status</div>", unsafe_allow_html=True)

    drug_ok  = st.session_state.drug_data  is not None
    gene_ok  = st.session_state.gene_data  is not None
    train_ok = st.session_state.model_trained
    pred_ok  = st.session_state.prioritized is not None

    for label, ok in [("Drug data loaded",  drug_ok),
                      ("Gene data loaded",   gene_ok),
                      ("Models trained",     train_ok),
                      ("Predictions ready",  pred_ok)]:
        icon  = "✅" if ok else "⬜"
        colour= "#3fb950" if ok else "#8b949e"
        st.markdown(f"<span style='color:{colour};font-size:.85rem'>{icon} {label}</span>",
                    unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Settings</div>", unsafe_allow_html=True)
    st.session_state["threshold"] = st.slider(
        "Interaction threshold", 0.3, 0.95, 0.5, 0.05,
        help="Minimum probability for a pair to be included in results")
    st.session_state["top_k"] = st.number_input(
        "Top-K results (0 = all)", 0, 500, 20, 5)
    st.session_state["random_state"] = st.number_input(
        "Random seed", 0, 9999, 42, 1)

    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · pandas")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🧬 Drug-Gene Interaction Prioritizer</h1>
        <p>Ensemble AI/ML tool for predicting and ranking drug–gene interactions
           using Random Forest, Gradient Boosting and Neural Networks.</p>
    </div>
    """, unsafe_allow_html=True)

    # Workflow pills
    st.markdown("#### How to use this tool")
    cols = st.columns(5)
    steps = [
        ("1", "📂 Load Data",      "Upload CSVs or use synthetic demo data"),
        ("2", "🎓 Train Models",   "Fit the 3-model ensemble on your data"),
        ("3", "🔬 Predict",        "Score every drug–gene pair"),
        ("4", "📊 Analyse",        "Explore results, charts and importances"),
        ("5", "💾 Export",         "Download CSV or save the trained model"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='section-card' style='text-align:center;'>
                <div style='font-size:2rem;'>{title.split()[0]}</div>
                <div style='color:#e6edf3;font-weight:600;margin:.4rem 0 .2rem;'>{title[2:]}</div>
                <div style='color:#8b949e;font-size:.8rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### What the models learn")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='section-card'>
            <div class='section-title'>🌲 Random Forest</div>
            <ul style='color:#8b949e;font-size:.85rem;margin:0;padding-left:1.2rem;'>
                <li>200 decision trees in parallel</li>
                <li>Provides feature importance</li>
                <li>Robust to outliers</li>
                <li>Handles missing data well</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='section-card'>
            <div class='section-title'>📈 Gradient Boosting</div>
            <ul style='color:#8b949e;font-size:.85rem;margin:0;padding-left:1.2rem;'>
                <li>150 sequential trees</li>
                <li>Each corrects previous errors</li>
                <li>High predictive accuracy</li>
                <li>Handles class imbalance</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='section-card'>
            <div class='section-title'>🧠 Neural Network</div>
            <ul style='color:#8b949e;font-size:.85rem;margin:0;padding-left:1.2rem;'>
                <li>128 → 64 → 32 architecture</li>
                <li>ReLU activation + Adam</li>
                <li>Learns non-linear patterns</li>
                <li>Early stopping prevents overfit</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Required CSV columns")
    dc, gc = st.columns(2)
    with dc:
        st.markdown("**Drug CSV**")
        st.dataframe(pd.DataFrame({
            "Column":      ["drug_id","drug_name","molecular_weight","logP",
                            "h_bond_donors","h_bond_acceptors","rotatable_bonds",
                            "polar_surface_area","aromatic_rings"],
            "Type":        ["str","str","float","float","int","int","int","float","int"],
            "Description": ["Unique ID","Name","Da","Lipophilicity","H-bond donors",
                            "H-bond acceptors","Flexibility","Polarity","Ring count"]
        }), width='stretch', hide_index=True)
    with gc:
        st.markdown("**Gene CSV**")
        st.dataframe(pd.DataFrame({
            "Column":      ["gene_id","gene_name","expression_level","protein_length",
                            "binding_domains","phosphorylation_sites",
                            "cellular_location_score","pathway_connectivity"],
            "Type":        ["str","str","float","int","int","int","float","int"],
            "Description": ["Unique ID","Symbol","FPKM/TPM","Amino-acids","Binding sites",
                            "Phospho sites","0–1 score","Pathway count"]
        }), width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Data Input":
    st.markdown("<div class='main-header'><h1>📂 Data Input</h1><p>Upload your CSV files or generate synthetic demo data.</p></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["⬆️  Upload CSVs", "🔬 Synthetic Demo Data"])

    # ── Upload tab ──────────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Drug Data CSV")
            drug_file = st.file_uploader("Upload drug CSV", type=["csv"], key="drug_upload")
            if drug_file:
                try:
                    df = pd.read_csv(drug_file)
                    st.session_state.drug_data = df
                    st.success(f"✅ Loaded {len(df)} drugs with {len(df.columns)} columns")
                    st.dataframe(df.head(8), width='stretch')
                    missing = [c for c in ["molecular_weight","logP","h_bond_donors",
                                           "h_bond_acceptors","rotatable_bonds",
                                           "polar_surface_area","aromatic_rings"]
                               if c not in df.columns]
                    if missing:
                        st.warning(f"⚠️ Missing columns: {', '.join(missing)} — defaults (0) will be used.")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        with c2:
            st.markdown("#### Gene Data CSV")
            gene_file = st.file_uploader("Upload gene CSV", type=["csv"], key="gene_upload")
            if gene_file:
                try:
                    df = pd.read_csv(gene_file)
                    st.session_state.gene_data = df
                    st.success(f"✅ Loaded {len(df)} genes with {len(df.columns)} columns")
                    st.dataframe(df.head(8), width='stretch')
                    missing = [c for c in ["expression_level","protein_length","binding_domains",
                                           "phosphorylation_sites","cellular_location_score",
                                           "pathway_connectivity"]
                               if c not in df.columns]
                    if missing:
                        st.warning(f"⚠️ Missing columns: {', '.join(missing)} — defaults (0) will be used.")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    # ── Synthetic tab ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("Generate realistic synthetic data to explore the tool without your own dataset.")
        c1, c2, c3 = st.columns(3)
        n_drugs  = c1.slider("Number of drugs",  10, 300, 80, 10)
        n_genes  = c2.slider("Number of genes",  5,  150, 40, 5)
        int_rate = c3.slider("Interaction rate", 0.1, 0.6, 0.3, 0.05)

        if st.button("🔬 Generate Synthetic Data"):
            with st.spinner("Generating…"):
                drug_df, gene_df, _ = generate_synthetic_data(n_drugs, n_genes, int_rate)
            st.session_state.drug_data = drug_df
            st.session_state.gene_data = gene_df
            st.success(f"✅ Generated {n_drugs} drugs × {n_genes} genes  ({n_drugs*n_genes:,} pairs)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Drug data preview**")
                st.dataframe(drug_df.head(6), width='stretch')
            with c2:
                st.markdown("**Gene data preview**")
                st.dataframe(gene_df.head(6), width='stretch')

    # ── Download sample templates ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Download sample CSV templates")
    tc1, tc2 = st.columns(2)
    sample_drug = pd.DataFrame({
        "drug_id": ["DRUG_0001","DRUG_0002"],
        "drug_name": ["Aspirin","Ibuprofen"],
        "molecular_weight": [180.2, 206.3],
        "logP": [1.19, 3.97],
        "h_bond_donors": [1, 1],
        "h_bond_acceptors": [4, 2],
        "rotatable_bonds": [3, 4],
        "polar_surface_area": [63.6, 37.3],
        "aromatic_rings": [1, 1]
    })
    sample_gene = pd.DataFrame({
        "gene_id": ["GENE_0001","GENE_0002"],
        "gene_name": ["EGFR","TP53"],
        "expression_level": [12.5, 8.3],
        "protein_length": [1210, 393],
        "binding_domains": [5, 2],
        "phosphorylation_sites": [15, 8],
        "cellular_location_score": [0.9, 0.8],
        "pathway_connectivity": [20, 25]
    })
    tc1.download_button("⬇️ Drug template CSV",
                        sample_drug.to_csv(index=False),
                        "drug_template.csv", "text/csv")
    tc2.download_button("⬇️ Gene template CSV",
                        sample_gene.to_csv(index=False),
                        "gene_template.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎓 Train Models":
    st.markdown("<div class='main-header'><h1>🎓 Train Models</h1><p>Build the ensemble and evaluate performance.</p></div>", unsafe_allow_html=True)

    if st.session_state.drug_data is None or st.session_state.gene_data is None:
        st.warning("⚠️ Please load drug and gene data first (Data Input page).")
        st.stop()

    drug_df = st.session_state.drug_data
    gene_df = st.session_state.gene_data

    st.info(f"📊 Dataset: **{len(drug_df)}** drugs × **{len(gene_df)}** genes = **{len(drug_df)*len(gene_df):,}** pairs")

    c1, c2 = st.columns(2)
    val_split = c1.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
    random_st = c2.number_input("Random seed", 0, 9999,
                                int(st.session_state.get("random_state", 42)), 1)

    if st.button("🚀 Train Ensemble Models"):
        prog  = st.progress(0, text="Initialising…")
        logs  = st.empty()

        # ── build prioritizer
        prioritizer = DrugGeneInteractionPrioritizer(random_state=int(random_st))
        prog.progress(5, "Building feature matrices…")
        log_lines = []

        try:
            X, y = build_training_data(prioritizer, drug_df, gene_df)
            log_lines.append(f"✅ Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
            log_lines.append(f"✅ Positive pairs: {y.sum():,} ({y.mean()*100:.1f}%)")
            logs.code("\n".join(log_lines))
            prog.progress(15, "Training Random Forest…")

            # ── train with per-model progress
            import io as _io, sys
            from contextlib import redirect_stdout

            model_progress = {"random_forest": 40, "gradient_boosting": 70, "neural_network": 90}
            results = {}
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import StandardScaler

            scaler = prioritizer.scaler
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=int(random_st))
            X_tr_s = scaler.fit_transform(X_train)
            X_va_s = scaler.transform(X_val)
            prioritizer.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            for name, model in prioritizer.models.items():
                prog.progress(model_progress[name] - 20, f"Training {name.replace('_',' ').title()}…")
                model.fit(X_tr_s, y_train)
                tr  = model.score(X_tr_s, y_train)
                va  = model.score(X_va_s, y_val)
                cv  = cross_val_score(model, X_tr_s, y_train, cv=5)
                auc = roc_auc_score(y_val, model.predict_proba(X_va_s)[:,1])
                results[name] = {
                    "train_accuracy": tr, "val_accuracy": va,
                    "cv_mean": cv.mean(), "cv_std": cv.std(), "auc": auc
                }
                log_lines.append(
                    f"  {name:20s}  train={tr:.3f}  val={va:.3f}  AUC={auc:.3f}  CV={cv.mean():.3f}±{cv.std():.3f}")
                logs.code("\n".join(log_lines))
                prog.progress(model_progress[name], f"{name.replace('_',' ').title()} done ✅")

            prioritizer.feature_importance = pd.DataFrame({
                "feature":    prioritizer.feature_names,
                "importance": prioritizer.models["random_forest"].feature_importances_
            }).sort_values("importance", ascending=False)
            prioritizer.is_trained = True

            # ── save to session
            st.session_state.prioritizer   = prioritizer
            st.session_state.train_results = results
            st.session_state.model_trained = True
            st.session_state.X_train       = X
            st.session_state.y_train       = y

            prog.progress(100, "Training complete ✅")
            st.success("🎉 All models trained successfully!")
            log_lines.append("\n✅ Training complete!")
            logs.code("\n".join(log_lines))

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # ── show metrics if trained ──────────────────────────────────────────────
    if st.session_state.model_trained and st.session_state.train_results:
        results = st.session_state.train_results
        st.markdown("### 📈 Model Performance")

        names = list(results.keys())
        mc = st.columns(len(names))
        for col, name in zip(mc, names):
            r = results[name]
            with col:
                st.markdown(f"""
                <div class='section-card' style='text-align:center;'>
                    <div style='color:#21d4fd;font-weight:700;font-size:.95rem;margin-bottom:.8rem;'>
                        {name.replace("_"," ").title()}
                    </div>
                    <div class='val' style='font-size:1.6rem;color:#3fb950;'>{r['auc']:.3f}</div>
                    <div class='lbl'>AUC-ROC</div>
                    <hr style='border-color:#30363d;margin:.6rem 0;'>
                    <div style='font-size:.82rem;color:#8b949e;'>
                        Train: <b style='color:#e6edf3;'>{r['train_accuracy']:.3f}</b><br>
                        Val:   <b style='color:#e6edf3;'>{r['val_accuracy']:.3f}</b><br>
                        CV:    <b style='color:#e6edf3;'>{r['cv_mean']:.3f}±{r['cv_std']:.3f}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── ROC curves ──────────────────────────────────────────────────────
        p = st.session_state.prioritizer
        if st.session_state.X_train is not None:
            X, y = st.session_state.X_train, st.session_state.y_train
            from sklearn.model_selection import train_test_split
            _, X_val, _, y_val = train_test_split(
                X, y, test_size=0.2, random_state=int(st.session_state.get("random_state", 42)))
            X_va_s = p.scaler.transform(X_val)

            fig, ax = plt.subplots(figsize=(7, 5))
            set_dark_style(fig, ax)
            colours = ["#21d4fd", "#3fb950", "#e3b341"]
            for (name, model), colour in zip(p.models.items(), colours):
                proba = model.predict_proba(X_va_s)[:, 1]
                fpr, tpr, _ = roc_curve(y_val, proba)
                auc = roc_auc_score(y_val, proba)
                ax.plot(fpr, tpr, label=f"{name.replace('_',' ').title()} (AUC={auc:.3f})",
                        color=colour, linewidth=2)
            ax.plot([0,1],[0,1],"--", color="#30363d", linewidth=1.5, label="Random")
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves – Ensemble Models")
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
            ax.grid(alpha=.15, color="#30363d")
            fig.tight_layout()
            st.pyplot(fig)
            st.download_button("⬇️ Download ROC plot",
                               fig_to_buf(fig), "roc_curves.png", "image/png")
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RUN PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Run Predictions":
    st.markdown("<div class='main-header'><h1>🔬 Run Predictions</h1><p>Score all drug–gene pairs and apply filters.</p></div>", unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first (Train Models page).")
        st.stop()

    p        = st.session_state.prioritizer
    drug_df  = st.session_state.drug_data
    gene_df  = st.session_state.gene_data
    threshold = float(st.session_state.get("threshold", 0.5))
    top_k     = int(st.session_state.get("top_k", 20))

    total_pairs = len(drug_df) * len(gene_df)
    st.info(f"📊 Will evaluate **{total_pairs:,}** drug–gene pairs  |  threshold: **{threshold}**  |  top-K: **{top_k if top_k > 0 else 'all'}**")

    if total_pairs > 5000:
        st.warning(f"⚠️ {total_pairs:,} pairs may take a while. Consider using smaller subsets for faster results.")

    col1, col2 = st.columns(2)
    with col1:
        use_subset = st.checkbox("Use subset of data (faster)", value=total_pairs > 2000)
    if use_subset:
        with col2:
            max_drugs = st.slider("Max drugs to use", 5, min(len(drug_df), 100),
                                  min(len(drug_df), 30))
            max_genes = st.slider("Max genes to use", 5, min(len(gene_df), 100),
                                  min(len(gene_df), 20))
        drug_sub = drug_df.head(max_drugs).reset_index(drop=True)
        gene_sub = gene_df.head(max_genes).reset_index(drop=True)
    else:
        drug_sub = drug_df.reset_index(drop=True)
        gene_sub = gene_df.reset_index(drop=True)

    if st.button("⚡ Run Predictions"):
        prog_bar = st.progress(0, "Starting…")
        status   = st.empty()
        results  = []
        d_feats  = p.create_drug_features(drug_sub)
        g_feats  = p.create_gene_features(gene_sub)
        total    = len(drug_sub) * len(gene_sub)
        done     = 0

        for i, drug_row in drug_sub.iterrows():
            for j, gene_row in gene_sub.iterrows():
                df_ = d_feats[i].reshape(1, -1)
                gf_ = g_feats[j].reshape(1, -1)
                combined = p.combine_features(df_, gf_)
                prob = p.predict_interaction(combined, return_proba=True)[0]
                if prob >= threshold:
                    results.append({
                        "drug_id":         drug_row.get("drug_id",   f"drug_{i}"),
                        "drug_name":       drug_row.get("drug_name", f"Drug_{i}"),
                        "gene_id":         gene_row.get("gene_id",   f"gene_{j}"),
                        "gene_name":       gene_row.get("gene_name", f"Gene_{j}"),
                        "interaction_score": round(float(prob), 4),
                        "confidence":      p._calculate_confidence(prob),
                        "priority_rank":   0
                    })
                done += 1
                if done % max(1, total // 40) == 0 or done == total:
                    pct = int(done / total * 100)
                    prog_bar.progress(pct, f"Evaluating pairs… {done:,}/{total:,}")

        if results:
            res_df = pd.DataFrame(results).sort_values(
                "interaction_score", ascending=False).reset_index(drop=True)
            res_df["priority_rank"] = range(1, len(res_df) + 1)
            if top_k > 0:
                res_df = res_df.head(top_k)
            st.session_state.prioritized = res_df
            prog_bar.progress(100, "Done ✅")
            st.success(f"🎉 Found **{len(res_df)}** interactions above threshold {threshold}!")
        else:
            st.error(f"No interactions found above threshold {threshold}. Try lowering it in the sidebar.")

    # Quick preview
    if st.session_state.prioritized is not None:
        res = st.session_state.prioritized
        st.markdown("### 🔍 Preview (top 10)")

        conf_col = res["confidence"].map({
            "Very High": "🟢", "High": "🔵", "Medium": "🟡", "Low": "🔴"
        }).fillna("⚪")
        preview = res.head(10).copy()
        preview.insert(0, " ", conf_col.head(10))
        st.dataframe(preview, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESULTS & PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results & Plots":
    st.markdown("<div class='main-header'><h1>📊 Results & Visualisations</h1><p>Explore prioritised interactions and model insights.</p></div>", unsafe_allow_html=True)

    if st.session_state.prioritized is None:
        st.warning("⚠️ No predictions yet — go to Run Predictions first.")
        st.stop()

    res = st.session_state.prioritized.copy()
    p   = st.session_state.prioritizer

    # ── summary metrics ──────────────────────────────────────────────────────
    mc = st.columns(4)
    conf_counts = res["confidence"].value_counts()
    mc[0].markdown(f"<div class='metric-card'><div class='val'>{len(res)}</div><div class='lbl'>Total Interactions</div></div>", unsafe_allow_html=True)
    mc[1].markdown(f"<div class='metric-card'><div class='val'>{res['interaction_score'].mean():.3f}</div><div class='lbl'>Mean Score</div></div>", unsafe_allow_html=True)
    mc[2].markdown(f"<div class='metric-card'><div class='val'>{conf_counts.get('Very High',0) + conf_counts.get('High',0)}</div><div class='lbl'>High+ Confidence</div></div>", unsafe_allow_html=True)
    mc[3].markdown(f"<div class='metric-card'><div class='val'>{res['drug_name'].nunique()}</div><div class='lbl'>Unique Drugs</div></div>", unsafe_allow_html=True)

    tab_res, tab_fi, tab_hm, tab_dist = st.tabs([
        "📋 Results Table", "🔑 Feature Importance", "🗺️ Heatmap", "📉 Score Distribution"])

    # ── tab: Results Table ───────────────────────────────────────────────────
    with tab_res:
        # filters
        fc1, fc2, fc3 = st.columns(3)
        drug_filter = fc1.multiselect("Filter drugs", sorted(res["drug_name"].unique()),
                                      default=None, placeholder="All drugs")
        gene_filter = fc2.multiselect("Filter genes", sorted(res["gene_name"].unique()),
                                      default=None, placeholder="All genes")
        conf_filter = fc3.multiselect("Confidence", ["Very High","High","Medium","Low"],
                                      default=None, placeholder="All levels")

        filtered = res.copy()
        if drug_filter: filtered = filtered[filtered["drug_name"].isin(drug_filter)]
        if gene_filter: filtered = filtered[filtered["gene_name"].isin(gene_filter)]
        if conf_filter: filtered = filtered[filtered["confidence"].isin(conf_filter)]

        st.markdown(f"Showing **{len(filtered)}** of **{len(res)}** interactions")
        st.dataframe(
            filtered.style
              .background_gradient(subset=["interaction_score"], cmap="YlOrRd")
              .format({"interaction_score": "{:.4f}"}),
            width='stretch', hide_index=True, height=420)

        dl_csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Download filtered CSV", dl_csv,
                           "prioritized_interactions.csv", "text/csv")

    # ── tab: Feature Importance ──────────────────────────────────────────────
    with tab_fi:
        if p.feature_importance is not None:
            top_n  = st.slider("Number of features to show", 5, 19, 15)
            fi_df  = p.feature_importance.head(top_n).copy()
            fi_df["label"] = fi_df["feature"].map(FEATURE_LABELS).fillna(fi_df["feature"])

            fig, ax = plt.subplots(figsize=(9, top_n * 0.42 + 1))
            set_dark_style(fig, ax)
            colours_bar = ["#21d4fd" if "Interaction" in lbl
                           else "#3fb950" if "Gene" in lbl
                           else "#e3b341"
                           for lbl in fi_df["label"]]
            ax.barh(fi_df["label"], fi_df["importance"], color=colours_bar, edgecolor="none")
            ax.set_xlabel("Importance Score")
            ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.15, color="#30363d")
            patches = [mpatches.Patch(color="#e3b341", label="Drug feature"),
                       mpatches.Patch(color="#3fb950", label="Gene feature"),
                       mpatches.Patch(color="#21d4fd", label="Interaction term")]
            ax.legend(handles=patches, facecolor="#161b22",
                      edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            st.download_button("⬇️ Download plot",
                               fig_to_buf(fig), "feature_importance.png", "image/png")
            plt.close(fig)
        else:
            st.info("Train the model to see feature importances.")

    # ── tab: Heatmap ─────────────────────────────────────────────────────────
    with tab_hm:
        st.markdown("Interaction score heatmap — darker = stronger predicted interaction")
        top_drugs = res.groupby("drug_name")["interaction_score"].max().nlargest(20).index.tolist()
        top_genes = res.groupby("gene_name")["interaction_score"].max().nlargest(15).index.tolist()

        hm_data = res[res["drug_name"].isin(top_drugs) & res["gene_name"].isin(top_genes)]
        if not hm_data.empty:
            pivot = hm_data.pivot_table(
                index="drug_name", columns="gene_name",
                values="interaction_score", aggfunc="max").fillna(0)

            fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*0.7),
                                            max(5, len(pivot)*0.4) + 1))
            set_dark_style(fig, ax)
            sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                        linewidths=0.3, linecolor="#0d1117",
                        cbar_kws={"shrink": 0.6},
                        annot=len(pivot) <= 15,
                        fmt=".2f" if len(pivot) <= 15 else "")
            ax.set_title("Drug–Gene Interaction Score Heatmap (top pairs)")
            ax.set_xlabel("Gene"); ax.set_ylabel("Drug")
            plt.xticks(rotation=45, ha="right", fontsize=8, color="#8b949e")
            plt.yticks(rotation=0, fontsize=8, color="#8b949e")
            ax.collections[0].colorbar.ax.tick_params(colors="#8b949e")
            fig.tight_layout()
            st.pyplot(fig)
            st.download_button("⬇️ Download heatmap",
                               fig_to_buf(fig), "heatmap.png", "image/png")
            plt.close(fig)
        else:
            st.info("Not enough data to generate heatmap.")

    # ── tab: Score Distribution ──────────────────────────────────────────────
    with tab_dist:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        set_dark_style(fig, *axes)

        # histogram
        axes[0].hist(res["interaction_score"], bins=30,
                     color="#21d4fd", edgecolor="#0d1117", alpha=0.85)
        axes[0].set_xlabel("Interaction Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Score Distribution")
        axes[0].grid(alpha=.15, color="#30363d")

        # confidence pie
        conf_order = ["Very High","High","Medium","Low"]
        conf_colours_pie = ["#3fb950","#58a6ff","#e3b341","#f85149"]
        counts = [conf_counts.get(c, 0) for c in conf_order]
        non_zero = [(c, v, col) for c, v, col in zip(conf_order, counts, conf_colours_pie) if v > 0]
        if non_zero:
            labels, vals, cols = zip(*non_zero)
            wedges, texts, autotexts = axes[1].pie(
                vals, labels=labels, colors=cols,
                autopct="%1.0f%%", startangle=90,
                textprops={"color": "#e6edf3", "fontsize": 9})
            for at in autotexts:
                at.set_color("#0d1117"); at.set_fontweight("bold")
        axes[1].set_title("Confidence Distribution")

        # top drugs bar
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        set_dark_style(fig2, ax2)
        top_d = res.groupby("drug_name")["interaction_score"].mean().nlargest(12)
        ax2.bar(top_d.index, top_d.values, color="#3fb950", edgecolor="none")
        ax2.set_xlabel("Drug"); ax2.set_ylabel("Mean Score")
        ax2.set_title("Top Drugs by Mean Interaction Score")
        plt.xticks(rotation=35, ha="right", fontsize=8, color="#8b949e")
        ax2.grid(axis="y", alpha=.15, color="#30363d")
        fig2.tight_layout()

        fig.tight_layout()
        st.pyplot(fig)
        st.pyplot(fig2)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button("⬇️ Distribution plot",
                                fig_to_buf(fig), "distribution.png", "image/png")
        col_dl2.download_button("⬇️ Top drugs plot",
                                fig_to_buf(fig2), "top_drugs.png", "image/png")
        plt.close(fig); plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL I/O
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💾 Model I/O":
    st.markdown("<div class='main-header'><h1>💾 Model Save / Load</h1><p>Persist your trained model or reload a previous session.</p></div>", unsafe_allow_html=True)

    tab_save, tab_load = st.tabs(["💾 Save Model", "📂 Load Model"])

    with tab_save:
        if not st.session_state.model_trained:
            st.warning("No trained model in session yet.")
        else:
            st.success("✅ A trained model is ready to save.")
            model_bytes = pickle.dumps(st.session_state.prioritizer)
            st.download_button(
                "⬇️ Download trained model (.pkl)",
                model_bytes, "dgi_model.pkl",
                "application/octet-stream"
            )
            if st.session_state.prioritized is not None:
                st.download_button(
                    "⬇️ Download results CSV",
                    st.session_state.prioritized.to_csv(index=False),
                    "prioritized_interactions.csv", "text/csv"
                )
            st.markdown("""
            <div class='section-card' style='margin-top:1rem;'>
                <div class='section-title'>📌 How to reload this model in Python</div>
                <pre style='color:#e6edf3;background:#0d1117;padding:1rem;border-radius:8px;font-size:.82rem;'>
import pickle
from main_script2 import DrugGeneInteractionPrioritizer

with open("dgi_model.pkl", "rb") as f:
    prioritizer = pickle.load(f)

# Now use it directly
results = prioritizer.prioritize_interactions(
    drug_df, gene_df, top_k=50, threshold=0.6
)
                </pre>
            </div>""", unsafe_allow_html=True)

    with tab_load:
        st.markdown("Upload a previously saved `.pkl` model file.")
        uploaded_model = st.file_uploader("Upload model (.pkl)", type=["pkl"])
        if uploaded_model:
            try:
                loaded = pickle.loads(uploaded_model.read())
                if isinstance(loaded, DrugGeneInteractionPrioritizer) and loaded.is_trained:
                    st.session_state.prioritizer   = loaded
                    st.session_state.model_trained = True
                    st.success("✅ Model loaded successfully! You can now go to Run Predictions.")
                    fi = loaded.feature_importance
                    if fi is not None:
                        st.markdown("**Top 5 loaded feature importances:**")
                        fi_show = fi.head(5).copy()
                        fi_show["label"] = fi_show["feature"].map(FEATURE_LABELS).fillna(fi_show["feature"])
                        st.dataframe(fi_show[["label","importance"]], width='stretch', hide_index=True)
                else:
                    st.error("File is not a valid trained DrugGeneInteractionPrioritizer object.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")


