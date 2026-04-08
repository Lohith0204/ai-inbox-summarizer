import io
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="AI Inbox Summarizer",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0A0F1E 0%, #0D1836 60%, #0A0F1E 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #080D1A; }

.hero-header { text-align: center; padding: 2.5rem 0 1rem; }
.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #4F8EF7 0%, #A78BFA 50%, #34D399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -1px; margin: 0;
}
.hero-sub { color: #94A3B8; font-size: 1.05rem; margin-top: 0.4rem; }

.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
    backdrop-filter: blur(10px); transition: border-color 0.2s;
}
.glass-card:hover { border-color: rgba(79,142,247,0.3); }

.metric-chip {
    display: inline-flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px; padding: 1.1rem 1.4rem; min-width: 130px; text-align: center;
}
.metric-chip .chip-label {
    font-size: 0.72rem; color: #64748B; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.metric-chip .chip-value { font-size: 1.5rem; font-weight: 700; margin-top: 0.3rem; }

.section-title {
    font-size: 0.78rem; font-weight: 700; color: #4F8EF7;
    text-transform: uppercase; letter-spacing: 0.12em; margin: 1.5rem 0 0.6rem;
}
.summary-text {
    color: #CBD5E1; font-size: 1rem; line-height: 1.7;
    padding: 1rem 1.2rem;
    background: rgba(79,142,247,0.06);
    border-left: 3px solid #4F8EF7;
    border-radius: 0 10px 10px 0; margin: 0.4rem 0;
}
.task-item {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.55rem 0.8rem; margin-bottom: 0.4rem;
    background: rgba(255,255,255,0.03); border-radius: 8px;
    font-size: 0.92rem; color: #CBD5E1;
}
.task-dot { color: #4F8EF7; font-size: 1.1rem; flex-shrink: 0; }

.urgency-bar-wrap {
    background: rgba(255,255,255,0.06); border-radius: 8px;
    height: 10px; width: 100%; overflow: hidden;
}
.urgency-bar-fill { height: 10px; border-radius: 8px; transition: width 0.5s ease; }

button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important; color: #94A3B8 !important;
    font-weight: 500 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    padding: 0.5rem 1.2rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(79,142,247,0.18) !important;
    color: #7AADFF !important;
    border-color: rgba(79,142,247,0.35) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4F8EF7, #7C3AED) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
textarea, input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #E6EDF3 !important; border-radius: 10px !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(79,142,247,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


def _urgency_bar(score, color):
    pct = int(score * 100)
    return (
        f'<div class="urgency-bar-wrap">'
        f'<div class="urgency-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
    )

def _card(html, style=""):
    return f'<div class="glass-card" style="{style}">{html}</div>'

def _plot_confusion_matrix(cm_data, labels, title):
    import numpy as _np
    z = _np.array(cm_data, dtype=float)
    z_pct = z / (_np.maximum(z.sum(axis=1, keepdims=True), 1))
    text = [
        [f"{int(z[i][j])}<br>{z_pct[i][j]*100:.0f}%" for j in range(len(labels))]
        for i in range(len(labels))
    ]
    fig = go.Figure(go.Heatmap(
        z=z_pct.tolist(),
        x=labels, y=labels,
        colorscale=[[0, "#0A0F1E"], [0.5, "#1E3A6E"], [1, "#4F8EF7"]],
        showscale=False,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#E6EDF3"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#94A3B8")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", size=11),
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="Predicted", title_font=dict(size=11)),
        yaxis=dict(title="Actual", title_font=dict(size=11), autorange="reversed"),
    )
    return fig


EXAMPLES = {
    "💼 Urgent work email": {
        "subject": "Action Required: Q3 Road-map Review — Thursday Deadline",
        "body": "Hi team, the Q3 road-map deliverables are due this Thursday by 5 PM. All engineering leads must submit their section updates to Jordan before the deadline. We have a board presentation on Friday morning — the deck must be finalised tonight. Please review the shared slides, address all open comments, and confirm your approval by reply. If you have blockers, escalate immediately. Action items: finalise estimates, update JIRA tickets, send your department summary to Alex.",
    },
    "💰 Finance invoice": {
        "subject": "Invoice #INV-2024-08921 Due in 3 Days",
        "body": "Dear Casey, please find attached Invoice #INV-2024-08921 for consulting services rendered in October 2024. Total amount due: $12,500. Payment must be received by November 15, 2024. Please transfer to Account #48291047, Routing #021000021. Reference the invoice number in your payment description. Late payments will incur a 2% monthly finance charge. Contact billing@company.com immediately if you have questions about this invoice.",
    },
    "⚡ System alert": {
        "subject": "CRITICAL: Payment Service is DOWN — Immediate Action Required",
        "body": "CRITICAL ALERT — Payment Service has been unreachable for 12 minutes. Error: Connection timed out. Last successful ping: 14:38 UTC. Affected region: us-east-1. Estimated user impact: 24,000 users. Auto-scaling triggered — issue persists. P0 incident declared. On-call engineer: please acknowledge within 5 minutes. Runbook: https://wiki.internal/runbooks/payment-service-outage Escalation path: L1 → L2 → Morgan. All hands required immediately.",
    },
    "🚨 Spam / phishing": {
        "subject": "Congratulations! You've been selected for a FREE iPhone 15 Pro",
        "body": "Dear valued customer, CONGRATULATIONS! You have been randomly selected to receive a free iPhone 15 Pro worth over $1,200! To claim your reward simply click the link below and enter your personal details. This offer expires in 24 hours — do not delay! Click here: http://totally-legit-prize.net/claim?id=38291 If you do not respond within 24 hours your prize will be forfeited permanently. Enter your name, credit card number and SSN to unlock your free gift.",
    },
    "📰 Newsletter": {
        "subject": "This Week in AI: Top 5 Stories You Need to Read",
        "body": "Welcome to this week's AI digest! Here are the top stories: 1. New open-source LLM outperforms GPT-4 on coding benchmarks — MIT publishes groundbreaking results. 2. New MLOps framework claims to cut deployment time by 40%. 3. AI startup funding hits a record $2.3B this quarter. 4. Top 10 machine learning tools of 2024. 5. Opinion: Why generative AI will reshape how we work. Read all articles on our website. Unsubscribe anytime below.",
    },
    "💬 Personal message": {
        "subject": "Hey, are we still on for Saturday?",
        "body": "Hey! Hope you're doing well. Just checking if we're still on for Saturday afternoon. I was thinking coffee around 3 PM at that new place on Main Street. Let me know if that works or if you need to reschedule — totally fine! Also, did you get to watch that documentary I mentioned? Would love to hear your thoughts. No rush, just text me!",
    },
}

def _models_exist():
    return os.path.exists(os.path.join("models", "classifier.pkl"))

def _run_training():
    try:
        sys.path.insert(0, os.path.abspath("."))
        from ml.trainer import run_training_pipeline
        run_training_pipeline(verbose=False)
        st.cache_resource.clear()
        return True
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        return False


st.markdown("""
<div class="hero-header">
  <p class="hero-title">📬 AI Inbox Summarizer</p>
  <p class="hero-sub">Intelligent email triage · ML classification · Summarisation · NER · SHAP explainability</p>
</div>
<hr style='border-color:rgba(255,255,255,0.07);margin:0 0 1.5rem'>
""", unsafe_allow_html=True)


if not _models_exist():
    st.markdown(_card(
        "<h3 style='color:#4F8EF7;margin:0 0 0.5rem'>🧠 First-run setup</h3>"
        "<p style='color:#94A3B8;margin:0'>No trained models detected. Click below to generate the synthetic dataset and train the ML classifiers (~30 seconds).</p>"
    ), unsafe_allow_html=True)

    if st.button("⚡ Train Models Now"):
        with st.spinner("Generating dataset & training models... (≈ 30 seconds)"):
            if _run_training():
                st.success("✅ Models ready! Reloading...")
                st.rerun()
    st.stop()


from ml.classifier import classify_with_confidence
from ml.priority import predict_priority_full
from ml.summarizer import summarize
from ml.ner import full_ner
from ml.explainer import explain_prediction


tab_analyze, tab_batch, tab_dashboard = st.tabs([
    "🔍  Analyze Email",
    "📂  Batch Analysis",
    "📊  Model Dashboard",
])


with tab_analyze:
    st.markdown('<p class="section-title">Load a Sample Email</p>', unsafe_allow_html=True)
    example_choice = st.selectbox(
        "Example", ["— select —"] + list(EXAMPLES.keys()),
        label_visibility="collapsed", key="eg_select"
    )

    if "s_subj" not in st.session_state:
        st.session_state.s_subj = ""
    if "s_body" not in st.session_state:
        st.session_state.s_body = ""

    if example_choice != "— select —":
        st.session_state.s_subj = EXAMPLES[example_choice]["subject"]
        st.session_state.s_body = EXAMPLES[example_choice]["body"]

    c1, c2 = st.columns([1, 2])
    with c1:
        subj = st.text_input("Subject", value=st.session_state.s_subj, placeholder="e.g. Action Required: Report Due Friday", key="t_subj")
    with c2:
        body = st.text_area("Email body", value=st.session_state.s_body, height=180, placeholder="Paste your email content here...", key="t_body")

    go_btn = st.button("⚡  Analyze Email", key="go_btn")

    if go_btn:
        full_text = f"{subj} {body}".strip()
        if len(full_text) < 10:
            st.warning("Please enter a subject or body before analysing.")
        else:
            with st.spinner("Running ML pipeline..."):
                cat_r = classify_with_confidence(full_text)
                pri_r = predict_priority_full(full_text)
                summ = summarize(body or full_text)
                ner_r = full_ner(body or full_text)
                shap_r = explain_prediction(full_text)

            st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:1rem 0'>", unsafe_allow_html=True)

            cat_meta = cat_r["meta"]
            pri_meta = pri_r["meta"]
            urgency = pri_r["urgency_score"]
            u_color = pri_meta.get("color", "#888")

            st.markdown(f"""
            <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.2rem">
              <div class="metric-chip">
                <span class="chip-label">Category</span>
                <span class="chip-value">{cat_meta['emoji']} {cat_r['prediction'].title()}</span>
                <span style="color:#64748B;font-size:.75rem;margin-top:4px">{cat_r['confidence']*100:.0f}% confidence</span>
              </div>
              <div class="metric-chip">
                <span class="chip-label">Priority</span>
                <span class="chip-value" style="color:{u_color}">{pri_meta.get('emoji','')} {pri_r['label'].title()}</span>
                <span style="color:#64748B;font-size:.75rem;margin-top:4px">{pri_r['confidence']*100:.0f}% confidence</span>
              </div>
              <div class="metric-chip" style="min-width:180px">
                <span class="chip-label">Urgency Score</span>
                <span class="chip-value" style="color:{u_color}">{urgency*100:.0f} / 100</span>
              </div>
            </div>
            <div style="margin-bottom:1.2rem">
              <span style="font-size:.75rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:.08em">Urgency</span>
              {_urgency_bar(urgency, u_color)}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📈 Category probability breakdown"):
                scores = cat_r["scores"]
                fig_b = go.Figure(go.Bar(
                    x=list(scores.values()), y=list(scores.keys()), orientation="h",
                    marker=dict(color=list(scores.values()), colorscale=[[0, "#1E293B"], [1, "#4F8EF7"]]),
                    text=[f"{v*100:.1f}%" for v in scores.values()],
                    textposition="outside", textfont=dict(color="#94A3B8"),
                ))
                fig_b.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"),
                    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.15]),
                    yaxis=dict(showgrid=False), height=260, margin=dict(l=10, r=80, t=10, b=10),
                )
                st.plotly_chart(fig_b, use_container_width=True)

            st.markdown('<p class="section-title">📝 Summary</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-text">{summ}</div>', unsafe_allow_html=True)

            st.markdown('<p class="section-title">✅ Action Items & Deadlines</p>', unsafe_allow_html=True)
            actions = ner_r.get("action_items", [])
            deadlines = ner_r.get("deadlines", [])

            if actions:
                st.markdown("".join(f'<div class="task-item"><span class="task-dot">›</span><span>{a}</span></div>' for a in actions), unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#64748B;font-size:.9rem">No action items detected.</p>', unsafe_allow_html=True)

            if deadlines:
                st.markdown('<p class="section-title" style="color:#F5D65B">⏰ Detected Deadlines</p>', unsafe_allow_html=True)
                for d in deadlines:
                    st.markdown(f'<div class="task-item"><span class="task-dot" style="color:#F5D65B">📅</span><span><strong style="color:#F5D65B">{d["date"]}</strong> — {d["context"]}</span></div>', unsafe_allow_html=True)

            with st.expander("🏷️ Named Entities"):
                entities = ner_r.get("entities", {})
                ecols = st.columns(3)
                for i, (etype, items) in enumerate(entities.items()):
                    if items:
                        with ecols[i % 3]:
                            st.markdown(f"**{etype.title()}**")
                            for it in items:
                                st.markdown(f"- {it}")

            st.markdown('<p class="section-title">🧠 Why This Prediction? (SHAP)</p>', unsafe_allow_html=True)

            if shap_r["status"] == "success" and shap_r["top_features"]:
                feats = shap_r["top_features"]
                words = [f["word"] for f in feats]
                shapvs = [f["shap"] for f in feats]
                colors = ["#4F8EF7" if v > 0 else "#FF4444" for v in shapvs]

                fig_s = go.Figure(go.Bar(
                    y=words, x=shapvs, orientation="h", marker_color=colors,
                    text=[f"{v:+.4f}" for v in shapvs], textposition="outside",
                    textfont=dict(color="#94A3B8", size=11),
                ))
                fig_s.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.15)"),
                    yaxis=dict(showgrid=False, autorange="reversed"), height=max(320, len(words) * 32),
                    margin=dict(l=10, r=110, t=20, b=10),
                )
                st.plotly_chart(fig_s, use_container_width=True)
                st.markdown('<p style="color:#64748B;font-size:.78rem">🔵 Blue = words pushing <em>toward</em> this category &nbsp;|&nbsp; 🔴 Red = words pushing <em>away</em></p>', unsafe_allow_html=True)
            elif shap_r["status"] == "no_features":
                st.info("Not enough distinctive words for a SHAP explanation.")
            else:
                st.warning(f"SHAP unavailable: {shap_r.get('message', 'unknown error')}")


with tab_batch:
    st.markdown('<p class="section-title">Upload a CSV for Batch Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B;font-size:.88rem">CSV must contain a <code>body</code> column. Optional: <code>subject</code>, <code>email_id</code>. Max 500 rows.</p>', unsafe_allow_html=True)

    sample_df = pd.DataFrame({
        "email_id": ["EML-001", "EML-002", "EML-003"],
        "subject":  ["Q3 deadline reminder", "Invoice #8821 due", "You won a prize!"],
        "body": [
            "Please submit Q3 deliverables to Jordan by Thursday 5 PM. Action required.",
            "Invoice #8821 for $5,400 is due November 10. Late fees apply after the deadline.",
            "Congratulations! You have been selected to receive a free iPhone. Click here now!",
        ],
    })
    st.download_button("⬇ Download Sample CSV", data=sample_df.to_csv(index=False).encode("utf-8"), file_name="sample_emails.csv", mime="text/csv", key="dl_sample")

    uploaded = st.file_uploader("CSV upload", type=["csv"], label_visibility="collapsed", key="batch_up")

    if uploaded:
        try:
            df_b = pd.read_csv(uploaded).head(500)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        if "body" not in df_b.columns:
            st.error("CSV must contain a `body` column.")
            st.stop()

        with st.spinner(f"Analysing {len(df_b)} emails..."):
            rows, prog = [], st.progress(0, text="Processing...")
            for i, row in df_b.iterrows():
                txt = f"{row.get('subject','')} {row.get('body','')}".strip()
                cat = classify_with_confidence(txt)
                pri = predict_priority_full(txt)
                summ = summarize(str(row.get("body", txt)), n_sentences=2)
                rows.append({
                    "email_id": row.get("email_id", f"EML-{i+1:04d}"),
                    "subject": row.get("subject", "—"),
                    "category": cat["prediction"],
                    "cat_conf": f"{cat['confidence']*100:.0f}%",
                    "priority": pri["label"],
                    "urgency": f"{pri['urgency_score']*100:.0f}",
                    "summary": summ,
                })
                prog.progress((i + 1) / len(df_b), text=f"Processing {i+1}/{len(df_b)}...")
            prog.empty()
            df_res = pd.DataFrame(rows)

        st.markdown(f"<p style='color:#5DDE9E;font-weight:600'>✅ Processed {len(df_res)} emails</p>", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            cc = df_res["category"].value_counts().reset_index()
            cc.columns = ["category", "count"]
            fig_pie = px.pie(cc, values="count", names="category", hole=0.55, color_discrete_sequence=["#4F8EF7","#FF4444","#9B59B6","#2ECC71","#F1C40F","#E74C3C"])
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), height=280, margin=dict(l=0,r=0,t=30,b=0), title=dict(text="Category distribution", font=dict(size=13,color="#94A3B8")))
            st.plotly_chart(fig_pie, use_container_width=True)
        with ch2:
            pc = df_res["priority"].value_counts().reset_index()
            pc.columns = ["priority", "count"]
            fig_pr = px.bar(pc, x="priority", y="count", color="priority", color_discrete_map={"high":"#FF4444","medium":"#F1C40F","low":"#2ECC71"})
            fig_pr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"), showlegend=False, height=280, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0,r=0,t=30,b=0), title=dict(text="Priority breakdown", font=dict(size=13,color="#94A3B8")))
            st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown('<p class="section-title">Results Table</p>', unsafe_allow_html=True)
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Results CSV", data=df_res.to_csv(index=False).encode("utf-8"), file_name="inbox_results.csv", mime="text/csv", key="dl_res")


with tab_dashboard:
    mpath = os.path.join("models", "metrics.json")
    if not os.path.exists(mpath):
        st.info("No metrics found yet — train the models first.")
        st.stop()

    with open(mpath) as f:
        mtr = json.load(f)

    clf_m = mtr["classifier"]
    pri_m = mtr["priority_model"]

    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    def _kpi(col, label, val, hint=""):
        col.markdown(f'<div class="metric-chip"><span class="chip-label">{label}</span><span class="chip-value">{val}</span>' + (f'<span style="color:#64748B;font-size:.72rem;margin-top:3px">{hint}</span>' if hint else "") + "</div>", unsafe_allow_html=True)

    _kpi(k1, "Cat. Accuracy", f"{clf_m['accuracy']*100:.1f}%")
    _kpi(k2, "Cat. Macro F1", f"{clf_m['macro_f1']*100:.1f}%")
    _kpi(k3, "5-Fold CV F1", f"{clf_m['cv_f1_mean']*100:.1f}%", f"±{clf_m['cv_f1_std']*100:.1f}%")
    _kpi(k4, "Pri. Accuracy", f"{pri_m['accuracy']*100:.1f}%")
    _kpi(k5, "Pri. Macro F1", f"{pri_m['macro_f1']*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.plotly_chart(_plot_confusion_matrix(clf_m["confusion_matrix"], clf_m["confusion_matrix_labels"], "Category Classifier — Confusion Matrix"), use_container_width=True)
    with cm2:
        st.plotly_chart(_plot_confusion_matrix(pri_m["confusion_matrix"], pri_m["confusion_matrix_labels"], "Priority Predictor — Confusion Matrix"), use_container_width=True)

    st.markdown('<p class="section-title">Per-Class F1 Scores</p>', unsafe_allow_html=True)
    f1c1, f1c2 = st.columns(2)

    def _f1_chart(col, per_class, title):
        labels = list(per_class.keys())
        f1s = [v["f1"] for v in per_class.values()]
        fig = go.Figure(go.Bar(
            x=labels, y=f1s, marker=dict(color=f1s, colorscale=[[0,"#FF4444"],[0.6,"#F1C40F"],[1,"#2ECC71"]]),
            text=[f"{v*100:.1f}%" for v in f1s], textposition="outside", textfont=dict(color="#94A3B8"),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color="#94A3B8")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"),
            yaxis=dict(range=[0,1.15], showgrid=True, gridcolor="rgba(255,255,255,0.05)"), xaxis=dict(showgrid=False), height=280, margin=dict(l=10,r=10,t=40,b=10), showlegend=False,
        )
        col.plotly_chart(fig, use_container_width=True)

    _f1_chart(f1c1, clf_m["per_class"], "Category Classifier — per-class F1")
    _f1_chart(f1c2, pri_m["per_class"], "Priority Predictor — per-class F1")

    st.markdown('<p class="section-title">Training Metadata</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card" style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem"><div><div style="color:#64748B;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em">Dataset</div><div style="color:#E6EDF3;font-weight:600;font-size:1rem;margin-top:4px">{mtr["dataset_size"]:,} emails</div><div style="color:#64748B;font-size:.8rem">{mtr["train_size"]:,} train / {mtr["test_size"]:,} test</div></div><div><div style="color:#64748B;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em">Vectorizer</div><div style="color:#E6EDF3;font-weight:600;font-size:1rem;margin-top:4px">TF-IDF · {mtr["vectorizer"]["vocab_size"]:,} features</div><div style="color:#64748B;font-size:.8rem">ngram (1,2) · sublinear TF</div></div><div><div style="color:#64748B;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em">Trained At</div><div style="color:#E6EDF3;font-weight:600;font-size:1rem;margin-top:4px">{mtr["trained_at"][:10]}</div><div style="color:#64748B;font-size:.8rem">{mtr["trained_at"][11:19]}</div></div></div>', unsafe_allow_html=True)

    experiment_log_path = os.path.join("logs", "experiments.csv")
    if os.path.exists(experiment_log_path):
        st.markdown('<p class="section-title">Experiment Logs (MLFlow Style)</p>', unsafe_allow_html=True)
        try:
            exp_df = pd.read_csv(experiment_log_path)
            exp_df["Timestamp"] = pd.to_datetime(exp_df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(exp_df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
        except Exception:
            pass

    with st.expander("☁️ Most Informative TF-IDF Features"):
        try:
            import pickle as _pkl
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            with open("models/vectorizer.pkl", "rb") as f:
                _vec = _pkl.load(f)

            freq = {w: float(idf) for w, idf in zip(_vec.get_feature_names_out(), _vec.idf_)}
            freq = {w: v for w, v in freq.items() if len(w) > 3}

            wc = WordCloud(
                width=900, height=340, background_color=None, mode="RGBA",
                colormap="Blues", max_words=120, prefer_horizontal=0.85, collocations=False,
            ).generate_from_frequencies(freq)

            fig_wc, ax = plt.subplots(figsize=(10, 3.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_alpha(0)
            st.pyplot(fig_wc)
        except Exception as e:
            st.info(f"Word cloud unavailable: {e}")