import os
import time
import tempfile
import html
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from agents import run_analysis, run_mock_analysis, get_market_scores
from pdf_generator import generate_pdf_from_markdown

load_dotenv()

st.set_page_config(
    page_title="AI Market Analyst Team",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════
MAX_TOPIC_LENGTH = 100
MAX_HISTORY_ITEMS = 20
VALID_DEPTHS = ("Quick", "Deep Dive")
RESET_KEYS = (
    "report_markdown", "orchestrator_mode", "topic_analyzed",
    "market_scores", "agent_steps", "loaded_from_history",
)
RESET_DEFAULTS = {
    "agent_steps": [],
    "market_scores": {},
}


def _reset_session_state(keys=RESET_KEYS, defaults=RESET_DEFAULTS):
    """Reset a set of session-state keys to their default values."""
    for k in keys:
        st.session_state[k] = defaults.get(k, "")


# ═══════════════════════════════════════════════
#  LOAD EXTERNAL CSS
# ═══════════════════════════════════════════════
def _load_css() -> str:
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


st.markdown(f"<style>{_load_css()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="orbs">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════
defaults = {
    "report_markdown": "",
    "orchestrator_mode": "",
    "topic_analyzed": "",
    "market_scores": {},
    "agent_steps": [],
    "history": [],
    "loaded_from_history": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
SUGGESTIONS = ["Supabase", "Vercel", "Figma", "Notion", "Linear", "Loom", "Stripe", "Retool"]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem;">
        <div style="font-size:2.6rem;margin-bottom:0.5rem;">🧠</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.15rem;font-weight:800;
                    background:linear-gradient(135deg,#6366f1,#a855f7);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            AI Market Analyst
        </div>
        <div style="color:#94a3b8;font-size:0.73rem;margin-top:4px;">
            Multi-Agent Intelligence Suite
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔑 API Keys")
    st.caption("Stored in process memory only — not persisted. "
               "For long-term use, set them in a `.env` file (see `.env.example`).")
    gemini_key = st.text_input("Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY",""), type="password",
        help="Google AI Studio — free tier available")
    tavily_key = st.text_input("Tavily API Key",
        value=os.environ.get("TAVILY_API_KEY",""), type="password",
        help="Tavily dashboard — free tier available")
    if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key
    if tavily_key: os.environ["TAVILY_API_KEY"] = tavily_key

    # Surface key status without ever echoing the secret back to the UI
    gemini_present = bool(os.environ.get("GEMINI_API_KEY"))
    tavily_present = bool(os.environ.get("TAVILY_API_KEY"))
    if gemini_present and tavily_present:
        st.markdown(
            "<div style='color:#10b981;font-size:0.78rem;'>"
            "✓ Both API keys detected</div>",
            unsafe_allow_html=True)
    else:
        missing = []
        if not gemini_present: missing.append("Gemini")
        if not tavily_present: missing.append("Tavily")
        st.markdown(
            f"<div style='color:#f59e0b;font-size:0.78rem;'>"
            f"⚠ Missing: {', '.join(missing)} — enable Demo Mode to run without keys.</div>",
            unsafe_allow_html=True)

    st.markdown("### ⚙️ Research Settings")
    research_depth = st.select_slider("Depth",
        options=list(VALID_DEPTHS), value="Quick",
        help="Quick=Fast Gemini · Deep Dive=Smarter Gemini")

    with st.expander("🔧 Model override (advanced)"):
        st.caption("Leave blank to use the auto-detected model. Override with any "
                   "model name your Gemini key has access to. If the chosen model "
                   "is unavailable, the app will automatically try a fallback.")
        fast_override = st.text_input("Quick-tier model", value="",
            placeholder="e.g. gemini-2.0-flash")
        pro_override = st.text_input("Deep Dive model", value="",
            placeholder="e.g. gemini-2.5-pro")
        if fast_override.strip():
            os.environ["GEMINI_MODEL_FAST"] = fast_override.strip()
        if pro_override.strip():
            os.environ["GEMINI_MODEL_PRO"] = pro_override.strip()

    demo_mode = st.checkbox("Demo / Simulation Mode", value=False,
        help="Full simulation — no API keys needed")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:1.2rem 0;'>",
                unsafe_allow_html=True)

    st.markdown("### 🕑 Analysis History")
    if not st.session_state.history:
        st.markdown("<div style='color:#475569;font-size:0.83rem;'>No analyses yet.</div>",
                    unsafe_allow_html=True)
    else:
        for i, item in enumerate(reversed(st.session_state.history[-6:])):
            col_h1, col_h2 = st.columns([4,1])
            with col_h1:
                st.markdown(f"""
                <div class='hist-item'>
                    <div class='hist-topic'>{html.escape(item['topic'])}</div>
                    <div class='hist-meta'>{html.escape(item['timestamp'])} · {html.escape(item['mode'])}</div>
                </div>""", unsafe_allow_html=True)
            with col_h2:
                if st.button("↩", key=f"hist_{i}", help=f"Reload {item['topic']}"):
                    st.session_state.report_markdown = item["markdown"]
                    st.session_state.topic_analyzed  = item["topic"]
                    st.session_state.orchestrator_mode= item["mode"]
                    st.session_state.market_scores   = item["scores"]
                    st.session_state.agent_steps     = []
                    st.session_state.loaded_from_history = True
                    st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:1.2rem 0;'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#475569;font-size:0.72rem;line-height:1.7;text-align:center;">
        CrewAI orchestration with automatic<br>LangChain fallback.<br>
        Powered by Gemini + Tavily.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge"><span class="hero-dot"></span> AI-Powered Research Platform</div>
    <div class="hero-title">Competitor Intelligence<br>& Market Analyst Team</div>
    <div class="hero-sub">
        Dispatch a crew of autonomous AI agents to crawl the web, build a SWOT matrix,
        benchmark your top competitors, and compile an executive PDF report — in minutes.
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="metrics-strip">
  <div class="metric-chip"><div class="metric-val">3</div><div class="metric-lbl">AI Agents</div></div>
  <div class="metric-chip"><div class="metric-val">12+</div><div class="metric-lbl">Searches</div></div>
  <div class="metric-chip"><div class="metric-val">SWOT</div><div class="metric-lbl">Analysis</div></div>
  <div class="metric-chip"><div class="metric-val">PDF</div><div class="metric-lbl">Export</div></div>
  <div class="metric-chip"><div class="metric-val">MD</div><div class="metric-lbl">Export</div></div>
  <div class="metric-chip"><div class="metric-val">∞</div><div class="metric-lbl">History</div></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-title">💡 Popular Topics</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(SUGGESTIONS))
chip_clicked = None
for i, s in enumerate(SUGGESTIONS):
    with chip_cols[i]:
        if st.button(s, key=f"chip_{s}"):
            chip_clicked = s

# ═══════════════════════════════════════════════
#  MAIN INPUT
# ═══════════════════════════════════════════════
col_inp, col_btn = st.columns([5, 1])
with col_inp:
    topic_val = chip_clicked if chip_clicked else ""
    topic = st.text_input("topic_input",
        value=topic_val,
        placeholder="🔍  Enter company name or product niche — e.g. Vercel, Supabase, Figma…",
        label_visibility="collapsed",
        max_chars=MAX_TOPIC_LENGTH)
with col_btn:
    run_btn = st.button("Analyse ⚡", use_container_width=True)

if st.session_state.report_markdown:
    if st.button("🗑️ Clear & New Analysis", use_container_width=False):
        _reset_session_state()
        st.rerun()

# ═══════════════════════════════════════════════
#  EXECUTION
# ═══════════════════════════════════════════════
if run_btn:
    topic_clean = topic.strip() if topic else ""
    if not topic_clean:
        st.warning(f"⚠️  Please enter a company name or product niche.")
    elif len(topic_clean) > MAX_TOPIC_LENGTH:
        st.warning(f"⚠️  Topic must be {MAX_TOPIC_LENGTH} characters or less.")
    elif not demo_mode and not os.environ.get("GEMINI_API_KEY"):
        st.error("🔑  Gemini API Key is missing. Add it in the sidebar or enable Demo Mode.")
    elif not demo_mode and not os.environ.get("TAVILY_API_KEY"):
        st.error("🔑  Tavily API Key is missing. Add it in the sidebar or enable Demo Mode.")
    else:
        st.session_state.agent_steps = []
        steps_placeholder = st.empty()

        def render_steps(steps):
            agent_icons = {"Researcher": "🔬", "Analyst": "⚖️", "Writer": "✍️"}
            agent_colors = {"Researcher": "#6366f1", "Analyst": "#a855f7", "Writer": "#ec4899"}
            cards = ""
            for step in steps:
                icon  = agent_icons.get(step["agent"], "⚡")
                color = agent_colors.get(step["agent"], "#6366f1")
                cards += f"""
                <div class="agent-step">
                    <div class="step-icon">{icon}</div>
                    <div class="step-body">
                        <div class="step-agent" style="color:{color} !important;">{html.escape(step['agent'])} Agent</div>
                        <div class="step-msg">{html.escape(step['msg'])}</div>
                        <div class="step-time">{html.escape(step['time'])}</div>
                    </div>
                </div>"""
            steps_placeholder.markdown(
                f'<div class="glass"><div class="section-title">🚀 Agent Activity Feed</div>{cards}</div>',
                unsafe_allow_html=True)

        def status_cb(agent, msg):
            st.session_state.agent_steps.append({
                "agent": agent,
                "msg":   msg,
                "time":  datetime.now().strftime("%H:%M:%S")
            })
            render_steps(st.session_state.agent_steps)

        try:
            if demo_mode:
                report  = run_mock_analysis(topic=topic_clean, status_callback=status_cb)
                mode    = "Simulation Mode"
            else:
                report, mode = run_analysis(
                    topic=topic_clean, depth=research_depth,
                    status_callback=status_cb
                )

            scores = get_market_scores(report, topic_clean)

            st.session_state.report_markdown  = report
            st.session_state.orchestrator_mode= mode
            st.session_state.topic_analyzed   = topic_clean
            st.session_state.market_scores    = scores

            st.session_state.history.append({
                "topic":     topic_clean,
                "mode":      mode,
                "markdown":  report,
                "scores":    scores,
                "timestamp": datetime.now().strftime("%d %b %H:%M"),
            })
            if len(st.session_state.history) > MAX_HISTORY_ITEMS:
                st.session_state.history = st.session_state.history[-MAX_HISTORY_ITEMS:]

            steps_placeholder.empty()
            st.success(f"✅  Report compiled via **{mode}** — topic: **{topic_clean}**")

        except Exception as e:
            st.error(f"❌  Orchestration error: {str(e)}")

# ═══════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════
if st.session_state.report_markdown:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    h_col1, h_col2, h_col3, h_col4 = st.columns([3, 1, 1, 1])
    with h_col1:
        wc   = len(st.session_state.report_markdown.split())
        mins = max(1, wc // 200)
        st.markdown(f"""
        <div style="margin-bottom:0.6rem;">
            <span style="font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:800;color:#e2e8f0;">
                📋 {html.escape(st.session_state.topic_analyzed)}
            </span>
            &nbsp;&nbsp;<span class="read-badge">⏱ ~{mins} min read &nbsp;·&nbsp; {wc:,} words</span>
        </div>""", unsafe_allow_html=True)

    with h_col2:
        pdf_fn = f"{st.session_state.topic_analyzed.lower().replace(' ','_')}_report.pdf"
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                pdf_path = tmp.name
            generate_pdf_from_markdown(
                st.session_state.report_markdown,
                st.session_state.topic_analyzed,
                pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button("📥 PDF Report", data=pdf_bytes,
                               file_name=pdf_fn, mime="application/pdf")
        except Exception as e:
            st.error(f"PDF error: {e}")
        finally:
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass

    with h_col3:
        md_fn = f"{st.session_state.topic_analyzed.lower().replace(' ','_')}_report.md"
        st.download_button("📄 Markdown",
            data=st.session_state.report_markdown.encode("utf-8"),
            file_name=md_fn, mime="text/markdown")

    with h_col4:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("🔄 New Report"):
            _reset_session_state()
            st.rerun()

    scores = st.session_state.market_scores
    if scores:
        st.markdown("<div class='section-title' style='margin-top:1.2rem;'>📊 Market Intelligence Dashboard</div>",
                    unsafe_allow_html=True)
        st.caption("Scores are illustrative — derived from the report content. "
                   "Use them as a quick signal, not ground truth.")

        score_items = [
            ("Market Opportunity",   scores["market_opportunity"],  "#10b981", "How strong is the market growth potential?"),
            ("Competitive Pressure", scores["competitive_pressure"],"#f59e0b", "How intense is competition in this space?"),
            ("Growth Trajectory",    scores["growth_trajectory"],   "#6366f1", "Momentum and expansion rate of the company"),
            ("Innovation Score",     scores["innovation_score"],    "#a855f7", "Differentiation through product innovation"),
            ("Risk Level",           scores["risk_level"],          "#ef4444", "Overall strategic and market risk exposure"),
            ("Market Maturity",      scores["market_maturity"],     "#0ea5e9", "How mature/saturated is the target market?"),
        ]

        sc_cols = st.columns(2)
        for i, (label, val, color, tip) in enumerate(score_items):
            with sc_cols[i % 2]:
                st.markdown(f"""
                <div class="score-card" title="{html.escape(tip)}">
                    <div class="score-header">
                        <span class="score-label">{html.escape(label)}</span>
                        <span class="score-value" style="color:{color} !important;">{val}</span>
                    </div>
                    <div class="score-bar-bg">
                        <div class="score-bar" style="width:{val}%;background:linear-gradient(90deg,{color}88,{color});"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.session_state.agent_steps and not st.session_state.loaded_from_history:
        st.markdown("<div class='section-title'>🤖 Agent Activity Log</div>", unsafe_allow_html=True)
        agent_icons  = {"Researcher":"🔬","Analyst":"⚖️","Writer":"✍️"}
        agent_colors = {"Researcher":"#6366f1","Analyst":"#a855f7","Writer":"#ec4899"}
        cards = ""
        for step in st.session_state.agent_steps:
            icon  = agent_icons.get(step["agent"],"⚡")
            color = agent_colors.get(step["agent"],"#6366f1")
            cards += f"""
            <div class="agent-step">
                <div class="step-icon">{icon}</div>
                <div class="step-body">
                    <div class="step-agent" style="color:{color} !important;">{html.escape(step['agent'])} Agent</div>
                    <div class="step-msg">{html.escape(step['msg'])}</div>
                    <div class="step-time">{html.escape(step['time'])}</div>
                </div>
            </div>"""
        st.markdown(f'<div class="glass">{cards}</div>', unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📄 Executive Report", "📊 SWOT Focus", "💻 Raw Markdown"])

    def _render_report_box(md_content: str, extra_style: str = ""):
        """Render markdown content inside a styled report-box div safely.

        Streamlit's st.markdown() (without unsafe_allow_html) sanitizes the
        content, so we open/close the styled div with static HTML and render
        the LLM-produced markdown safely in between.
        """
        style = f' style="{extra_style}"' if extra_style else ""
        st.markdown(f'<div class="report-box"{style}>', unsafe_allow_html=True)
        st.markdown(md_content)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab1:
        _render_report_box(st.session_state.report_markdown)

    with tab2:
        md = st.session_state.report_markdown
        swot_start = md.find("## SWOT")
        if swot_start == -1:
            swot_start = md.find("## Swot")
        swot_end   = md.find("\n## ", swot_start + 1) if swot_start != -1 else -1

        if swot_start != -1:
            swot_section = md[swot_start:swot_end if swot_end != -1 else len(md)]
            _render_report_box(swot_section)

            rec_start = md.find("## Strategic")
            if rec_start == -1:
                rec_start = md.find("## Recommendation")
            if rec_start != -1:
                rec_end = md.find("\n## ", rec_start + 1)
                rec_section = md[rec_start:rec_end if rec_end != -1 else len(md)]
                _render_report_box(rec_section, extra_style="margin-top:1rem;")
        else:
            st.info("SWOT section not found in this report. Check the Executive Report tab.")

    with tab3:
        st.code(st.session_state.report_markdown, language="markdown")
        st.markdown(
            "<div style='text-align:right;margin-top:0.5rem'>"
            "<small style='color:#475569'>Tip: Use the Markdown download button above to save this file</small>"
            "</div>",
            unsafe_allow_html=True)

    st.session_state.loaded_from_history = False
