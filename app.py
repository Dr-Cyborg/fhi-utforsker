"""FHI Datautforsker — interaktiv Streamlit-app for FHI Statistikk Open API.

Kjør med:  streamlit run fhi_utforsker.py
"""
from __future__ import annotations

import itertools
import json
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_URL = "https://statistikk-data.fhi.no/api/open/v1"
USER_AGENT = "fhi-utforsker (contact: nyborgchristoffer@gmail.com)"
HTTP_TIMEOUT = 60.0


# -------- API-laget (cachet) --------------------------------------------------

@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        timeout=HTTP_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    )


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_kilder() -> list[dict]:
    return _client().get("/Common/source").json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_tabeller(kilde: str) -> list[dict]:
    return _client().get(f"/{kilde}/Table").json()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_dimensjoner(kilde: str, tabell_id: int | str) -> dict:
    return _client().get(f"/{kilde}/Table/{tabell_id}/dimension").json()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_template(kilde: str, tabell_id: int | str) -> dict:
    return _client().get(f"/{kilde}/Table/{tabell_id}/query").json()


@st.cache_data(ttl=3600, show_spinner=False)
def hent_data(kilde: str, tabell_id: int | str, sporring_json: str) -> dict:
    body = json.loads(sporring_json)
    r = _client().post(f"/{kilde}/Table/{tabell_id}/data", json=body)
    r.raise_for_status()
    return r.json()


# -------- Hjelpere ------------------------------------------------------------

def flat_kategorier(cats: list[dict], depth: int = 0, out: list | None = None) -> list[tuple[str, str, int]]:
    """Tre → flat liste av (value, label, depth). Hopper over rene parent-noder med children=alle."""
    if out is None:
        out = []
    for c in cats:
        out.append((c["value"], c["label"], depth))
        if c.get("children"):
            flat_kategorier(c["children"], depth + 1, out)
    return out


def jsonstat_til_dataframe(ds: dict) -> pd.DataFrame:
    """json-stat2 → tidy pandas DataFrame med én rad per verdi."""
    dim_koder = ds["id"]
    indekser = [ds["dimension"][d]["category"]["index"] for d in dim_koder]
    labels = [ds["dimension"][d]["category"]["label"] for d in dim_koder]
    dim_navn = [ds["dimension"][d]["label"] for d in dim_koder]
    verdier = ds["value"]

    rows = []
    for v, combo in zip(verdier, itertools.product(*indekser)):
        row = {dim_navn[i]: labels[i].get(combo[i], combo[i]) for i in range(len(combo))}
        if v == ":" or v is None or v == "":
            row["Verdi"] = None
        else:
            try:
                row["Verdi"] = float(v)
            except (TypeError, ValueError):
                row["Verdi"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def format_label(value: str, label: str, depth: int) -> str:
    indent = "  " * depth + ("└ " if depth > 0 else "")
    return f"{indent}{label}"


def smart_default(dim_label: str, dim_kode: str, flat: list[tuple[str, str, int]],
                  viste_labels: list[str]) -> list[str]:
    """Velg fornuftige defaultverdier for en dimensjon ved første visning."""
    lbl_low = dim_label.lower()
    kode_low = dim_kode.lower()

    # Tids-dim: ta nyeste N (52 hvis uker, ellers 20 hvis år, ellers 12)
    is_time = any(t in lbl_low for t in ["år", "uke", "tid", "dato", "måned"]) \
              or any(t in kode_low for t in ["aar", "uke", "tidspunkt", "periode"])
    if is_time:
        n = 52 if "uke" in lbl_low or "uke" in kode_low else 20
        return viste_labels[-min(n, len(viste_labels)):]

    # Måltall: bare ÉN, prefer rate/per1000/andel/prosent
    if "måltall" in lbl_low or "maltall" in lbl_low or kode_low.endswith("measure_type") or "measure" in kode_low:
        for kw in ["per1000", "per_1000", "rate", "prosent", "andel", "_per_", "smr"]:
            for v, l, _ in flat:
                if kw in v.lower() or kw.replace("_", "") in l.lower().replace(" ", ""):
                    return [format_label(v, l, 0)]
        return viste_labels[:1]

    # Kategorisk med én rotnode + barn → bruk barna (typisk: TOTALT-rot med 14 ATC-er under)
    depth0 = [it for it in flat if it[2] == 0]
    if len(depth0) == 1 and any(it[2] == 1 for it in flat):
        depth1 = [format_label(*it) for it in flat if it[2] == 1]
        if 1 <= len(depth1) <= 30:
            return depth1
        elif len(depth1) > 30:
            return depth1[:25]

    # Liten liste: alle
    if len(flat) <= 15:
        return viste_labels

    # Stor liste: bare topp-nivå (capped)
    return [format_label(*it) for it in flat if it[2] == 0][:15]


def er_tids_dim(dim_label: str, dim_kode: str) -> bool:
    lbl = dim_label.lower()
    kode = dim_kode.lower()
    return any(t in lbl for t in ["år", "uke", "tid", "dato", "måned"]) \
        or any(t in kode for t in ["aar", "uke", "tidspunkt", "periode"])


def er_hierarkisk(flat: list[tuple[str, str, int]]) -> bool:
    return any(it[2] > 0 for it in flat)


def bygg_sporring(dim_data: dict, template: dict, valgte_per_dim: dict[str, list[str]]) -> dict:
    """Bygg en spørring fra template + valgte verdier per dim."""
    sporring = {
        "dimensions": [],
        "response": template.get("response", {"format": "json-stat2", "maxRowCount": 50000}),
    }
    for d in template["dimensions"]:
        sporring["dimensions"].append({
            "code": d["code"],
            "filter": d.get("filter", "item"),
            "values": valgte_per_dim.get(d["code"], d["values"]),
        })
    return sporring


def auto_velg_per_dim(dim_data: dict) -> dict[str, list[str]]:
    """Smart auto-utvalg for hver dim — for Oversikt-fanen."""
    out = {}
    for dim in dim_data["dimensions"]:
        flat = flat_kategorier(dim["categories"])
        if not flat:
            continue
        viste = [format_label(*it) for it in flat]
        label_til_verdi = {format_label(v, l, d): v for v, l, d in flat}
        valgte_labels = smart_default(dim["label"], dim["code"], flat, viste)
        out[dim["code"]] = [label_til_verdi[lab] for lab in valgte_labels]
    return out


def auto_velg_akser(df: pd.DataFrame, dim_kolonner: list[str]) -> tuple[str | None, str | None]:
    """Velg fornuftig x-akse + fargedimensjon for auto-grafen."""
    varierende = [c for c in dim_kolonner if df[c].nunique() > 1]
    if not varierende:
        return None, None
    x = next((c for c in varierende if any(t in c.lower() for t in ["år", "uke", "tid", "dato"])), None)
    if x is None:
        x = max(varierende, key=lambda c: df[c].nunique())
    rest = [c for c in varierende if c != x]
    color = max(rest, key=lambda c: df[c].nunique()) if rest else None
    return x, color


def lag_graf(plot_df: pd.DataFrame, x_akse: str, farge: str, stil: str,
             graftype: str, palett: list, tittel: str, ymerke: str | None = None):
    common: dict[str, Any] = dict(x=x_akse, y="Verdi", color_discrete_sequence=palett)
    if farge and farge != "(ingen)":
        common["color"] = farge
    if stil and stil != "(ingen)":
        if "Linje" in graftype:
            common["line_dash"] = stil
            common["symbol"] = stil
        elif graftype == "Scatter (punkter)":
            common["symbol"] = stil
        elif "søyle" in graftype.lower() or "Område" in graftype:
            common["pattern_shape"] = stil

    if graftype == "Linje":
        fig = px.line(plot_df, **common)
    elif graftype == "Linje + markører":
        fig = px.line(plot_df, markers=True, **common)
    elif graftype == "Søyle":
        fig = px.bar(plot_df, **common)
    elif graftype == "Gruppert søyle":
        fig = px.bar(plot_df, barmode="group", **common)
    elif graftype == "Stablet søyle":
        fig = px.bar(plot_df, barmode="stack", **common)
    elif graftype == "Område":
        fig = px.area(plot_df, **common)
    elif graftype == "Stablet område":
        fig = px.area(plot_df, **common)
    elif graftype == "Scatter (punkter)":
        fig = px.scatter(plot_df, **common)
    else:
        fig = px.line(plot_df, **common)

    fig.update_layout(
        height=780,
        title=tittel,
        legend=dict(title=None, orientation="v"),
        hovermode="x unified" if "Linje" in graftype or "Område" in graftype else "closest",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if "Linje" in graftype or graftype == "Scatter (punkter)":
        fig.update_traces(line=dict(width=2.2), marker=dict(size=7))
    if ymerke:
        fig.update_yaxes(title=ymerke)

    # KRITISK: alltid kategorisk x-akse for å unngå at plotly tolker
    # "2025-17" (uke 17) som "november 2025" (ISO år-måned).
    # Dataene fra FHI er strenger; vi vil aldri ha auto-datoparsing.
    n_kategorier = len(plot_df[x_akse].unique()) if x_akse in plot_df.columns else 0
    fig.update_xaxes(
        type="category",
        tickangle=-45 if n_kategorier > 12 else 0,
        # Bevar sorteringsrekkefølgen vi har i plot_df
        categoryorder="array",
        categoryarray=list(plot_df[x_akse].unique()) if x_akse in plot_df.columns else None,
    )
    return fig


# -------- UI ------------------------------------------------------------------

st.set_page_config(page_title="FHI Datautforsker", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

      /* Sidebar — bruk Streamlits eget tema som bunn, legg på semi-transparent
         tint + tydelig høyrekant. Fungerer i både lys og mørk modus. */
      [data-testid="stSidebar"] > div:first-child {
        background-image: linear-gradient(180deg,
          rgba(74, 123, 168, 0.14) 0%,
          rgba(74, 123, 168, 0.05) 100%);
        border-right: 3px solid #4a7ba8;
      }
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] h4 {
        color: #4a7ba8 !important;
        border-bottom: 1px solid rgba(128, 128, 128, 0.3);
        padding-bottom: 4px;
        margin-top: 0.6rem;
      }
      /* Filter-expander = kort med tema-adaptiv overlay */
      [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        margin-bottom: 6px;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        font-size: 0.92rem;
      }
      /* Faner — gjør dem tydeligere */
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid rgba(74, 123, 168, 0.4);
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
        padding: 0.55rem 1.1rem;
        background: rgba(128, 128, 128, 0.08);
        border-radius: 8px 8px 0 0;
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: #2c5282 !important;
        color: white !important;
      }

      h1 { color: #4a7ba8; margin-bottom: 0.2rem; }
      .small-muted { color: rgba(128, 128, 128, 0.85); font-size: 0.85em; }

      /* Native trevelger: gjør expanders kompakte i sidebaren */
      [data-testid="stSidebar"] [data-testid="stExpander"] {
        margin-bottom: 2px;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        padding: 4px 8px;
      }

      /* Bruk Streamlits egen header som topplinje — spenner over sidebar +
         hovedinnhold automatisk og lar hamburger-menyen være intakt */
      [data-testid="stHeader"] {
        background: linear-gradient(135deg, #14365c 0%, #2c5282 100%) !important;
        height: 56px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
      }
      [data-testid="stHeader"]::before {
        content: "📊 FHI Datautforsker  ·  Interaktiv visualisering av FHI Statistikk Open API";
        position: absolute;
        left: 1.4rem;
        top: 50%;
        transform: translateY(-50%);
        color: #fff;
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: calc(100vw - 6rem);
      }
      /* La toolbar-ikonene (kebab/run) bli synlige mot mørkblå bakgrunn */
      [data-testid="stHeader"] [data-testid="stToolbar"] svg,
      [data-testid="stHeader"] button { color: #fff !important; }

      /* Padding under stHeader så øverste innhold ikke ligger under den */
      [data-testid="stAppViewContainer"] > section.main > div.block-container {
        padding-top: 4.5rem !important;
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 4.5rem !important; }

      /* Venstre sidebar: kompakt kilde→tabell-tre */
      [data-testid="stSidebar"] .stButton button {
        text-align: left;
        justify-content: flex-start;
        white-space: normal;
        height: auto;
        padding: 4px 8px;
        font-size: 0.82rem;
        font-weight: 400;
        background: transparent;
        color: inherit;
        border: 1px solid transparent;
      }
      [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(74, 123, 168, 0.18);
        border-color: rgba(74, 123, 168, 0.35);
      }
      /* Aktiv kilde/tabell-knapp — bruk type=primary fra Streamlit */
      [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: #2c5282 !important;
        color: #fff !important;
        font-weight: 600;
        border-color: #14365c;
      }
      /* Innrykk for tabell-knapper (under kilde) */
      [data-testid="stSidebar"] .tabell-rad .stButton button {
        padding-left: 1.3rem;
        font-size: 0.78rem;
      }

      /* Høyre filterkolonne: kompakte filterbokser */
      .filter-col [data-testid="stContainer"] {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 6px 10px;
        margin-bottom: 6px;
      }
      .filter-col .filter-tittel {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4a7ba8;
        margin: 0 0 2px 0;
      }
      .filter-col [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        font-size: 0.72rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tittel rendres i Streamlits header via CSS — se [data-testid="stHeader"]::before

try:
    kilder = get_kilder()
except Exception as e:
    st.error(f"Kunne ikke hente datakilder: {e}")
    st.stop()

# === Venstre sidebar: kilde→tabell-tre ===
with st.sidebar:
    sb_sok = st.text_input(
        "🔍 Søk i kilder/tabeller", "",
        placeholder="Filter på navn…",
        key="sok_kilde_tabell",
        label_visibility="collapsed",
    )

    # Hvilken kilde er åpnet i treet (kan avvike fra aktiv)
    if "open_source" not in st.session_state:
        st.session_state["open_source"] = st.session_state.get("aktiv_kilde") or kilder[0]["id"]

    for k in kilder:
        sid = k["id"]
        is_active_src = (st.session_state.get("aktiv_kilde") == sid)
        is_open = (st.session_state["open_source"] == sid) or bool(sb_sok)

        prefix = "▼ " if is_open else "▶ "
        if st.button(
            f"{prefix}📚 {k['title']} ({sid})",
            key=f"src_{sid}",
            use_container_width=True,
            type="primary" if is_active_src else "secondary",
        ):
            st.session_state["open_source"] = sid
            st.rerun()

        if is_open:
            try:
                tabs_for_src = get_tabeller(sid)
            except Exception:
                continue
            filt_t = [
                t for t in tabs_for_src
                if not sb_sok or sb_sok.lower() in t["title"].lower()
            ]
            if not filt_t:
                continue
            st.markdown('<div class="tabell-rad">', unsafe_allow_html=True)
            for t in filt_t:
                tid = t["tableId"]
                is_active_tab = is_active_src and (st.session_state.get("aktiv_tabell") == tid)
                if st.button(
                    f"{'▶ ' if is_active_tab else '· '}{t['title']}",
                    key=f"tab_{sid}_{tid}",
                    use_container_width=True,
                    type="primary" if is_active_tab else "secondary",
                    help=f"Tabell #{tid}",
                ):
                    st.session_state["aktiv_kilde"] = sid
                    st.session_state["aktiv_tabell"] = tid
                    st.session_state["open_source"] = sid
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# Aktiv kilde + tabell — første gang: bruk første kilde + første tabell
kilde_id = st.session_state.get("aktiv_kilde") or kilder[0]["id"]
st.session_state["aktiv_kilde"] = kilde_id

if not st.session_state.get("aktiv_tabell"):
    try:
        first_tabs = get_tabeller(kilde_id)
    except Exception as e:
        st.error(f"Kunne ikke hente tabeller: {e}")
        st.stop()
    if not first_tabs:
        st.warning("Ingen tabeller for denne kilden.")
        st.stop()
    st.session_state["aktiv_tabell"] = first_tabs[0]["tableId"]

tabell_id = st.session_state["aktiv_tabell"]

try:
    dim_data = get_dimensjoner(kilde_id, tabell_id)
except Exception as e:
    st.error(f"Kunne ikke hente dimensjoner: {e}")
    st.stop()

# === Hovedlayout: graf til venstre, filter-sidebar til høyre ===
graph_col, filter_col = st.columns([3.5, 1.3], gap="medium")

with filter_col:
    st.markdown('<div class="filter-col">', unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    bc1.button(
        "✓ Bruk valg", type="primary", use_container_width=True,
        key=f"apply_{tabell_id}",
        help="Tving en oppfriskning av grafen (filtre virker reaktivt allerede)",
    )
    if bc2.button(
        "↺ Nullstill", use_container_width=True, key=f"reset_{tabell_id}",
        help="Tilbake til standardvalg for alle filtre",
    ):
        prefixes = (
            f"dim_v2_{tabell_id}_",
            f"slider_dim_v2_{tabell_id}_",
        )
        for k in list(st.session_state.keys()):
            if any(k.startswith(p) for p in prefixes):
                del st.session_state[k]
        st.rerun()

    st.caption(f"{len(dim_data['dimensions'])} variabler")

    for dim in dim_data["dimensions"]:
        kode = dim["code"]
        lbl = dim["label"]
        flat = flat_kategorier(dim["categories"])
        if not flat:
            continue
        viste_labels = [format_label(*item) for item in flat]
        verdi_til_label = {v: l for v, l, _ in flat}
        label_til_verdi = {format_label(v, l, d): v for v, l, d in flat}
        depth_per_v = {v: d for v, l, d in flat}
        all_values = [v for v, l, d in flat]

        state_key = f"dim_v2_{tabell_id}_{kode}"
        if state_key not in st.session_state:
            default_labels = smart_default(lbl, kode, flat, viste_labels)
            st.session_state[state_key] = [label_til_verdi[lab] for lab in default_labels]
        else:
            st.session_state[state_key] = [
                v for v in st.session_state[state_key] if v in label_til_verdi.values()
            ]

        is_time = er_tids_dim(lbl, kode) and len(all_values) > 4
        n_valgt = len(st.session_state[state_key])

        with st.container(border=True):
            tc1, tc2 = st.columns([5, 1])
            tc1.markdown(
                f'<div class="filter-tittel">{lbl} — {n_valgt}/{len(all_values)}</div>',
                unsafe_allow_html=True,
            )
            if tc2.button("✕", key=f"clr_{state_key}", help=f"Tøm {lbl}"):
                st.session_state[state_key] = []
                if is_time and f"slider_{state_key}" in st.session_state:
                    del st.session_state[f"slider_{state_key}"]
                st.rerun()

            if is_time:
                ts_values = (
                    [v for v, l, d in flat if d > 0]
                    if er_hierarkisk(flat) else all_values
                )
                ts_labels = {v: l for v, l, d in flat}
                eksisterende = [v for v in st.session_state[state_key] if v in ts_values]
                if eksisterende:
                    start_def, end_def = eksisterende[0], eksisterende[-1]
                else:
                    start_def, end_def = ts_values[max(0, len(ts_values) - 20)], ts_values[-1]
                start, end = st.select_slider(
                    "Periode",
                    options=ts_values,
                    value=(start_def, end_def),
                    format_func=lambda v, tl=ts_labels: tl.get(v, v),
                    key=f"slider_{state_key}",
                    label_visibility="collapsed",
                )
                i, j = ts_values.index(start), ts_values.index(end)
                st.session_state[state_key] = ts_values[i:j + 1]
                st.caption(f"📅 {ts_labels.get(start)} → {ts_labels.get(end)}")
            else:
                st.multiselect(
                    "Verdier",
                    options=all_values,
                    format_func=lambda v, vl=verdi_til_label, dp=depth_per_v: format_label(v, vl[v], dp[v]),
                    key=state_key,
                    label_visibility="collapsed",
                )

    st.markdown('</div>', unsafe_allow_html=True)

# === Bygg valgte_per_dim + meta fra committed state ===
valgte_per_dim: dict[str, list[str]] = {}
dim_meta: list[dict] = []
for dim in dim_data["dimensions"]:
    kode = dim["code"]
    lbl = dim["label"]
    flat = flat_kategorier(dim["categories"])
    if not flat:
        continue
    verdi_til_label = {v: l for v, l, _ in flat}
    state_key = f"dim_v2_{tabell_id}_{kode}"
    valgte_per_dim[kode] = list(st.session_state.get(state_key, []))
    dim_meta.append({
        "kode": kode,
        "label": lbl,
        "antall_valgt": len(valgte_per_dim[kode]),
        "valgte_labels": [verdi_til_label.get(v, v) for v in valgte_per_dim[kode]],
    })

with graph_col:
    # --- Hovedområde ---

    # Hvis brukeren har "tømt" en dim, auto-fyll med smart-default i stedet for å stoppe.
    tomme_dim = [m["label"] for m in dim_meta if m["antall_valgt"] == 0]
    if tomme_dim:
        st.info(
            f"ℹ️ Tomme filtre: **{', '.join(tomme_dim)}** — bruker standardvalg for disse. "
            "Velg verdier i venstremenyen for å overstyre."
        )
        auto_fyll = auto_velg_per_dim(dim_data)
        for kode, verdier in valgte_per_dim.items():
            if not verdier:
                valgte_per_dim[kode] = auto_fyll.get(kode, [])

    # Bygg sporring fra template, override values per dim
    try:
        template = get_template(kilde_id, tabell_id)
    except Exception as e:
        st.error(f"Kunne ikke hente template: {e}")
        st.stop()

    sporring = bygg_sporring(dim_data, template, valgte_per_dim)

    # Hent data
    estimert = 1
    for v in valgte_per_dim.values():
        estimert *= max(1, len(v))

    if estimert > 50000:
        st.error(f"Du har valgt {estimert:,} kombinasjoner — over taket på 50 000. Reduser filtrene.")
        st.stop()

    with st.spinner(f"Henter {estimert:,} datapunkter fra FHI…"):
        try:
            ds = hent_data(kilde_id, tabell_id, json.dumps(sporring))
        except httpx.HTTPStatusError as e:
            st.error(f"FHI API-feil ({e.response.status_code}): {e.response.text[:500]}")
            st.stop()
        except Exception as e:
            st.error(f"Henting feilet: {e}")
            st.stop()

    df = jsonstat_til_dataframe(ds)
    df = df.dropna(subset=["Verdi"])

    if df.empty:
        st.warning("Ingen tall i resultatet (alt undertrykket eller manglende).")
        st.stop()

    st.success(f"✅ {len(df):,} rader fra **{ds.get('label', '?')}**, oppdatert {ds.get('updated', '?')[:10]}")

    dim_kolonner = [d["label"] for d in dim_data["dimensions"] if d["label"] in df.columns]
    varierende = [c for c in dim_kolonner if df[c].nunique() > 1]
    konstante = [c for c in dim_kolonner if df[c].nunique() == 1]

    PALETT_MAP = {
        "Plotly": px.colors.qualitative.Plotly, "D3": px.colors.qualitative.D3,
        "G10": px.colors.qualitative.G10, "T10": px.colors.qualitative.T10,
        "Set1": px.colors.qualitative.Set1, "Set2": px.colors.qualitative.Set2,
        "Set3": px.colors.qualitative.Set3, "Pastel": px.colors.qualitative.Pastel,
        "Bold": px.colors.qualitative.Bold, "Vivid": px.colors.qualitative.Vivid,
        "Dark24": px.colors.qualitative.Dark24, "Light24": px.colors.qualitative.Light24,
        "Viridis": px.colors.sequential.Viridis, "Plasma": px.colors.sequential.Plasma,
        "Turbo": px.colors.sequential.Turbo,
    }

    # === Faner ===
    tab_min, tab_oversikt, tab_data, tab_query, tab_meta = st.tabs([
        "📈 Min graf", "🔭 Oversikt", "📋 Data", "🧬 Spørring", "ℹ️ Om tabellen",
    ])

    # ---------- Min graf ---------------------------------------------------------
    with tab_min:
        if not varierende:
            st.info("Bare én verdi i hver dimensjon — ingenting å plotte. Endre filtrene.")
        else:
            with st.expander("🎨 Visualiseringsvalg", expanded=False):
                c0, c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.4, 1.4, 1.6])
                with c0:
                    foretrukket = next(
                        (c for c in varierende if any(t in c.lower() for t in ["år", "uke", "tid", "dato", "ar"])),
                        varierende[0],
                    )
                    x_akse = st.selectbox("X-akse", varierende, index=varierende.index(foretrukket))
                with c1:
                    fargekandidater = ["(ingen)"] + [c for c in varierende if c != x_akse]
                    farge = st.selectbox("Farge etter", fargekandidater, index=1 if len(fargekandidater) > 1 else 0)
                with c2:
                    stilkandidater = ["(ingen)"] + [c for c in varierende if c != x_akse and c != farge]
                    stil = st.selectbox("Stil/markør etter", stilkandidater, index=0)
                with c3:
                    graftype = st.selectbox("Graftype", [
                        "Linje", "Linje + markører", "Søyle", "Gruppert søyle",
                        "Stablet søyle", "Område", "Stablet område", "Scatter (punkter)",
                    ])
                with c4:
                    palett_navn = st.selectbox("Fargepalett", list(PALETT_MAP.keys()))

            agg_kols = [x_akse] + ([farge] if farge != "(ingen)" else []) + ([stil] if stil != "(ingen)" else [])
            agg_kols = list(dict.fromkeys(agg_kols))
            plot_df = df.groupby(agg_kols, dropna=False, as_index=False)["Verdi"].sum()
            plot_df = plot_df.sort_values(by=agg_kols)

            ymerke = None
            maltall_kol = next((c for c in ["Måltall", "Maltall"] if c in df.columns), None)
            if maltall_kol and df[maltall_kol].nunique() == 1:
                ymerke = df[maltall_kol].iloc[0]

            try:
                fig = lag_graf(plot_df, x_akse, farge, stil, graftype,
                                PALETT_MAP[palett_navn], ds.get("label", ""), ymerke)
                st.plotly_chart(fig, width="stretch", theme="streamlit")
            except Exception as e:
                st.error(f"Kunne ikke bygge graf: {e}")

            if konstante:
                st.markdown("**Faste filtre:** " + " · ".join(f"`{c}: {df[c].iloc[0]}`" for c in konstante))

    # ---------- Oversikt — auto-illustrasjon av hele datasettet ------------------
    with tab_oversikt:
        st.caption(
            "Auto-generert oversikt over hele datasettet med smarte standardvalg "
            "(siste 20 år/52 uker, hovednivåer, ett måltall). Uavhengig av filtrene dine til venstre."
        )

        auto_dim = auto_velg_per_dim(dim_data)
        auto_estimert = 1
        for v in auto_dim.values():
            auto_estimert *= max(1, len(v))

        if auto_estimert > 50000:
            st.warning(
                f"Standard-oversikten ville hentet {auto_estimert:,} celler — for mye. "
                "Tabellen er svært bred; gå til 'Min graf'-fanen og innsnevre selv."
            )
        else:
            try:
                auto_template = template
                auto_sporring = bygg_sporring(dim_data, auto_template, auto_dim)
                with st.spinner(f"Henter {auto_estimert:,} celler for oversikt…"):
                    auto_ds = hent_data(kilde_id, tabell_id, json.dumps(auto_sporring))
                auto_df = jsonstat_til_dataframe(auto_ds).dropna(subset=["Verdi"])
            except Exception as e:
                st.error(f"Kunne ikke hente oversiktsdata: {e}")
                auto_df = pd.DataFrame()

            if auto_df.empty:
                st.info("Ingen tall i oversikten.")
            else:
                auto_dim_kol = [d["label"] for d in dim_data["dimensions"] if d["label"] in auto_df.columns]
                auto_x, auto_color = auto_velg_akser(auto_df, auto_dim_kol)
                if auto_x is None:
                    st.info("Ingen varierende dimensjoner — ingenting å plotte.")
                else:
                    # heuristikk: linje hvis tid på x, ellers gruppert søyle
                    is_tid = any(t in auto_x.lower() for t in ["år", "uke", "tid", "dato"])
                    gtype = "Linje + markører" if is_tid else "Gruppert søyle"

                    agg_kols2 = [auto_x] + ([auto_color] if auto_color else [])
                    pdf = auto_df.groupby(agg_kols2, dropna=False, as_index=False)["Verdi"].sum()
                    pdf = pdf.sort_values(by=agg_kols2)

                    ymerke2 = None
                    mc = next((c for c in ["Måltall", "Maltall"] if c in auto_df.columns), None)
                    if mc and auto_df[mc].nunique() == 1:
                        ymerke2 = auto_df[mc].iloc[0]

                    fig2 = lag_graf(pdf, auto_x, auto_color or "(ingen)", "(ingen)",
                                     gtype, px.colors.qualitative.Bold,
                                     auto_ds.get("label", ""), ymerke2)
                    st.plotly_chart(fig2, width="stretch", theme="streamlit")

                    # Stikkord-oversikt
                    kol1, kol2, kol3 = st.columns(3)
                    kol1.metric("Datapunkter", f"{len(auto_df):,}")
                    kol2.metric("Dimensjoner", len(auto_dim_kol))
                    if is_tid and auto_x in auto_df.columns:
                        tid_verdier = sorted(auto_df[auto_x].unique())
                        kol3.metric("Tidsspenn", f"{tid_verdier[0]} → {tid_verdier[-1]}")
                    else:
                        kol3.metric("Verdier på x-akse", auto_df[auto_x].nunique())

                    st.markdown(
                        "**Auto-valg:** "
                        f"x-akse = `{auto_x}`"
                        + (f", farge = `{auto_color}`" if auto_color else "")
                        + f", graftype = `{gtype}`"
                    )

    # ---------- Data, Spørring, Om ----------------------------------------------
    with tab_data:
        st.dataframe(df, width="stretch", height=420)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Last ned som CSV", csv, file_name=f"fhi_{kilde_id}_{tabell_id}.csv", mime="text/csv")

    with tab_query:
        st.code(json.dumps(sporring, ensure_ascii=False, indent=2), language="json")
        st.caption(
            f"POST {BASE_URL}/{kilde_id}/Table/{tabell_id}/data — "
            f"{estimert:,} kombinasjoner forespurt, {len(df):,} med tall."
        )

    with tab_meta:
        st.json({
            "kilde": kilde_id,
            "tabell_id": tabell_id,
            "tittel": ds.get("label"),
            "oppdatert": ds.get("updated"),
            "dimensjoner": [
                {"kode": d["code"], "label": d["label"], "antall_kategorier": len(flat_kategorier(d["categories"]))}
                for d in dim_data["dimensions"]
            ],
        })
