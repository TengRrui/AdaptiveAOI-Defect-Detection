from __future__ import annotations

import streamlit as st


DESIGN_TOKENS = {
    "color_primary": "#0F172A",
    "color_secondary": "#1E293B",
    "color_accent": "#06B6D4",
    "color_accent_dark": "#0891B2",
    "color_success": "#16A34A",
    "color_warning": "#F59E0B",
    "color_danger": "#EF4444",
    "color_bg": "#F5F7FA",
    "color_surface": "#FFFFFF",
    "color_surface_soft": "#F8FBFF",
    "color_border": "#E5E7EB",
    "color_text": "#0F172A",
    "color_muted": "#64748B",
    "radius_sm": "6px",
    "radius_md": "8px",
    "radius_lg": "12px",
    "shadow_card": "0 8px 24px rgba(15, 23, 42, 0.06)",
}


GLOBAL_CSS = f"""
<style>
:root {{
  --aoi-primary: {DESIGN_TOKENS["color_primary"]};
  --aoi-secondary: {DESIGN_TOKENS["color_secondary"]};
  --aoi-accent: {DESIGN_TOKENS["color_accent"]};
  --aoi-accent-dark: {DESIGN_TOKENS["color_accent_dark"]};
  --aoi-success: {DESIGN_TOKENS["color_success"]};
  --aoi-warning: {DESIGN_TOKENS["color_warning"]};
  --aoi-danger: {DESIGN_TOKENS["color_danger"]};
  --aoi-bg: {DESIGN_TOKENS["color_bg"]};
  --aoi-surface: {DESIGN_TOKENS["color_surface"]};
  --aoi-surface-soft: {DESIGN_TOKENS["color_surface_soft"]};
  --aoi-border: {DESIGN_TOKENS["color_border"]};
  --aoi-text: {DESIGN_TOKENS["color_text"]};
  --aoi-muted: {DESIGN_TOKENS["color_muted"]};
  --aoi-radius-sm: {DESIGN_TOKENS["radius_sm"]};
  --aoi-radius-md: {DESIGN_TOKENS["radius_md"]};
  --aoi-radius-lg: {DESIGN_TOKENS["radius_lg"]};
  --aoi-shadow-card: {DESIGN_TOKENS["shadow_card"]};
}}

html {{
  scrollbar-gutter: stable;
}}

html, body, [class*="stApp"] {{
  background: var(--aoi-bg);
  color: var(--aoi-text);
  font-family: "Inter", "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
}}

/* Reduce visible Streamlit development chrome for local application presentation. */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"], .stDeployButton {{
  display: none !important;
}}

/*
 * The native app header is empty after the toolbar and native sidebar controls
 * are replaced by the formal AdaptiveAOI shell. Streamlit still keeps it as a
 * 60px, high-z-index layer over the main scroll container, so content passing
 * underneath is obscured. Remove that empty layer instead of making it
 * transparent and leaving its hit area/stacking context in place.
 */
header[data-testid="stHeader"] {{
  display: none !important;
}}

[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(circle at top right, rgba(6, 182, 212, 0.08), transparent 28rem),
    linear-gradient(180deg, #F8FAFC 0%, var(--aoi-bg) 42%);
}}

[data-testid="stMain"] .block-container {{
  max-width: 1180px;
  min-width: 0;
  min-height: calc(100vh - 1rem);
  padding-top: 1rem;
  padding-bottom: 4rem;
}}

[data-testid="stMain"] {{
  min-width: 0;
  min-height: 100vh;
}}

/*
 * Streamlit deliberately fades elements that belong to the previous script
 * run.  On a dense inspection page that looks like a full-page white flash,
 * even though the old DOM is still usable until the replacement deltas arrive.
 * Keep the last committed view fully opaque; operation-specific status rows
 * and spinners communicate the pending work beside the initiating control.
 */
[data-testid="stAppViewContainer"] [data-stale="true"],
[data-testid="stAppViewContainer"] .stale {{
  opacity: 1 !important;
  filter: none !important;
  visibility: visible !important;
  transition: none !important;
}}

/* Prevent a short empty block-container frame while Streamlit reconciles a
 * fragment or a page script.  The previous content remains the visual floor. */
[data-testid="stMain"] .block-container:empty {{
  min-height: calc(100vh - 1rem);
}}

h1, h2, h3 {{
  color: var(--aoi-primary);
  letter-spacing: 0;
}}

h1 {{
  font-size: 2.35rem !important;
  line-height: 1.18 !important;
  font-weight: 780 !important;
  margin-bottom: 1.35rem !important;
}}

h2 {{
  font-size: 1.45rem !important;
  font-weight: 700 !important;
  margin-top: 1.6rem !important;
}}

h3 {{
  font-size: 1.12rem !important;
  font-weight: 680 !important;
}}

p, label, span, li {{
  letter-spacing: 0;
}}

.runtime-hero {{
  margin: 0 0 1rem;
  padding: 0.1rem 0;
}}

.runtime-eyebrow {{
  color: #168AAD;
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
}}

.runtime-title {{
  margin: 0.18rem 0 0.34rem;
  color: #102A43;
  font-size: clamp(1.8rem, 3vw, 2.35rem);
  font-weight: 820;
  line-height: 1.12;
}}

.runtime-copy {{
  max-width: 900px;
  color: #62758A;
  font-size: 0.96rem;
  line-height: 1.65;
}}

/* Sidebar and navigation */
[data-testid="stSidebar"] {{
  background:
    linear-gradient(180deg, #F7FAFC 0%, #F2F7FA 54%, #EEF5F8 100%);
  border-right: 1px solid #DCE7EE;
}}

[data-testid="stSidebar"][aria-expanded="true"] {{
  min-width: 17.25rem !important;
  max-width: 17.25rem !important;
}}

[data-testid="stSidebarContent"] {{
  padding: 0.62rem 0.72rem 0 !important;
  min-height: 100vh;
  overflow-x: hidden;
  scrollbar-width: thin;
  scrollbar-color: #B6C7D6 transparent;
}}

[data-testid="stSidebarHeader"] {{
  display: none !important;
}}

[data-testid="stSidebarUserContent"] {{
  padding: 0 0 1rem !important;
  margin: 0 !important;
}}

[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{
  gap: 0 !important;
}}

/* The custom grid-aligned control proxies Streamlit's native controls. */
[data-testid="stToolbar"],
button[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {{
  display: none !important;
}}

button:focus-visible, a:focus-visible, input:focus-visible {{
  outline: 3px solid rgba(6, 182, 212, 0.62) !important;
  outline-offset: 2px !important;
}}

[data-testid="stSidebarNav"] {{
  display: none !important;
}}

.aoi-sidebar-head {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 2.55rem;
  gap: 0.55rem;
  align-items: center;
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  margin: 0;
  padding: 0;
}}

[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] >
[data-testid="stElementContainer"]:has(.aoi-sidebar-head) {{
  margin-bottom: 1rem !important;
}}

[data-testid="stMarkdownContainer"]:has(.aoi-sidebar-head) {{
  margin-bottom: 0 !important;
}}

.aoi-sidebar-brand {{
  display: flex;
  align-items: center;
  gap: 0.60rem;
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  height: 3.48rem;
  margin: 0;
  padding: 0.42rem 0.54rem;
  border: 1px solid rgba(125, 211, 252, 0.20);
  border-radius: 0.70rem;
  background:
    linear-gradient(125deg, #143852 0%, #15566A 62%, #176D78 100%);
  box-shadow: 0 3px 9px rgba(15, 39, 71, 0.085);
  overflow: hidden;
  transition: border-color 180ms ease, box-shadow 180ms ease;
}}

.aoi-sidebar-brand:hover {{
  border-color: rgba(165, 243, 252, 0.32);
  box-shadow: 0 4px 11px rgba(15, 39, 71, 0.105);
}}

.aoi-sidebar-brand-mark {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 2.26rem;
  height: 2.26rem;
  border: 1px solid rgba(224, 242, 254, 0.22);
  border-radius: 0.60rem;
  background: rgba(9, 35, 59, 0.56);
  color: #FFFFFF;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12);
  font-size: 0.73rem;
  font-weight: 820;
  letter-spacing: 0.04em;
}}

.aoi-sidebar-brand-title {{
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  color: #FFFFFF;
  font-size: 0.92rem;
  font-weight: 790;
  line-height: 1;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.aoi-sidebar-toggle {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  align-self: center;
  box-sizing: border-box;
  width: 2.55rem;
  height: 2.55rem;
  min-width: 2.55rem;
  padding: 0;
  border: 1px solid #C7D9E4;
  border-radius: 0.70rem;
  background: #F9FCFD;
  color: #496578;
  box-shadow: 0 2px 7px rgba(15, 39, 71, 0.05);
  cursor: pointer;
  transition: background 170ms ease, border-color 170ms ease, color 170ms ease;
}}

.aoi-sidebar-toggle:hover {{
  border-color: #9FC2D3;
  background: #EEF7FA;
  color: #155E75;
}}

.aoi-sidebar-toggle:active {{
  border-color: #7EADBF;
  background: #E3F1F5;
  color: #0E5269;
  box-shadow: inset 0 1px 2px rgba(15, 39, 71, 0.08);
}}

.aoi-sidebar-toggle:focus-visible {{
  outline: 3px solid rgba(6, 182, 212, 0.48);
  outline-offset: 2px;
}}

.aoi-sidebar-toggle:disabled {{
  border-color: #D9E4EA;
  background: #F3F6F8;
  color: #9AAAB6;
  box-shadow: none;
  cursor: not-allowed;
}}

.aoi-sidebar-toggle-icon {{
  display: block;
  font-family: "Segoe UI Symbol", "Segoe UI", sans-serif;
  font-size: 1.20rem;
  font-weight: 760;
  line-height: 1;
  letter-spacing: -0.16em;
  transform: translateX(-0.05rem);
}}

[data-testid="stSidebar"][aria-expanded="false"] .aoi-sidebar-head {{
  position: fixed;
  top: 0.72rem;
  left: 0.72rem;
  z-index: 999997;
  grid-template-columns: 2.70rem 2.70rem;
  gap: 0.44rem;
  width: auto;
  margin: 0;
  pointer-events: auto;
}}

[data-testid="stSidebar"][aria-expanded="false"] .aoi-sidebar-brand {{
  width: 2.70rem;
  height: 2.70rem;
  margin: 0;
  padding: 0.28rem;
  justify-content: center;
  pointer-events: auto;
}}

[data-testid="stSidebar"][aria-expanded="false"] .aoi-sidebar-brand-mark {{
  width: 2.04rem;
  height: 2.04rem;
  border-radius: 0.52rem;
}}

[data-testid="stSidebar"][aria-expanded="false"] .aoi-sidebar-toggle {{
  width: 2.70rem;
  height: 2.70rem;
  min-width: 2.70rem;
}}

[data-testid="stSidebar"][aria-expanded="false"] .aoi-sidebar-brand-title {{
  display: none;
}}

[data-testid="stSidebar"][aria-expanded="false"] {{
  position: fixed !important;
  inset: 0 auto auto 0 !important;
  min-width: 0 !important;
  max-width: 0 !important;
  width: 0 !important;
  transform: none !important;
  overflow: visible !important;
}}

[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {{
  width: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}}

[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"],
[data-testid="stSidebar"][aria-expanded="false"] .st-key-aoi_nav_scroll {{
  display: none !important;
}}

.st-key-aoi_nav_scroll {{
  max-height: calc(100vh - 5.9rem);
  min-height: 10rem;
  margin: 0;
  padding: 0.04rem 0 0.24rem;
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  scrollbar-color: #B6C7D6 transparent;
}}

.aoi-current-route-marker {{
  display: none;
}}

.st-key-aoi_nav_scroll [data-testid="stElementContainer"]:has(.aoi-current-route-marker) {{
  display: none !important;
}}

.st-key-aoi_nav_scroll::-webkit-scrollbar {{
  width: 0.34rem;
}}

.st-key-aoi_nav_scroll::-webkit-scrollbar-thumb {{
  border-radius: 999px;
  background: #B6C7D6;
}}

.st-key-aoi_nav_scroll[data-testid="stVerticalBlock"] {{
  gap: 0.38rem !important;
}}

.st-key-aoi_nav_group_overview,
.st-key-aoi_nav_group_inspection,
.st-key-aoi_nav_group_optimization,
.st-key-aoi_nav_group_analysis {{
  gap: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 1px solid rgba(179, 199, 213, 0.50) !important;
  border-radius: 0.74rem !important;
  background: rgba(255, 255, 255, 0.58) !important;
  box-shadow: none !important;
  overflow: hidden;
  transition: border-color 180ms ease, background 180ms ease, box-shadow 180ms ease;
}}

.st-key-aoi_nav_group_overview:has(.aoi-nav-group-open),
.st-key-aoi_nav_group_inspection:has(.aoi-nav-group-open),
.st-key-aoi_nav_group_optimization:has(.aoi-nav-group-open),
.st-key-aoi_nav_group_analysis:has(.aoi-nav-group-open) {{
  border-color: rgba(8, 145, 178, 0.22) !important;
  background: rgba(245, 251, 253, 0.92) !important;
  box-shadow: 0 3px 10px rgba(15, 63, 86, 0.055) !important;
}}

.st-key-aoi_nav_group_overview > [data-testid="stElementContainer"]:has(.aoi-nav-group-open),
.st-key-aoi_nav_group_inspection > [data-testid="stElementContainer"]:has(.aoi-nav-group-open),
.st-key-aoi_nav_group_optimization > [data-testid="stElementContainer"]:has(.aoi-nav-group-open),
.st-key-aoi_nav_group_analysis > [data-testid="stElementContainer"]:has(.aoi-nav-group-open),
.st-key-aoi_nav_group_overview > [data-testid="stElementContainer"]:has(.aoi-nav-group-closed),
.st-key-aoi_nav_group_inspection > [data-testid="stElementContainer"]:has(.aoi-nav-group-closed),
.st-key-aoi_nav_group_optimization > [data-testid="stElementContainer"]:has(.aoi-nav-group-closed),
.st-key-aoi_nav_group_analysis > [data-testid="stElementContainer"]:has(.aoi-nav-group-closed) {{
  display: none !important;
}}

.aoi-nav-group-open,
.aoi-nav-group-closed {{ display: none; }}

.aoi-nav-group-toggle {{
  display: flex;
  align-items: center;
  gap: 0.58rem;
  justify-content: flex-start;
  appearance: none;
  width: 100%;
  min-height: 2.72rem;
  padding: 0.54rem 0.64rem !important;
  border: 0 !important;
  border-radius: 0.70rem;
  background: transparent;
  box-shadow: none !important;
  color: #28445D;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.94rem;
  font-weight: 760;
  line-height: 1.2;
  text-align: left;
  white-space: nowrap;
  transition: background 180ms ease, color 180ms ease, box-shadow 180ms ease;
}}

.aoi-nav-group-icon,
.aoi-nav-group-arrow {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 1.18rem;
  height: 1.18rem;
  color: currentColor;
}}

.aoi-nav-group-icon svg,
.aoi-nav-group-arrow svg {{
  display: block;
  width: 100%;
  height: 100%;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}}

.aoi-nav-group-label {{
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.aoi-nav-group-arrow {{
  margin-left: auto;
  color: #5B7890;
  width: 1rem;
  height: 1rem;
  transform: rotate(-90deg);
  transform-origin: center;
  transition: transform 210ms ease, color 180ms ease;
}}

.st-key-aoi_nav_group_overview:has(.aoi-nav-group-open) .aoi-nav-group-arrow,
.st-key-aoi_nav_group_inspection:has(.aoi-nav-group-open) .aoi-nav-group-arrow,
.st-key-aoi_nav_group_optimization:has(.aoi-nav-group-open) .aoi-nav-group-arrow,
.st-key-aoi_nav_group_analysis:has(.aoi-nav-group-open) .aoi-nav-group-arrow {{
  color: #087B9A;
  transform: rotate(0deg);
}}

.aoi-nav-group-toggle:hover {{
  background: rgba(8, 145, 178, 0.075);
  color: #0C4A6E;
}}

.aoi-nav-group-toggle:focus-visible {{
  outline: 2px solid rgba(8, 145, 178, 0.48);
  outline-offset: -2px;
}}

.st-key-aoi_nav_group_overview:has(.aoi-nav-group-open) .aoi-nav-group-toggle,
.st-key-aoi_nav_group_inspection:has(.aoi-nav-group-open) .aoi-nav-group-toggle,
.st-key-aoi_nav_group_optimization:has(.aoi-nav-group-open) .aoi-nav-group-toggle,
.st-key-aoi_nav_group_analysis:has(.aoi-nav-group-open) .aoi-nav-group-toggle {{
  border-radius: 0.70rem 0.70rem 0.38rem 0.38rem;
  background: linear-gradient(90deg, rgba(224, 242, 254, 0.66), rgba(240, 249, 255, 0.46));
  color: #075985;
  box-shadow: none !important;
}}

.st-key-aoi_nav_group_body_overview,
.st-key-aoi_nav_group_body_inspection,
.st-key-aoi_nav_group_body_optimization,
.st-key-aoi_nav_group_body_analysis {{
  /* Streamlit's vertical blocks otherwise compress adjacent links at render time. */
  gap: 0.38rem !important;
  max-height: 14rem;
  padding: 0.34rem 0.34rem 0.42rem !important;
  border: 0 !important;
  background: transparent !important;
  opacity: 1;
  overflow: hidden;
  pointer-events: auto;
  transition: max-height 210ms ease, opacity 170ms ease, padding 210ms ease;
}}

.st-key-aoi_nav_group_overview:has(.aoi-nav-group-closed) .st-key-aoi_nav_group_body_overview,
.st-key-aoi_nav_group_inspection:has(.aoi-nav-group-closed) .st-key-aoi_nav_group_body_inspection,
.st-key-aoi_nav_group_optimization:has(.aoi-nav-group-closed) .st-key-aoi_nav_group_body_optimization,
.st-key-aoi_nav_group_analysis:has(.aoi-nav-group-closed) .st-key-aoi_nav_group_body_analysis {{
  max-height: 0;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  opacity: 0;
  pointer-events: none;
}}

/*
 * Streamlit adds margin:-6px 0 to the element wrapper around each widget.
 * Page links then overflow into the next item (about 8px at the current
 * scale), making the active and hover backgrounds overlap.  Only the formal
 * sidebar navigation wrappers are normalized here; other Streamlit widgets
 * retain their native spacing.
 */
[data-testid="stSidebar"] .st-key-aoi_nav_group_body_overview [data-testid="stElementContainer"]:has(> .stPageLink),
[data-testid="stSidebar"] .st-key-aoi_nav_group_body_inspection [data-testid="stElementContainer"]:has(> .stPageLink),
[data-testid="stSidebar"] .st-key-aoi_nav_group_body_optimization [data-testid="stElementContainer"]:has(> .stPageLink),
[data-testid="stSidebar"] .st-key-aoi_nav_group_body_analysis [data-testid="stElementContainer"]:has(> .stPageLink) {{
  margin: 0 !important;
}}

[data-testid="stSidebar"] [data-testid="stPageLink"] {{
  margin: 0;
}}

[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
  position: relative;
  box-sizing: border-box;
  width: 100%;
  height: 2.48rem;
  min-height: 2.48rem;
  margin: 0;
  padding: 0.57rem 0.64rem;
  border: 1px solid transparent;
  border-radius: 0.62rem;
  box-shadow: inset 3px 0 0 transparent;
  color: #334155;
  font-size: 0.93rem;
  font-weight: 640;
  line-height: 1.2;
  white-space: nowrap;
  transition: background 180ms ease, border-color 180ms ease, color 180ms ease;
}}

[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
  border-color: transparent;
  background: rgba(8, 145, 178, 0.09);
  color: #0F4C5C;
  transform: none;
}}

[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="overview"]) .st-key-aoi_nav_item_overview [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="single"]) .st-key-aoi_nav_item_single [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="batch"]) .st-key-aoi_nav_item_batch [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="video"]) .st-key-aoi_nav_item_video [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="feedback_optimization"]) .st-key-aoi_nav_item_feedback_optimization [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="domain_adaptation"]) .st-key-aoi_nav_item_domain_adaptation [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="status"]) .st-key-aoi_nav_item_status [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="runtime_config"]) .st-key-aoi_nav_item_runtime_config [data-testid="stPageLink"] a,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="statistics"]) .st-key-aoi_nav_item_statistics [data-testid="stPageLink"] a {{
  border-color: transparent;
  background: linear-gradient(90deg, #DDF3FC 0%, #EAF7FB 100%);
  color: #075985;
  font-weight: 730;
  box-shadow: inset 3px 0 0 #0891B2, 0 1px 3px rgba(15, 63, 86, 0.04);
}}

[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="overview"]) .st-key-aoi_nav_item_overview [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="single"]) .st-key-aoi_nav_item_single [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="batch"]) .st-key-aoi_nav_item_batch [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="video"]) .st-key-aoi_nav_item_video [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="feedback_optimization"]) .st-key-aoi_nav_item_feedback_optimization [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="domain_adaptation"]) .st-key-aoi_nav_item_domain_adaptation [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="status"]) .st-key-aoi_nav_item_status [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="runtime_config"]) .st-key-aoi_nav_item_runtime_config [data-testid="stPageLink"] a svg,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-page="statistics"]) .st-key-aoi_nav_item_statistics [data-testid="stPageLink"] a svg {{
  color: #0369A1;
}}

[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-group="overview"]) .st-key-aoi_nav_group_overview [data-testid="stButton"] button,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-group="inspection"]) .st-key-aoi_nav_group_inspection [data-testid="stButton"] button,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-group="optimization"]) .st-key-aoi_nav_group_optimization [data-testid="stButton"] button,
[data-testid="stSidebar"] .st-key-aoi_nav_scroll:has(.aoi-current-route-marker[data-aoi-current-group="analysis"]) .st-key-aoi_nav_group_analysis [data-testid="stButton"] button {{
  color: #075985;
  box-shadow: inset 2px 0 0 rgba(8, 145, 178, 0.58) !important;
}}

[data-testid="stSidebar"] [data-testid="stPageLink"] a svg {{
  width: 1.12rem !important;
  height: 1.12rem !important;
}}

.aoi-sidebar-runtime-heading {{
  margin: 0.28rem 0 0.30rem;
  padding: 0.78rem 0.14rem 0;
  border-top: 1px solid #D9E5ED;
  color: #29485F;
  font-size: 0.84rem;
  font-weight: 780;
}}

.aoi-sidebar-runtime-copy {{
  margin: 0 0.14rem 0.48rem;
  color: #718096;
  font-size: 0.69rem;
  line-height: 1.35;
}}

[data-testid="stSidebar"] label {{
  color: #334155 !important;
  font-size: 0.78rem !important;
  font-weight: 680 !important;
}}

[data-testid="stSidebar"] [data-testid="stSelectbox"],
[data-testid="stSidebar"] [data-testid="stTextInput"] {{
  margin: 0.08rem 0;
}}

[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] > div > div {{
  min-height: 2.32rem;
  border-radius: 0.58rem;
}}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
  min-width: 0 !important;
  max-width: 100% !important;
}}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span {{
  display: block;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

@media (prefers-reduced-motion: reduce) {{
  [data-testid="stSidebar"] [data-testid="stPageLink"] a,
  .st-key-aoi_nav_group_overview [data-testid="stButton"] button,
  .st-key-aoi_nav_group_inspection [data-testid="stButton"] button,
  .st-key-aoi_nav_group_optimization [data-testid="stButton"] button,
  .st-key-aoi_nav_group_analysis [data-testid="stButton"] button,
  .aoi-sidebar-brand,
  .aoi-sidebar-toggle,
  [data-testid="stMain"].aoi-main-route-transition,
  .aoi-card,
  .aoi-panel,
  .aoi-kpi-card {{
    animation: none !important;
    transition: none !important;
  }}
}}

@media (max-width: 860px) {{
  [data-testid="stSidebar"][aria-expanded="true"] {{
    min-width: min(86vw, 17.25rem) !important;
    max-width: min(86vw, 17.25rem) !important;
  }}
}}

/* Cards and metrics */
div[data-testid="stMetric"] {{
  background: var(--aoi-surface);
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  padding: 1rem 1.05rem;
  box-shadow: var(--aoi-shadow-card);
  min-height: 6.1rem;
}}

div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
  color: var(--aoi-muted) !important;
  font-size: 0.78rem !important;
  font-weight: 650 !important;
}}

div[data-testid="stMetricValue"] {{
  color: var(--aoi-primary);
  font-size: 1.65rem !important;
  font-weight: 740 !important;
}}

.aoi-card {{
  background: var(--aoi-surface);
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  box-shadow: var(--aoi-shadow-card);
  padding: 1.15rem;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}}

.aoi-card:hover,
.aoi-panel:hover,
.aoi-kpi-card:hover,
.aoi-command-card:hover,
.aoi-score-card:hover,
.aoi-flow-step:hover {{
  border-color: rgba(6, 182, 212, 0.28);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}}

.aoi-muted {{
  color: var(--aoi-muted);
  font-size: 0.86rem;
}}

.aoi-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
  border: 1px solid var(--aoi-border);
  background: #F8FAFC;
  color: #334155;
}}

.aoi-badge-success {{
  color: #166534;
  background: #DCFCE7;
  border-color: #BBF7D0;
}}

.aoi-badge-warning {{
  color: #92400E;
  background: #FEF3C7;
  border-color: #FDE68A;
}}

.aoi-badge-danger {{
  color: #991B1B;
  background: #FEE2E2;
  border-color: #FECACA;
}}

/* Buttons */
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{
  border-radius: var(--aoi-radius-md) !important;
  min-height: 2.4rem;
  font-weight: 680 !important;
  letter-spacing: 0;
  border: 1px solid var(--aoi-border) !important;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}}

[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, var(--aoi-accent) 0%, var(--aoi-accent-dark) 100%) !important;
  border-color: rgba(8, 145, 178, 0.40) !important;
  color: #FFFFFF !important;
}}

[data-testid="stBaseButton-primary"]:hover {{
  filter: brightness(0.98);
  box-shadow: 0 8px 18px rgba(6, 182, 212, 0.18);
}}

.stButton > button:disabled {{
  background: #F1F5F9 !important;
  color: #94A3B8 !important;
  border-color: #E2E8F0 !important;
  box-shadow: none;
}}

/* Detection actions must keep their four-character labels intact even when
 * the surrounding workbench is rendered at a compact width. */
.st-key-single_start_detection button,
.st-key-batch_start_detection button {{
  white-space: nowrap;
}}

.st-key-single_start_detection button p,
.st-key-batch_start_detection button p {{
  flex: 0 0 auto;
  margin: 0 !important;
  white-space: nowrap;
  word-break: keep-all;
  overflow-wrap: normal;
}}

/* Inputs and uploaders */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stFileUploader"] section {{
  border-radius: var(--aoi-radius-md) !important;
  border-color: var(--aoi-border) !important;
  background: #FFFFFF !important;
}}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {{
  border-color: var(--aoi-accent) !important;
  box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.12) !important;
}}

.st-key-batch_input_dir input {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.st-key-batch_choose_folder button {{
  min-height: 2.52rem !important;
  padding-inline: 0.62rem !important;
  border-color: rgba(8, 145, 178, 0.30) !important;
  background: linear-gradient(180deg, #FFFFFF 0%, #F0F9FF 100%) !important;
  color: #075985 !important;
  gap: 0.36rem !important;
  white-space: nowrap;
  transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease, transform 100ms ease;
}}

.st-key-batch_choose_folder button p {{
  flex: 0 0 auto;
  margin: 0 !important;
  overflow: visible;
  font-size: 0.92rem;
  white-space: nowrap;
  word-break: keep-all;
}}

.st-key-batch_choose_folder button [data-testid="stIconMaterial"] {{
  flex: 0 0 auto;
}}

.st-key-batch_choose_folder button:hover:not(:disabled) {{
  border-color: rgba(8, 145, 178, 0.58) !important;
  background: #E0F2FE !important;
  box-shadow: 0 7px 16px rgba(8, 145, 178, 0.13);
}}

.st-key-batch_choose_folder button:active:not(:disabled) {{
  transform: translateY(1px);
  background: #CFF0FB !important;
  box-shadow: 0 2px 6px rgba(8, 145, 178, 0.12);
}}

.st-key-batch_choose_folder button:focus-visible {{
  outline: 3px solid rgba(6, 182, 212, 0.48) !important;
  outline-offset: 2px !important;
}}

.st-key-batch_choose_folder button:disabled {{
  border-color: #D7E2EA !important;
  background: #F1F5F9 !important;
  color: #94A3B8 !important;
  box-shadow: none !important;
  cursor: not-allowed;
}}

@media (max-width: 900px) {{
  .st-key-batch_folder_picker_row [data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap;
  }}

  .st-key-batch_folder_picker_row [data-testid="stColumn"] {{
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }}
}}

.st-key-batch_folder_picker_row {{
  container-type: inline-size;
}}

@container (max-width: 27rem) {{
  .st-key-batch_folder_picker_row [data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap;
  }}

  .st-key-batch_folder_picker_row [data-testid="stColumn"] {{
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }}
}}

[data-testid="stFileUploader"] section {{
  background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
  border-style: dashed !important;
}}

/* Tables and dataframes */
table {{
  border-collapse: separate !important;
  border-spacing: 0 !important;
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  overflow: hidden;
  background: var(--aoi-surface);
}}

thead tr th {{
  background: #F1F5F9 !important;
  color: #334155 !important;
  font-size: 0.80rem !important;
  font-weight: 700 !important;
}}

tbody tr td {{
  border-color: #EEF2F7 !important;
  font-size: 0.84rem !important;
}}

[data-testid="stDataFrame"] {{
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  overflow: hidden;
  box-shadow: var(--aoi-shadow-card);
}}

/* Alerts and expanders */
[data-testid="stAlert"] {{
  border-radius: var(--aoi-radius-lg);
  border: 1px solid var(--aoi-border);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
}}

[data-testid="stExpander"] {{
  border: 1px solid var(--aoi-border) !important;
  border-radius: var(--aoi-radius-lg) !important;
  background: var(--aoi-surface) !important;
  box-shadow: var(--aoi-shadow-card);
}}

/* Code, logs, and images */
code, pre {{
  border-radius: var(--aoi-radius-md) !important;
  background: #F1F5F9 !important;
  color: #334155 !important;
  font-size: 0.78rem !important;
}}

textarea {{
  border-radius: var(--aoi-radius-md) !important;
  border-color: var(--aoi-border) !important;
  font-family: "Cascadia Mono", Consolas, monospace !important;
  font-size: 0.78rem !important;
}}

[data-testid="stImage"] img {{
  display: block;
  width: auto !important;
  max-width: 100% !important;
  height: auto !important;
  margin-inline: auto;
  object-fit: contain;
  border-radius: var(--aoi-radius-md);
  border: 1px solid var(--aoi-border);
  box-shadow: var(--aoi-shadow-card);
}}

[data-testid="stImage"],
[data-testid="stVideo"],
[data-testid="stVideo"] > div {{
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  overflow: hidden;
}}

[data-testid="stVideo"] video {{
  display: block;
  width: 100% !important;
  max-width: 100% !important;
  height: auto !important;
  object-fit: contain;
  border-radius: var(--aoi-radius-md);
  background: #0F172A;
}}

.js-plotly-plot,
.plot-container,
.svg-container,
[data-testid="stVegaLiteChart"] {{
  max-width: 100% !important;
  min-width: 0 !important;
}}

.aoi-home-hero {{
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(240px, 0.8fr);
  gap: 1.2rem;
  align-items: stretch;
  margin: 0 0 1.15rem 0;
  padding: 1.35rem;
  border: 1px solid rgba(14, 116, 144, 0.16);
  border-radius: 16px;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 41, 59, 0.94) 56%, rgba(14, 116, 144, 0.92) 100%);
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.16);
  overflow: hidden;
}}

.aoi-home-hero::after {{
  content: "";
  position: absolute;
  inset: auto 1rem -4rem auto;
  width: 12rem;
  height: 12rem;
  border: 1px solid rgba(125, 211, 252, 0.28);
  border-radius: 50%;
  opacity: 0.62;
  pointer-events: none;
}}

.aoi-hero-content,
.aoi-hero-panel {{
  position: relative;
  z-index: 1;
  min-width: 0;
}}

.aoi-hero-kicker {{
  color: #67E8F9;
  font-size: 0.78rem;
  font-weight: 760;
  letter-spacing: 0;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
}}

.aoi-hero-title {{
  color: #FFFFFF;
  font-size: 1.96rem;
  line-height: 1.16;
  font-weight: 790;
  margin-bottom: 0.7rem;
  overflow-wrap: anywhere;
}}

.aoi-hero-copy {{
  color: rgba(226, 232, 240, 0.84);
  font-size: 0.96rem;
  line-height: 1.72;
  max-width: 44rem;
}}

.aoi-hero-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 1rem;
}}

.aoi-hero-chip {{
  display: inline-flex;
  align-items: center;
  min-height: 1.8rem;
  padding: 0.32rem 0.68rem;
  border: 1px solid rgba(148, 163, 184, 0.30);
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.28);
  color: rgba(248, 250, 252, 0.92);
  font-size: 0.78rem;
  font-weight: 680;
  max-width: 100%;
}}

.aoi-hero-panel {{
  padding: 1rem;
  border: 1px solid rgba(226, 232, 240, 0.18);
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.42);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
}}

.aoi-hero-panel-title {{
  color: rgba(248, 250, 252, 0.76);
  font-size: 0.78rem;
  font-weight: 720;
  margin-bottom: 0.8rem;
}}

.aoi-health-score {{
  color: #FFFFFF;
  font-size: 2rem;
  line-height: 1.05;
  font-weight: 790;
}}

.aoi-health-caption {{
  color: rgba(226, 232, 240, 0.72);
  font-size: 0.78rem;
  margin: 0.25rem 0 0.85rem 0;
}}

.aoi-status-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.42rem;
  padding: 0.32rem 0.62rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 740;
  border: 1px solid rgba(148, 163, 184, 0.32);
  color: #E2E8F0;
  background: rgba(15, 23, 42, 0.34);
}}

.aoi-status-dot {{
  width: 0.48rem;
  height: 0.48rem;
  border-radius: 999px;
  background: #94A3B8;
}}

.aoi-status-ok .aoi-status-dot {{
  background: #22C55E;
  box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.14);
}}

.aoi-status-warn .aoi-status-dot {{
  background: #F59E0B;
  box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.14);
}}

.aoi-section-header {{
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin: 1.35rem 0 0.7rem 0;
}}

.aoi-section-title {{
  color: var(--aoi-primary);
  font-size: 1.08rem;
  font-weight: 760;
  line-height: 1.25;
}}

.aoi-section-desc {{
  color: var(--aoi-muted);
  font-size: 0.82rem;
  margin-top: 0.2rem;
}}

.aoi-kpi-card {{
  min-height: 8.15rem;
  padding: 1.05rem;
  border: 1px solid rgba(203, 213, 225, 0.84);
  border-radius: var(--aoi-radius-lg);
  background:
    linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
  box-shadow: var(--aoi-shadow-card);
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}}

.aoi-kpi-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}}

.aoi-kpi-icon {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border-radius: var(--aoi-radius-md);
  background: #E0F2FE;
  color: #0369A1;
  font-size: 0.82rem;
  font-weight: 800;
}}

.aoi-kpi-label {{
  color: var(--aoi-muted);
  font-size: 0.78rem;
  font-weight: 710;
}}

.aoi-kpi-value {{
  color: var(--aoi-primary);
  font-size: 1.76rem;
  line-height: 1.08;
  font-weight: 790;
  white-space: nowrap;
}}

.aoi-kpi-foot {{
  color: var(--aoi-muted);
  font-size: 0.76rem;
  margin-top: 0.45rem;
  line-height: 1.42;
}}

.aoi-panel {{
  padding: 1.1rem;
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  background: var(--aoi-surface);
  box-shadow: var(--aoi-shadow-card);
  transition: border-color 160ms ease, box-shadow 160ms ease;
}}

.aoi-panel-title {{
  color: var(--aoi-primary);
  font-size: 0.96rem;
  font-weight: 760;
  margin-bottom: 0.25rem;
}}

.aoi-panel-subtitle {{
  color: var(--aoi-muted);
  font-size: 0.78rem;
  margin-bottom: 0.9rem;
}}

.aoi-mini-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.62rem 0;
  border-bottom: 1px solid #EEF2F7;
}}

.aoi-mini-row:last-child {{
  border-bottom: 0;
}}

.aoi-mini-label {{
  color: #334155;
  font-size: 0.84rem;
  font-weight: 670;
}}

.aoi-mini-value {{
  color: var(--aoi-primary);
  font-size: 0.84rem;
  font-weight: 720;
  text-align: right;
  word-break: break-word;
}}

.aoi-progress-row {{
  margin: 0.72rem 0;
}}

.aoi-progress-meta {{
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
  color: #334155;
  font-size: 0.82rem;
  font-weight: 660;
  margin-bottom: 0.36rem;
}}

.aoi-progress-track {{
  height: 0.52rem;
  border-radius: 999px;
  background: #E2E8F0;
  overflow: hidden;
}}

.aoi-progress-fill {{
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--aoi-accent), #2563EB);
}}

.aoi-empty-state {{
  padding: 1.1rem;
  border: 1px dashed #CBD5E1;
  border-radius: var(--aoi-radius-lg);
  background: #F8FAFC;
  color: var(--aoi-muted);
  font-size: 0.84rem;
  line-height: 1.6;
}}

.aoi-detection-card {{
  padding: 1.1rem;
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
  box-shadow: var(--aoi-shadow-card);
}}

.aoi-decision-normal {{
  color: #166534;
  background: #DCFCE7;
  border-color: #BBF7D0;
}}

.aoi-decision-defect {{
  color: #991B1B;
  background: #FEE2E2;
  border-color: #FECACA;
}}

.aoi-path-line {{
  color: #475569;
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 0.75rem;
  line-height: 1.52;
  word-break: break-all;
}}

.aoi-flow {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.8rem;
}}

.aoi-flow-step {{
  padding: 0.95rem;
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  background: #FFFFFF;
  box-shadow: var(--aoi-shadow-card);
}}

.aoi-flow-index {{
  color: var(--aoi-accent-dark);
  font-size: 0.76rem;
  font-weight: 800;
  margin-bottom: 0.42rem;
}}

.aoi-flow-title {{
  color: var(--aoi-primary);
  font-size: 0.9rem;
  font-weight: 760;
}}

.aoi-flow-desc {{
  color: var(--aoi-muted);
  font-size: 0.76rem;
  line-height: 1.5;
  margin-top: 0.35rem;
}}

.aoi-workbench-hero {{
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin: 0 0 1rem 0;
  padding: 1.15rem 1.25rem;
  border: 1px solid rgba(14, 116, 144, 0.14);
  border-radius: 16px;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(241, 245, 249, 0.96) 54%, rgba(224, 242, 254, 0.92) 100%);
  box-shadow: var(--aoi-shadow-card);
}}

.aoi-workbench-kicker {{
  color: var(--aoi-accent-dark);
  font-size: 0.76rem;
  font-weight: 800;
  text-transform: uppercase;
  margin-bottom: 0.45rem;
}}

.aoi-workbench-title {{
  color: var(--aoi-primary);
  font-size: 1.82rem;
  line-height: 1.18;
  font-weight: 790;
}}

.aoi-workbench-copy {{
  color: var(--aoi-muted);
  font-size: 0.88rem;
  line-height: 1.62;
  max-width: 48rem;
  margin-top: 0.52rem;
}}

.aoi-workbench-meta {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.45rem;
  min-width: 18rem;
}}

.aoi-command-card {{
  padding: 1.1rem;
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  background: #FFFFFF;
  box-shadow: var(--aoi-shadow-card);
  transition: border-color 160ms ease, box-shadow 160ms ease;
}}

.aoi-command-title {{
  color: var(--aoi-primary);
  font-size: 1rem;
  font-weight: 760;
  margin-bottom: 0.22rem;
}}

.aoi-command-desc {{
  color: var(--aoi-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  margin-bottom: 0.85rem;
}}

.aoi-verdict-card {{
  min-height: 12rem;
  padding: 1.15rem;
  border: 1px solid var(--aoi-border);
  border-radius: 16px;
  background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
  box-shadow: var(--aoi-shadow-card);
}}

.aoi-verdict-card-danger {{
  border-color: rgba(239, 68, 68, 0.26);
  background:
    linear-gradient(180deg, #FFFFFF 0%, #FFF7F7 100%);
}}

.aoi-verdict-card-normal {{
  border-color: rgba(34, 197, 94, 0.24);
  background:
    linear-gradient(180deg, #FFFFFF 0%, #F0FDF4 100%);
}}

.aoi-verdict-label {{
  color: var(--aoi-muted);
  font-size: 0.78rem;
  font-weight: 760;
  margin-bottom: 0.62rem;
}}

.aoi-verdict-value {{
  color: var(--aoi-primary);
  font-size: 2.15rem;
  line-height: 1.05;
  font-weight: 800;
  margin-bottom: 0.6rem;
}}

.aoi-verdict-card-danger .aoi-verdict-value {{
  color: #991B1B;
}}

.aoi-verdict-card-normal .aoi-verdict-value {{
  color: #166534;
}}

.aoi-verdict-copy {{
  color: #475569;
  font-size: 0.82rem;
  line-height: 1.55;
}}

.aoi-score-strip {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
}}

.aoi-score-card {{
  padding: 0.92rem;
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  background: #FFFFFF;
  box-shadow: var(--aoi-shadow-card);
  transition: border-color 160ms ease, box-shadow 160ms ease;
}}

.aoi-score-label {{
  color: var(--aoi-muted);
  font-size: 0.76rem;
  font-weight: 730;
}}

.aoi-score-value {{
  color: var(--aoi-primary);
  font-size: 1.36rem;
  font-weight: 790;
  line-height: 1.2;
  margin-top: 0.32rem;
}}

.aoi-score-foot {{
  color: var(--aoi-muted);
  font-size: 0.72rem;
  margin-top: 0.28rem;
}}

.aoi-image-tile-title {{
  color: var(--aoi-primary);
  font-size: 0.88rem;
  font-weight: 760;
  margin: 0.3rem 0 0.45rem 0;
}}

.aoi-image-tile-note {{
  color: var(--aoi-muted);
  font-size: 0.74rem;
  line-height: 1.48;
  margin: -0.15rem 0 0.55rem 0;
}}

.aoi-compare-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.8rem;
}}

.aoi-result-toolbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin: 1.1rem 0 0.65rem 0;
}}

.aoi-table-panel {{
  padding: 1rem;
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  background: #FFFFFF;
  box-shadow: var(--aoi-shadow-card);
}}

.aoi-batch-summary {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.8rem;
  margin: 1rem 0;
}}

.aoi-status-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.8rem;
}}

.aoi-log-box {{
  border: 1px solid var(--aoi-border);
  border-radius: var(--aoi-radius-lg);
  background: #0F172A;
  box-shadow: var(--aoi-shadow-card);
  overflow: hidden;
}}

.aoi-log-box textarea {{
  background: #0F172A !important;
  color: #D1FAE5 !important;
  border: 0 !important;
}}

.aoi-inline-actions {{
  display: flex;
  align-items: center;
  gap: 0.65rem;
  flex-wrap: wrap;
  margin-top: 0.75rem;
}}

.aoi-soft-divider {{
  height: 1px;
  background: linear-gradient(90deg, transparent, #CBD5E1, transparent);
  margin: 1rem 0;
}}

button[kind="tab"],
[data-testid="stTabs"] button {{
  border-radius: var(--aoi-radius-md) !important;
  font-weight: 680 !important;
}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 0.35rem;
}}

@media (max-width: 900px) {{
  .aoi-workbench-hero {{
    flex-direction: column;
    align-items: stretch;
  }}

  .aoi-workbench-meta {{
    justify-content: flex-start;
    min-width: 0;
  }}

  .aoi-score-strip,
  .aoi-batch-summary {{
    grid-template-columns: 1fr;
  }}
}}

@media (max-width: 900px) {{
  .aoi-home-hero {{
    grid-template-columns: 1fr;
  }}

  .aoi-hero-title {{
    font-size: 1.72rem;
  }}

  .aoi-flow {{
    grid-template-columns: 1fr;
  }}
}}

hr {{
  border-color: var(--aoi-border);
}}
</style>
"""


def apply_global_styles() -> None:
    # Streamlit 1.58 renders style-only ``st.html`` in its event container,
    # so the CSS applies to the application document without a zero-height
    # iframe.  This avoids the deprecated iframe resizer whose body-level
    # MutationObserver could run after the iframe was unmounted.
    st.html(GLOBAL_CSS)


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="aoi-sidebar-head" data-testid="app-sidebar">
          <div class="aoi-sidebar-brand" title="AdaptiveAOI" aria-label="AdaptiveAOI">
            <div class="aoi-sidebar-brand-mark" title="AdaptiveAOI">AOI</div>
            <div class="aoi-sidebar-brand-title">AdaptiveAOI</div>
          </div>
          <button class="aoi-sidebar-toggle" data-aoi-sidebar-toggle type="button" aria-label="收起侧栏" title="收起侧栏">
            <span class="aoi-sidebar-toggle-icon" aria-hidden="true">‹‹</span>
          </button>
        </div>
        """,
        unsafe_allow_html=True,
    )
