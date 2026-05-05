# === DASHBOARD.PY: LAKE OKEECHOBEE LITTORAL ZONE PRODUCTIVITY MONITOR ===
#
# GitHub deployment version — uses relative paths and environment detection
#
# Inputs (from data/ folder):
#   - master_table.csv
#   - terraclimate_monthly.csv
#   - productivity_composite.csv
#   - phenology_metrics.csv
#   - productivity_monthly.csv
#   - area_productivity_summary.csv   (replaces shapefiles)
#   - productivity_2023/2024/2025.png
#   - productivity_bounds_2023/2024/2025.json
#
# Launch locally : panel serve dashboard.py --show --port 5006
# Deploy on Render: connect GitHub repo, Render reads Procfile automatically

import pandas as pd
import numpy as np
import folium
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
import panel as pn
import os
import json
import base64
import logging
import warnings
import nest_asyncio

warnings.filterwarnings('ignore')
logging.getLogger('bokeh').setLevel(logging.ERROR)
nest_asyncio.apply()

pn.extension('plotly', sizing_mode='stretch_width', notifications=True)

# ══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT-AWARE PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

LOCAL_OUTPUT_DIR = r'D:\amr\proposals\SAR-Research-Ideas\LitorialVeg\Data\Outputs'

if os.path.exists(LOCAL_OUTPUT_DIR):
    OUTPUT_DIR = LOCAL_OUTPUT_DIR
    DEPLOY_MODE = 'local'
else:
    # Server mode — data folder is relative to this file
    OUTPUT_DIR = os.path.join(os.getcwd(), 'data')
    DEPLOY_MODE = 'server'

print(f"   Mode       : {DEPLOY_MODE}")
print(f"   Output dir : {OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

YEARS      = [2023, 2024, 2025]
DEFAULT_YR = 2024

BIOMASS_OPTIONS = {
    'Conservative  (0.3 kg/m² — wetland grass)' : 0.3,
    'Moderate      (0.6 kg/m² — emergent marsh)' : 0.6,
    'Optimistic    (1.2 kg/m² — dense Cattail)'  : 1.2,
}

TC_VAR_LABELS = {
    'precip_mm'        : 'Precipitation (mm)',
    'tmax_c'           : 'Max Temperature (°C)',
    'soil_moisture_mm' : 'Soil Moisture (mm)',
    'pet_mm'           : 'PET (mm)',
    'pdsi'             : 'PDSI (Drought Index)',
}

CLASS_COLORS = [
    '#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading data...")

def load_csv(fname, label):
    path = os.path.join(OUTPUT_DIR, fname)
    try:
        df = pd.read_csv(path)
        for col in ['year', 'month']:
            if col in df.columns:
                df[col] = df[col].astype('Int64')
        print(f"   OK {label:<40} : {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"   MISSING {label:<40} : not found")
        return pd.DataFrame()

master       = load_csv('master_table.csv',           'master_table')
tc_df        = load_csv('terraclimate_monthly.csv',    'terraclimate_monthly')
prod_df      = load_csv('productivity_composite.csv',  'productivity_composite')
pheno_df     = load_csv('phenology_metrics.csv',       'phenology_metrics')
area_summary = load_csv('area_productivity_summary.csv', 'area_productivity_summary')

# Load raster PNG overlays and bounds
raster_data = {}
for yr in YEARS:
    png_path    = os.path.join(OUTPUT_DIR, f'productivity_{yr}.png')
    bounds_path = os.path.join(OUTPUT_DIR, f'productivity_bounds_{yr}.json')
    if os.path.exists(png_path) and os.path.exists(bounds_path):
        with open(bounds_path) as f:
            bounds = json.load(f)
        with open(png_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        raster_data[yr] = {'bounds': bounds, 'img_b64': img_b64}
        print(f"   OK raster_{yr:<37} : "
              f"{bounds['width_px']}x{bounds['height_px']} px")
    else:
        print(f"   MISSING raster_{yr} — run Cell 11 first")

# Class color mapping
all_classes  = sorted(master['veg_class'].dropna().unique()
                       if not master.empty else [])
class_colors = {cls: CLASS_COLORS[i % len(CLASS_COLORS)]
                for i, cls in enumerate(all_classes)}

print(f"\n   Classes : {all_classes}")
print(f"   Years   : {YEARS}")
print(f"   Default : {DEFAULT_YR}")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def find_col(df, prefix):
    """Find first column matching prefix."""
    cols = [c for c in df.columns if c.lower().startswith(prefix.lower())]
    return cols[0] if cols else None

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — RASTER IMAGE OVERLAY MAP
# ══════════════════════════════════════════════════════════════════════════════

def build_map(year, opacity=0.75):
    year = int(year)
    if year not in raster_data:
        return (f"<p style='padding:20px;color:red'>"
                f"Raster not available for {year} — run Cell 11 first.</p>")

    rd     = raster_data[year]
    bounds = rd['bounds']
    img_b64= rd['img_b64']

    center = [
        (bounds['lat_min'] + bounds['lat_max']) / 2,
        (bounds['lon_min'] + bounds['lon_max']) / 2
    ]

    m = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')

    folium.raster_layers.ImageOverlay(
        image       = f"data:image/png;base64,{img_b64}",
        bounds      = [[bounds['lat_min'], bounds['lon_min']],
                       [bounds['lat_max'], bounds['lon_max']]],
        opacity     = opacity,
        name        = f'Productivity {year}',
        interactive = False,
        cross_origin= False,
        zindex      = 1
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Legend
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:6px;
                border:1px solid #aaa;font-size:11px;line-height:1.8">
        <b>Composite Productivity Score</b><br>
        <div style="background:linear-gradient(
                    to right,#ffffcc,#c2e699,#78c679,#31a354,#006837);
                    width:150px;height:14px;margin:4px 0;
                    border-radius:3px;border:1px solid #ccc"></div>
        <div style="display:flex;justify-content:space-between;
                    width:150px;font-size:10px">
            <span>0.0 Low</span><span>1.0 High</span>
        </div><br>
        <span style="color:#888;font-size:10px">
            Transparent = No vegetation / Water
        </span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    # Mean stage annotation
    if not master.empty and 'stage_ft' in master.columns:
        stage_yr = master[master['year'] == year]['stage_ft'].mean()
        if not np.isnan(stage_yr):
            note = f"""
            <div style="position:fixed;top:10px;right:10px;z-index:1000;
                        background:rgba(255,255,255,0.9);padding:6px 12px;
                        border-radius:4px;border:1px solid #aaa;font-size:11px">
                {year} Mean Stage: <b>{stage_yr:.2f} ft</b> (L OKEE)
            </div>"""
            m.get_root().html.add_child(folium.Element(note))

    # Placeholder banner
    if year == 2023:
        banner = """
        <div style="position:fixed;top:10px;left:50%;
                    transform:translateX(-50%);z-index:1000;
                    background:#fff3cd;padding:6px 16px;
                    border:1px solid #ffc107;border-radius:4px;font-size:11px">
            2023 land cover map is flagged as a placeholder dataset
        </div>"""
        m.get_root().html.add_child(folium.Element(banner))

    return m._repr_html_()

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — MONTHLY PRODUCTIVITY + WATER STAGE TIMELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_timeline(selected_classes, index_type):
    if not selected_classes or master.empty:
        return go.Figure()

    prod_col = 'composite_productivity_norm'
    if prod_col not in master.columns:
        prod_col = ('composite_productivity'
                    if index_type == 'Optical' else 'sar_productivity')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for cls in selected_classes:
        sub = (master[master['veg_class'] == cls]
               .copy().sort_values(['year', 'month']))
        if sub.empty or prod_col not in sub.columns:
            continue
        sub['time'] = (sub['year'].astype(str) + '-' +
                       sub['month'].astype(str).str.zfill(2))
        fig.add_trace(
            go.Scatter(
                x=sub['time'], y=sub[prod_col],
                mode='lines+markers',
                name=cls,
                line=dict(color=class_colors.get(cls, '#888'), width=2),
                marker=dict(size=4),
                connectgaps=False
            ),
            secondary_y=False
        )

    if 'stage_ft' in master.columns:
        stage = (master[['year', 'month', 'stage_ft']]
                 .dropna(subset=['stage_ft'])
                 .drop_duplicates(['year', 'month'])
                 .sort_values(['year', 'month']))
        stage['time'] = (stage['year'].astype(str) + '-' +
                         stage['month'].astype(str).str.zfill(2))
        fig.add_trace(
            go.Scatter(
                x=stage['time'], y=stage['stage_ft'],
                mode='lines', name='Water Stage (ft)',
                line=dict(color='#1a9eff', width=1.5, dash='dot'),
                opacity=0.8
            ),
            secondary_y=True
        )

    fig.update_layout(
        title='Monthly Productivity + Water Stage (L OKEE)',
        template='plotly_white', height=430,
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.28, x=0),
        margin=dict(l=55, r=55, t=55, b=90)
    )
    fig.update_yaxes(title_text='Productivity Score (normalized)',
                     secondary_y=False)
    fig.update_yaxes(title_text='Water Stage (ft)', secondary_y=True)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — CLIMATE DRIVER CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_climate_scatter(cls, climate_var, season_filter):
    if pheno_df.empty or tc_df.empty:
        return go.Figure()

    pheno_cls = pheno_df[
        (pheno_df['veg_class'] == cls) &
        (pheno_df['index']     == 'NDVI')
    ][['year', 'AUC']].copy()

    tc_sub = tc_df.copy()
    if season_filter == 'Dry (Jan-Apr)':
        tc_sub = tc_sub[tc_sub['month'].isin([1, 2, 3, 4])]
    elif season_filter == 'Wet (Jun-Sep)':
        tc_sub = tc_sub[tc_sub['month'].isin([6, 7, 8, 9])]

    if climate_var not in tc_sub.columns:
        return go.Figure()

    tc_yr = (tc_sub.groupby('year')[climate_var]
             .mean().reset_index()
             .rename(columns={climate_var: 'climate_val'}))

    merged = pheno_cls.merge(tc_yr, on='year').dropna()
    fig    = go.Figure()

    if len(merged) < 2:
        fig.add_annotation(
            text="Insufficient data for correlation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=12)
        )
        return fig

    rho, pval = spearmanr(merged['AUC'], merged['climate_val'])

    fig.add_trace(go.Scatter(
        x=merged['climate_val'], y=merged['AUC'],
        mode='markers+text',
        text=merged['year'].astype(str),
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(size=12, color=class_colors.get(cls, '#888')),
        name=cls
    ))

    z    = np.polyfit(merged['climate_val'], merged['AUC'], 1)
    xfit = np.linspace(merged['climate_val'].min(),
                       merged['climate_val'].max(), 50)
    fig.add_trace(go.Scatter(
        x=xfit, y=np.polyval(z, xfit),
        mode='lines',
        line=dict(color='grey', dash='dash', width=1),
        showlegend=False
    ))

    fig.update_layout(
        title=(f'{cls} — NDVI AUC vs '
               f'{TC_VAR_LABELS.get(climate_var, climate_var)}<br>'
               f'<sup>Spearman rho = {rho:.2f}  |  p = {pval:.3f}'
               f'  |  {season_filter}</sup>'),
        xaxis_title=TC_VAR_LABELS.get(climate_var, climate_var),
        yaxis_title='NDVI AUC (Annual Integrated Greenness)',
        template='plotly_white', height=420,
        margin=dict(l=55, r=30, t=90, b=55)
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — WATER STAGE x PRODUCTIVITY SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def build_stage_scatter(year, selected_classes):
    if master.empty or not selected_classes:
        return go.Figure()

    year     = int(year)
    prod_col = 'composite_productivity_norm'
    if prod_col not in master.columns:
        prod_col = 'composite_productivity'

    fig = go.Figure()
    sub = master[
        (master['year'] == year) &
        (master['veg_class'].isin(selected_classes))
    ].dropna(subset=[prod_col, 'stage_ft'])

    if sub.empty:
        fig.add_annotation(
            text=f"No data available for {year}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=12)
        )
        return fig

    for cls in selected_classes:
        cd = sub[sub['veg_class'] == cls]
        if cd.empty:
            continue
        rho, pval = (spearmanr(cd['stage_ft'], cd[prod_col])
                     if len(cd) >= 3 else (np.nan, np.nan))
        rho_str = f'rho={rho:.2f}' if not np.isnan(rho) else 'rho=N/A'
        fig.add_trace(go.Scatter(
            x=cd['stage_ft'], y=cd[prod_col],
            mode='markers+text',
            name=f'{cls}  ({rho_str})',
            text=cd['month'].map(
                lambda m: MONTH_NAMES[m - 1] if 1 <= m <= 12 else ''
            ),
            textposition='top center',
            textfont=dict(size=8),
            marker=dict(size=10,
                        color=class_colors.get(cls, '#888'),
                        opacity=0.85)
        ))

    fig.update_layout(
        title=(f'Water Stage x Productivity — {year}<br>'
               f'<sup>L OKEE gauge (ft)  |  Spearman rho per class</sup>'),
        xaxis_title='Water Stage (ft)',
        yaxis_title='Productivity Score (normalized)',
        template='plotly_white', height=440,
        hovermode='closest',
        legend=dict(orientation='h', y=-0.28, x=0),
        margin=dict(l=55, r=30, t=80, b=90)
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — MANAGEMENT SUMMARY
# Uses pre-computed area_productivity_summary.csv — no shapefiles needed
# ══════════════════════════════════════════════════════════════════════════════

def build_management(year, biomass_label):
    year = int(year)

    if area_summary.empty:
        return pd.DataFrame(), go.Figure()

    factor    = BIOMASS_OPTIONS[biomass_label]
    short_lbl = biomass_label.split('(')[0].strip()

    # Filter to selected year and biomass scenario
    scenario_key = {
        'Conservative  (0.3 kg/m² — wetland grass)' : 'conservative_0p3',
        'Moderate      (0.6 kg/m² — emergent marsh)' : 'moderate_0p6',
        'Optimistic    (1.2 kg/m² — dense Cattail)'  : 'optimistic_1p2',
    }[biomass_label]

    yr_df = area_summary[
        (area_summary['year']             == year) &
        (area_summary['biomass_scenario'] == scenario_key)
    ].copy()

    if yr_df.empty:
        return pd.DataFrame(), go.Figure()

    # Year-over-year change flag
    prev_df = area_summary[
        (area_summary['year']             == year - 1) &
        (area_summary['biomass_scenario'] == scenario_key)
    ][['veg_class', 'composite_productivity']].rename(
        columns={'composite_productivity': 'prev_prod'}
    )

    yr_df = yr_df.merge(prev_df, on='veg_class', how='left')
    yr_df['YoY'] = yr_df.apply(
        lambda r: ('Up'   if pd.notna(r.get('prev_prod')) and
                             r['composite_productivity'] - r['prev_prod'] >  0.02
                   else 'Down' if pd.notna(r.get('prev_prod')) and
                             r['composite_productivity'] - r['prev_prod'] < -0.02
                   else 'Stable'),
        axis=1
    )

    summary_df = (yr_df[['veg_class', 'area_acres',
                          'composite_productivity', 'sar_productivity',
                          'prod_tier', 'biomass_mt', 'YoY']]
                  .rename(columns={
                      'veg_class'             : 'Vegetation Class',
                      'area_acres'            : 'Area (acres)',
                      'composite_productivity': 'Productivity Score',
                      'sar_productivity'      : 'SAR Productivity',
                      'prod_tier'             : 'Tier',
                      'biomass_mt'            : f'Biomass {short_lbl} (mt)',
                  })
                  .sort_values('Productivity Score', ascending=False)
                  .reset_index(drop=True))

    bm_col = f'Biomass {short_lbl} (mt)'
    fig = px.bar(
        summary_df.sort_values('Area (acres)', ascending=False),
        x='Vegetation Class', y=bm_col,
        color='Productivity Score',
        color_continuous_scale='YlGn',
        range_color=[0, 1],
        title=(f'Estimated Biomass by Class — {year}<br>'
               f'<sup>{biomass_label}</sup>'),
        template='plotly_white', height=400,
        labels={'Vegetation Class': 'Class', bm_col: 'Biomass (mt)'}
    )
    fig.update_xaxes(tickangle=32)
    fig.update_layout(margin=dict(l=55, r=30, t=80, b=130))
    return summary_df, fig

# ══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

year_sel     = pn.widgets.Select(
    name='Year', options=YEARS, value=DEFAULT_YR, width=130)
opacity_sl   = pn.widgets.FloatSlider(
    name='Map Opacity', start=0.3, end=1.0,
    step=0.05, value=0.75, width=200)
class_multi  = pn.widgets.MultiChoice(
    name='Classes (Panels 2 & 4)',
    options=all_classes, value=all_classes[:4], width=300)
clim_cls_sel = pn.widgets.Select(
    name='Class', options=all_classes,
    value=all_classes[0] if all_classes else '', width=230)
clim_var_sel = pn.widgets.Select(
    name='Climate Variable',
    options=list(TC_VAR_LABELS.keys()),
    value='precip_mm', width=230)
season_sel   = pn.widgets.Select(
    name='Season Filter',
    options=['Full Year', 'Dry (Jan-Apr)', 'Wet (Jun-Sep)'],
    value='Full Year', width=175)
biomass_sel  = pn.widgets.Select(
    name='Biomass Conversion',
    options=list(BIOMASS_OPTIONS.keys()),
    value=list(BIOMASS_OPTIONS.keys())[1],
    width=390)
export_btn   = pn.widgets.Button(
    name='Export Summary CSV',
    button_type='primary', width=210)

# ══════════════════════════════════════════════════════════════════════════════
# REACTIVE PANELS
# ══════════════════════════════════════════════════════════════════════════════

@pn.depends(year_sel, opacity_sl)
def panel1(year, opacity):
    return pn.Column(
        pn.pane.Markdown(
            f"### Panel 1 — Land Cover x Productivity Map  |  {year}"
        ),
        pn.pane.HTML(
            build_map(year, opacity),
            height=540,
            sizing_mode='stretch_width',
            sanitize_html=False
        )
    )

@pn.depends(class_multi, year_sel)
def panel2(selected_classes, year):
    if not selected_classes:
        return pn.pane.Markdown("*Select at least one class in the sidebar.*")
    return pn.Column(
        pn.pane.Markdown(
            "### Panel 2 — Monthly Productivity + Water Stage Timeline"
        ),
        pn.pane.Plotly(
            build_timeline(selected_classes, 'Optical'),
            sizing_mode='stretch_width'
        )
    )

@pn.depends(clim_cls_sel, clim_var_sel, season_sel)
def panel3(cls, climate_var, season_filter):
    return pn.Column(
        pn.pane.Markdown("### Panel 3 — Climate Driver Correlations"),
        pn.pane.Plotly(
            build_climate_scatter(cls, climate_var, season_filter),
            sizing_mode='stretch_width'
        )
    )

@pn.depends(year_sel, class_multi)
def panel4(year, selected_classes):
    if not selected_classes:
        return pn.pane.Markdown("*Select at least one class in the sidebar.*")
    return pn.Column(
        pn.pane.Markdown(
            f"### Panel 4 — Water Stage x Productivity  |  {year}"
        ),
        pn.pane.Plotly(
            build_stage_scatter(year, selected_classes),
            sizing_mode='stretch_width'
        )
    )

@pn.depends(year_sel, biomass_sel)
def panel5(year, biomass_label):
    summary_df, fig = build_management(year, biomass_label)

    def on_export(event):
        path = os.path.join(OUTPUT_DIR,
                            f'management_summary_{int(year)}.csv')
        summary_df.to_csv(path, index=False)
        export_btn.name = f'Saved management_summary_{int(year)}.csv'

    export_btn.on_click(on_export)

    if summary_df.empty:
        return pn.pane.Markdown(f"*No data available for {int(year)}.*")

    return pn.Column(
        pn.pane.Markdown(
            f"### Panel 5 — Management Summary  |  {int(year)}\n\n"
            f"*Tier 3 classes shown for completeness — interpret with caution. "
            f"Tier 1 = non-vegetated (productivity = 0).*"
        ),
        export_btn,
        pn.pane.Plotly(fig, sizing_mode='stretch_width'),
        pn.widgets.DataFrame(
            summary_df,
            sizing_mode='stretch_width',
            height=320
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

header = pn.pane.Markdown("""
# Lake Okeechobee Littoral Zone — Productivity Monitor
**University of Florida**  |  Sentinel-1 SAR + Sentinel-2 Optical  |
TerraClimate  |  SFWMD Stage Gauge (L OKEE)
---
""", sizing_mode='stretch_width')

sidebar = pn.Column(
    pn.pane.Markdown("## Controls"),
    pn.pane.Markdown("**Global**"),
    year_sel,
    pn.layout.Divider(),
    pn.pane.Markdown("**Map**"),
    opacity_sl,
    pn.layout.Divider(),
    pn.pane.Markdown("**Panels 2 & 4**"),
    class_multi,
    pn.layout.Divider(),
    pn.pane.Markdown("**Panel 3 — Climate**"),
    clim_cls_sel,
    clim_var_sel,
    season_sel,
    pn.layout.Divider(),
    pn.pane.Markdown("**Panel 5 — Biomass**"),
    biomass_sel,
    width=330
)

tabs = pn.Tabs(
    ('Map',        panel1),
    ('Timeline',   panel2),
    ('Climate',    panel3),
    ('Stage',      panel4),
    ('Management', panel5),
    sizing_mode='stretch_width',
    dynamic=True
)

dashboard = pn.template.FastListTemplate(
    title             = 'Lake Okeechobee Littoral Zone — Productivity Monitor',
    sidebar           = [sidebar],
    main              = [header, tabs],
    accent_base_color = '#006837',
    header_background = '#006837',
    theme             = 'default'
)

dashboard.servable()
