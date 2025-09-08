import os
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# NOTE: Script extended to include static capital market account statistics
# and PSX market capitalization vs GDP comparison (Dec 2024 figures provided).

# --- Configuration ---------------------------------------------------------
CSV_FILE = Path(r"d:\projects\Pakistan's First Financial Literacy App_ Finasaan - Form responses 1.csv")
OUTPUT_DIR = Path(r"d:\projects\finasaan_charts")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")  # smaller base fonts

# Canonical logical column labels mapped from fuzzy survey headers
COLUMN_KEYS = {
    'age': re.compile(r"age group", re.I),
    'savings': re.compile(r"save monthly", re.I),  # proxy for earnings / capacity
    'knowledge': re.compile(r"financial knowledge", re.I),
    'current_do': re.compile(r"currently do", re.I),
    'learn_topics': re.compile(r"want to learn", re.I),
    'learn_sources': re.compile(r"learn about personal finance", re.I),
    'app_interest': re.compile(r"would you be interested", re.I),
    'features_useful': re.compile(r"features would you find most useful", re.I),
    'barrier': re.compile(r"biggest barrier", re.I),
    'willing_pay_features': re.compile(r"worth paying", re.I),
}

# Ordered knowledge levels for better visual narrative
KNOWLEDGE_ORDER = [
    "I know very little",
    "I understand the basics (budgeting, savings)",
    "I actively try to learn about finance/investing",
    "I consider myself financially literate",
]

# Remap verbose knowledge responses -> concise labels for plots
KNOWLEDGE_REMAP = {
    "I know very little": "Very little",
    "I understand the basics (budgeting, savings)": "Basics",
    "I actively try to learn about finance/investing": "Learning",
    "I consider myself financially literate": "Literate",
}

"""Label formatting without truncation.

Wrap long labels across multiple lines (at word boundaries) instead of
adding ellipses so that full meaning is preserved while remaining compact.
"""
def _format_labels(index, remap=None, max_width=34):
    wrapped = []
    for raw in index:
        lab = remap.get(raw, raw) if remap else raw
        if len(lab) > max_width and ' ' in lab:
            words = lab.split()
            line = ''
            lines = []
            for w in words:
                tentative = (line + ' ' + w).strip()
                if len(tentative) <= max_width:
                    line = tentative
                else:
                    if line:
                        lines.append(line)
                    line = w
            if line:
                lines.append(line)
            lab = '\n'.join(lines)
        wrapped.append(lab)
    return wrapped

def load_csv():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    # Keep original header (some have trailing spaces / newlines)
    df = pd.read_csv(CSV_FILE, encoding="utf-8")
    # Strip whitespace/newlines from column names
    df.columns = [c.replace('\n', ' ').strip() for c in df.columns]
    return df

def map_columns(df):
    mapped = {}
    for logical, pattern in COLUMN_KEYS.items():
        for col in df.columns:
            if pattern.search(col):
                mapped[logical] = col
                break
    return mapped

def simple_count_plot(series, title, filename, order=None, rotate=0, horizontal=False, palette="viridis", remap=None):
    counts = series.value_counts(dropna=True)
    if order:
        counts = counts.reindex([o for o in order if o in counts.index], fill_value=0)
    plt.figure(figsize=(5.8,3.2))
    if horizontal:
        display_idx = _format_labels(counts.index, remap)
        sns.barplot(x=counts.values, y=display_idx, palette=palette)
        for i,v in enumerate(counts.values):
            plt.text(v+0.05, i, str(v), va='center', fontsize=6.5)
        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)
    else:
        display_idx = _format_labels(counts.index, remap)
        sns.barplot(x=display_idx, y=counts.values, palette=palette)
        for i,v in enumerate(counts.values):
            plt.text(i, v+0.05, str(v), ha='center', fontsize=6.5)
        plt.xticks(rotation=rotate, ha='right' if rotate else 'center', fontsize=7)
        plt.yticks(fontsize=7)
    plt.title(title, fontsize=10.5)
    plt.ylabel('Resp', fontsize=8)
    plt.xlabel('' if horizontal else '')
    out = OUTPUT_DIR / filename
    plt.tight_layout(pad=0.8); plt.savefig(out, dpi=170); plt.close()
    return counts

def multi_select_breakdown(series, title, filename, top_n=10, palette="magma", remap=None, max_len=34):
    exploded = []
    for raw in series.dropna():
        # Split on commas, handle quotes
        parts = [p.strip().strip('"').strip() for p in str(raw).split(',') if p.strip()]
        exploded.extend(parts)
    counts = pd.Series(exploded).value_counts().head(top_n)
    display_idx = _format_labels(counts.index, remap, max_width=max_len)
    plt.figure(figsize=(6.0,3.4))
    sns.barplot(x=counts.values, y=display_idx, palette=palette)
    plt.title(title, fontsize=10.5)
    plt.xlabel('Mentions', fontsize=8)
    plt.yticks(fontsize=7)
    plt.xticks(fontsize=7)
    for i,v in enumerate(counts.values):
        plt.text(v+0.05, i, str(v), va='center', fontsize=6.5)
    out = OUTPUT_DIR / filename
    plt.tight_layout(); plt.savefig(out, dpi=170); plt.close()
    return counts

def build_visualizations():
    df = load_csv()
    mapped = map_columns(df)
    print("Mapped columns:")
    for k,v in mapped.items():
        print(f"  {k}: {v}")

    # Age Distribution (already categorical like 18-24, Under 18 etc.)
    age_counts = None
    if 'age' in mapped:
        age_counts = simple_count_plot(
            df[mapped['age']].str.strip(),
            'Age',
            'age_distribution.png',
            order=['Under 18','18-24','25-34','35-44','45+']
        )

    # Savings (proxy for earnings capacity)
    savings_counts = None
    if 'savings' in mapped:
        ordered_savings = [
            '0 PKR','0','1-1000 PKR','1-1000','1000-5000 PKR','1000-5000',
            '5000-10k PKR','10k-25k PKR','25k-50k PKR','50k+'  # some may not appear
        ]
        savings_counts = simple_count_plot(
            df[mapped['savings']].str.strip(),
            'Monthly Savings',
            'income_distribution.png',
            order=ordered_savings,
            rotate=35
        )

    # Knowledge Levels
    knowledge_counts = None
    if 'knowledge' in mapped:
        knowledge_series = df[mapped['knowledge']].str.strip()
        knowledge_counts = simple_count_plot(
            knowledge_series,
            'Knowledge Level',
            'investing_knowledge.png',
            order=KNOWLEDGE_ORDER,
            rotate=20,
            remap=KNOWLEDGE_REMAP
        )

    # Topics they want to learn (Investment intent proxy)
    topics_counts = None
    if 'learn_topics' in mapped:
        topics_counts = multi_select_breakdown(
            df[mapped['learn_topics']],
            'Learning Topics',
            'investment_intent.png',
            top_n=10
        )

    # Features they find useful (App usage intention)
    features_counts = None
    if 'features_useful' in mapped:
        features_counts = multi_select_breakdown(
            df[mapped['features_useful']],
            'Desired Features',
            'app_feature_usage.png',
            top_n=10,
            palette='Blues_r'
        )

    # Barrier (optional, can help narrative)
    barrier_counts = None
    if 'barrier' in mapped:
        barrier_counts = simple_count_plot(
            df[mapped['barrier']].str.strip(),
            'Barriers',
            'barriers.png',
            rotate=25,
            palette='cubehelix'
        )

    # Simple summary text aligning with requested narrative
    if age_counts is not None:
        dominant_age = age_counts.idxmax()
    else:
        dominant_age = 'N/A'
    if knowledge_counts is not None:
        low_knowledge_share = knowledge_counts.get('I know very little', 0) + knowledge_counts.get('I understand the basics (budgeting, savings)', 0)
        total_resp = knowledge_counts.sum()
        low_knowledge_pct = (low_knowledge_share / total_resp * 100) if total_resp else 0
    else:
        low_knowledge_pct = 0
    if savings_counts is not None:
        top_savings = savings_counts.idxmax()
    else:
        top_savings = 'N/A'
    top_topic = topics_counts.idxmax() if topics_counts is not None and not topics_counts.empty else 'N/A'
    top_feature = features_counts.idxmax() if features_counts is not None and not features_counts.empty else 'N/A'

    summary_lines = [
        f"Age peak: {dominant_age}",
        f"Savings mode: {top_savings}",
        f"Low/basic knowledge: {low_knowledge_pct:.1f}%", 
        f"Top learn: {top_topic}",
        f"Top feature: {top_feature}",
    ]
    (OUTPUT_DIR / 'summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
    print("\n" + "\n".join(summary_lines))

    # --- Additional static investing statistics visualizations -------------
    generate_investing_stats()


def generate_investing_stats():
    """Generate charts from provided PSX / account statistics.

    Figures added:
      1. accounts_overview.png - Bar of high-level account categories
      2. investor_breakdown.png - Individual (RDA vs Non-RDA) vs Corporate investors
      3. sub_accounts_breakdown.png - Individual vs Corporate sub accounts
      4. market_cap_vs_gdp.png - PSX market cap vs GDP with ratio annotation
    """
    # Raw provided numbers
    data = {
        'Sahulat Accounts': 75485,
        'Investor Accounts (Individual incl. RDA)': 100870,
        'Investor Accounts (Corporate)': 2536,
        'RDA Accounts': 15640,  # subset of individual investor accounts
        'Sub Accounts (Individual)': 355164,
        'Sub Accounts (Corporate)': 10182,
    }

    # 1. Accounts overview (selected comparable categories)
    overview_keys = [
        'Sahulat Accounts',
        'Investor Accounts (Individual incl. RDA)',
        'Investor Accounts (Corporate)',
        'Sub Accounts (Individual)',
        'Sub Accounts (Corporate)'
    ]
    overview_vals = [data[k] for k in overview_keys]
    plt.figure(figsize=(6.2,3.2))
    sns.barplot(x=overview_keys, y=overview_vals, palette='crest')
    for i,v in enumerate(overview_vals):
        plt.text(i, v + max(overview_vals)*0.01, f"{v:,}", ha='center', fontsize=7)
    plt.ylabel('Count', fontsize=8)
    plt.xticks(rotation=25, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.title('Account Category Scale', fontsize=11)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'accounts_overview.png', dpi=170); plt.close()

    # 2. Investor breakdown (stacked bar for Individual = RDA + Non-RDA, separate Corporate)
    individual_total = data['Investor Accounts (Individual incl. RDA)']
    rda = data['RDA Accounts']
    non_rda = individual_total - rda
    corp = data['Investor Accounts (Corporate)']
    fig, ax = plt.subplots(figsize=(4.6,3.2))
    ax.bar(['Individual'], [non_rda], label='Non-RDA', color='#4c72b0')
    ax.bar(['Individual'], [rda], bottom=[non_rda], label='RDA', color='#dd8452')
    ax.bar(['Corporate'], [corp], label='Corporate', color='#55a868')
    ax.set_ylabel('Accounts', fontsize=8)
    ax.set_title('Investor Accounts Breakdown', fontsize=11)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=7)
    # Annotations
    ax.text(0, non_rda/2, f"Non-RDA\n{non_rda:,}", ha='center', va='center', fontsize=7, color='white')
    ax.text(0, non_rda + rda/2, f"RDA\n{rda:,}", ha='center', va='center', fontsize=7, color='white')
    ax.text(1, corp/2, f"{corp:,}", ha='center', va='center', fontsize=7, color='white')
    rda_pct = rda / individual_total * 100 if individual_total else 0
    ax.text(0, individual_total + individual_total*0.03, f"RDA {rda_pct:.1f}% of Individual", ha='center', fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'investor_breakdown.png', dpi=170); plt.close()

    # 3. Sub accounts breakdown
    sub_keys = ['Sub Accounts (Individual)', 'Sub Accounts (Corporate)']
    sub_vals = [data[k] for k in sub_keys]
    plt.figure(figsize=(4.8,3.0))
    sns.barplot(x=sub_keys, y=sub_vals, palette='viridis')
    for i,v in enumerate(sub_vals):
        plt.text(i, v + max(sub_vals)*0.015, f"{v:,}", ha='center', fontsize=7)
    plt.ylabel('Count', fontsize=8)
    plt.xticks(rotation=20, fontsize=7)
    plt.yticks(fontsize=7)
    plt.title('Sub Accounts Split', fontsize=11)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'sub_accounts_breakdown.png', dpi=170); plt.close()

    # 4. Market cap vs GDP comparison
    psx_market_cap_usd_bn = 50.0  # bn USD (Dec 2024)
    gdp_usd_bn = 373.1            # bn USD
    ratio_pct = psx_market_cap_usd_bn / gdp_usd_bn * 100 if gdp_usd_bn else 0
    plt.figure(figsize=(4.6,3.0))
    sns.barplot(x=['PSX MCap', 'GDP'], y=[psx_market_cap_usd_bn, gdp_usd_bn], palette=['#2b8cbe','#a6bddb'])
    plt.ylabel('USD (Bn)', fontsize=8)
    plt.title(f'Market Cap vs GDP (Ratio {ratio_pct:.1f}%)', fontsize=11)
    for i,v in enumerate([psx_market_cap_usd_bn, gdp_usd_bn]):
        plt.text(i, v + gdp_usd_bn*0.01, f"{v:.1f}", ha='center', fontsize=7)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'market_cap_vs_gdp.png', dpi=170); plt.close()

    # Append investing stats summary
    invest_summary = [
        f"Investor indiv (incl RDA): {individual_total:,}",
        f" - of which RDA: {rda:,} ({rda_pct:.1f}%)",
        f"Investor corporate: {corp:,}",
        f"Sahulat: {data['Sahulat Accounts']:,}",
        f"Sub indiv: {data['Sub Accounts (Individual)']:,}",
        f"Sub corp: {data['Sub Accounts (Corporate)']:,}",
        f"PSX MCap/GDP: {ratio_pct:.1f}% (50 / 373.1 bn USD)",
    ]
    summary_file = OUTPUT_DIR / 'summary.txt'
    existing = summary_file.read_text(encoding='utf-8') if summary_file.exists() else ''
    summary_file.write_text(existing + ('\n--- Investing Stats ---\n' if existing else '') + '\n'.join(invest_summary), encoding='utf-8')
    print("Added investing statistics charts and updated summary.")

    # Country comparison chart (India vs Pakistan vs USA)
    generate_country_comparison()


def generate_country_comparison():
        """Generate a simple bar chart comparing approximate investor / active
        client account counts across India, Pakistan, USA.

        Data sources (mixed / approximate):
            - India: "Client Accounts Active" = 4,14,26,038 (converted to 41,426,038)
            - Pakistan: Investor Accounts (Individual incl. RDA) + Corporate = 103,406
                (excludes Sahulat & sub-accounts to avoid double counting)
            - USA: Approx retail investors / adults owning stocks ~150,000,000
                (public survey estimates; illustrative only)

        A disclaimer is appended to summary.txt.
        """
        india_active = 41426038
        pakistan_investors = 100870 + 2536  # individual incl RDA + corporate
        usa_investors_est = 150_000_000  # illustrative estimate

        countries = ['Pakistan','India','USA']
        values = [pakistan_investors, india_active, usa_investors_est]

        plt.figure(figsize=(5.6,3.2))
        palette = ['#1b9e77','#7570b3','#d95f02']
        sns.barplot(x=countries, y=values, palette=palette)
        for i,v in enumerate(values):
                plt.text(i, v*1.01, f"{v/1_000_000:.2f}M", ha='center', fontsize=7)
        plt.ylabel('Investors / Active Accounts', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=7)
        plt.title('Investing Population Scale (Illustrative)', fontsize=11)
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'country_investing_comparison.png', dpi=170); plt.close()

        # Append disclaimer
        note = ("Figures are approximate; India = depository active client accounts, "
                        "Pakistan = investor (individual+corporate) accounts, USA = est. retail investors owning stocks.")
        summary_file = OUTPUT_DIR / 'summary.txt'
        existing = summary_file.read_text(encoding='utf-8') if summary_file.exists() else ''
        if '--- Country Comparison ---' not in existing:
                existing += ('\n--- Country Comparison ---\n' + note)
                summary_file.write_text(existing, encoding='utf-8')
        print("Added country investing comparison chart.")

        # After country comparison also produce pension allocation vs funds chart
        generate_pension_gap()


def generate_pension_gap():
    """Compare Federal pension budget allocation vs Voluntary Pension Scheme assets.

    Data provided:
      - Federal pension allocation FY 2025-26: Rs 1,055 billion (1.055 trillion PKR)
      - Voluntary Pension Schemes AUM (Jul 2025): Rs 109,170 million = 109.17 billion PKR

    Output: pension_allocation_vs_funds.png and appended summary note.
    """
    govt_pension_alloc_bn = 1055.0        # billion PKR
    vps_assets_bn = 109.170               # billion PKR
    ratio_pct = vps_assets_bn / govt_pension_alloc_bn * 100 if govt_pension_alloc_bn else 0

    labels = ['Govt Pension Allocation', 'Voluntary Pension AUM']
    values = [govt_pension_alloc_bn, vps_assets_bn]
    palette = ['#b2182b', '#2166ac']
    plt.figure(figsize=(6.0,3.2))
    sns.barplot(x=labels, y=values, palette=palette)
    for i,v in enumerate(values):
        plt.text(i, v * 1.01, f"{v:,.0f} bn", ha='center', fontsize=7)
    plt.ylabel('PKR (Billion)', fontsize=8)
    plt.xticks(rotation=15, fontsize=8)
    plt.yticks(fontsize=7)
    plt.title(f'Pension Outlay vs VPS Assets (VPS {ratio_pct:.1f}% of Govt)', fontsize=11)
    # Visual gap annotation
    gap = values[0] - values[1]
    plt.text(0.5, max(values)*0.6, f"Gap ≈ {gap:,.0f} bn", ha='center', fontsize=8, color='#444')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'pension_allocation_vs_funds.png', dpi=170); plt.close()

    # Append to summary
    summary_file = OUTPUT_DIR / 'summary.txt'
    existing = summary_file.read_text(encoding='utf-8') if summary_file.exists() else ''
    pension_note = (f"Federal pension allocation: {govt_pension_alloc_bn:,.0f} bn PKR; "
                    f"VPS AUM: {vps_assets_bn:,.2f} bn PKR ({ratio_pct:.1f}% of allocation)")
    if '--- Pension Comparison ---' not in existing:
        existing += (('\n' if existing else '') + '--- Pension Comparison ---\n' + pension_note)
        summary_file.write_text(existing, encoding='utf-8')
    print("Added pension allocation vs VPS assets chart.")

def main():
    build_visualizations()
    print(f"Charts saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
