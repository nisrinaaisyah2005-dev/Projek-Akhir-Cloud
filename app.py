import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ─── Konstanta Warna — Tema Kuning Cerah ──────────────────────────────────────
PALETTE_MAIN  = ['#F4A900', '#E07B00', '#3D7AB5', '#E05C5C', '#7BC67E', '#A78BFA']
COLOR_BLUE    = '#F4A900'   # kuning utama (menggantikan "biru")
COLOR_ORANGE  = '#E07B00'   # oranye gelap
COLOR_GREEN   = '#3D7AB5'   # biru untuk kontras
COLOR_RED     = '#E05C5C'   # merah untuk median / peringatan

TRIP_MIN       = 60
TRIP_MAX       = 86400
REFERENCE_YEAR = 2018
AGE_MIN        = 16
AGE_MAX        = 90
BY_MIN         = REFERENCE_YEAR - AGE_MAX   # 1928
BY_MAX         = REFERENCE_YEAR - AGE_MIN   # 2002

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 12,
                     'axes.labelsize': 10, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Citi Bike Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #fffbe6;
        border-left: 4px solid #F4A900;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #3a2500;
        border-bottom: 3px solid #F4A900;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }
    .insight-box {
        background: #fffbe6;
        border-left: 4px solid #E07B00;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.92rem;
        margin-top: 8px;
    }
    /* warna tab aktif */
    div[data-baseweb="tab-highlight"] {
        background-color: #F4A900 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── URL Dataset dari GitHub ──────────────────────────────────────────────────
# Ganti URL di bawah dengan raw URL file CSV kamu di GitHub
# Format: https://raw.githubusercontent.com/<username>/<repo>/<branch>/<path/file.csv>
GITHUB_DATA_URL = "https://raw.githubusercontent.com/nisrinaaisyah2005-dev/Projek-Akhir-Cloud/refs/heads/main/NYC%20Citi%20Bike%20Trips.csv"

# ─── Load & Cache Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Memuat data dari GitHub...")
def load_and_clean(url: str):
    try:
        df_raw = pd.read_csv(url)
    except Exception as e:
        st.error(f"❌ Gagal memuat data dari GitHub.\n\nPastikan URL sudah benar di variabel `GITHUB_DATA_URL`.\n\nError: {e}")
        st.stop()

    df = df_raw.copy()

    # Konversi datetime
    df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
    df['stoptime']  = pd.to_datetime(df['stoptime'],  errors='coerce')
    df = df.dropna(subset=['starttime', 'stoptime'])

    # Validasi logika waktu
    df = df[df['stoptime'] > df['starttime']].copy()

    # Filter outlier tripduration
    df = df[(df['tripduration'] >= TRIP_MIN) & (df['tripduration'] <= TRIP_MAX)].copy()

    # Filter outlier birth_year
    df = df[(df['birth_year'] >= BY_MIN) & (df['birth_year'] <= BY_MAX)].copy()

    # Standardisasi kategorik
    for col in ['usertype', 'gender', 'start_station_name', 'end_station_name']:
        df[col] = df[col].str.strip().str.lower()

    # Perbaiki tripduration vs selisih waktu aktual
    df['_dur_check'] = (df['stoptime'] - df['starttime']).dt.total_seconds().round(0).astype(int)
    df['_dur_diff']  = (df['tripduration'] - df['_dur_check']).abs()
    mask = df['_dur_diff'] > 60
    df.loc[mask, 'tripduration'] = df.loc[mask, '_dur_check']
    df.drop(columns=['_dur_check', '_dur_diff'], inplace=True)

    # Feature engineering
    df['start_date']         = df['starttime'].dt.date
    df['start_year']         = df['starttime'].dt.year
    df['start_month']        = df['starttime'].dt.month
    df['start_month_name']   = df['starttime'].dt.strftime('%B')
    df['start_day']          = df['starttime'].dt.day
    df['start_weekday']      = df['starttime'].dt.dayofweek
    df['start_weekday_name'] = df['starttime'].dt.strftime('%A')
    df['start_hour']         = df['starttime'].dt.hour
    df['is_weekend']         = df['start_weekday'].isin([5, 6])
    df['tripduration_min']   = (df['tripduration'] / 60).round(2)
    df['age']                = REFERENCE_YEAR - df['birth_year']
    df['is_round_trip']      = (df['start_station_name'] == df['end_station_name'])

    def categorize_hour(h):
        if   h < 6:  return 'dini_hari'
        elif h < 12: return 'pagi'
        elif h < 17: return 'siang'
        elif h < 21: return 'sore'
        else:        return 'malam'

    def categorize_duration(m):
        if   m < 5:  return 'sangat_pendek'
        elif m < 15: return 'pendek'
        elif m < 30: return 'sedang'
        elif m < 60: return 'panjang'
        else:        return 'sangat_panjang'

    def categorize_age(a):
        if   a < 25: return 'remaja/dewasa_muda'
        elif a < 35: return 'dewasa_25_34'
        elif a < 45: return 'dewasa_35_44'
        elif a < 55: return 'dewasa_45_54'
        elif a < 65: return 'menengah_55_64'
        else:        return 'lansia_65+'

    df['time_of_day']       = df['start_hour'].apply(categorize_hour)
    df['duration_category'] = df['tripduration_min'].apply(categorize_duration)
    df['age_group']         = df['age'].apply(categorize_age)

    DURATION_ORDER = ['sangat_pendek', 'pendek', 'sedang', 'panjang', 'sangat_panjang']
    df['duration_category'] = pd.Categorical(df['duration_category'],
                                              categories=DURATION_ORDER, ordered=True)
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #F4A900, #E07B00);
        border-radius: 12px;
        padding: 18px 14px 14px 14px;
        text-align: center;
        margin-bottom: 8px;
    ">
        <div style="font-size: 2.8rem; line-height: 1;">🚲</div>
        <div style="color: #3a2500; font-size: 1.1rem; font-weight: 700; margin-top: 6px;">
            NYC Citi Bike
        </div>
        <div style="color: #6b3d00; font-size: 0.78rem; margin-top: 3px;">
            Dashboard Analisis
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Navigasi**")
    page = st.radio("Pilih Halaman", [
        "📊 Overview",
        "🔍 EDA",
        "📈 Analisis Statistik",
        "🗺️ Analisis Stasiun",
        "🤖 Machine Learning",
        "💡 Insight & Ringkasan",
        "🎨 Panduan Kustomisasi"
    ])

# ─── Load Data ────────────────────────────────────────────────────────────────
df = load_and_clean(GITHUB_DATA_URL)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Overview Dataset")

    # KPI
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Perjalanan", f"{len(df):,}")
    col2.metric("Median Durasi", f"{df['tripduration_min'].median():.1f} mnt")
    col3.metric("Subscriber", f"{(df['usertype']=='subscriber').mean()*100:.1f}%")
    col4.metric("Median Usia", f"{df['age'].median():.0f} thn")
    col5.metric("Round-Trip", f"{df['is_round_trip'].mean()*100:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Distribusi Durasi Perjalanan</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['tripduration_min'], bins=60, color=COLOR_BLUE, edgecolor='white', alpha=0.85)
        ax.axvline(df['tripduration_min'].median(), color=COLOR_RED, lw=2, linestyle='--',
                   label=f'Median = {df["tripduration_min"].median():.1f} mnt')
        ax.set_xlabel('Durasi (menit)')
        ax.set_ylabel('Frekuensi')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-title">Proporsi User Type & Gender</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        ut = df['usertype'].value_counts()
        axes[0].pie(ut, labels=ut.index, autopct='%1.1f%%',
                    colors=[COLOR_BLUE, COLOR_ORANGE], startangle=90)
        axes[0].set_title('User Type')

        gd = df['gender'].value_counts()
        axes[1].bar(gd.index, gd.values, color=PALETTE_MAIN[:len(gd)], edgecolor='white')
        axes[1].set_title('Gender')
        axes[1].set_ylabel('Jumlah')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Perjalanan per Jam</div>', unsafe_allow_html=True)
        hour_cnt = df['start_hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hour_cnt.index, hour_cnt.values, color=COLOR_BLUE, edgecolor='white', alpha=0.85)
        ax.set_xlabel('Jam')
        ax.set_ylabel('Jumlah Trip')
        ax.set_xticks(range(0, 24, 2))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.markdown('<div class="section-title">Weekday vs Weekend</div>', unsafe_allow_html=True)
        wd = df['is_weekend'].map({True: 'Weekend', False: 'Weekday'}).value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(wd.index, wd.values, color=[COLOR_ORANGE, COLOR_BLUE], edgecolor='white', alpha=0.85)
        ax.set_ylabel('Jumlah Trip')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{bar.get_height()/1000:.1f}k', ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Mayoritas pengguna adalah <b>subscriber</b> yang menggunakan sepeda sebagai moda komuter harian. Pola bimodal (pagi & sore) terlihat jelas di grafik jam keberangkatan.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Distribusi Data", "Pola Temporal", "Demografis"])

    with tab1:
        st.markdown('<div class="section-title">Distribusi Variabel Kunci</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Distribusi Data — NYC Citi Bike', fontsize=14, fontweight='bold')

        axes[0,0].hist(df['tripduration_min'], bins=60, color=COLOR_BLUE, edgecolor='white', alpha=0.85)
        axes[0,0].axvline(df['tripduration_min'].median(), color=COLOR_RED, lw=1.8, linestyle='--',
                          label=f'Median={df["tripduration_min"].median():.1f} mnt')
        axes[0,0].set_title('Distribusi Durasi (menit)')
        axes[0,0].set_xlabel('Durasi (menit)'); axes[0,0].legend(fontsize=8)

        axes[0,1].hist(df['age'], bins=30, color=COLOR_GREEN, edgecolor='white', alpha=0.85)
        axes[0,1].axvline(df['age'].median(), color=COLOR_RED, lw=1.8, linestyle='--',
                          label=f'Median={df["age"].median():.0f} thn')
        axes[0,1].set_title('Distribusi Usia'); axes[0,1].set_xlabel('Usia'); axes[0,1].legend(fontsize=8)

        dur_order  = ['sangat_pendek', 'pendek', 'sedang', 'panjang', 'sangat_panjang']
        dur_counts = df['duration_category'].value_counts().reindex(dur_order)
        axes[0,2].bar(dur_counts.index, dur_counts.values, color=PALETTE_MAIN[:5], edgecolor='white')
        axes[0,2].set_title('Kategori Durasi')
        axes[0,2].tick_params(axis='x', rotation=20)

        age_order  = ['remaja/dewasa_muda','dewasa_25_34','dewasa_35_44','dewasa_45_54','menengah_55_64','lansia_65+']
        age_counts = df['age_group'].value_counts().reindex(age_order)
        axes[1,0].bar(age_counts.index, age_counts.values, color=PALETTE_MAIN, edgecolor='white')
        axes[1,0].set_title('Kelompok Usia')
        axes[1,0].tick_params(axis='x', rotation=25)

        bp_data  = [df[df['usertype']==u]['tripduration_min'].values for u in df['usertype'].unique()]
        bp_labels = df['usertype'].unique()
        axes[1,1].boxplot(bp_data, labels=bp_labels, patch_artist=True,
                          boxprops=dict(facecolor=COLOR_BLUE, alpha=0.6),
                          medianprops=dict(color=COLOR_RED, lw=2))
        axes[1,1].set_title('Boxplot Durasi per User Type')
        axes[1,1].set_ylim(0, 90); axes[1,1].set_ylabel('Durasi (menit)')

        missing_pct = df.isnull().sum() / len(df) * 100
        missing_pct = missing_pct.sort_values(ascending=False).head(10)
        axes[1,2].barh(missing_pct.index, missing_pct.values,
                       color=[COLOR_RED if v > 0 else COLOR_GREEN for v in missing_pct.values])
        axes[1,2].set_title('Missing Value (%)'); axes[1,2].set_xlabel('Missing (%)')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Statistik Deskriptif**")
        st.dataframe(df[['tripduration_min', 'age']].describe().round(2))

    with tab2:
        st.markdown('<div class="section-title">Pola Temporal</div>', unsafe_allow_html=True)
        DAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        trend_hour    = df['start_hour'].value_counts().sort_index()
        trend_weekday = (df.groupby('start_weekday_name').size()
                          .reindex([d for d in DAY_ORDER if d in df['start_weekday_name'].unique()]))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(trend_hour.index, trend_hour.values, color=COLOR_BLUE, edgecolor='white', alpha=0.85)
        axes[0].set_title('Volume Trip per Jam')
        axes[0].set_xlabel('Jam'); axes[0].set_ylabel('Jumlah Trip')
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))

        axes[1].bar(range(len(trend_weekday)), trend_weekday.values, color=PALETTE_MAIN[:7], edgecolor='white')
        axes[1].set_xticks(range(len(trend_weekday)))
        axes[1].set_xticklabels(trend_weekday.index, rotation=25, ha='right')
        axes[1].set_title('Volume Trip per Hari')
        axes[1].set_ylabel('Jumlah Trip')
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Heatmap weekday vs weekend
        st.markdown("**Heatmap: Jam × Hari (Weekday vs Weekend)**")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, is_we, label in zip(axes, [False, True], ['Weekday', 'Weekend']):
            pivot = (df[df['is_weekend']==is_we]
                     .groupby(['start_weekday_name', 'start_hour'])
                     .size().unstack(fill_value=0))
            pivot = pivot.reindex([d for d in DAY_ORDER if d in pivot.index])
            sns.heatmap(pivot, ax=ax, cmap='YlOrRd', fmt='.0f',
                        linewidths=0.3, cbar_kws={'shrink': 0.8})
            ax.set_title(f'Heatmap Volume Trip — {label}')
            ax.set_xlabel('Jam'); ax.set_ylabel('Hari')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown('<div class="section-title">Analisis Demografis</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        age_order = ['remaja/dewasa_muda','dewasa_25_34','dewasa_35_44','dewasa_45_54','menengah_55_64','lansia_65+']
        age_gender = (df[df['gender'].isin(['male','female'])]
                      .groupby(['age_group','gender'])['tripduration_min']
                      .median().unstack())
        age_gender = age_gender.reindex([a for a in age_order if a in age_gender.index])
        x = np.arange(len(age_gender)); w = 0.35
        bars_m = axes[0].bar(x - w/2, age_gender.get('male', 0), w, color=COLOR_BLUE, label='Male', alpha=0.85)
        bars_f = axes[0].bar(x + w/2, age_gender.get('female', 0), w, color=COLOR_ORANGE, label='Female', alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(age_gender.index, rotation=25, ha='right', fontsize=8)
        axes[0].set_title('Median Durasi per Kelompok Usia & Gender')
        axes[0].set_ylabel('Durasi Median (menit)')
        axes[0].legend()

        pivot_ut = df.pivot_table(index='age_group', columns='usertype', values='tripduration_min',
                                   aggfunc='median').reindex([a for a in age_order if a in df['age_group'].unique()])
        x2 = np.arange(len(pivot_ut))
        for i, (col, color) in enumerate(zip(pivot_ut.columns, PALETTE_MAIN)):
            axes[1].bar(x2 + i*(0.8/len(pivot_ut.columns)) - 0.4,
                        pivot_ut[col], 0.8/len(pivot_ut.columns),
                        label=col, color=color, alpha=0.85)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(pivot_ut.index, rotation=25, ha='right', fontsize=8)
        axes[1].set_title('Median Durasi per Kelompok Usia & User Type')
        axes[1].set_ylabel('Durasi Median (menit)')
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALISIS STATISTIK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analisis Statistik":
    st.title("📈 Analisis Statistik")

    tab1, tab2, tab3, tab4 = st.tabs(["Uji Mann-Whitney", "Korelasi Spearman", "Agregasi", "Tren Musiman"])

    dur_sub = df[df['usertype']=='subscriber']['tripduration_min']
    dur_cus = df[df['usertype']=='customer']['tripduration_min']
    dur_wd  = df[df['is_weekend']==False]['tripduration_min']
    dur_we  = df[df['is_weekend']==True]['tripduration_min']

    with tab1:
        st.markdown('<div class="section-title">Uji Mann-Whitney U</div>', unsafe_allow_html=True)

        stat1, p1 = stats.mannwhitneyu(dur_sub, dur_cus, alternative='two-sided')
        stat2, p2 = stats.mannwhitneyu(dur_wd,  dur_we,  alternative='two-sided')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Subscriber vs Customer**")
            st.markdown(f"""
            | | Subscriber | Customer |
            |---|---|---|
            | n | {len(dur_sub):,} | {len(dur_cus):,} |
            | Median (mnt) | {dur_sub.median():.2f} | {dur_cus.median():.2f} |
            | Mean (mnt) | {dur_sub.mean():.2f} | {dur_cus.mean():.2f} |
            """)
            status1 = "✅ SIGNIFIKAN (p < 0.05)" if p1 < 0.05 else "❌ Tidak signifikan"
            st.metric("Statistik U", f"{stat1:,.0f}")
            st.metric("P-value", f"{p1:.6f}")
            st.success(status1) if p1 < 0.05 else st.warning(status1)

        with col2:
            st.markdown("**Weekday vs Weekend**")
            st.markdown(f"""
            | | Weekday | Weekend |
            |---|---|---|
            | n | {len(dur_wd):,} | {len(dur_we):,} |
            | Median (mnt) | {dur_wd.median():.2f} | {dur_we.median():.2f} |
            | Mean (mnt) | {dur_wd.mean():.2f} | {dur_we.mean():.2f} |
            """)
            status2 = "✅ SIGNIFIKAN (p < 0.05)" if p2 < 0.05 else "❌ Tidak signifikan"
            st.metric("Statistik U", f"{stat2:,.0f}")
            st.metric("P-value", f"{p2:.6f}")
            st.success(status2) if p2 < 0.05 else st.warning(status2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, d1, d2, l1, l2, title in [
            (axes[0], dur_sub, dur_cus, 'Subscriber', 'Customer', 'Durasi: Subscriber vs Customer'),
            (axes[1], dur_wd,  dur_we,  'Weekday',    'Weekend',  'Durasi: Weekday vs Weekend'),
        ]:
            ax.boxplot([d1.clip(upper=60), d2.clip(upper=60)], labels=[l1, l2],
                       patch_artist=True,
                       boxprops=dict(facecolor=COLOR_BLUE, alpha=0.6),
                       medianprops=dict(color=COLOR_RED, lw=2))
            ax.set_title(title)
            ax.set_ylabel('Durasi (menit)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown('<div class="section-title">Korelasi Spearman</div>', unsafe_allow_html=True)
        num_vars = ['tripduration_min', 'age', 'start_hour', 'start_weekday']
        corr = df[num_vars].corr(method='spearman').round(3)
        st.dataframe(corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title('Matriks Korelasi Spearman')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        groups_g = [df[df['gender']==g]['tripduration_min'].values for g in df['gender'].unique()]
        kw_h, kw_p = stats.kruskal(*groups_g)
        st.markdown(f"**Kruskal-Wallis Test (Durasi antar Gender):** H = {kw_h:.2f}, p = {kw_p:.6f}")
        if kw_p < 0.05:
            st.success("✅ Terdapat perbedaan durasi signifikan antar gender (p < 0.05)")
        else:
            st.warning("Tidak ada perbedaan signifikan antar gender")

    with tab3:
        st.markdown('<div class="section-title">Agregasi Lanjutan</div>', unsafe_allow_html=True)
        agg_hour = (df.groupby('start_hour')['tripduration_min']
                      .agg(['mean', 'median', 'count']).round(2)
                      .rename(columns={'mean':'rata2_durasi','median':'median_durasi','count':'n_trip'}))
        st.markdown("**Agregasi Durasi per Jam**")
        st.dataframe(agg_hour)

        agg_tod = (df.groupby('time_of_day')
                     .agg(n_trip=('tripduration_min','count'),
                          rata2_durasi=('tripduration_min','mean'),
                          median_durasi=('tripduration_min','median'))
                     .round(2).sort_values('n_trip', ascending=False))
        st.markdown("**Agregasi per Periode Waktu**")
        st.dataframe(agg_tod)

    with tab4:
        st.markdown('<div class="section-title">Pola Musiman</div>', unsafe_allow_html=True)
        trend_monthly = (df.groupby(['start_month','start_month_name'])
                           .agg(n_trip=('tripduration_min','count'),
                                median_durasi=('tripduration_min','median'))
                           .reset_index().sort_values('start_month'))
        trend_monthly['rolling_3m'] = trend_monthly['n_trip'].rolling(3, center=True).mean().round(0)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        colors = [PALETTE_MAIN[i % len(PALETTE_MAIN)] for i in range(len(trend_monthly))]
        bars = ax1.bar(trend_monthly['start_month_name'], trend_monthly['n_trip'],
                       color=colors, edgecolor='white', alpha=0.85)
        ax2 = ax1.twinx()
        ax2.plot(trend_monthly['start_month_name'], trend_monthly['median_durasi'],
                 color=COLOR_RED, marker='D', lw=2, ms=7, label='Median Durasi (mnt)')
        ax1.set_title('Pola Musiman: Volume Trip & Durasi Median per Bulan',
                      fontsize=13, fontweight='bold')
        ax1.set_xlabel('Bulan'); ax1.set_ylabel('Jumlah Trip')
        ax2.set_ylabel('Median Durasi (menit)', color=COLOR_RED)
        ax2.tick_params(axis='y', labelcolor=COLOR_RED)
        ax1.tick_params(axis='x', rotation=30)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
        ax2.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALISIS STASIUN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Analisis Stasiun":
    st.title("🗺️ Analisis Stasiun")

    top_n = st.slider("Tampilkan Top N Stasiun", 5, 20, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Top Stasiun Keberangkatan</div>', unsafe_allow_html=True)
        top_start = df['start_station_name'].value_counts().head(top_n).reset_index()
        top_start.columns = ['Stasiun', 'Jumlah Trip']
        fig, ax = plt.subplots(figsize=(7, top_n * 0.45 + 1))
        ax.barh(top_start['Stasiun'][::-1], top_start['Jumlah Trip'][::-1],
                color=COLOR_BLUE, edgecolor='white', alpha=0.85)
        ax.set_title(f'Top {top_n} Stasiun Keberangkatan')
        ax.set_xlabel('Jumlah Trip')
        ax.tick_params(axis='y', labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(top_start, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Top Stasiun Kedatangan</div>', unsafe_allow_html=True)
        top_end = df['end_station_name'].value_counts().head(top_n).reset_index()
        top_end.columns = ['Stasiun', 'Jumlah Trip']
        fig, ax = plt.subplots(figsize=(7, top_n * 0.45 + 1))
        ax.barh(top_end['Stasiun'][::-1], top_end['Jumlah Trip'][::-1],
                color=COLOR_ORANGE, edgecolor='white', alpha=0.85)
        ax.set_title(f'Top {top_n} Stasiun Kedatangan')
        ax.set_xlabel('Jumlah Trip')
        ax.tick_params(axis='y', labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(top_end, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Top 10 Rute Terpopuler</div>', unsafe_allow_html=True)
    top_routes = (df.groupby(['start_station_name', 'end_station_name'])
                    .size().reset_index(name='Jumlah Trip')
                    .sort_values('Jumlah Trip', ascending=False)
                    .head(10)
                    .reset_index(drop=True))
    top_routes.index += 1
    st.dataframe(top_routes, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Machine Learning":
    st.title("🤖 Segmentasi Pengguna — K-Means Clustering")

    st.info("Segmentasi berbasis perilaku: durasi perjalanan, usia, jam keberangkatan, dan hari penggunaan.")

    sample_size = st.slider("Ukuran Sample untuk Clustering", 5000, 50000, 20000, 5000)

    @st.cache_data(show_spinner="Menjalankan K-Means...")
    def run_clustering(data_hash, sample_size):
        features  = ['tripduration_min', 'age', 'start_hour', 'start_weekday']
        df_ml     = df[features].dropna().copy()
        df_sample = df_ml.sample(n=min(sample_size, len(df_ml)), random_state=42)
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(df_sample)

        inertia_list, sil_list = [], []
        k_range = range(2, 8)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia_list.append(km.inertia_)
            sil_list.append(silhouette_score(X_scaled, km.labels_, sample_size=3000, random_state=42))

        best_k   = list(k_range)[sil_list.index(max(sil_list))]
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df_sample = df_sample.copy()
        df_sample['cluster'] = km_final.fit_predict(X_scaled)

        return df_sample, list(k_range), inertia_list, sil_list, best_k

    df_sample, k_range, inertia_list, sil_list, best_k = run_clustering(len(df), sample_size)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Elbow Method & Silhouette Score</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(k_range, inertia_list, 'bo-', lw=2, ms=8)
        axes[0].axvline(best_k, color=COLOR_RED, linestyle='--', label=f'Optimal k={best_k}')
        axes[0].set_title('Elbow Method (Inertia)'); axes[0].set_xlabel('k'); axes[0].legend()

        axes[1].plot(k_range, sil_list, 'go-', lw=2, ms=8)
        axes[1].axvline(best_k, color=COLOR_RED, linestyle='--', label=f'Optimal k={best_k}')
        axes[1].set_title('Silhouette Score'); axes[1].set_xlabel('k'); axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(f'<div class="section-title">Profil Cluster (k={best_k})</div>', unsafe_allow_html=True)
        profile = (df_sample.groupby('cluster')[['tripduration_min','age','start_hour','start_weekday']]
                             .agg(['mean','median']).round(2))
        st.dataframe(profile)

    st.markdown('<div class="section-title">Visualisasi Cluster</div>', unsafe_allow_html=True)
    palette_cl = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'K-Means Clustering: Segmentasi Pengguna (k={best_k})', fontsize=13, fontweight='bold')

    for cl in sorted(df_sample['cluster'].unique()):
        mask = df_sample['cluster'] == cl
        axes[0].scatter(df_sample.loc[mask,'start_hour'], df_sample.loc[mask,'tripduration_min'],
                        c=palette_cl[cl], alpha=0.3, s=8, label=f'Cluster {cl}')
        axes[1].scatter(df_sample.loc[mask,'age'], df_sample.loc[mask,'tripduration_min'],
                        c=palette_cl[cl], alpha=0.3, s=8, label=f'Cluster {cl}')

    axes[0].set_title('Cluster: Jam vs Durasi'); axes[0].set_xlabel('Jam'); axes[0].set_ylabel('Durasi (mnt)')
    axes[0].set_ylim(0, 90); axes[0].legend(markerscale=3, fontsize=8)
    axes[1].set_title('Cluster: Usia vs Durasi'); axes[1].set_xlabel('Usia'); axes[1].set_ylabel('Durasi (mnt)')
    axes[1].set_ylim(0, 90); axes[1].legend(markerscale=3, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHT & RINGKASAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insight & Ringkasan":
    st.title("💡 Insight & Ringkasan Eksekutif")

    st.markdown("""
    ## Ringkasan Eksekutif — NYC Citi Bike Trips
    
    Analisis dataset menghasilkan empat tema utama:
    """)

    insights = [
        ("🕐 Pola Komuter Urban", COLOR_BLUE,
         "Pola bimodal rush-hour (07:00-09:00 & 17:00-19:00) dan dominasi subscriber membuktikan Citi Bike berhasil "
         "menjadi moda transportasi komuter harian. Fokus strategis: keandalan ketersediaan sepeda saat jam puncak."),
        ("👥 Segmentasi Pengguna Jelas", COLOR_ORANGE,
         "Dua persona dominan: Subscriber (komuter cepat, trip pendek) dan Customer (rekreasi/wisata, trip lebih panjang ~2-3x). "
         "Strategi pricing dan promosi harus dibedakan untuk tiap segmen."),
        ("📊 Pola Musiman Kuat", COLOR_GREEN,
         "Penggunaan meningkat signifikan di musim semi/panas dan menurun drastis di musim dingin. "
         "Rasio puncak/terendah mencerminkan kebutuhan perencanaan kapasitas armada musiman."),
        ("♿ Potensi Inklusi Belum Maksimal", COLOR_RED,
         "Kelompok lansia (65+) dan remaja sangat rendah representasinya meski lansia justru bersepeda lebih lama. "
         "Program khusus dapat meningkatkan inklusi dan memperluas basis pengguna."),
    ]

    for title, color, text in insights:
        st.markdown(f"""
        <div style="border-left: 5px solid {color}; background: #f9f9f9;
                    padding: 14px 18px; border-radius: 8px; margin-bottom: 14px;">
            <b style="font-size:1.05rem">{title}</b>
            <p style="margin-top:6px; margin-bottom:0; color:#333">{text}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Implikasi Operasional")
    ops = {
        "Rebalancing Armada": "Prioritaskan jam 07-09 dan 17-19 serta stasiun bisnis ↔ permukiman",
        "Pricing Strategy": "Loyalty reward untuk subscriber; paket wisata untuk customer",
        "Ekspansi Infrastruktur": "Perbesar kapasitas stasiun dengan volume keberangkatan ≠ kedatangan",
        "Pemasaran Inklusi": "Program khusus untuk lansia, remaja, dan gender non-binary",
        "Perencanaan Musiman": "Siapkan armada cadangan di bulan puncak; maintenance di bulan sepi",
    }
    for k, v in ops.items():
        st.markdown(f"- **{k}:** {v}")

    st.markdown("---")
    st.caption("Dashboard dibuat dengan Streamlit | Data: NYC Citi Bike Trips")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PANDUAN KUSTOMISASI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎨 Panduan Kustomisasi":
    st.title("🎨 Panduan Kustomisasi Warna Dashboard")

    st.info("Semua kustomisasi warna dilakukan di **2 tempat**: file `.streamlit/config.toml` dan konstanta warna di `app.py`.")

    # ── Bagian 1: config.toml ────────────────────────────────────────────────
    st.markdown("## 1️⃣ Warna Tema Streamlit — `.streamlit/config.toml`")
    st.markdown("File ini mengontrol warna **UI Streamlit** (tombol, tab, sidebar, slider, dll).")

    st.code("""[theme]
primaryColor             = "#4C72B0"   # ← warna aksen utama (tombol, tab aktif, garis bawah)
backgroundColor          = "#FFFFFF"   # ← latar halaman utama
secondaryBackgroundColor = "#F0F4FF"   # ← latar sidebar & card
textColor                = "#1a1a2e"   # ← warna teks utama

font = "sans serif"   # pilihan: "sans serif" | "serif" | "monospace"

[server]
maxUploadSize = 500""", language="toml")

    st.markdown("### 🎨 Preset Tema Siap Pakai")

    presets = {
        "🔵 Biru Navy (Default)": {
            "primaryColor": "#4C72B0",
            "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#F0F4FF",
            "textColor": "#1a1a2e",
        },
        "🟢 Hijau Teal": {
            "primaryColor": "#2A9D8F",
            "backgroundColor": "#F0FAF8",
            "secondaryBackgroundColor": "#D4EFEB",
            "textColor": "#1a2e2b",
        },
        "🔴 Merah Modern": {
            "primaryColor": "#E63946",
            "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#FFF0F1",
            "textColor": "#2b0a0a",
        },
        "🌙 Dark Mode": {
            "primaryColor": "#BB86FC",
            "backgroundColor": "#121212",
            "secondaryBackgroundColor": "#1E1E1E",
            "textColor": "#E0E0E0",
        },
        "🌅 Oranye Hangat": {
            "primaryColor": "#DD8452",
            "backgroundColor": "#FFFBF5",
            "secondaryBackgroundColor": "#FFF0DC",
            "textColor": "#2e1a00",
        },
    }

    selected_preset = st.selectbox("Pilih preset untuk lihat kodenya:", list(presets.keys()))
    p = presets[selected_preset]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Preview warna:**")
        st.markdown(f"""
        <div style="border-radius:10px; overflow:hidden; border: 1px solid #ddd; font-size:0.85rem;">
            <div style="background:{p['primaryColor']}; color:white; padding:10px 14px; font-weight:700;">
                primaryColor — tombol & aksen
            </div>
            <div style="background:{p['backgroundColor']}; color:{p['textColor']}; padding:10px 14px;">
                backgroundColor — latar utama
            </div>
            <div style="background:{p['secondaryBackgroundColor']}; color:{p['textColor']}; padding:10px 14px;">
                secondaryBackgroundColor — sidebar
            </div>
            <div style="background:{p['textColor']}; color:{p['backgroundColor']}; padding:10px 14px;">
                textColor — teks
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Kode untuk config.toml:**")
        st.code(f"""[theme]
primaryColor             = "{p['primaryColor']}"
backgroundColor          = "{p['backgroundColor']}"
secondaryBackgroundColor = "{p['secondaryBackgroundColor']}"
textColor                = "{p['textColor']}"

font = "sans serif"

[server]
maxUploadSize = 500""", language="toml")

    # ── Bagian 2: Warna Grafik di app.py ────────────────────────────────────
    st.markdown("---")
    st.markdown("## 2️⃣ Warna Grafik Matplotlib/Seaborn — `app.py`")
    st.markdown("Konstanta ini mengontrol warna **bar chart, line, scatter, dan boxplot** di dalam grafik.")

    st.code("""# Temukan baris ini di bagian atas app.py (sekitar baris 13-17)

PALETTE_MAIN  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
COLOR_BLUE    = '#4C72B0'   # ← warna bar utama
COLOR_ORANGE  = '#DD8452'   # ← warna aksen / perbandingan
COLOR_GREEN   = '#55A868'   # ← warna positif / distribusi usia
COLOR_RED     = '#C44E52'   # ← warna garis median & peringatan""", language="python")

    st.markdown("### 🖌️ Contoh Kombinasi Warna Grafik")
    combos = {
        "Biru & Oranye (Default)": ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
        "Hijau & Ungu":            ["#2A9D8F", "#8338EC", "#06D6A0", "#EF233C"],
        "Merah & Biru Tua":        ["#E63946", "#1D3557", "#457B9D", "#A8DADC"],
        "Pastel Soft":             ["#6A9FB5", "#E8B4B8", "#A8D5BA", "#F4C7AB"],
    }
    combo_sel = st.selectbox("Pilih kombinasi:", list(combos.keys()))
    c = combos[combo_sel]
    cols = st.columns(4)
    labels = ["COLOR_BLUE (utama)", "COLOR_ORANGE (aksen)", "COLOR_GREEN (positif)", "COLOR_RED (median)"]
    for col, hex_c, lbl in zip(cols, c, labels):
        col.markdown(f"""
        <div style="background:{hex_c}; height:60px; border-radius:8px;"></div>
        <div style="font-size:0.75rem; text-align:center; margin-top:4px; color:#555;">{hex_c}<br>{lbl}</div>
        """, unsafe_allow_html=True)

    st.code(f"""# Ganti konstanta warna di app.py dengan:
PALETTE_MAIN  = ['{c[0]}', '{c[1]}', '{c[2]}', '{c[3]}']
COLOR_BLUE    = '{c[0]}'
COLOR_ORANGE  = '{c[1]}'
COLOR_GREEN   = '{c[2]}'
COLOR_RED     = '{c[3]}'""", language="python")

    # ── Bagian 3: Sidebar header ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 3️⃣ Warna Header Sidebar")
    st.markdown("Cari blok ini di `app.py` untuk mengubah warna header pojok kiri:")

    st.code("""# Cari teks: background: linear-gradient(135deg, ...
# Ganti dua warna gradien sesuai selera:

background: linear-gradient(135deg, #4C72B0, #2a4a8a);
#                                    ↑ warna mulai   ↑ warna akhir

# Contoh gradien lain:
# Hijau teal  → linear-gradient(135deg, #2A9D8F, #1a6b64)
# Merah       → linear-gradient(135deg, #E63946, #9b1c28)
# Ungu        → linear-gradient(135deg, #8338EC, #3a0ca3)
# Oranye      → linear-gradient(135deg, #DD8452, #a0522d)""", language="css")

    st.markdown("---")
    st.success("✅ **Urutan deploy setelah edit warna:** simpan file → `git add . && git commit -m 'custom warna' && git push` → Streamlit Cloud otomatis redeploy dalam ~1 menit.")
