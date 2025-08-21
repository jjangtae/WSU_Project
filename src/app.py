import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
from datetime import datetime
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="MEMS 센서 데이터 대시보드",
    page_icon="🔬",
    layout="wide",  # Use wide layout for better data display
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Constants ---
# IMPORTANT: Change this to the actual path where your data folders are located
# Example: DATA_DIR = "/path/to/your/data/folders"
#          DATA_DIR = "C:/Users/YourUser/Documents/MEMS_Data"
# For demonstration, we'll assume a 'data' subdirectory exists where the script runs
DATA_DIR = "data"
DEFAULT_POINT_COLOR = "#1f77b4" # Default Plotly blue

# --- Helper Functions ---

@st.cache_data # Cache data loading for performance
def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        # Attempt to find and parse a datetime column
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    # Store the name of the first successfully parsed datetime column
                    st.session_state['time_col'] = col
                    break # Use the first one found
                except Exception:
                    continue # Try next column if parsing fails
        return df
    except FileNotFoundError:
        st.error(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return None
    except Exception as e:
        st.error(f"오류: CSV 파일 로딩 중 문제 발생 ({file_path}): {e}")
        return None

def find_data_files(base_dir):
    """
    Scans the base directory for date-named folders (YYYY-MM-DD)
    and returns a dictionary mapping dates to lists of CSV files.
    Returns an empty dict if base_dir doesn't exist.
    """
    data_structure = {}
    if not os.path.isdir(base_dir):
        st.sidebar.warning(f"데이터 디렉토리 '{base_dir}'를 찾을 수 없습니다.")
        st.sidebar.warning("스크립트 내 'DATA_DIR' 변수를 수정해주세요.")
        # Create the directory if it doesn't exist for demonstration purposes
        try:
            os.makedirs(base_dir)
            st.sidebar.info(f"'{base_dir}' 디렉토리를 생성했습니다. 샘플 데이터를 넣어주세요.")
            # Create sample data for demonstration
            create_sample_data(base_dir)
        except Exception as e:
            st.sidebar.error(f"'{base_dir}' 디렉토리 생성 실패: {e}")
            return {}

    # Find folders matching YYYY-MM-DD pattern
    date_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    valid_date_folders = []
    for folder in date_folders:
        try:
            datetime.strptime(folder, '%Y-%m-%d')
            valid_date_folders.append(folder)
        except ValueError:
            continue # Ignore folders not matching the date format

    # Sort dates chronologically (most recent first)
    valid_date_folders.sort(reverse=True)

    for date_folder in valid_date_folders:
        folder_path = os.path.join(base_dir, date_folder)
        # Find all .csv files within the date folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        # Sort files alphabetically or by modification time if needed
        csv_files.sort()
        if csv_files:
            # Store only the file names, not the full path initially
            data_structure[date_folder] = [os.path.basename(f) for f in csv_files]

    return data_structure

def create_sample_data(base_dir):
    """Creates sample data folders and CSV files for demonstration."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday = datetime.now() - pd.Timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')

    for date_str in [today_str, yesterday_str]:
        date_path = os.path.join(base_dir, date_str)
        os.makedirs(date_path, exist_ok=True)
        for i in range(1, 3):
            file_path = os.path.join(date_path, f"experiment_{i}.csv")
            if not os.path.exists(file_path):
                sample_df = pd.DataFrame({
                    'Timestamp': pd.to_datetime(pd.date_range(start=f'{date_str} 09:00', periods=100, freq='s')),
                    'SensorA_Voltage': np.random.rand(100) * 5 + np.linspace(0, i, 100),
                    'SensorB_Pressure': np.random.rand(100) * 10 + 50 + np.sin(np.linspace(0, i * np.pi, 100)) * 5,
                    'Temperature_C': np.random.normal(25, 0.5, 100)
                })
                sample_df.to_csv(file_path, index=False)
    st.sidebar.info("샘플 데이터 파일 생성 완료.")


# --- Initialize Session State ---
if 'selected_file_path' not in st.session_state:
    st.session_state['selected_file_path'] = None
if 'anomalies' not in st.session_state:
    st.session_state['anomalies'] = {} # Store anomalies per file {file_path: anomaly_df}
if 'time_col' not in st.session_state:
    st.session_state['time_col'] = None # To store the auto-detected time column name

# --- Sidebar: File Explorer ---
st.sidebar.title("🔬 데이터 탐색기")
st.sidebar.markdown("---")

data_files_structure = find_data_files(DATA_DIR)

if not data_files_structure:
    st.sidebar.warning("표시할 데이터 파일이 없습니다.")
    st.warning(f"'{DATA_DIR}' 폴더에 'YYYY-MM-DD' 형식의 하위 폴더를 만들고 CSV 파일을 넣어주세요.")
else:
    # Keep track if a file selection button was clicked
    file_selected = False
    selected_file_display_name = "선택된 파일 없음"

    # Iterate through dates (sorted) and create expanders
    for date_folder, file_list in data_files_structure.items():
      with st.sidebar.expander(
    f"📅 {date_folder}", expanded=bool(st.session_state.get('selected_file_path') and date_folder in st.session_state.selected_file_path)):
            for file_name in file_list:
                full_file_path = os.path.join(DATA_DIR, date_folder, file_name)
                # Use unique keys for buttons
                button_key = f"btn_{full_file_path}"
                # Highlight the selected button
                button_type = "primary" if st.session_state.selected_file_path == full_file_path else "secondary"

                if st.button(f"📄 {file_name}", key=button_key, use_container_width=True, type=button_type):
                    st.session_state.selected_file_path = full_file_path
                    st.session_state.time_col = None # Reset time column on new file selection
                    file_selected = True # Mark that a selection happened in this run
                    # Clear previous anomalies when selecting a new file
                    st.session_state.anomalies = {}
                    st.rerun() # Rerun to update the UI immediately

    # Display the name of the currently selected file
    if st.session_state.selected_file_path:
        selected_file_display_name = os.path.basename(st.session_state.selected_file_path)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"현재 선택: {selected_file_display_name}")


# --- Main Area ---
st.title("📊 압저항형 MEMS 센서 데이터 대시보드")

if st.session_state.selected_file_path:
    # Load the selected data
    df_original = load_data(st.session_state.selected_file_path)

    if df_original is not None and not df_original.empty:
        # --- Tabs for Researcher and Client Views ---
        tab_researcher, tab_client = st.tabs(["👩‍🔬 연구자 뷰", "👨‍💼 클라이언트 뷰"])

        # =========================
        # Researcher View Tab
        # =========================
        with tab_researcher:
            st.header("데이터 탐색 및 분석")
            st.markdown(f"**파일:** `{os.path.basename(st.session_state.selected_file_path)}`")

            # Make a copy for filtering and manipulation
            df = df_original.copy()

            # --- Data Filtering Section ---
            with st.expander("🔍 데이터 필터링", expanded=False):
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()

                # Allow filtering by multiple columns
                filter_cols = st.multiselect("필터링할 컬럼 선택:", options=df.columns.tolist())

                for col in filter_cols:
                    st.markdown(f"**'{col}' 필터 조건:**")
                    if col in numeric_cols:
                        min_val, max_val = st.slider(
                            f"값 범위 ({col}):",
                            min_value=float(df[col].min()),
                            max_value=float(df[col].max()),
                            value=(float(df[col].min()), float(df[col].max())),
                            key=f"slider_{col}_{st.session_state.selected_file_path}" # Unique key per file
                        )
                        df = df[df[col].between(min_val, max_val)]
                    elif col in non_numeric_cols:
                        unique_values = df[col].unique().tolist()
                        selected_values = st.multiselect(
                            f"포함할 값 ({col}):",
                            options=unique_values,
                            default=unique_values,
                             key=f"multi_{col}_{st.session_state.selected_file_path}" # Unique key per file
                        )
                        df = df[df[col].isin(selected_values)]
                    else: # Fallback for other types if necessary
                         st.write(f"'{col}' 컬럼은 현재 필터링을 지원하지 않는 타입입니다.")

                st.metric(label="필터링 후 데이터 개수", value=f"{len(df)} / {len(df_original)}")
                if len(df) != len(df_original):
                     st.info("필터가 적용되었습니다. 아래 데이터와 차트에 반영됩니다.")


            # --- Anomaly Detection Section ---
            with st.expander("🚨 이상치 탐지 및 경고", expanded=False):
                anomaly_col = st.selectbox(
                    "이상치 탐지 대상 컬럼:",
                    options=[None] + numeric_cols, # Allow selecting None
                    index=0, # Default to None
                    key=f"anomaly_col_{st.session_state.selected_file_path}"
                )

                threshold = None
                comparison = None
                anomalies_found = pd.DataFrame() # Empty dataframe initially

                if anomaly_col:
                    col1, col2 = st.columns(2)
                    with col1:
                        comparison = st.radio(
                            "비교 조건:",
                            options=[">", "<", ">=", "<=", "=="],
                            horizontal=True,
                            key=f"anomaly_comp_{st.session_state.selected_file_path}"
                        )
                    with col2:
                        threshold = st.number_input(
                            f"임계값 ({anomaly_col}):",
                            value=float(df_original[anomaly_col].mean()), # Default to mean
                            step=float(df_original[anomaly_col].std() / 10) if df_original[anomaly_col].std() > 0 else 0.1,
                            key=f"anomaly_thresh_{st.session_state.selected_file_path}"
                        )

                    if threshold is not None:
                        # Detect anomalies on the ORIGINAL dataframe
                        try:
                            if comparison == ">":
                                anomalies_found = df_original[df_original[anomaly_col] > threshold]
                            elif comparison == "<":
                                anomalies_found = df_original[df_original[anomaly_col] < threshold]
                            elif comparison == ">=":
                                anomalies_found = df_original[df_original[anomaly_col] >= threshold]
                            elif comparison == "<=":
                                anomalies_found = df_original[df_original[anomaly_col] <= threshold]
                            elif comparison == "==":
                                # Use tolerance for float comparison
                                tolerance = 1e-6
                                anomalies_found = df_original[np.isclose(df_original[anomaly_col], threshold, atol=tolerance)]

                            if not anomalies_found.empty:
                                st.warning(f"**경고:** '{anomaly_col}' 컬럼에서 임계값({comparison} {threshold:.2f})을 벗어난 데이터 {len(anomalies_found)}건 발견!")
                                # Store anomalies in session state, keyed by file path and column
                                anomaly_key = f"{st.session_state.selected_file_path}_{anomaly_col}"
                                st.session_state.anomalies[anomaly_key] = anomalies_found
                            else:
                                st.success(f"'{anomaly_col}' 컬럼에서 설정된 임계값({comparison} {threshold:.2f})을 벗어난 데이터가 없습니다.")
                                # Clear previous anomalies for this column if none found now
                                anomaly_key = f"{st.session_state.selected_file_path}_{anomaly_col}"
                                if anomaly_key in st.session_state.anomalies:
                                    del st.session_state.anomalies[anomaly_key]

                        except Exception as e:
                            st.error(f"이상치 탐지 중 오류 발생: {e}")

                # Display anomaly history/details
                if st.session_state.anomalies:
                    st.subheader("⚠️ 경고 내역")
                    for key, df_anomaly in st.session_state.anomalies.items():
                         # Extract file path and column name from the key
                         parts = key.split('_')
                         file_path_from_key = '_'.join(parts[:-1]) # Handle potential underscores in file path
                         col_name_from_key = parts[-1]
                         # Only show anomalies for the currently selected file
                         if file_path_from_key == st.session_state.selected_file_path:
                             st.markdown(f"**파일:** `{os.path.basename(file_path_from_key)}`, **컬럼:** `{col_name_from_key}` ({len(df_anomaly)} 건)")
                             st.dataframe(df_anomaly)


            # --- Data Visualization Section ---
            st.subheader("📈 데이터 시각화")

            # Use the auto-detected time column if available, otherwise let user choose
            time_col_options = [st.session_state['time_col']] if st.session_state['time_col'] else df.columns.tolist()
            x_axis = st.selectbox(
                "X축 선택 (시간 권장):",
                options=time_col_options,
                index=0, # Default to the first option (auto-detected or first column)
                key=f"xaxis_{st.session_state.selected_file_path}"
            )

            # Select columns to plot (numeric only)
            plot_cols = st.multiselect(
                "Y축 선택 (다중 선택 가능):",
                options=numeric_cols,
                default=numeric_cols[:min(len(numeric_cols), 3)], # Default to first 3 numeric cols
                key=f"plotcols_{st.session_state.selected_file_path}"
            )

            if x_axis and plot_cols:
                try:
                    # Use the filtered dataframe (df) for plotting
                    fig = px.line(
                        df,
                        x=x_axis,
                        y=plot_cols,
                        title="데이터 트렌드",
                        markers=True, # Show markers for better point identification
                        template="plotly_white" # Clean template
                    )

                    # Enhance interactivity (hover details)
                    fig.update_traces(
                        mode='lines+markers',
                        hoverinfo='all', # Show all info on hover
                        hovertemplate=f"<b>{x_axis}</b>: %{{x}}<br><b>Value</b>: %{{y}}<br><b>Variable</b>: %{{fullData.name}}<extra></extra>"
                    )

                    # Highlight anomalies on the plot if detected for a plotted column
                    if anomaly_col and anomaly_col in plot_cols and not anomalies_found.empty:
                         fig.add_scatter(
                             x=anomalies_found[x_axis],
                             y=anomalies_found[anomaly_col],
                             mode='markers',
                             marker=dict(color='red', size=8, symbol='x'),
                             name=f'이상치 ({anomaly_col})',
                             hoverinfo='skip' # Optional: disable hover for anomaly markers if too cluttered
                         )


                    fig.update_layout(
                        hovermode="x unified", # Show hover for all lines at a given x
                        legend_title_text='측정 항목',
                        xaxis_title=x_axis,
                        yaxis_title="측정 값",
                        title_font_size=20,
                        legend_font_size=12
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"차트 생성 중 오류 발생: {e}")
                    st.error("X축 또는 Y축으로 선택된 컬럼의 데이터 타입을 확인해주세요.")

            else:
                st.info("시각화를 위해 X축과 Y축 컬럼을 선택해주세요.")

            # --- Raw Data Display Section ---
            st.subheader("📄 데이터 테이블 (필터링 적용)")
            st.dataframe(df, use_container_width=True) # Display filtered data

        # =========================
        # Client View Tab
        # =========================
        with tab_client:
            st.header("📊 결과 요약 (클라이언트용)")
            st.markdown(f"**데이터 출처:** `{os.path.basename(st.session_state.selected_file_path)}`")
            st.markdown("---")

            st.subheader("주요 지표 요약")
            # Select key numeric columns for summary
            summary_cols = df_original.select_dtypes(include=np.number).columns.tolist()
            if not summary_cols:
                st.warning("요약할 숫자형 데이터 컬럼이 없습니다.")
            else:
                # Use st.metric for key stats
                cols = st.columns(min(len(summary_cols), 4)) # Show up to 4 metrics side-by-side
                for i, col_name in enumerate(summary_cols[:4]):
                    with cols[i]:
                        st.metric(
                            label=f"{col_name} (평균)",
                            value=f"{df_original[col_name].mean():.2f}",
                            delta=f"{df_original[col_name].std():.2f} (표준편차)",
                            delta_color="off" # Neutral color for std dev
                        )
                        st.metric(
                            label=f"{col_name} (최대)",
                            value=f"{df_original[col_name].max():.2f}"
                        )
                        st.metric(
                            label=f"{col_name} (최소)",
                            value=f"{df_original[col_name].min():.2f}"
                        )

                # Display descriptive statistics table
                st.subheader("통계 요약 테이블")
                try:
                    st.table(df_original[summary_cols].describe().round(2))
                except Exception as e:
                    st.error(f"통계 요약 테이블 생성 중 오류: {e}")


            st.subheader("핵심 트렌드 시각화")
            # Select one or two key columns for the client view plot
            client_plot_cols = st.multiselect(
                "클라이언트 뷰 차트에 표시할 주요 컬럼 선택:",
                options=numeric_cols,
                default=numeric_cols[:min(len(numeric_cols), 2)], # Default to first 1 or 2
                key=f"client_plotcols_{st.session_state.selected_file_path}"
            )

            client_x_axis = st.session_state.get('time_col', None) # Use detected time col if available
            if not client_x_axis and not df_original.empty: # Fallback if no time col detected
                client_x_axis = df_original.columns[0]

            if client_x_axis and client_plot_cols:
                try:
                    # Use the ORIGINAL dataframe for client summary plot
                    client_fig = px.line(
                        df_original,
                        x=client_x_axis,
                        y=client_plot_cols,
                        title="주요 데이터 변화 추이",
                        template="plotly_white"
                    )
                    client_fig.update_layout(
                        hovermode="x unified",
                        legend_title_text='측정 항목',
                        xaxis_title=client_x_axis,
                        yaxis_title="측정 값"
                    )
                    st.plotly_chart(client_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"클라이언트용 차트 생성 중 오류 발생: {e}")
            else:
                st.info("차트를 표시할 컬럼을 선택해주세요.")


            # Optional: Allow client to view raw data
            st.markdown("---")
            if st.checkbox("전체 실험 데이터 보기 (클라이언트 요청 시)", key=f"show_raw_{st.session_state.selected_file_path}"):
                st.subheader("전체 원본 데이터")
                st.dataframe(df_original, use_container_width=True)

    elif df_original is not None and df_original.empty:
        st.warning(f"선택한 파일 '{os.path.basename(st.session_state.selected_file_path)}'이 비어있습니다.")
    # else: Error message handled by load_data

else:
    # Initial state when no file is selected
    st.info("👈 사이드바에서 분석할 데이터 파일을 선택하세요.")
    if not data_files_structure:
         st.warning(f"'{DATA_DIR}' 폴더를 확인하거나 샘플 데이터를 생성해주세요.")


# --- Footer or additional info ---
st.markdown("---")
st.caption("압저항형 MEMS 센서 데이터 분석 대시보드 v1.0")
