"""Giao diện Web tương tác Demo Phân loại Bình luận Toxic / Non-toxic Điện Máy Xanh.
Chạy ứng dụng: streamlit run app.py
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Thêm đường dẫn project
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocess import VietnameseTextPreprocessor

# Cấu hình trang
st.set_page_config(
    page_title="Phân loại Bình luận DMX - BTL NLP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card-toxic {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #ef4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-card-nontoxic {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 6px solid #22c55e;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .badge-toxic {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.95rem;
    }
    .badge-nontoxic {
        background-color: #22c55e;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.95rem;
    }
    .feature-tag-toxic {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px 4px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .feature-tag-nontoxic {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px 4px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_components():
    """Tải preprocessor, vectorizer và các mô hình machine learning đã lưu."""
    preprocessor = VietnameseTextPreprocessor(remove_stopwords=True, lowercase=True)

    models_dir = os.path.join(project_root, "outputs", "models")
    bow_ext = joblib.load(os.path.join(models_dir, "bow_extractor.pkl"))
    tfidf_ext = joblib.load(os.path.join(models_dir, "tfidf_extractor.pkl"))

    models = {
        "TF-IDF + Linear SVM (Mô hình tối ưu)": {
            "ext": tfidf_ext,
            "model": joblib.load(os.path.join(models_dir, "best_traditional_svm_tfidf.pkl")),
            "desc": "Accuracy 98.51% | Precision 96.15% | F1 91.15% | ROC-AUC 99.76%",
            "type": "svm",
        },
        "TF-IDF + Multinomial Naive Bayes": {
            "ext": tfidf_ext,
            "model": joblib.load(os.path.join(models_dir, "tfidf_multinomial_nb.pkl")),
            "desc": "Accuracy 94.42% | Recall 87.62% | F1 73.60% | ROC-AUC 97.65%",
            "type": "nb",
        },
        "Bag of Words (BoW) + Linear SVM": {
            "ext": bow_ext,
            "model": joblib.load(os.path.join(models_dir, "bow_linear_svm.pkl")),
            "desc": "Accuracy 98.16% | Precision 95.45% | F1 88.89% | ROC-AUC 99.69%",
            "type": "svm",
        },
        "Bag of Words (BoW) + Multinomial Naive Bayes": {
            "ext": bow_ext,
            "model": joblib.load(os.path.join(models_dir, "bow_multinomial_nb.pkl")),
            "desc": "Accuracy 95.13% | Recall 89.11% | F1 76.43% | ROC-AUC 97.10%",
            "type": "nb",
        },
    }
    return preprocessor, models


preprocessor, models_dict = load_components()

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<div class="main-title">🛡️ Hệ Thống Nhận Diện Bình Luận Độc Hại (Toxic Comments)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Bài tập lớn Xử lý Ngôn ngữ Tự nhiên — Nhóm 6 (Dữ liệu đánh giá sản phẩm Điện Máy Xanh)</div>', unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("⚙️ Cấu hình thử nghiệm")

model_choice = st.sidebar.selectbox(
    "1. Lựa chọn mô hình suy luận:",
    options=list(models_dict.keys()) + ["⚡ So sánh đồng thời cả 4 mô hình"],
    index=0,
    help="Chọn mô hình Machine Learning cụ thể hoặc chế độ so sánh đối sánh song song",
)

threshold = st.sidebar.slider(
    "2. Ngưỡng phân loại Toxic (Threshold):",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Điểm cắt xác suất để phân định nhãn Toxic (Mặc định 0.50)",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Chọn nhanh câu mẫu kiểm thử:")

preset_examples = {
    "1. Khen ngợi dịch vụ (5 sao)": "Tivi Sony 65 inch hình ảnh rất sắc nét, âm thanh sống động. Nhân viên Điện Máy Xanh giao hàng nhanh và lắp đặt chu đáo 👍",
    "2. Khiếu nại kỹ thuật 1 sao (Non-toxic)": "Máy giặt mới mua 1 tuần vắt kêu rất to và bị rò nước ra sàn. Gọi bảo hành hẹn 3 lần chưa thấy kỹ thuật xuống xử lý, quá thất vọng.",
    "3. Xúc phạm, lăng mạ nhân viên (Toxic)": "Thợ lắp máy lạnh làm ăn cẩu thả, thái độ mất dạy vô học! Nói chuyện với khách hàng như bố đời mẹ thiên hạ, đồ vô văn hóa.",
    "4. Viết tắt teencode lách luật (Toxic)": "Đmx làm ăn l.ừ.a đ.ả.o vkl, nv giao hàng thái độ bố láo mất d.ạ.y 😡 bán đồ như cc!",
    "5. Vu khống gian lận (Toxic)": "Bọn điện máy xanh này toàn một lũ lừa đảo ăn cướp tiền của dân, bán đồ phế phẩm tráo đổi linh kiện trộm cắp!",
}

default_text = preset_examples["4. Viết tắt teencode lách luật (Toxic)"]

for label, text in preset_examples.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["input_text"] = text

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Mẹo:** Thử nghiệm câu khiếu nại kỹ thuật 1 sao để thấy mô hình phân biệt chuẩn xác giữa phàn nàn dịch vụ (Non-toxic) và công kích thô tục (Toxic)!"
)

# ==============================================================================
# MAIN CONTENT TABS
# ==============================================================================
tab_demo, tab_benchmark, tab_charts, tab_errors = st.tabs(
    ["🚀 Thử nghiệm trực tiếp (Live Demo)", "📊 Bảng đo lường hiệu năng", "📈 Biểu đồ thực nghiệm", "🔍 Phân tích lỗi (Error Analysis)"]
)

with tab_demo:
    col_input, col_action = st.columns([4, 1])

    with col_input:
        current_val = st.session_state.get("input_text", default_text)
        user_input = st.text_area(
            "Nhập nội dung bình luận cần kiểm tra (hoặc chọn mẫu ở thanh bên trái):",
            value=current_val,
            height=110,
            placeholder="Nhập bất kỳ câu đánh giá hoặc khiếu nại nào tại đây...",
        )

    with col_action:
        st.write("")
        st.write("")
        btn_predict = st.button("🔍 Phân tích ngay", type="primary", use_container_width=True)
        btn_clear = st.button("🗑️ Xóa trắng", use_container_width=True)
        if btn_clear:
            st.session_state["input_text"] = ""
            st.rerun()

    if btn_predict or user_input:
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản bình luận để kiểm tra!")
        else:
            t0 = time.time()
            clean_text = preprocessor.preprocess_text(user_input, for_transformer=False)
            latency = (time.time() - t0) * 1000

            st.markdown("### 📋 Kết quả phân tích")

            # Trường hợp: So sánh đồng thời cả 4 mô hình
            if model_choice == "⚡ So sánh đồng thời cả 4 mô hình":
                results_comparison = []
                for m_name, m_info in models_dict.items():
                    ext = m_info["ext"]
                    clf = m_info["model"]
                    t_inf = time.time()
                    vec = ext.transform([clean_text])
                    prob_toxic = float(clf.predict_proba(vec)[0, 1])
                    t_inf_ms = (time.time() - t_inf) * 1000

                    is_toxic = prob_toxic >= threshold
                    label_str = "TOXIC (1)" if is_toxic else "NON-TOXIC (0)"
                    results_comparison.append({
                        "Mô hình": m_name.replace(" (Mô hình tối ưu)", ""),
                        "Kết quả dự đoán": label_str,
                        "Xác suất Toxic (%)": f"{prob_toxic * 100:.2f}%",
                        "Xác suất Non-toxic (%)": f"{(1 - prob_toxic) * 100:.2f}%",
                        "Thời gian suy luận": f"{t_inf_ms:.2f} ms",
                    })

                df_cmp = pd.DataFrame(results_comparison)

                # Hiển thị tóm tắt đồng thuận
                all_toxic = all(r["Kết quả dự đoán"] == "TOXIC (1)" for r in results_comparison)
                all_nontoxic = all(r["Kết quả dự đoán"] == "NON-TOXIC (0)" for r in results_comparison)

                if all_toxic:
                    st.error("🚨 **Cả 4 mô hình đều ĐỒNG THUẬN dự đoán: TOXIC (Bình luận độc hại)**")
                elif all_nontoxic:
                    st.success("✅ **Cả 4 mô hình đều ĐỒNG THUẬN dự đoán: NON-TOXIC (Bình luận hợp lệ)**")
                else:
                    st.warning("⚠️ **Có sự sai lệch ranh giới quyết định (Borderline) giữa các mô hình!**")

                st.table(df_cmp)

            # Trường hợp: Dự đoán bằng một mô hình cụ thể
            else:
                m_info = models_dict[model_choice]
                ext = m_info["ext"]
                clf = m_info["model"]

                t_inf = time.time()
                vec = ext.transform([clean_text])
                prob_toxic = float(clf.predict_proba(vec)[0, 1])
                prob_nontoxic = 1.0 - prob_toxic
                inf_time = (time.time() - t_inf) * 1000

                is_toxic = prob_toxic >= threshold

                col_res1, col_res2, col_res3 = st.columns([2, 2, 1])

                with col_res1:
                    if is_toxic:
                        st.markdown(
                            f"""
                            <div class="metric-card-toxic">
                                <span class="badge-toxic">TOXIC (ĐỘC HẠI)</span>
                                <h3 style="color: #b91c1c; margin-top: 0.5rem; margin-bottom: 0.2rem;">Phát hiện nội dung công kích</h3>
                                <p style="color: #7f1d1d; margin: 0;">Xác suất độc hại: <b>{prob_toxic * 100:.2f}%</b> (Ngưỡng: {threshold})</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="metric-card-nontoxic">
                                <span class="badge-nontoxic">NON-TOXIC (HỢP LỆ)</span>
                                <h3 style="color: #15803d; margin-top: 0.5rem; margin-bottom: 0.2rem;">Bình luận tiêu chuẩn</h3>
                                <p style="color: #14532d; margin: 0;">Xác suất hợp lệ: <b>{prob_nontoxic * 100:.2f}%</b> (Ngưỡng: {threshold})</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with col_res2:
                    st.write("**Phân bố độ tin cậy (Confidence):**")
                    st.progress(prob_toxic)
                    st.caption(f"Toxic: {prob_toxic * 100:.1f}% | Non-toxic: {prob_nontoxic * 100:.1f}%")

                with col_res3:
                    st.metric("Tốc độ xử lý", f"{inf_time:.2f} ms")
                    st.caption(f"Tiền xử lý: {latency:.1f} ms")

            # Quy trình xử lý ngôn ngữ chi tiết
            with st.expander("🔍 Xem chi tiết các bước tiền xử lý & Trích xuất đặc trưng", expanded=True):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**1. Văn bản gốc người dùng nhập:**")
                    st.code(user_input, language="text")
                with col_p2:
                    st.markdown("**2. Văn bản sau chuẩn hóa (Unicode NFC, Teencode, Tách từ PyVi):**")
                    st.code(clean_text if clean_text else "(chuỗi rỗng sau khi lọc stopwords)", language="text")

                # Phân tích từ khóa đóng góp (Explainable AI - XAI)
                st.markdown("**3. Phân rã từ vựng / N-grams được phát hiện trong câu:**")
                tokens = clean_text.split()
                if tokens:
                    toxic_keywords = {
                        "lừa_đảo", "lừa", "mất_dạy", "bố_láo", "vãi", "cứt", "địt", "mẹ", "đéo",
                        "chó", "khốn_nạn", "ăn_cướp", "ăn_giật", "phế_thải", "phế_phẩm", "thất_đức",
                        "vô_học", "vô_văn_hóa", "lũ", "bọn", "đm", "vkl", "vcl", "cc", "bố_đời"
                    }
                    tags_html = ""
                    for tok in tokens:
                        if tok.lower() in toxic_keywords or any(k in tok.lower() for k in ["lừa", "mất_dạy", "láo", "cứt", "địt", "cướp"]):
                            tags_html += f'<span class="feature-tag-toxic">⚠️ {tok}</span> '
                        else:
                            tags_html += f'<span class="feature-tag-nontoxic">{tok}</span> '
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.caption("Không có từ vựng đặc trưng còn lại.")

with tab_benchmark:
    st.markdown("### 📊 Kết quả đo lường đối sánh trên tập Test độc lập (2.278 mẫu)")
    st.markdown(
        "Tập kiểm thử gồm **2.076 mẫu Non-toxic (91.13%)** và **202 mẫu Toxic (8.87%)**, được thẩm định thủ công 100% (Human Verified)."
    )

    results_csv = os.path.join(project_root, "outputs", "results_summary.csv")
    if os.path.exists(results_csv):
        df_bench = pd.read_csv(results_csv)
        st.dataframe(df_bench, use_container_width=True)

    st.markdown(
        """
        > **Nhận xét chính:**
        > - **Mô hình tối ưu vận hành thực tế**: **TF-IDF + Linear SVM** đạt Accuracy **98.51%**, Precision **96.15%**, F1-Score **91.15%**, Macro-F1 **95.17%**, ROC-AUC **99.76%** cùng tốc độ suy luận nhanh nhất (0.26s trên 2.278 mẫu).
        > - **Mô hình độ nhạy cao**: **PhoBERT Transformer** đạt Recall **90.10%** (nhận diện 182/202 bình luận độc hại), tối ưu cho màng lọc rà soát sâu.
        """
    )

with tab_charts:
    st.markdown("### 📈 Biểu đồ trực quan hóa dữ liệu & Đánh giá mô hình")
    fig_dir = os.path.join(project_root, "outputs", "figures")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.image(os.path.join(fig_dir, "03_model_comparison_metrics.png"), caption="Hình 1: So sánh Accuracy, F1 và AUC 5 mô hình")
        st.image(os.path.join(fig_dir, "05_roc_curves.png"), caption="Hình 2: Đường cong ROC và diện tích AUC")
    with col_f2:
        st.image(os.path.join(fig_dir, "04_confusion_matrices_grid.png"), caption="Hình 3: Ma trận nhầm lẫn (Confusion Matrix)")
        st.image(os.path.join(fig_dir, "01_eda_distribution.png"), caption="Hình 4: Phân bố sao và nhãn theo 6 ngành hàng")

with tab_errors:
    st.markdown("### 🔍 Phân tích định tính các ca dự đoán sai (Error Analysis)")
    error_csv = os.path.join(project_root, "outputs", "error_analysis.csv")
    if os.path.exists(error_csv):
        df_err = pd.read_csv(error_csv)
        st.write(f"Tổng số trường hợp sai lệch hoặc ranh giới không đồng thuận giữa các mô hình: **{len(df_err)} mẫu**.")
        show_cols = [c for c in ["category", "product_name", "raw_text", "label", "pred_tfidf_svm", "pred_phobert"] if c in df_err.columns]
        st.dataframe(df_err[show_cols].head(30), use_container_width=True)
