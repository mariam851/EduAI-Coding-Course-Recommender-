import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from src.model.gru_kt import GRUKT
from src.recommender import recommend_next_topn
import os

# Page Config

st.set_page_config(
    page_title="EduAI Interactive Platform",
    layout="wide"
)

# Load Data
@st.cache_data
def load_platform_data():
    path = r"C:\Users\MASTER\OneDrive\Desktop\projects for master\large projects that prove that you a researcher\Intelligent Recommendation System for Coding Courses Based on Learner Behavior\Data\data\processed\ednet_sequences.csv"
    if not os.path.exists(path):
        return None, None, None

    df = pd.read_csv(path)
    items = sorted(df['item_id'].unique())
    item2idx = {item: idx + 1 for idx, item in enumerate(items)}
    idx2item = {idx: item for item, idx in item2idx.items()}
    return df, item2idx, idx2item

df, item2idx, idx2item = load_platform_data()

# Header + Student Selection
st.title("EduAI – Interactive Coding Course Recommender")
st.markdown(
    "Predict next learning steps using **GRU Knowledge Tracing** and learner behavior analytics."
)

if df is None:
    st.error("Dataset not found.")
    st.stop()

student_ids = sorted(df['subject_id'].unique())

selected_student = st.selectbox(
    "Select Student ID",
    options=student_ids,
    key="student_select"
)
st.success(f"Selected Student: **{selected_student}**")

# Student Data
student_data = df[df['subject_id'] == selected_student].sort_values("timestamp")

# Metrics
m1, m2, m3, m4 = st.columns(4)
accuracy = student_data['is_correct'].mean() * 100

with m1:
    st.metric("Total Interactions", len(student_data))
with m2:
    st.metric("Accuracy", f"{accuracy:.1f}%")
with m3:
    status_text = "Stable " if accuracy >= 60 else "At Risk "
    st.metric("Learner Status", status_text)
with m4:
    st.metric("Unique Problems", student_data['item_id'].nunique())

st.markdown("---")

# Tabs
tab_rec, tab_trace, tab_heat, tab_data = st.tabs(
    [" Recommendations", " Learning Curve", " Knowledge Heatmap", " Raw Data"]
)

# ================== Recommendations ==================
with tab_rec:
    st.subheader("Top-N Next Recommended Problems")

    model = GRUKT(len(item2idx) + 1, 64, 128)
    if os.path.exists("models/gru_model.pt"):
        model.load_state_dict(torch.load("models/gru_model.pt", map_location="cpu"))
    model.eval()

    hist_items = [item2idx[i] for i in student_data['item_id']]
    hist_corrs = student_data['is_correct'].astype(int).tolist()

    recs, probs = recommend_next_topn(model, idx2item, hist_items, hist_corrs, top_n=5)
    probs = np.array(probs)
    norm_probs = probs / probs.sum() if probs.sum() > 0 else probs

    for rank, (item, prob) in enumerate(zip(recs, norm_probs), start=1):
        st.markdown(f"### #{rank} `{item}`")
        st.progress(float(prob))
        st.caption(f"Relative relevance inside Top-N: **{prob*100:.1f}%**")

    st.markdown(
        "**Explanation:** These problems are recommended based on the student's recent mistakes and weak areas. "
        "Higher relevance means higher priority for mastery."
    )

# ================== Interactive Learning Curve ==================
with tab_trace:
    st.subheader("Student Learning Progress")

    student_data["EMA"] = student_data["is_correct"].ewm(span=5).mean()
    student_data["Cumulative Accuracy"] = student_data["is_correct"].expanding().mean()

    # Hover info
    student_data['hover_text'] = student_data.apply(
        lambda row: f"Problem: {row['item_id']}<br>Correct: {row['is_correct']}<br>Timestamp: {row['timestamp']}", axis=1
    )

    fig = px.line(
        student_data,
        y=["EMA", "Cumulative Accuracy"],
        labels={"value": "Mastery Level", "variable": "Metric"},
        title="Knowledge Growth Over Time",
        hover_data={'hover_text': True}
    )
    fig.update_traces(hovertemplate='%{customdata[0]}<br>%{y:.2f}')
    st.plotly_chart(fig, use_container_width=True)

    # Identify major drops in EMA
    student_data['EMA_diff'] = student_data['EMA'].diff()
    major_drops = student_data[student_data['EMA_diff'] < -0.2]  # Drop > 20% EMA
    drop_items = major_drops['item_id'].tolist()

    explanation = f"""
**Learning Curve Analysis**:

- Current Estimated Mastery (EMA): **{student_data['EMA'].iloc[-1]:.2f}**
- Overall Accuracy: **{accuracy:.1f}%**
- Trend: {'Stable' if student_data['EMA'].iloc[-1] >= 0.6 else 'Needs Improvement'}
- Major Drops (EMA decrease >20%): {', '.join(map(str, drop_items)) if drop_items else 'None'}
"""
    st.markdown(explanation)

    # ================== Learning Curve Interpretation ==================
student_data["EMA_diff"] = student_data["EMA"].diff()

major_drops = student_data[student_data["EMA_diff"] < -0.2]
drop_points = major_drops["item_id"].tolist()

st.markdown("### Learning Curve Interpretation")

if len(drop_points) > 0:
    st.info(
        f"""
        **What does this mean?**
        
        - The blue curve (EMA) shows *short-term learning stability*.
        - The orange curve (Cumulative Accuracy) shows *overall performance*.
        - A **sharp drop** indicates a concept that caused confusion or cognitive overload.
        
        **Significant learning drops detected at:**
        `{', '.join(map(str, drop_points))}`
        
        This suggests the student struggled suddenly after these problems.
        """
    )
else:
    st.success(
        """
        **Stable Learning Pattern**
        
        - No sharp drops detected.
        - Learning progress is consistent.
        - The student is adapting well to increasing difficulty.
        """
    )


# ================== Knowledge Heatmap ==================
with tab_heat:
    st.subheader("Knowledge State Heatmap")

    heat_df = student_data.groupby("item_id")["is_correct"].mean().reset_index().sort_values("is_correct", ascending=False)
    heat_df['hover_text'] = heat_df.apply(
        lambda row: f"Problem: {row['item_id']}<br>Mastery: {row['is_correct']*100:.1f}%", axis=1
    )

    fig = px.imshow(
        heat_df[["is_correct"]].T,
        labels=dict(x="Problems", color="Mastery"),
        x=heat_df["item_id"],
        color_continuous_scale="RdYlGn",
        title="Top Knowledge Components (Green = Mastered, Red = Weak)",
        text_auto=True
    )
    fig.update_traces(hovertemplate="%{customdata[0]}")
    fig.update_traces(customdata=heat_df['hover_text'])
    st.plotly_chart(fig, use_container_width=True)

    # Categorize mastery
    weak = heat_df[heat_df["is_correct"] < 0.5]["item_id"].tolist()
    medium = heat_df[(heat_df["is_correct"] >= 0.5) & (heat_df["is_correct"] < 0.8)]["item_id"].tolist()
    strong = heat_df[heat_df["is_correct"] >= 0.8]["item_id"].tolist()

    # ================== Personalized Learning Plan ==================
    st.subheader("Personalized Learning Plan")

    # Combine major drops with weak items
    plan_items = list(set(drop_items + weak))
    if plan_items:
        st.markdown("Based on your performance and problem mastery, focus on the following problems:")
        for i, item in enumerate(plan_items, start=1):
            # Priority: if item in drop_items AND weak → high priority
            priority = "High" if item in drop_items and item in weak else "Medium"
            st.markdown(f"{i}. `{item}` — Priority: {priority}")
    else:
        st.markdown("No urgent focus areas. Keep up the good work!")

    # Original heatmap insights
    heat_explanation = f"""
**Heatmap Insights**:

- Weak (Needs practice): {', '.join(map(str, weak)) if weak else 'None'}
- Medium (Reinforce): {', '.join(map(str, medium)) if medium else 'None'}
- Strong (Mastered): {', '.join(map(str, strong)) if strong else 'None'}
"""
    st.markdown(heat_explanation)


# ================== Raw Data ==================
with tab_data:
    st.subheader("Student Interaction Log")
    st.dataframe(student_data, use_container_width=True)

# --------------------------------------------------
st.caption("EduAI Platform • Interactive Interactive Prototype")
