from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_traffic


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AI Network Traffic Analysis",
    page_icon="🛡️",
    layout="wide",
)


st.title("AI-Based Network Traffic Analysis")
st.caption(
    "Unsupervised anomaly detection using Isolation Forest"
)

st.info(
    "This application identifies unusual network flows. "
    "An anomaly is not proof of a cyberattack."
)


@st.cache_data
def load_csv(file_bytes):
    """Read an uploaded CSV file."""

    from io import BytesIO

    return pd.read_csv(
        BytesIO(file_bytes),
        low_memory=False
    )


uploaded_file = st.file_uploader(
    "Upload a network traffic CSV",
    type=["csv"],
)


if uploaded_file is not None:

    try:
        df = load_csv(uploaded_file.getvalue())

        st.subheader("Uploaded Traffic Data")

        st.write("Dataset shape:", df.shape)

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        if st.button(
            "Analyse Network Traffic",
            type="primary"
        ):

            with st.spinner(
                "Analysing network traffic..."
            ):

                results = predict_traffic(df)
                st.subheader("Hybrid Model Analysis")

                st.write(
                    "Selected decision strategy:",
                    results["selected_strategy"].iloc[0]
                )

                st.write(
                    "Random Forest estimates the likelihood of the "
                    "attack class from its learned classification model. "
                    "Isolation Forest identifies unusual traffic patterns."
                )

                st.caption(
                    "The Random Forest probability is model-estimated, "
                    "not necessarily a calibrated real-world probability. "
                    "The Isolation Forest score is an anomaly score, "
                    "not an attack probability."
                )

                st.dataframe(
                    results[
                        [
                            "rf_attack_probability",
                            "if_anomaly_score",
                            "rf_prediction",
                            "if_prediction",
                            "prediction",
                            "status",
                        ]
                    ],
                    use_container_width=True
                )

            total_flows = len(results)

            anomaly_mask = (
                results["status"] == "Potential Anomaly"
            )

            anomalous_flows = int(anomaly_mask.sum())

            normal_flows = (
                total_flows - anomalous_flows
            )

            anomaly_rate = (
                anomalous_flows / total_flows * 100
                if total_flows > 0
                else 0
            )

            st.subheader("Detection Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Flows",
                f"{total_flows:,}"
            )

            col2.metric(
                "Normal Flows",
                f"{normal_flows:,}"
            )

            col3.metric(
                "Potential Anomalies",
                f"{anomalous_flows:,}"
            )

            col4.metric(
                "Anomaly Rate",
                f"{anomaly_rate:.2f}%"
            )

            st.subheader("Anomaly Distribution")

            distribution = (
                results["status"]
                .value_counts()
                .rename_axis("Status")
                .to_frame("Number of Flows")
            )

            st.bar_chart(distribution)

            st.subheader("Traffic Analysis Results")

            status_filter = st.selectbox(
                "Filter results",
                [
                    "All",
                    "Normal",
                    "Potential Anomaly",
                ],
            )

            if status_filter == "All":
                filtered_results = results

            else:
                filtered_results = results[
                    results["status"] == status_filter
                ]

            st.dataframe(
                filtered_results,
                use_container_width=True
            )

            st.subheader("Most Unusual Flows")

            suspicious = results.sort_values(
                "anomaly_score",
                ascending=False
            )

            st.dataframe(
                suspicious.head(20),
                use_container_width=True
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Analysis Results",
                data=csv_data,
                file_name="network_traffic_analysis.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(
            f"Unable to analyse this file: {exc}"
        )

else:
    st.warning(
        "Upload a network traffic CSV to begin."
    )