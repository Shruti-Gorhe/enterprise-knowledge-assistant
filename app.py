import streamlit as st

from src.graph import build_graph


st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    h1, h2, h3 {
        color: #071a3d;
    }

    .title {
        color: #071a3d;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .answer-box {
        background-color: white;
        border-left: 5px solid #071a3d;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #dce3ee;
        margin-top: 10px;
    }

    .source-box {
        background-color: white;
        padding: 12px 15px;
        border-radius: 7px;
        border: 1px solid #dce3ee;
        margin-bottom: 8px;
    }

    div.stButton > button {
        background-color: #071a3d;
        color: white;
        border-radius: 7px;
        border: none;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #123b78;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">📚 Enterprise Knowledge Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your enterprise documents and policies.'
    '</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Ask a question",
    placeholder="Example: What are the key requirements for employees working remotely?",
    height=100,
)

ask = st.button(
    "Ask Assistant",
    use_container_width=True,
)

if ask:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                graph = build_graph()

                result = graph.invoke(
                    {
                        "question": question.strip()
                    }
                )

            except Exception as e:

                st.error("Something went wrong.")

                st.exception(e)

                st.stop()

        st.markdown("## Answer")

        answer = result.get(
            "answer",
            "No answer generated."
        )

        st.markdown(
            '<div class="answer-box">',
            unsafe_allow_html=True,
        )

        st.markdown(answer)

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("## Evaluation")

        faithfulness = result.get(
            "faithfulness",
            0.0
        )

        relevancy = result.get(
            "answer_relevancy",
            0.0
        )

        interpretation = result.get(
            "evaluation",
            {}
        ).get(
            "interpretation",
            "N/A"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Faithfulness",
                f"{faithfulness:.2f}"
            )

        with col2:

            st.metric(
                "Answer Relevancy",
                f"{relevancy:.2f}"
            )

        with col3:

            st.metric(
                "Quality",
                interpretation
            )

        st.markdown("## Sources")

        sources = result.get(
            "sources",
            []
        )

        if sources:

            seen = set()

            for source in sources:

                filename = source.get(
                    "source",
                    "Unknown document"
                )

                page = source.get(
                    "page",
                    "Unknown"
                )

                key = (
                    filename,
                    page
                )

                if key in seen:
                    continue

                seen.add(key)

                st.markdown(
                    f"""
                    <div class="source-box">
                        📄 <b>{filename}</b>
                        &nbsp; — &nbsp;
                        Page {page}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info("No sources found.")

else:

    st.info(
        "Enter a question above to search the enterprise knowledge base."
    )