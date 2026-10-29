import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pandas as pd
import streamlit as st

ART = os.environ.get("EMOJI_ART", "emoji_sentiment_artifacts")
sys.path.insert(0, ART)
import emoji_sentiment_lib as esl

st.set_page_config(page_title="Emoji Sentiment", page_icon="😊", layout="centered")

@st.cache_resource(show_spinner="Loading models from disk…")
def load_predictor():
    return esl.SentimentPredictor(ART)

predictor = load_predictor()
NAMES = esl.SentimentPredictor.MODEL_NAMES
FACE = {"negative": "😠", "neutral": "😐", "positive": "😊"}

st.title("Emoji-based Sentiment Classification")



text = st.text_area("Your text", height=90, placeholder="type a sentence with emojis…")

if st.button("Predict", type="primary") and text.strip():
    r = predictor.predict(text.strip())
    d = r["D"]
    st.subheader(f"{FACE[d['label']]}  {d['label'].upper()}")
    st.caption("Prediction from model D (ours)")
    for lab, p in zip(esl.LABELS, d["probs"]):
        st.progress(float(p), text=f"{lab}: {p:.1%}")

    # if d["emoji_weights"]:
    #     st.markdown(f"**Gate:** {d['gate']:.0%} emoji branch · {1 - d['gate']:.0%} text branch")
    #     cols = st.columns(len(d["emoji_weights"]))
    #     for col, (e, w) in zip(cols, d["emoji_weights"]):
    #         col.metric(e, f"{w:.2f}")
    #     st.caption("Number under each emoji = its attention weight inside the emoji branch")
    # else:
    #     st.markdown("**Gate:** no emojis found → 100% text branch")

    # st.markdown("#### All three models")
    # rows = [{"model": NAMES[k], "prediction": f"{FACE[v['label']]} {v['label']}",
    #          **{l: round(float(x), 3) for l, x in zip(esl.LABELS, v["probs"])}} for k, v in r.items()]
    # st.dataframe(pd.DataFrame(rows).set_index("model"))


