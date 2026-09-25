
import streamlit as st
import joblib
import requests


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Language Detection AI",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# =========================================================
# LANGUAGE CODES
# =========================================================

LANGUAGE_CODES = {

    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Italian": "it",
    "Russian": "ru",
    "Dutch": "nl",
    "Arabic": "ar",
    "Turkish": "tr",
    "Danish": "da",
    "Greek": "el",
    "Swedish": "sv"
}


# =========================================================
# HISTORY
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        🌍 Language Detection AI
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
        Detect and translate text using NLP and Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# TEXT INPUT
# =========================================================

st.subheader("✍️ Enter Your Text")

user_text = st.text_area(
    "Type or paste a sentence below:",
    height=160,
    placeholder="Example: Hello, how are you today?"
)


# =========================================================
# DETECT BUTTON
# =========================================================

detect_button = st.button(
    "🔍 Detect Language",
    width="stretch"
)


# =========================================================
# DETECTION
# =========================================================

if detect_button:

    if user_text.strip() == "":

        st.warning(
            "⚠️ Please enter some text first."
        )

    else:

        text_vector = vectorizer.transform(
            [user_text]
        )

        detected_language = model.predict(
            text_vector
        )[0]

        probabilities = model.predict_proba(
            text_vector
        )[0]

        confidence = (
            max(probabilities) * 100
        )

        st.session_state["detected_language"] = (
            detected_language
        )

        st.session_state["confidence"] = (
            confidence
        )

        st.session_state["input_text"] = (
            user_text
        )


# =========================================================
# DETECTION RESULT
# =========================================================

if "detected_language" in st.session_state:

    detected_language = (
        st.session_state["detected_language"]
    )

    confidence = (
        st.session_state["confidence"]
    )


    st.success(
        f"🌍 Detected Language: {detected_language}"
    )

    st.info(
        f"📊 Confidence: {confidence:.2f}%"
    )

    st.divider()


    # =====================================================
    # TRANSLATION
    # =====================================================

    st.subheader("🔄 Translate Text")

    target_language = st.selectbox(
        "Select language to translate:",
        list(LANGUAGE_CODES.keys()),
        index=0
    )


    translate_button = st.button(
        "🌐 Translate",
        width="stretch"
    )


    # =====================================================
    # TRANSLATE
    # =====================================================

    if translate_button:

        original_text = (
            st.session_state["input_text"]
        )

        source_code = LANGUAGE_CODES.get(
            detected_language
        )

        target_code = LANGUAGE_CODES.get(
            target_language
        )


        # -------------------------------------------------
        # SAME LANGUAGE
        # -------------------------------------------------

        if detected_language == target_language:

            translated_text = original_text

            st.success(
                "✅ Text is already in the selected language."
            )


        # -------------------------------------------------
        # TRANSLATE
        # -------------------------------------------------

        elif source_code and target_code:

            try:

                if len(
                    original_text.encode("utf-8")
                ) > 500:

                    st.warning(
                        "⚠️ Please enter a shorter sentence."
                    )

                    translated_text = None

                else:

                    response = requests.get(

                        "https://api.mymemory.translated.net/get",

                        params={
                            "q": original_text,
                            "langpair":
                            f"{source_code}|{target_code}",
                            "mt": "1"
                        },

                        timeout=20
                    )


                    if response.status_code == 200:

                        data = response.json()

                        translated_text = (
                            data.get(
                                "responseData",
                                {}
                            ).get(
                                "translatedText"
                            )
                        )

                    else:

                        translated_text = None


            except Exception:

                translated_text = None


        else:

            translated_text = None


        # =================================================
        # SHOW TRANSLATION
        # =================================================

        if translated_text:

            st.success(
                "✅ Translation completed!"
            )


            st.text_area(
                "Translated Text:",
                translated_text,
                height=150
            )


            # =================================================
            # COPY / DOWNLOAD
            # =================================================

            st.subheader(
                "📥 Translation Options"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.download_button(

                    label="📥 Download Translation",

                    data=translated_text,

                    file_name="translated_text.txt",

                    mime="text/plain",

                    width="stretch"

                )


            with col2:

                st.download_button(

                    label="📋 Save Full Result",

                    data=(
                        "Language Detection Result\n"
                        "===========================\n\n"
                        f"Original Text:\n"
                        f"{original_text}\n\n"
                        f"Detected Language:\n"
                        f"{detected_language}\n\n"
                        f"Confidence:\n"
                        f"{confidence:.2f}%\n\n"
                        f"Target Language:\n"
                        f"{target_language}\n\n"
                        f"Translation:\n"
                        f"{translated_text}\n"
                    ),

                    file_name="language_detection_result.txt",

                    mime="text/plain",

                    width="stretch"

                )


            # =================================================
            # SAVE HISTORY
            # =================================================

            history_item = {

                "Text": original_text,

                "Detected Language":
                detected_language,

                "Confidence":
                f"{confidence:.2f}%",

                "Target Language":
                target_language,

                "Translation":
                translated_text
            }


            st.session_state.history.append(
                history_item
            )


        elif translated_text == "":

            st.error(
                "❌ Translation was empty."
            )

        elif not (
            detected_language == target_language
        ):

            st.error(
                "❌ Translation failed. "
                "Please try again."
            )


# =========================================================
# HISTORY SECTION
# =========================================================

st.divider()

st.subheader(
    "🕒 Translation History"
)


if len(st.session_state.history) == 0:

    st.info(
        "No translation history yet."
    )

else:

    for number, item in enumerate(
        reversed(
            st.session_state.history
        ),
        start=1
    ):

        st.markdown(
            f"""
            ### {number}. 🌐
            **Original Text:** {item["Text"]}

            **Detected Language:** {item["Detected Language"]}

            **Confidence:** {item["Confidence"]}

            **Translated To:** {item["Target Language"]}

            **Translation:** {item["Translation"]}
            """
        )

        st.divider()


    if st.button(
        "🗑️ Clear History",
        width="stretch"
    ):

        st.session_state.history = []

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <p style="text-align:center;">
        🌍 NLP Language Detection & Translation System
        <br>
        Machine Learning • TF-IDF • Logistic Regression
    </p>
    """,
    unsafe_allow_html=True
)

