
// ============================================
// LANGUAGE DETECTION
// ============================================

async function detectLanguage() {

    const text =
        document.getElementById("textInput").value.trim();

    const result =
        document.getElementById("language");

    const confidence =
        document.getElementById("confidence");

    const resultBox =
        document.getElementById("result");

    if (!text) {
        alert("Please enter some text.");
        return;
    }

    resultBox.classList.remove("hidden");

    result.innerText = "Detecting...";
    confidence.innerText = "Please wait...";

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                data.message || "Detection failed."
            );
        }

        result.innerText =
            data.language;

        confidence.innerText =
            data.confidence + "%";

    }

    catch (error) {

        console.error(error);

        result.innerText = "Error";

        confidence.innerText =
            error.message;
    }
}


// ============================================
// FILE UPLOAD + DETECTION + TRANSLATION
// ============================================

async function uploadFile() {

    const fileInput =
        document.getElementById("fileInput");

    const file =
        fileInput.files[0];

    if (!file) {

        alert(
            "Please select a PDF, DOCX or TXT file."
        );

        return;
    }

    const fileResult =
        document.getElementById("fileResult");

    const fileName =
        document.getElementById("fileName");

    const fileLanguage =
        document.getElementById("fileLanguage");

    const fileConfidence =
        document.getElementById("fileConfidence");

    const resultBox =
        document.getElementById("translationResult");

    const translatedText =
        document.getElementById("translatedText");


    fileResult.classList.remove("hidden");

    fileName.innerText =
        file.name;

    fileLanguage.innerText =
        "Reading file...";

    fileConfidence.innerText =
        "Please wait...";


    try {

        // =====================================
        // UPLOAD FILE
        // =====================================

        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );


        const uploadResponse =
            await fetch(
                "http://127.0.0.1:5000/upload",
                {
                    method: "POST",
                    body: formData
                }
            );


        const uploadData =
            await uploadResponse.json();


        if (!uploadData.success) {

            throw new Error(
                uploadData.message ||
                "File processing failed."
            );
        }


        // =====================================
        // SHOW DETECTED LANGUAGE
        // =====================================

        fileLanguage.innerText =
            uploadData.language;

        fileConfidence.innerText =
            uploadData.confidence + "%";


        // =====================================
        // GET EXTRACTED TEXT
        // =====================================

        const extractedText =
            uploadData.extracted_text;


        if (
            !extractedText ||
            extractedText.trim() === ""
        ) {

            throw new Error(
                "No readable text found in this file."
            );
        }


        // =====================================
        // TARGET LANGUAGE
        // =====================================

        const targetLanguage =
            document.getElementById(
                "targetLanguage"
            ).value;


        // =====================================
        // SHOW TRANSLATING
        // =====================================

        resultBox.classList.remove(
            "hidden"
        );

        translatedText.value =
            "Translating document...";


        // =====================================
        // TRANSLATE DOCUMENT
        // =====================================

        const translationResponse =
            await fetch(
                "http://127.0.0.1:5000/translate",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        text:
                            extractedText,

                        source_language:
                            uploadData.language,

                        target_language:
                            targetLanguage

                    })
                }
            );


        const translationData =
            await translationResponse.json();


        if (!translationData.success) {

            throw new Error(
                translationData.message ||
                "Translation failed."
            );
        }


        // =====================================
        // DISPLAY TRANSLATION
        // =====================================

        translatedText.value =
            translationData.translated_text;

    }


    catch (error) {

        console.error(
            "Document translation error:",
            error
        );

        translatedText.value =
            "Translation failed: " +
            error.message;
    }
}


// ============================================
// TEXT TRANSLATION
// ============================================

async function translateText() {

    const text =
        document.getElementById(
            "textInput"
        ).value.trim();

    const targetLanguage =
        document.getElementById(
            "targetLanguage"
        ).value;

    const resultBox =
        document.getElementById(
            "translationResult"
        );

    const translatedText =
        document.getElementById(
            "translatedText"
        );


    if (!text) {

        alert(
            "Please enter text first."
        );

        return;
    }


    translatedText.value =
        "Translating...";

    resultBox.classList.remove(
        "hidden"
    );


    try {

        // Detect source language
        const detectResponse =
            await fetch(
                "http://127.0.0.1:5000/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })
                }
            );


        const detectData =
            await detectResponse.json();


        if (!detectData.success) {

            throw new Error(
                detectData.message ||
                "Language detection failed."
            );
        }


        const sourceLanguage =
            detectData.language;


        // Translate
        const response =
            await fetch(
                "http://127.0.0.1:5000/translate",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        text: text,

                        source_language:
                            sourceLanguage,

                        target_language:
                            targetLanguage

                    })
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            throw new Error(
                data.message ||
                "Translation failed."
            );
        }


        translatedText.value =
            data.translated_text;

    }


    catch (error) {

        console.error(error);

        translatedText.value =
            "Translation failed: " +
            error.message;
    }
}


// ============================================
// DOWNLOAD TRANSLATION
// ============================================

function downloadTranslation() {

    const translatedText =
        document.getElementById(
            "translatedText"
        ).value.trim();


    if (!translatedText) {

        alert(
            "No translated text available."
        );

        return;
    }


    try {

        const blob =
            new Blob(
                [translatedText],
                {
                    type:
                        "text/plain;charset=utf-8"
                }
            );


        const url =
            window.URL.createObjectURL(
                blob
            );


        const link =
            document.createElement("a");


        link.href = url;

        link.download =
            "translated_text.txt";


        document.body.appendChild(
            link
        );

        link.click();

        document.body.removeChild(
            link
        );


        window.URL.revokeObjectURL(
            url
        );

    }


    catch (error) {

        console.error(
            "Download error:",
            error
        );

        alert(
            "Download failed: " +
            error.message
        );
    }
}

