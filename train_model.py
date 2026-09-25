
import csv
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# FILE SETTINGS
# =========================================================

DATASET_PATH = "language_dataset.csv"
CLEAN_DATASET_PATH = "language_dataset_clean.csv"


# =========================================================
# VALID LANGUAGES
# =========================================================

VALID_LANGUAGES = {
    "English",
    "Hindi",
    "Telugu",
    "Tamil",
    "Kannada",
    "Malayalam",
    "Spanish",
    "French",
    "German",
    "Portuguese",
    "Italian",
    "Russian",
    "Dutch",
    "Arabic",
    "Turkish",
    "Danish",
    "Greek",
    "Swedish"
}


# =========================================================
# LANGUAGE NAME CORRECTIONS
# =========================================================

LANGUAGE_FIXES = {
    "Portugeese": "Portuguese",
    "portugeese": "Portuguese",
    "PORTUGEESE": "Portuguese",

    "Sweedish": "Swedish",
    "sweedish": "Swedish",
    "SWEEDISH": "Swedish"
}


# =========================================================
# START
# =========================================================

print("\n========================================")
print("      LANGUAGE DETECTION TRAINING")
print("========================================")


# =========================================================
# READ DATASET
# =========================================================

print("\nReading dataset...")

rows = []

try:

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.reader(file)
        rows = list(reader)

except UnicodeDecodeError:

    print("UTF-8 encoding not detected.")
    print("Using Latin-1 encoding...")

    with open(
        DATASET_PATH,
        "r",
        encoding="latin1",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.reader(file)
        rows = list(reader)


# =========================================================
# CHECK DATASET
# =========================================================

if len(rows) <= 1:

    raise ValueError(
        "Dataset is empty or invalid."
    )


print(
    "\nOriginal columns:",
    rows[0]
)


# =========================================================
# REPAIR CSV ROWS
# =========================================================

clean_rows = []

for row in rows[1:]:

    if len(row) < 2:
        continue

    # Last column = Language
    language = str(
        row[-1]
    ).strip()

    # All previous columns = Text
    text = ",".join(
        row[:-1]
    ).strip()

    text = text.strip('"').strip()
    language = language.strip('"').strip()

    if not text:
        continue

    if not language:
        continue

    clean_rows.append(
        [text, language]
    )


print(
    "\nRows read:",
    len(clean_rows)
)


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(
    clean_rows,
    columns=[
        "Text",
        "Language"
    ]
)


# =========================================================
# CLEAN TEXT
# =========================================================

df["Text"] = (
    df["Text"]
    .astype(str)
    .str.strip()
)

df["Language"] = (
    df["Language"]
    .astype(str)
    .str.strip()
)


# =========================================================
# FIX LANGUAGE NAMES
# =========================================================

df["Language"] = df["Language"].replace(
    LANGUAGE_FIXES
)


# =========================================================
# REMOVE INVALID / CORRUPTED LANGUAGES
# =========================================================

before_invalid = len(df)

df = df[
    df["Language"].isin(
        VALID_LANGUAGES
    )
]

invalid_removed = (
    before_invalid - len(df)
)

print(
    "\nInvalid/corrupted rows removed:",
    invalid_removed
)


# =========================================================
# REMOVE VERY SHORT TEXT
# =========================================================

df = df[
    df["Text"].str.len() >= 3
]


# =========================================================
# REMOVE DUPLICATES
# =========================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=[
        "Text",
        "Language"
    ]
)

duplicates_removed = (
    before_duplicates - len(df)
)

print(
    "Duplicate records removed:",
    duplicates_removed
)


# =========================================================
# BALANCE DATASET
# =========================================================

TARGET_PER_LANGUAGE = 300

balanced_data = []

for language in sorted(
    VALID_LANGUAGES
):

    language_rows = df[
        df["Language"] == language
    ]

    if len(language_rows) == 0:

        print(
            f"WARNING: {language} not found."
        )

        continue

    if len(language_rows) >= TARGET_PER_LANGUAGE:

        selected = language_rows.sample(
            n=TARGET_PER_LANGUAGE,
            random_state=42
        )

    else:

        selected = language_rows.sample(
            n=TARGET_PER_LANGUAGE,
            replace=True,
            random_state=42
        )

    balanced_data.append(
        selected
    )


# =========================================================
# COMBINE BALANCED DATA
# =========================================================

df = pd.concat(
    balanced_data,
    ignore_index=True
)


# =========================================================
# SHUFFLE
# =========================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# =========================================================
# SAVE CLEAN DATASET
# =========================================================

df.to_csv(
    CLEAN_DATASET_PATH,
    index=False,
    encoding="utf-8"
)


print("\n========================================")
print(
    "FINAL DATASET RECORDS:",
    len(df)
)
print("========================================")


print("\nFinal language distribution:")

print(
    df["Language"].value_counts()
)


# =========================================================
# INPUT AND TARGET
# =========================================================

X = df["Text"]
y = df["Language"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# =========================================================
# WORD TF-IDF
# =========================================================

word_vectorizer = TfidfVectorizer(

    analyzer="word",

    ngram_range=(1, 2),

    min_df=1,

    sublinear_tf=True,

    max_features=30000
)


# =========================================================
# CHARACTER TF-IDF
# =========================================================

char_vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(2, 5),

    min_df=1,

    sublinear_tf=True,

    max_features=40000
)


# =========================================================
# COMBINE FEATURES
# =========================================================

vectorizer = FeatureUnion([

    (
        "word",
        word_vectorizer
    ),

    (
        "character",
        char_vectorizer
    )

])


print(
    "\nCreating TF-IDF features..."
)


X_train_vectorized = (
    vectorizer.fit_transform(
        X_train
    )
)


X_test_vectorized = (
    vectorizer.transform(
        X_test
    )
)


print(
    "Feature matrix:",
    X_train_vectorized.shape
)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

print(
    "\nCreating Logistic Regression model..."
)


model = LogisticRegression(

    max_iter=1000,

    C=5,

    class_weight="balanced",

    solver="lbfgs"

)


# =========================================================
# TRAIN MODEL
# =========================================================

print(
    "\nTraining language detection model..."
)


model.fit(
    X_train_vectorized,
    y_train
)


print(
    "Training completed successfully!"
)


# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(
    X_test_vectorized
)


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========================================")
print(
    f"MODEL ACCURACY: {accuracy * 100:.2f}%"
)
print("========================================")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n========================================")
print("MODEL FILES CREATED")
print("========================================")

print("✓ model.pkl")
print("✓ vectorizer.pkl")
print("✓ language_dataset_clean.csv")

print("\nTraining completed successfully!")

print("\nNext command:")
print("streamlit run app.py")

print("========================================")
