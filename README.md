# ✨ DermaDecision: AI-Powered Skincare Intelligence

> **Stop guessing. Start glowing.**  
> A precision recommendation engine using Cosine Similarity to map your skin profile to the perfect product match.

---

## 🧬 The Problem & The Science
Finding the right skincare in a sea of thousands of products is overwhelming. **DermaDecision** removes human bias by treating skincare data as high-dimensional vectors. 

By utilizing **TF-IDF Vectorization**, we convert complex ingredient lists and product claims into numerical signatures, then use **Cosine Similarity** to calculate the mathematical "distance" between your skin's needs and the ideal product.

---

## 🚀 Key Features

*   **🎯 Hyper-Personalized:** Matches your specific skin type (Oily, Dry, Sensitive) and concerns (Acne, Aging, Redness).
*   **🧠 Vector-Based Matching:** Uses `scikit-learn` to analyze product DNA beyond just simple keywords.
*   **⚡ Instant Analysis:** Real-time processing of user inputs to generate a ranked "Top Picks" list.
*   **🎨 Streamlit UI:** A sleek, interactive interface designed for a seamless user journey.

---

## 🛠️ Technology Stack


| Category | Tools |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io) |
| **Data Science** | `Pandas` • `NumPy` • `Scikit-Learn` |
| **Interface** | `Streamlit` • `Jupyter Notebook` |
| **Logic** | TF-IDF Vectorization • Cosine Similarity |

---

## 📂 Project Architecture

```text
DermaDecision/
├── data/
│   └── skincare_products.csv   # Categorized product database
├── models/
│   └── vectorizer.pkl          # Pre-trained TF-IDF weights
├── scripts/
│   ├── processor.py            # Data cleaning & NLP pipeline
│   └── recommender.py          # Similarity logic & ranking
├── app.py                      # Streamlit Interactive UI
└── requirements.txt            # Project dependencies



🧪 How It Works
Vectorization: We transform product descriptions into a "Feature Matrix."
Input Mapping: Your preferences are converted into a single "User Vector."
Similarity Score: We calculate the angle between your vector and the database.
The Reveal: Products with the highest Cosine Similarity Score (closest to 1.0) are recommended.


🔮 Future Roadmap
API Integration: Real-time pricing and stock from Sephora/Ulta.
Reinforcement Learning: Improving matches based on "User Liked/Disliked" feedback.
Deep Scan: Using Computer Vision to analyze skin photos for automatic profile building.


# Install dependencies
pip install -r requirements.txt

# Launch the engine
streamlit run app.py
















