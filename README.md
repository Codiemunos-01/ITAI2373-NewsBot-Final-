# NewsBot Intelligence System 2.0 
**Student: Codie Munos**
<br />**Dataset:** BBC News Train.csv 
<br />
<br />**📊 OVERVIEW OF BBC NEWS TRAIN DATA**
<br />**=============================================**
<br />📰**Total articles:** 1490
<br />🧠**Unique categories:** 5
<br />🗞️**Names of Categories:** ['business', 'tech', 'politics', 'sport', 'entertainment']
<br />
<br />**📈 CATEGORY DISTRIBUTION**
<br />**==============================================**
<br />Categories:
<br />⛹️‍♂️ Sport:            346
<br />©️ Business:         336
<br />💼 Politics:        274
<br />📺 Entertainment:    273
<br />📱 Tech:             261
<br /> 
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/b0708046-192c-47e6-9d29-685d9a2764f7" />
<br />
<br /> 📝 TEXT LENGTH DISTRIBUTION
==============================================
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/30630dd9-54d9-4f22-965d-4e3ba0dc82d4" />
<br />
<br /> **🤖 STRUCTURE OF THE NEWS BOT PROJECT**:
ITAI2373-NewsBot-Final/
<br />├── README.md                    # This file
<br />├── requirements.txt             # Python dependencies
<br />├── newsbot_main.py             # Main system entry point
<br />├── config/
<br />│   ├── settings.py             # Configuration management
<br />│   └── api_keys_template.txt   # API key template
<br />├── src/
<br />│   ├── data_processing/        # Text preprocessing and validation
<br />│   ├── analysis/               # Classification, sentiment, NER, topics
<br />│   ├── language_models/        # Summarization and embeddings
<br />│   ├── multilingual/           # Language detection and translation
<br />│   ├── conversation/           # Query processing and responses
<br />│   └── utils/                  # Visualization, evaluation, export
<br />├── data/
<br />│   ├── raw/                    # Original BBC News dataset
<br />│   ├── processed/              # Cleaned and prepared data
<br />│   ├── models/                 # Trained model files
<br />│   └── results/                # Analysis outputs
<br />├── notebooks/
<br />│   ├── 01_Data_Exploration.ipynb
<br />│   ├── 02_Advanced_Classification.ipynb
<br />│   ├── 03_Topic_Modeling.ipynb
<br />│   ├── 04_Language_Models.ipynb
<br />│   ├── 05_Multilingual_Analysis.ipynb
<br />│   ├── 06_Conversational_Interface.ipynb
<br />│   └── 07_System_Integration.ipynb
<br />├── tests/                      # Comprehensive test suite
<br />├── docs/                       # Complete documentation
<br />└── reports/                    # Executive summary and reports



<br />
<br /> **PREPROCESSING RESULTS**
<br />🧹 Preprocessing all articles...
<br />✅ Preprocessing complete!

<br />📝 BEFORE AND AFTER EXAMPLES
<br />========================================================

<br />Example 1:
<br />Original: worldcom ex-boss launches defence lawyers defending former worldcom chieF bernie ebbers against a ba...
<br />Processed: worldcom exboss launch defence lawyer defending former worldcom chief bernie ebbers battery fraud ch...

<br />Example 2:
<br />Original: german business confidence slides german business confidence fell in february knocking hopes of a sp...
<br />Processed: german business confidence slide german business confidence fell february knocking hope speedy recov...

<br />Example 3:
<br />Original: bbc poll indicates economic gloom citizens in a majority of nations surveyed in a bbc world service ...
<br />Processed: bbc poll indicates economic gloom citizen majority nation surveyed bbc world service poll believe wo...

<br />Average original text length: 2233.46
<br />Average processed text length: 1481.36
<br />Unique words in original text: 35594
<br />Unique words in processed text: 22486

<br />🔥 Most common words after preprocessing:
<br />  said: 4838
<br />  year: 1872
<br />  would: 1711
<br />also: 1426
<br /> new: 1334 <br /> people: 1323
<br /> one: 1190
<br />  could: 1032
<br />  game: 949
<br /> time: 940
<br />
<br />📊 DETAILED EVALUATION: Naive Bayes
========================================================

📋 Classification Report:

             **precision    recall  f1-score   support**
     business       0.94      0.97      0.96        67
    entertainment       1.00      1.00      1.00        55
     politics       0.96      0.96      0.96        55
        sport       1.00      1.00      1.00        69
         tech       0.98      0.94      0.96        52

     accuracy                           0.98       298
    macro avg       0.98      0.98      0.98       298
     weighted avg       0.98      0.98      0.98       298

<img width="913" height="790" alt="image" src="https://github.com/user-attachments/assets/a7a6dff0-32d9-4345-8367-955017247ef9" />

<br />**RESEARCH SUMMARY: DIFFERENT TEXT CLASSIFICATION TECHNIQUES**
<br />**========================================================**
<br />📊 Multinomial Naive Bayes:
<br />- **Technique:** Feature Log-Probabilities (log_prob) and Log-Likelihood Ratios (feature_log_prob_).
-<br /> **Explanation:** The log-probabilities of features given a class (feature_log_prob_) indicate how likely a feature is to appear in documents of a specific class. The difference in log-probabilities between classes can highlight features that are discriminative.
<br />- **Implementation (scikit-learn):** Access the `feature_log_prob_` attribute of the trained `MultinomialNB` model. This gives log probabilities P(feature | class).
- **Pros:** Simple, directly available from the model, computationally efficient.
- **Cons:** Assumes feature independence (Naive Bayes assumption), explanation is based on feature presence/absence, not interaction.

📈 Support Vector Machines (SVC):
- **Technique:** Inspecting coefficients (for linear SVMs) or using permutation importance/SHAP/LIME (for kernel SVMs).
- **Explanation:** For linear SVMs, the coefficients (`coef_` attribute) represent the weights assigned to each feature, indicating their importance in separating classes. For non-linear kernel SVMs, direct coefficient interpretation is not possible.
- **Implementation (scikit-learn):** For linear SVC, access the `coef_` attribute. For kernel SVC, model-agnostic methods are generally needed.
- **Pros (Linear SVM):** Provides direct feature weights, relatively easy to understand.
- **Cons (Linear SVM):** Only applicable to linear kernels. Interpretation can be tricky with high-dimensional data.
- **Pros (Kernel SVM + Agnostic):** Can explain complex non-linear models.
- **Cons (Kernel SVM + Agnostic):** More computationally intensive, explanations are often local (per prediction), may require separate libraries (LIME, SHAP).

🌐 Model-Agnostic Techniques (LIME, SHAP):
- **Technique:** Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP).
- **Explanation:** LIME approximates the complex model locally around a specific prediction with a simple interpretable model (e.g., linear model) to show which features influenced that *single* prediction. SHAP uses concepts from cooperative game theory to attribute the prediction of a model to each feature.
- **Implementation:** Requires installing separate libraries (`lime`, `shap`). Involves wrapping the prediction function of the trained model and using explainer objects.
- **Pros:** Can explain *any* black-box model (including complex neural networks or ensemble methods), provide local explanations (per prediction), generally considered robust.
- **Cons:** Can be computationally expensive, especially for SHAP on large datasets. LIME's local approximation can sometimes be unstable. Requires careful setup for text data (e.g., using text-specific explainers or handling feature representation).

Key Considerations for Text Classification:
- **Feature Representation:** Explanations often work best on interpretable features (like TF-IDF terms or word counts), not necessarily on raw text.
- **Model Complexity:** Simple models like Naive Bayes have inherently interpretable components. Complex models benefit more from model-agnostic methods.
- **Explanation Scope:** Do you need global explanations (overall feature importance) or local explanations (why a specific article was classified in a certain way)?
- **Computational Cost:** LIME and SHAP can add significant overhead, especially when explaining many predictions.

<br />📊 NAMED ENTITY ANALYSIS
================================================
<img width="1489" height="1190" alt="image" src="https://github.com/user-attachments/assets/8506db5f-fc3d-4518-ad92-b00a53826aeb" />

  
