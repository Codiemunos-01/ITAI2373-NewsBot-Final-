# NewsBot Intelligence System 2.0 
**Student: Codie Munos**
<br />
<br />**Dataset:** BBC News Train.csv 
<br />**📊 OVERVIEW OF BBC NEWS TRAIN DATA**
==================================================
<br />📰**Total articles:** 1490
<br />🧠**Unique categories:** 5
<br />🗞️**Names of Categories:** ['business', 'tech', 'politics', 'sport', 'entertainment']
**📈 CATEGORY DISTRIBUTION**
==================================================
<br />Category
<br />⛹️‍♂️ Sport:            346
<br />©️ Business:         336
<br />💼 Politics:        274
<br />📺 Entertainment:    273
<br />📱 Tech:             261
<br />
Research Summary: Text Classification Explanation Techniques
============================================================
📊 Multinomial Naive Bayes:
- **Technique:** Feature Log-Probabilities (log_prob) and Log-Likelihood Ratios (feature_log_prob_).
- **Explanation:** The log-probabilities of features given a class (feature_log_prob_) indicate how likely a feature is to appear in documents of a specific class. The difference in log-probabilities between classes can highlight features that are discriminative.
- **Implementation (scikit-learn):** Access the `feature_log_prob_` attribute of the trained `MultinomialNB` model. This gives log probabilities P(feature | class).
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
