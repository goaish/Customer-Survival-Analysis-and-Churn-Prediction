# Customer-Survival-Analysis-and-Churn-Prediction
Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.

Telephone service companies, Internet service providers, pay TV companies, insurance firms, and alarm monitoring services, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.

Predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn. Since these models generate a small prioritized list of potential defectors, they are effective at focusing customer retention marketing programs on the subset of the customer base who are most vulnerable to churn.

In this project I aim to perform customer survival analysis and build a model which can predict customer churn. I also aim to build an app which can be used to understand why a specific customer would stop the service and to know his/her expected lifetime value.

# **Final Customer Churn Prediction App**

<img width="1920" height="1116" alt="app-pic" src="https://github.com/user-attachments/assets/3fdc29bb-ff25-4662-807e-13a8b29d09fe" />

## Project Organization
```text
.
├── Images/                             # contains images
├── static/                             # plots used in Flask App
│   └── images/
│       ├── hazard.png
│       ├── surv.png
│       ├── shap.png
│       └── new_plot.png
├── templates/                          # HTML templates for Flask App
│   └── index.html
├── Customer Survival Analysis.ipynb    # Kaplan-Meier Curve, Log-Rank Test, Cox PH Model
├── Exploratory Data Analysis.ipynb     # Data Analysis and Visualization
├── Churn Prediction Model.ipynb        # Random Forest Churn Prediction Model
├── app.py                              # Flask Application
├── app-pic.png                         # Final App Screenshot
├── explainer.bz2                       # SHAP Explainer
├── model.pkl                           # Random Forest Model
├── survivemodel.pkl                    # Cox Proportional Hazard Model
├── requirements.txt                    # Required Python Libraries
├── Procfile                            # Deployment Configuration
├── LICENSE.md                          # MIT License
└── README.md                           # Project Documentation
```
## Customer Survival Analysis
Survival Analysis: Survival analysis is generally defined as a set of methods for analyzing data where the outcome variable is the time until the occurrence of an event of interest. The event can be death, occurrence of a disease, marriage, divorce, etc. The time to event or survival time can be measured in days, weeks, years, etc.

For example, if the event of interest is heart attack, then the survival time can be the time in years until a person develops a heart attack.

Objective: The objective of this analysis is to utilize non-parametric and semi-parametric methods of survival analysis to answer the following questions.

How the likelihood of the customer churn changes over time?
How we can model the relationship between customer churn, time, and other customer characteristics?
What are the significant factors that drive customer churn?
What is the survival and Hazard curve of a specific customer?
What is the expected lifetime value of a customer?

### Kaplan-Meier Survival Curve:
<img width="794" height="544" alt="image" src="https://github.com/user-attachments/assets/ad13917b-fd87-499c-986b-24f9cf7be3a9" />

From above graph, we can say that

AS expected, for telcom, churn is relatively low. The company was able to retain more than 60% of its customers even after 72 months.
There is a constant decrease in survival probability probability between 3-60 months.
After 60 months or 5 years, survival probability decreases with a higher rate.
Log-Rank Test:

Log-rank test is carried out to analyze churning probabilities group wise and to find if there is statistical significance between groups. The plots show survival curve group wise.

<img width="264" height="187" alt="gender" src="https://github.com/user-attachments/assets/067fc43b-c25f-439b-8750-4a0a12b26259" />
<img width="260" height="185" alt="Senior Citizen" src="https://github.com/user-attachments/assets/e8f358e1-40f8-456b-8e29-d37e1867d9bf" />
<img width="272" height="185" alt="partner_1" src="https://github.com/user-attachments/assets/21c00bbe-24a3-4fad-961a-8108fd5be385" />
<img width="275" height="190" alt="dependents" src="https://github.com/user-attachments/assets/989d8067-8360-4508-bf30-eac7fbe1ac28" />
<img width="272" height="193" alt="phoneservice" src="https://github.com/user-attachments/assets/3c4bb1f0-582b-49a0-9255-5bbcbbc6ab57" />
<img width="268" height="185" alt="MultipleLines" src="https://github.com/user-attachments/assets/315a0945-29e4-41e9-99d5-11e9e89b14ee" />
<img width="272" height="194" alt="InternetService" src="https://github.com/user-attachments/assets/bb597ee5-b099-4fb4-b05c-9890fc86f2ff" />
<img width="264" height="194" alt="DeviceProtection" src="https://github.com/user-attachments/assets/bf5a6fb9-8c7c-4508-8317-d8e69a8a078e" /><img width="283" height="197" alt="PaperlessBilling" src="https://github.com/user-attachments/assets/7f3c020c-6fb8-493f-a427-1fee0fca8048" />
<img width="276" height="196" alt="paymentmethod" src="https://github.com/user-attachments/assets/d64b5688-d7a4-45ae-b336-84abf2a3d5ad" />
<img width="275" height="197" alt="StreamingMovies" src="https://github.com/user-attachments/assets/64fdcb4b-5a10-4596-b96d-32b0d0463999" />
<img width="276" height="193" alt="Contract" src="https://github.com/user-attachments/assets/9c14f5e7-f674-4590-9cb6-41ae01a21a1d" />
<img width="272" height="194" alt="TechSupport" src="https://github.com/user-attachments/assets/5e1a8ff5-422c-4360-9e42-01306d2a9c07" />
<img width="269" height="194" alt="OnlineBackup" src="https://github.com/user-attachments/assets/96464e56-d3c0-4350-9d4a-a48cbc02a979" />
<img width="271" height="192" alt="OnlineSecurity" src="https://github.com/user-attachments/assets/4d6eb10c-6f5c-48b5-b49e-592d8fcca6b1" />

## Key Findings

* Gender and phone service type are not significant predictors of customer churn, as their Log-Rank Test p-values exceed the significance threshold of 0.05.

* Younger customers with families tend to have a lower churn rate, possibly due to greater service dependence, stable income, or long-term usage patterns.

* Customers who have internet service but do not subscribe to value-added services such as:

  * Online Backup
  * Online Security
  * Device Protection
  * Tech Support
  * Streaming TV
  * Streaming Movies

  exhibit lower survival probabilities and are more likely to churn.

* Customers using internet services show a steadily declining survival probability, indicating a higher likelihood of churn over time.

* Fiber Optic internet users have higher churn rates compared to DSL users, potentially due to higher service costs despite faster speeds.

* Customers on month-to-month contracts are more likely to churn than those on long-term contracts.

* Offering incentives and promotional plans to month-to-month subscribers may improve customer retention.

* Customers using automatic payment methods demonstrate lower churn rates compared to those paying via electronic checks or mailed checks.

* Automatic payment methods reduce customer effort and improve payment consistency, contributing to higher retention.

## Survival Regression Analysis

* A Cox Proportional Hazards Model was used to perform survival regression analysis on customer data.

* The model evaluates the impact of multiple customer attributes simultaneously on the likelihood of churn over time.

* In this model, the hazard rate represents the probability of a customer churning at a specific time, given that they have remained active until that point.

* The Cox model effectively identifies risk factors associated with customer churn and provides interpretable coefficients for each feature.

* Features with positive coefficients increase churn risk, while features with negative coefficients improve customer retention and survival probability.

* The model provides valuable insights for designing targeted customer retention strategies and reducing churn.

<img width="1486" height="806" alt="image" src="https://github.com/user-attachments/assets/1088fe86-0983-4a29-b9bd-c05d8fc61112" />

Using this model we can calculate the survival curve and hazard curve of any customer as shown below. These plots are useful to know the remaining life of a customer.

<img width="299" height="207" alt="hazard" src="https://github.com/user-attachments/assets/0ea79331-0682-4be4-8682-16840d135ed8" />
<img width="299" height="207" alt="hazard" src="https://github.com/user-attachments/assets/8562d2b0-11cc-4312-a0af-4f272d05ca5c" />

## Customer Churn Prediction
I aim to implement a machine learning model to accurately predict if the customer will churn or not.
# Analysis
# Churn and Tenure Relationship:
<img width="647" height="329" alt="tenure-churn" src="https://github.com/user-attachments/assets/3f8266e6-64d1-4a29-9e0a-6a9a03c8407f" />

As we can see the higher the tenure, the lesser the churn rate. This tells us that the customer becomes loyal with the tenure.

# Tenure Distrbution by Various Services:
<img width="262" height="187" alt="tenure-dist" src="https://github.com/user-attachments/assets/820285ae-cd71-46bf-8563-548a0b0a4044" />

When the customers are new they do not opt for various services and their churning rate is very high. This can be seen in above plot for Streaming Movies and this holds true for all various services.

# Internet Service By Contract Type:
<img width="277" height="186" alt="internetservice-contract" src="https://github.com/user-attachments/assets/4252fd01-cc68-45b7-a8e3-0c5e9cd80b75" />

Many of the people of who opt for month-to-month Contract choose Fiber optic as Internet service and this is the reason for higher churn rate for fiber optic Internet service type.

# Payment method By Contract Type:
<img width="499" height="223" alt="payment-contract" src="https://github.com/user-attachments/assets/6c7b8a3a-ae29-41e3-a60e-4d997e3ce0a5" />

People having month-to-month contract prefer paying by Electronic Check mostly or mailed check. The reason might be short subscription cancellation process compared to automatic payment.

# Monthly Charges:

<img width="272" height="181" alt="monthlycharges" src="https://github.com/user-attachments/assets/4e5cfc61-bc88-43fd-a010-33711ee04f03" />

As we can see the customers paying high monthly fees churn more.

# Modelling
For the modelling, I will use tress based Ensemble method as we do not have linearity in this classification problem. Also, we have a class imbalance of 1:3 and to combat it I will assign class weightage of 1:3 which means false negatives are 3 times costlier than false positives. I built a model on 80% of data and validated model on remaining 20% of data keeping in mind that I do not have data leakage. The random forest model has many hyperparameters and I tuned them using Grid Search Cross Validation while making sure that I do not overfit.

The final model resulted in 0.62 F1 score and 0.85 ROC-AUC. The resulting plots can be seen below.
<img width="548" height="267" alt="model_1" src="https://github.com/user-attachments/assets/52be7a4b-4c77-424b-9e01-aa96d2439014" />
<img width="554" height="401" alt="model_feat_imp" src="https://github.com/user-attachments/assets/cc499ade-e856-4d42-aede-cd48170b69e2" />

From the feature importance plot, we can see which features govern the customer churn.

# Explainability
We can explain and understand the Random forest model using explainable AI modules such as Permutation Importance, Partial Dependence plots and Shap values.

Permutation Importance shows feature importance by randomly shuffling feature values and measuring how much it degrades our performance.
<img width="289" height="348" alt="eli51" src="https://github.com/user-attachments/assets/6392faf9-8d13-4de9-98dc-65ebfd03ca6c" />
<img width="290" height="177" alt="eli52" src="https://github.com/user-attachments/assets/e50e2727-d42f-4850-ab1f-a4721fa56930" />

Partial dependence plot is used to see how churning probability changes across the range of particular feature. For example, in below graph of tenure group, the churn probability decreases at a higher rate if a person is in tenure group 2 compared to 1.
<img width="635" height="401" alt="pdp_tenure" src="https://github.com/user-attachments/assets/a7899ccc-5317-4840-86a0-2de45c0dfabc" />
<img width="674" height="396" alt="pdp_contract" src="https://github.com/user-attachments/assets/78b563de-cda9-41df-a420-2112fb0e8dea" />
<img width="670" height="402" alt="pdp_monthly_charges" src="https://github.com/user-attachments/assets/c25c1a05-8362-477f-be39-b8eb8ea0ea62" />
<img width="672" height="398" alt="pdp_total_charges" src="https://github.com/user-attachments/assets/a461bcfb-3ed2-4acd-97c8-0d483677ab9b" />

Shap values (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. In below plot we can see that why a particual customer's churning probability is less than baseline value and which features are causing them.

<img width="1577" height="366" alt="shap" src="https://github.com/user-attachments/assets/0a8e46bc-57fc-4a12-8ec9-19173b0f4964" />


## Flask App
I saved the final tuned Random Forest model and deployed it using Flask web app. Flask is a micro web framework written in Python. It is designed to make getting started quick and easy, with the ability to scale up to complex applications. I saved the shap value explainer tuned using random forest model to show shap plots in app. I have also utilized the cox-proportional hazard model to show survival curve and hazard curve, and to calculate expected customer lifetime value.

The final app shows churning probability, gauge chart of how severe a customer is and shap values based on customer's data. The final app layout can be seen above.

