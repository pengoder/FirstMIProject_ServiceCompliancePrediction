# First MI Project - HEDIS Service Compliance Prediction

#### HEDIS initiated a member outreach program to incentivize members to take preventive tests/screenings
#### I was curious if we can know the likelihood of member's compliance rate in a specific measure; if we know, we can modify our outreach strategy to promote the rate.

#### This is an end-to-end data product, from data collection to story telling
- Feature Extraction
  - Demographic
  - Medical/Rx Claims
    - IP/ER
    - High Cost
    - Chronic Condition
    - etc.
  - Benefit
  - PCP/IPA HEDIS Rates
- Data Collection
  - Time frame to choose
  - Measures to choose
- Data Preprcessing
  - Scaling
  - Sampling
  - Imputation
- Model Engineering
  - Try linear regressions first to how they fit
  - Then use MI classification algorithms
    - SVC Linear
    - Random Forest
    - Gradient Boost
 - Results Interpretation
