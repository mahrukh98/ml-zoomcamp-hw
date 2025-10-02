# ML Zoomcamp

## Session 1 Notes

*by Mahrukh Tariq*

### Machine Learning (ML)

1. ML is the **process** of extracting **patterns** from data (*features* and *targets*).
2. The output of ML is the **model** that *encapsulates* the **contents** of these **patterns**.
3. The data (*features* without *targets*) is fed into the **model**, which then outputs the **predictions** of *targets*.

### ML vs Rule-Based System

1. Traditionally, **Rule-Based Systems** were employed for tasks such as *spam classification*.
2. But the rules need to be constantly updated, cannot cover the whole scope, and become unmaintainable when they grow too complex.
3. **ML** can solve this problem.
4. Get **data**.
5. Design and define rules → extract *features* → *feature vectors* and *targets/labels* (as defined by the user).
6. Use data and perform ML → train/fit **model** → classify/predict *SPAM (0.8)* or *GOOD (0.2)* → final decision based on threshold (e.g., ≥ 0.5).

**Rule-Based Approach:**

Data + Code → **Software (Rules)** → Output

**ML Approach:**

Data + Outcome → **ML → Model**

Data → **Model** → Outcome

### Supervised ML

1. We show examples and *teach/supervise* machines to find patterns in them, helping them predict *targets* for new examples.
2. Examples include *car price prediction* and *spam vs. non-spam mail classification*.
3. The process not only includes showing examples, but also extracting *features* and specifying the *desired output*.
4. **ML** is a branch of computer science and applied mathematics, so the model tries finding patterns through mathematics and statistics.
5. **Feature (2D array: matrix X)** — rows are *observations*, columns are *features*.
6. **Target (1D array: vector y)** — one target value for each row of the matrix.
7. **Training the model** means making predictions as close to the target *y* as possible, by training the function *g*, i.e., the ML model in reality.
    
    The essence of Supervised ML is:
    
    $$
    g(X) \approx y
    $$
    

### Types of Supervised ML

1. **Regression** → when $g(X)$ returns *continuous numerical target variables* $(-\infty, +\infty)$.
2. **Binary Classification** → when $g(X)$ returns *categorical target variables* with two classes (e.g., arrays of 1s and 0s).
3. **Multi-class Classification** → when $g(X)$ returns *more than two categorical target variables*.
4. **Ranking** → when $g(X)$ returns *ranks*, such as in recommendation systems:
    
    *“Top 6 items I would like to buy”* based on probabilistic ranking scores (e.g., Google PageRank algorithm).
    

### CRISP-DM

1 A special methodology for organizing ML projects

2 Cross-Industry Standard Process for Data Mining, invented by IBM

3 First. Business Understanding — problem statement (what), method to solve: rule-based? ML? DL ? etc. (how), success measure (how much? the extent)

e.g., using ML algorithms to reduce the amount of spam mails by 50%

4 Second. Data Understanding — analyze available data or collect more, how it is (and will be) collected — datasource(s)?  clean? reliable? large enough? being tracked/collected correctly?

e.g., report spam button for users, surveys, etc.,

5 business understanding $\leftrightarrow$ data understanding:

    improve problem statement

6 Third. Data Preparation — transformation :  data → pipelines (raw → transform → clean)  → tabular format → features matrix $X$ and labels vector $y$ 

7 Fourth. Model — training the model — essence of ML, try different models and select the best one (Logistic Regression, Decision Tree, Neural Network, etc)

8 data preparation $\leftrightarrow$ model 

    add new features, fix data issues

9 Fifth. Evaluation — test data, measure how well the model solves our problem ?

10 Evaluation $\rightarrow$ Business understanding :

adjust problem statement / stop working

11 Sixth. Evaluation + Deployment — online evaluation and deploy to production, focus: monitor and maintain service, good quality

12 Seventh. Iterate — to improve

### CRISP-DM

1. A **special methodology** for organizing ML projects.
2. **Cross-Industry Standard Process for Data Mining**, invented by IBM.
3. **First: Business Understanding** — define:
    - *Problem statement* (**what**)
    - *Method to solve* (rule-based? ML? DL?) (**how**)
    - *Success measure* (extent of success) (**how much**)
    
    *Example:* using **ML algorithms** to **reduce spam mails** by **50%**.
    
4. **Second: Data Understanding** — analyze available data or collect more:
    - How is it (and will it be) collected?
    - Data sources?
    - Clean? Reliable? Large enough?
    - Being tracked/collected correctly?
    
    *Example:* report spam button for users, surveys, etc.
    
5. **Business Understanding ↔ Data Understanding**
    - Improve problem statement iteratively.
6. **Third: Data Preparation** — transformations:
    
    ```
    raw → transform → clean → tabular format → features matrix X and labels vector y
    
    ```
    
7. **Fourth: Modeling** — train the model (essence of ML).
    - Try different models and select the best one (*Logistic Regression, Decision Tree, Neural Network*, etc.).
8. **Data Preparation ↔ Modeling**
    - Add new features, fix data issues.
9. **Fifth: Evaluation** — use test data, measure how well the model solves the problem.
10. **Evaluation → Business Understanding**
    - Adjust problem statement or stop the project.
11. **Sixth: Deployment** — online evaluation and deployment to production.
    - Focus: monitor and maintain service quality.
12. **Seventh: Iterate** — repeat steps to improve results.

### Model Selection Process

1. Essence of ML.
2. Try different models and choose the best one considering the *problem, data,* and *constraints*.
3. To ensure model performance before deployment, mimic the evaluation that happens at deployment — **validation**.
4. **Dataset split** — 80% (train) : 20% (validation).
5. $X_{\text{train}},\ y_{\text{train}}$ → from training data.
6. $X_{\text{val}},\ y_{\text{val}}$ → from validation data.
7. $g(X_{\text{val}}) = \hat{y}_{\text{val}}$
8. Compare $\hat{y}_{val}$ with $y_{val}$.
If the prediction is greater than a certain threshold, we assign it to predefined label categories:

$$
\hat{y}_{val,i} \geq 0.5 \;\Rightarrow\; 1
$$

$$
\hat{y}_{val,i} < 0.5 \;\Rightarrow\; 0
$$
    
9. **Accuracy** — ratio of correctly predicted observations to total observations.
    - Used to compare performance of different models.
    - *Example:* Neural Network = 80% accuracy vs. Decision Tree = 60% accuracy → NN is the better model.
10. Performing the same comparisons many times introduces the **multiple comparison problem** in statistics.
- Many different models evaluated against the same validation dataset.
1. Since ML models are **probabilistic**, good results can sometimes occur by chance.
- To combat this, two parts of the dataset are held out: **validation** and **test**.

**PROCESS**

1. **First:** Dataset split — 60% (train) : 20% (validation) : 20% (test).
2. Create **3 non-overlapping subsets** of data.
3. **Second:** Hide test data. Perform training (train data) and model selection (validation data).
4. **Third:** Verify that the selected model is truly the best and didn’t just get lucky on validation data → test on the **test data**.
5. After model selection, combine train and validation data → new **train dataset**.
6. Train the **final model** (the best one) → evaluate on **test data**.