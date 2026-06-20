# Introduction to Machine Learning 

Dor Bank 

Lecture: ML project & preprocessing 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## The learning diagram – ML course 

**==> picture [755 x 511] intentionally omitted <==**

**----- Start of picture text -----**<br>
Unknown Target Distribution<br>Probability<br>𝑃(𝑦|𝐱)<br>Distribution<br>Target function  𝑓: 𝒳→𝒴 plus noise<br>𝑃 on 𝒳<br>(ideal credit approval function)<br>(customers distribution)<br>Training Examples<br>𝐷= 𝑥 𝑥! … 𝑥" 𝐱<br>!, 𝑦! … (𝑥", 𝑦")<br>_/<br>(historical records of credit customers)<br>Error Measure '𝑓 𝑥≈𝑓(𝑥)<br>𝑒{}<br>Learning<br>Trained model<br>Algorithm<br>:<br>2𝑓  𝒳→𝒴<br>A<br>(final formula to be used)<br>Model<br>ℱ<br>**----- End of picture text -----**<br>


(set of candidate formulas) 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Today - CRISP DM 

- CRoss Industry Standard Process for Data Mining 

- Several other diagrams exist, but this is the most common 

- Real data is not simply sampled from 𝑃(𝑦|𝐱)! 

- Business context, missing values, etc… 

# • The key is “to tell the data story” 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Business understanding 

- What problem are we trying to solve? 

- Decide on clear measures to assess the success of the project 

- For our course project, this does not take much place, as it is already defined for you 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Data Understanding / Exploration 

- Usually known as EDA – Exploratory data analysis 

- Business perspective – what is the meaning of each feature? How do we expect each feature to correlate with the others? With the labels? etc. 

- Statistics & visualization: 

   - How does the data distribute? 

   - What can we learn from the data? 

   - BIG room for visualizations! 

   - Plot for purpose, not for plotting J 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Data Understanding / Exploration 

• Distribution examples: 

## Histogram 

## Boxplots 

## Statistics 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Data Understanding / Exploration 

## • More examples: 

## Categorical Features Features Correlation Distributions 

## Missing Values 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Data Preparation / Preprocessing 

- After we get a good understanding, we start to process the data and get it ready for the ML model 

- It basically includes: 

   - Outlier removal 

   - Filling missing values 

   - Dimensionality reduction 

   - Data transformation / normalization 

   - Feature engineering 

   - etc. 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Preprocessing - Outlier removal 

- These are samples that do not represent the true distribution of the data, and we do not want our model to learn from them 

- Various methods for doing so 

- Example – using boxplots (assuming the data distributes normally) 

- TIP: Do not forget to plot the data!! 

- Visualization usually helps 

from scipy import **stats** df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] 

if 𝒙 𝒊𝒌 < 𝑸𝟏 −𝟏. 𝟓7 𝑰𝑸𝑹𝑜𝑟𝒙𝒊𝒌 > 𝑸𝟑 + 𝟏. 𝟓7 𝑰𝑸𝑹 

**==> picture [86 x 14] intentionally omitted <==**

**----- Start of picture text -----**<br>
→𝑶𝒖𝒕𝒍𝒊𝒆𝒓<br>**----- End of picture text -----**<br>


Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Preprocessing – filling missing values 

- How does a missing value look like? 

- A sample/feature with many missing values – remove it 

- Fill missing values 

   - By average \ median (numerical) 

   - By most frequent \ new category (categorical) 

   - Constant / zero 

   - Serialized data (i.e with dates) – use the previous and the next 

   - KNN imputation – use the nearest neighbors data 

   - Hidden gems in sklearn! 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Preprocessing – dimensionality reduction 

- Why reduce the dimensionality? 

- Having ‘too much’ dimensions: 

   - Increases model variance 

   - Exposure to more noise than signal 

   - Curse of dimensionality – the space is sparser 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

Feature selection types • Filter method: Ranks features or feature subsets independently of the classifier • Low computational power • Independent of model type Now J a • Wrapper method: Uses a predictive model (machine learning) to score feature subsets • Requires training a model for each feature set • Lecture 4 Commonly used AFTER filter methods a • Embedded method: Performs variable selection (implicitly) in the course of model training (e.g. decision tree\Lasso) Meet along the course a Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Feature selection – Filter methods 

- Select subsets of variables as a pre-processing step, by ranking according to some scoring metric, independently of the learning model 

Input Feature variables set Input Feature Learning features subset selection algorithm 

- Relatively fast & not tuned by a given learner 

- Very commonly used 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Feature selection – Filter methods 

- Examples: 

   - Label association: 

      - Example: Choose the top 10 features correlated with the  label 

   - Low variation (sparse) features: 

      - Remove features with little variation in their value 

   - Correlated features (redundancy): 

      - Keep only one out of two highly correlated features 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Feature selection – Filter methods: scoring functions 

## • Pearson correlation 

- Measures the linear relationship between two features [continuous not categorical] 

**==> picture [410 x 45] intentionally omitted <==**

- Sample Correlation definition: 

**In python: scipy.stats.pearsonr** 

**==> picture [312 x 58] intentionally omitted <==**

- Range: [-1,1] (what does the -1,0,1 values mean?) 

• No correlation is not necessarily independent. The opposite is true however. 

- If r(X,Y) = 0.8, what does it means? 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Feature selection – Filter methods: scoring functions 

## • Mutual information 

- Measures the amount of uncertainty in X which is removed by knowing Y 

**In python: Sklearn.metrics.mutual_info_score** 

• Discrete random variables X and Y: 𝑝 ( 𝑥, 𝑦 ) 𝐼 ( 𝑋, 𝑌= 𝑝 𝑥, 𝑦log 8 8 ) C ) ~~(3)~~ 𝑝 𝑥𝑝 𝑦 Q∈F T∈H • Continuous random variables X and Y 𝑝 𝑥, 𝑦 𝐼 𝑋, 𝑌= 𝑝 𝑥, 𝑦log 𝑑𝑦𝑑𝑥 ( ) =F =H ( ) ( 𝑝 ~~Cy)~~ 𝑥𝑝 ~~()~~ 𝑦 ) 

- Nonnegative (equal 0 if 𝑋, 𝑌 are independent) 

- Isn’t restricted to linear dependency 

- Works for both discrete and continuous variables 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Preprocessing – Data transformation 

- Normalization : 10 years is not like 10,000$.... 

   - 0-1 : make all features between 0 and 1 by reducing the minimal value and dividing by the max 

      - Bounded, but sensitive to outliers 

**==> picture [238 x 36] intentionally omitted <==**

   - Bounded, less sensitive, almost linear at the center, but squeezes the edges 

- Standardize : make all features to have 0 mean and 1 variance be reducing the mean and dividing by the variance. 

   - Highly intuitive, but not bounded 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Preprocessing – Data transformation 

- Box & Cox : used to reduce the skewness 

- OneHotEncoding / Dummy variables – turning categorical features to numerical 

   - To avoid bugs, use sklearn.OneHotencoding 

   - Not pandas.dummies 

- Discretization – turning numerical features to categorical 

• 𝑥["] = 0 𝑖𝑓𝑥> 10.12.2005 𝑒𝑙𝑠𝑒1 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Preprocessing – Data transformation 

- In practice – we usually use sklearn transformers 

- Instead of ‘fit’ and ‘predict’, we have ‘fit’ and ‘transform’ 

   - The ‘fit’ learns needed parameters (like mean/variance for StandardScaler, or categories for OneHotEncoder) 

- DO NOT USE ‘fit’ ON THE TEST DATA! 

- To be extra careful, do not even fit on the validation in model selection 

Standardization (or Z-score normalization) 

**==> picture [86 x 33] intentionally omitted <==**

## MinMax Scaling 

**==> picture [155 x 34] intentionally omitted <==**

from sklearn.preprocessing import StandardScaler 

# We initialize our scaler standard_scaler = StandardScaler() 

# We fit our scaler standard_scaler.fit(X) 

- # We transform our X using the scaler we have just fit. 

from sklearn.preprocessing import MinMaxScaler 

# We initialize our scaler min_max_scaler = MinMaxScaler() 

# We fit our scaler 

min_max_scaler_scaler.fit(X) 

# We transform our X using the scaler we have just fit. 

scaled_X = min_max_scaler.transform(X) 

scaled_X = standard_scaler.transform(X) Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Preprocessing – Feature engineering 

- Create new features for the existing ones (instead / on top) 

- PCA is an example for both dimensionality reduction & feature engineering 

- Others maybe used with business understanding / domain knowledge 

- Examples 

   - Dates -> day of week 

   - Weight & Height -> BMI 

   - Grade1, grade2, grade3 -> grade average 

   - Etc. 

- Word of caution – correlation does not imply causation 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Data Preparation / Preprocessing 

- We have covered some examples 

- Feel free to use other methods which make sense 

- A lot of room for creativity 

- Key to success! 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Modeling 

- Try out different models 

- Did our preprocessing match the model? 

   - Example: using mutual information and using linear model 

- Make sure to “exploit” the full capacity of each model 

   - Hyper parameter tuning 

   - Regularization 

   - etc. 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Evaluation 

- Check the performance of the model 

- Validation, Cross Validation 

- Notice the difference between the loss function and the business metric 

   - For example, minimizing the cross entropy VS maximizing AUC 

- Common mistake:  training/fitting on the test set 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Deployment 

- After we are sure about our results we “go to production” 

- In practice, Not that simple! 

- In our project, it simply means to submit predictions for the test set 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Important notes 

- The CRISP-DM is cyclic, with the ability to go back! 

- A DS project is an iterative process 

- Tip: get as fast and naively through the first iteration 

   - Get a first benchmark for evaluation 

   - Get past technical difficulties 

   - Get initial insights on the data 

   - This holds for our project and in general! 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Important notes 

- For our project, a great project is one where the notebook is almost not needed 

- The report (and the results) tell complete story 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

## Good Luck! 

Introduction to Machine Learning – Digital Sciences for High-Tech, Dor Bank 

