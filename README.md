# log

## Papers 
- **[Predicting the Survival of Patients With Cancer From Their Initial Oncology Consultation Document Using Natural Language Processing](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2801709)**
  - Authors: Nunez et al. 
  - tag:survival_analysis, tag:nlp, tag:llm
  - [github link](https://github.com/jjnunez11/scar_nlp_survival)
  - BOW: feature is standard BOW vector, target is binary. Model is logistic regression, not survival analysis. Instead of predicting P(event in horizon) or E(num events in horizon), we set the horizon at different values (6, 12 months), and do logistic regression for each. 
  - CNN: features are word embeddings of length 300.
  - BERT: `best-base-uncased` checkpoint. Classifier head is `nn.Linear(self.bert.config.hidden_size, out_features=1)`. Loss is [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html). 

- **[Survival prediction models: an introduction to discrete-time modeling](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-022-01679-6)**
  - Authors: Suresh et al.  
  - tag:survival_analysis, tag:machine_learning
  - [github link](https://github.com/ksuresh17/autoSurv)
  - Continuous-time framing:
    - Existing ML algorithms for survival include penalized Cox regression, boosted Cox regression, survival trees and random survival forests, support vector regression, and neural networks.
    - Neural networks for survival have expanded on the Cox PH model [14, 42, 43], but again only output a prognostic index and not the survival probability, thus requiring additional estimation of the baseline hazard using the Breslow estimator
    - With random survival forests, we take b=1,…,B bootstrap samples from the original data set. For each bootstrap sample b, we grow a survival tree, where for each node we consider a set of randomly selected p candidate predictors rather than the full set of predictors, and split the node on the predictor that maximizes the dissimilarity measure. We grow the survival tree until each terminal node has no fewer than d0 unique deaths. In each terminal node, we compute the conditional cumulative hazard function using the Nelson-Aalen estimator, a non-parametric estimator of the survival function, using the subjects that are in the bootstrap sample b whose predictors place them in that terminal node.
  - Discrete-time survival models:
    - In the discrete-time framework, we assume that the available data is the same but we define the hazard function and the link between the hazard and survival functions differently. We divide the continuous survival time into a sequence of J contiguous time intervals (t0,t1],(t1,t2],…,(tJ−1,tJ], where t0=0. Within this framework the hazard, or instantaneous risk of the event, in a particular interval is the probability of an individual experiencing the event during that interval given that they have survived up to the start of that interval. **So, in discrete time, the hazard is a conditional probability rather than a rate, and as such its value lies between zero and one**.
    - Dataset format: transform from continuous-time data to a "person-period data set". See Fig 1. 
    - Due to the binomial structure of the likelihood function in Eq. (2) the discrete survival time formulation is general and any algorithm that can optimize a binomial log-likelihood can be used to obtain parameter estimates. Thus, within this approach we can apply any method for computing the probability of a binary event and can choose from various binary classification methods, from traditional regression methods to more complex machine learning approaches
    - The advantage of a discrete-time survival approach is that it does not require a proportional hazards assumption for the survival time distribution. As well, it provides a more intuitive interpretation since the hazard function represents the probability of experiencing the event in an interval given the person is alive at the start of the interval. Discrete-time models are also able to handle tied failure times without adjustments [26], as is required in Cox PH modeling due to its assumption of a continuous hazard in which ties are not possible [3].
    - Hyperparameter tuning: For the discrete-time prediction models, we additionally treat the number of intervals as a hyperparameter [49]. Tuning can be performed by identifying a reasonable range for the hyperparameter values, selecting a method by which to sample the values, and selecting a metric to assess performance. The model is fit for all of the sampled hyperparameter values and evaluated on the validation data. The tuned hyperparameter values are selected as those that optimize the performance metric. Methods of sampling the values include grid search, random search, and Bayesian optimization that uses the results from the previous iteration to improve the sampling of hyperparameter values for the current iteration [57–59]. 
    - Fig 4: survival estimates from the discrete-time models are a step function with the number of steps being equivalent to the number of tuned intervals. The steps in the Cox PH model and RSF correspond to event times in the data set.

- **[Empirical Comparison of Continuous and Discrete-time Representations for Survival Prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8232898/)**
  - Authors: Sloma et al.  
  - tag:survival_analysis, tag:machine_learning
  - The main difference between survival prediction and other prediction problems in machine learning is the presence of incomplete observations, where we have only partial information about the outcomes for some examples due to censoring. Typical machine learning algorithms cannot incorporate this partial information, so survival prediction algorithms need to be created to accommodate censoring. Classical methods for survival analysis have typically treated the time to event as a continuous outcome. The classical survival prediction problem is thus formulated as a censored regression problem. Such approaches require some assumptions on the survival times and include both semi-parametric and parametric models. The most commonly used semi-parametric model is the Cox Proportional Hazards (CoxPH) model (Cox, 1972), which makes the proportional hazards assumption about survival times.
  - An alternative approach to survival prediction is to discretize the survival times into a set of time bins. This is done by assuming some maximum time or horizon (e.g. 20 years) and then dividing time into equally-spaced bins (e.g. 20 bins each representing 1 year). This reformulates the survival prediction problem as a sequence of binary classification problems, which is a type of multi-task learning problem. Such an approach is both convenient and does not require any assumptions on the distribution of the survival times. This discrete-time approach forms the basis for many recently-proposed survival prediction algorithms (Yu et al., 2011; Li et al., 2016a; Lee et al., 2018; Giunchiglia et al., 2018; Ren et al., 2019; Wulczyn et al., 2020).
  - We examine three research questions in this paper:
    - RQ1 How much does discretizing time decrease the accuracy of a continuous-time survival prediction algorithm?
      - we compare the C-indices for the CoxPH model fit to the continuous survival times and for the CoxPH models fit to the discretized times. Across all data sets, there is very little difference between the two C-indices regardless of the number of time bins, both for the validation and the test sets. The minor differences are likely as a result of hyperparameter tuning rather than the time discretization. Thus, it appears that discretizing time has minimal effect on the accuracy of continuous-time survival prediction on real data provided that a reasonable number of bins are used.
      - Our findings for RQ1 were somewhat surprising to us, as they suggest that the actual survival times provide little value beyond their grouping into time bins! A possible explanation for this finding is that the continuous-time prediction model is not able to predict the order of closely-timed events, and thus providing finer-grained time information does not significantly improve prediction accuracy. 
    - RQ2 How does the number of discrete time bins affect the accuracy of a discrete-time survival prediction algorithm?
      - As the number of time bins gets extremely small (2 bins in the most extreme case), we are discarding a lot of information about timing of events by combining events that are not closely timed into the same bin. Thus, one might expect to see prediction accuracy increase as the number of bins increases. In the multi-task binary classification formulation used in MTLR, however, increasing the number of bins also increases the number of classification tasks, which increases the number of parameters in the model. As we found for RQ1, there is very little information to be gained in survival times beyond a certain level of granularity. Hence, we eventually begin to add more parameters without adding additional signal, which suggests that prediction accuracy should begin to decrease if the number of time bins gets too high, which we do observe.
      - **This suggests that the number of time bins should be treated as a hyperparameter to be optimized in discrete-time survival prediction models.**
    - RQ3 Does the added flexibility of the discrete-time formulation lead to an increase in accuracy that compensates for any decreases in accuracy to discretizing time?
  - Fig 1: different number of bins for same model
  - There are several advantages of using discrete time to event setting. They do not require any proportional hazard-like assumptions on the distribution of the survival times because any discrete distribution is valid. They also allow the survival prediction problem to be formulated as a sequence of binary classification problems. Finally, compared to continuous time models, interpreting the hazard functions in discrete time models becomes easier as they are expressed as conditional probabilities, and they can handle ties easily.
  - To overcome the proportional hazards assumption in the CoxPH model, Yu et al. (2011) proposed a multi-task logistic regression (MTLR) approach to survival analysis and demonstrated superior performance compared to the CoxPH model on several real data sets. Rather than the hazard function, it directly models the survival function by combining local logistic regression models so that censored observations and time varying effects of features are naturally handled. The survival time s is encoded as a binary sequence of survival statuses y whose probability of observation is represented as a generalization of a logistic regression model
  - We used the CoxPH model from the scikit-survival package (Pölsterl, 2020) and the MTLR model from PySurvival (Fotso et al., 2019–)
    - See https://square.github.io/pysurvival/models/linear_mtlr.html 

- **Deep Neural Networks for Survival Analysis Based on a Multi-Task Framework (pdf: https://arxiv.org/pdf/1801.05512)**
  - Authors: S. Fotso  
  - tag:survival_analysis, tag:machine_learning
  - The most common survival analysis modeling techniques are the Kaplan-Meier (KM) model [16] and Cox Proportional Hazard (CoxPH) model [3]. The KM model provides a very easy way to compute the survival function of an entire cohort, but doesn’t do so for a specific individual. The CoxPH model’s approach enables the end-user to predict the survival and hazard functions of an individual based on its feature vector, but exhibits the following limitations:
    - It assumes that the hazard function, or more precisely the log of the hazard ratio, is powered by a linear combination of features of an individual.
    - It relies on the proportional hazard assumption, which specifies that the hazard function of two individuals has to be constant over time.
    - The exact formula of the model that can handle ties isn’t computationally efficient, and is often rewritten using approximations, such as the Efron’s[5] or Breslow’s[2] approximations, in order to fit the model in a reasonable time.
    - The fact that the time component of the hazard function (i.e., the baseline function) remains unspecified makes the CoxPH model ill-suited for actual survival function predictions.
  - Multi-task logistic regression (MTLR) approach:
    - ...because we are not analyzing the effects of recurrent events, we need to make sure that when a unit experiences an event on interval as with s ∈ [[1, J]], its status for the remaining intervals stays the same
   
- **DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks (pdf:http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit)**
  - Authors: C. Lee at al. 
  - tag:survival_analysis, tag:machine_learning
 
 - **Analysis of recurrent event data (doi:https://doi.org/10.1016/S0169-7161(03)23034-0)**
   - Authors: Cai and Schaubel
   - tag:survival_analysis
   - Classification of semiparametric models for recurrent event data:
     - Conditional regression models
       - AG
       - PWP
       - Chang and Wang 
     - Others:
       - WLW marginal hazards
       - Pepe and Cai rate models
       - Marginal means/rates models
         - The majority of the models previously discussed in this chapter have focused on the hazard or intensity function. In the context of recurrent event data, the mean number of events is a more interpretable quantity, particularly for non-statisticians, and is often of direct interest to investigators. For these reasons, a marginal means/rates model may be
preferred, where the rate function is the derivative of the mean function.

- **[Deep recurrent survival analysis](https://arxiv.org/abs/1809.02403)**
  - Authors: Ren et al. 
  - tag:survival_analysis, tag:machine_learning
  - [github link](https://github.com/rk2900/DRSA)
  - Evaluation on three real-world datasets:
    - CLINIC
    - MUSIC
    - BIDDING
  - Compared against KM, lasso-Cox, ... DeepSurv, DeepHit
    - Metrics: c-index and average negative log probability (ANLP)
   
- **Assessment and comparison of prognostic classification schemes for survival data**
  - Authors: Graf et al.
  - tag:survival_analysis
  - Mentions Brier score and Integrated Brier Score. **Lower scores are better**. 
    - Brier score can be calculated for any time horizon t*.
      - E.g. a test set TBE is (100 FH, event_indicator=1). We can set t* = 50, calculate S_hat(t*), and incorporate this into the BS. The contribution to BS will be different at t*=60, 70, 80, ... up to 99. 
        - The above example is __Category 2: when TBE > t* (and event_indicator is either 0 or 1)__. These can always be incorporated into the BS. 
        - __Category 1: TBE <= t* and event_indicator=1__. These can always be incorporated into the BS. 
        - __Category 3: TBE <= T* and event_indicator=0__. This cannot be incorporated into the BS without reweighting.
        - **Reweighting using the KM curve**: this is done so that we can incorporate information from all three categories above 
      - We can also integrate the BS over all t* values to give a single value - integrated Brier score (IBS). 
  - Interesting methodology: fit a Cox PH model, use it to define sub-groups (based on which variables are significant). Then fit separate KM curves based on those sub-groups (instead of just doing a pooled KM curve) 
  - KM curve vs Cox PH (and other models) are evaluated in terms of BS(t*=5), and IBS (and other metrics).
  - Also see https://stats.stackexchange.com/questions/507633/intuition-behind-brier-score-weighing-step-for-censored-data

 - **[Informer model for multivariate time series forecasting](https://huggingface.co/blog/informer)**
   - Authors: Eli Simhayev et. al.
   - tag: time_series
   - Uses the `traffic_hourly` dataset, as part of the Monash Time Series Forecasting repo
   - Uses GluonTS library




## Blog posts/Videos/Slides
- **[Liquidity modeling in real estate using survival analysis](https://www.opendoor.com/articles/liquidity-modeling-real-estate-survival-analysis)**
  - tags: survival_analysis
  - Three framings:
    - regression
    - classification
    - "survival analysis": expand the dataset and then use classification. This is the "discrete-time modeling" approach. See these links:
      - https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-022-01679-6

- **[Predicting Time-to-Event Outcomes - A Tour of Survival Analysis from Classical to Modern](https://www.youtube.com/watch?v=G5Q-JuVzFE0)**
  - Presenter: George H. Chen: https://www.andrew.cmu.edu/user/georgech/ 
  - Presentation materials: https://sites.google.com/view/survival-analysis-tutorial 
  - See https://github.com/havakv/pycox for discrete-time models
  - Censoring percentage of 40% or more is highly censored, and results will not be good 
  - Deep learning models based on Cox PH:
    - DeepSurv
    - Cox-Time
    - Cox-CC (case-control)
  - Deep learning models based on discrete-time modeling:
    - Note: KM is an example of a discrete-time model.
    - Multi-task logistic regression
    - nnet-survival
    - Deep kernel survival analysis
    - DeepHit
      - Can also handle competing events
      - One of the most popular ones now 
    - Generative models:
      - Deep survival analysis (Ranganath et al, 2016)
      - Deep survival machines (Nagpal et al, 2019)
      - Neural survival-supervised topic models (Li et al, 2020)
  - KKBox dataset is one of the largest ones out there. See https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/discussion/45926
  - In healthcare, predicting time to death with ~10s of features generally gives c-index scores under 0.80

 - **[Simulation-based optimization]()**
   - Presenter: Nanjing Jian
  
  - **[Probability management - ORMS today](https://www.probabilitymanagement.org/s/Probability_Management_Part1s.pdf)**
    - Author: Sam Savage et. al.
    - Use of PowerPoint slides was highlighted as a contributing factor to the Columbia space shuttle disaster.
    - Eight examples where using point estimates give wrong answers:
      - Drunk on the highway: avg position is in the middle (alive), but actually, on average he is dead, because he doesn't stay in one position in each "iteration" - he moves back and forth
      - ...
    - SIP: stochastic information packet 


## Cards 
- 423: logits
- 186, 222: conditional independence
- 424: nonparametric conditional survival
- 308: kernel function as a generalization of inner product
- 64: overview of logistic regression
- 425, 308: kernel functions as local neighborhoods
- 426: Brier score without censoring
- 427: Brier score, with censoring
- 428: time-dependent AUC and Brier score
- 429, 430: cross-validation for prediction error and hyperparameter selection - ESL, p242
- 431: two uses of cross-validation - ISL, p175 (first page of chapter on resampling methods)
- 402: cross-entropy loss for multi-class classification
- 178: random forest training and variable importance 
- 432: mixture of experts transformers (cf. card 406 on main sub-layers in standard transformer). 
- 433: nested cross validation: https://inria.github.io/scikit-learn-mooc/python_scripts/cross_validation_nested.html (done) 
- 434: how machine learning differs from optimization: Goodfellow et. al., p268
- 199 to ~217: notes from book "Euler's Gem" 
- 435: inference vs prediction: Gerds & Kattan *Medical Risk Prediction Models*, p27
- 436: in `lifelines`, how do you get a conditional survival curve, and conditional prediction of e.g. median lifetime. See [here](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#prediction-on-censored-subjects)
- 437: should a non-significant variable be dropped from a predictive regression model? See [this](https://discourse.datamethods.org/t/model-selection-and-assessment-of-model-fit/321/5)
- 438: one-hot encoding vs BOW for text - see j29, p89 


