# log

## Courses 
- **Coursera/deeplearning.ai - Data engineering**
   - Started Course 2!
 
- **Anthropic - MCP course**
   - L5, code overview + video 
 
- **MCP course"
  - Up to Video 2? 

## Books 
- **The practitioner's guide to graph data**
  - Authors:Gosnell and Broecheler 
  - up to "When do we use properties??"  

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
   - tag:time_series
   - Uses the `traffic_hourly` dataset, as part of the Monash Time Series Forecasting repo
   - Uses GluonTS library
  
 - **[Maximum likelihood estimation of the weibull distribution with reduced bias](https://arxiv.org/pdf/2209.14567)**
   - Authors: Makalic and Schmidt
   - tag:mle, weibull
   - Also see:
     - R. Ross. Formulas to describe the bias and standard deviation of the ML-estimated Weibull shape parameter. IEEE Transactions on Dielectrics and Electrical Insulation. doi:10.1109/94.300257.

- **[Failure rate analysis and mx plan optimization for aircraft](https://www.sciencedirect.com/science/article/pii/S100093612400356X?fr=RR-2&ref=pdf_download&rr=8f177cd9cea42f39)**
  - Authors: Cao et al
  - tag:reliability
  - x

 - **Mx interval determination and optimization tool (Boeing Technical Journal, 2012)**
   - Author: Shuguang Song
   - tag:reliability
   - Uses data availability file to find right-censored current TOW intervals
   - 

  - **[Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices](https://arxiv.org/pdf/2203.10716)**
     - Authors: Hewamalage et. al.
   
  - **[Global models for time series forecasting](https://arxiv.org/pdf/2012.12485)**
     - Authors: Hewamalage et. al. 



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
   
  - **[Embeddings - Vicki Boykis](https://vickiboykis.com/what_are_embeddings/)**
    - p1-24, p24-28, p28-33
    - Encoding categorical variables: 
      - [sklearn doc on comparing encoders](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder.html#sphx-glr-auto-examples-preprocessing-plot-target-encoder-py)
      - [sklearn doc on cross fitting in TargetEncoder](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html#sphx-glr-auto-examples-preprocessing-plot-target-encoder-cross-val-py)
      - [TDS post](https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69) explaining smoothing in TargetEncoder
      - [kaggle post](https://www.kaggle.com/code/ryanholbrook/target-encoding)
      - [TargetEncoder example implementation](https://gist.github.com/lmassaron/6695171ff45bae7ef7ddcdad2ad493ca)
      - 

  - **[Kaggle M5 competition](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion?sort=hotness)**
    - https://www.kaggle.com/code/kyakovlev/m5-three-shades-of-dark-darker-magic
    - https://www.kaggle.com/code/kyakovlev/m5-simple-fe
    - [Converting time series problems into supervised learning problems](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)

  - **[MOIRAI transformer-based forecasting model](https://aihorizonforecast.substack.com/p/moirai-zero-shot-forecasting-without)**
    - [sktime tutorial](https://www.sktime.net/en/latest/examples/03_transformers.html)
    - [nixtla's libraries for forecasting](https://nixtlaverse.nixtla.io/)
      - [NBEATS](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeats.html#usage-example), [NHITS](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html), [TimeLLM](https://nixtlaverse.nixtla.io/neuralforecast/models.timellm.html)

  - **[Jay Alammar - Hands on LLMs](https://youtu.be/RVxl9u7rt9w?si=FuuCdoWJy6W6Atqo)**
    - Part of Toronto Data Workshop

  - **[PyMC3 Bayesian multilevel modeling](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html#conventional-approaches)**
    - Radon example from Gelman's book
   
  - **[Context-Aided Forecasting: Enhancing Forecasting with Textual Data](https://aihorizonforecast.substack.com/p/context-aided-forecasting-enhancing?utm_source=substack&publication_id=1940355&post_id=152058185&utm_medium=email&utm_content=share&utm_campaign=email-share&triggerShare=true&isFreemail=false&r=38w78&triedRedirect=true)**
    - From AI Horizon Forecast substack, by Nikos Kafritsas
    - github: https://github.com/ServiceNow/context-is-key-forecasting 
    - Mentions Continuous Ranked Probability Score (CRPS), a proper scoring rule that provides a comprehensive assessment of forecast quality by evaluating the entire predictive distribution rather than focusing solely on summary statistics.
    - Compares general LLMs (e.g. Llama), time series foundation models (e.g. MOIRAI), and statistical methods (e.g. ARIMA)
    - From the paper: "We manually curate and release 71 forecasting tasks (Sec. 3) spanning 7 domains, which cover various kinds of contextual information (Sec. 3.2), and in addition to basic natural languageprocessing and time-series analysis, require various capabilities (e.g. retrieval, reasoning, etc.)"
    - When using LLMs we need to take steps to avoid a situation where the model has seen the time series in its training data. One way to do this is to take very recent data.
    - Model types: 
      - LLMs used include: GPT-4o, Llama-3.1-405b and other Llamas, and Mixtral-8x7B
      - Multimodal forecasting models used: UniTime and Time-LLM
      - Time series foundation models: lag-llama, chronos, timeGEN, MOIRAI
      - Statistical models: ARIMA, ETS, exponential smoothing
      - 

  - **[VulnWatch: AI-Enhanced Prioritization of Vulnerabilities](https://www.databricks.com/blog/vulnwatch-ai-enhanced-prioritization-vulnerabilities)**
    - From databricks blog 
    - Our engineering team has designed an AI-based system that can proactively detect, classify, and prioritize vulnerabilities as soon as they are disclosed, based on their severity, potential impact, and relevance to Databricks infrastructure. Our system achieves an accuracy rate of approximately 85% in identifying business-critical vulnerabilities. By leveraging our prioritization algorithm, the security team has significantly reduced their manual workload by over 95%. They are now able to focus their attention on the 5% of vulnerabilities that require immediate action, rather than sifting through hundreds of issues.
    - Steps:
      - Extract data from APIs
      - Extract data like description, CVSS score, EPSS score, etc. 
      - Create three main features: severity score, component score, topic score
        - Severity score: This score's high value corresponds to CVEs deemed critical to the community and our organization. It's a simple weighted average
        - Component score: Quantitatively measures how important the CVE is to our organization. To do this, they have to take a CVE, extract what library it's related to, and then match against a list of all libraries used in databricks. 
          - Step 1: Converting each word in the library name into an embedding for comparison.
          - Step 2: Vector similarity search to find a list of databricks libraries that might be vulnerable
          - Step 3: Prompt a fine-tuned LLM to classify which libraries within that list are actually vulnerable.
            - We fine-tuned various models using a ground truth dataset to improve accuracy in identifying vulnerable dependent packages. A ground truth dataset comprising 300 manually labeled examples was utilized for fine-tuning purposes. The tested LLMs included gpt-4o, gpt-3.5-Turbo, llama3-70B, and llama-3.1-405b-instruct.
            - The prompt was automatically optimized in an iterative process. See [this notebook](https://colab.research.google.com/drive/1Bn11v5X85PEgWnn3Rz_4GJRIUh3S0aEB?usp=sharing%23scrollTo%3DaAgLfkPqNB7G#scrollTo=aAgLfkPqNB7G)
            - 300 labelled examples used in fine-tuning gpt-3.5-turbo, but this was still slightly worse than gpt-4o. 
            - Also see [Databricks API for fine-tuning](https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html)
          - Step 4: Once the Databricks libraries in a CVE are identified, the corresponding score of the library (library_score as described above) is assigned as the component score of the CVE
        - Topic score:
       
  - **[Andrej Karpathy: Software Is Changing (Again)](https://www.youtube.com/watch?v=LCEmiRjPEtQ)**
     - Think of LLMs as the operating system, and consider how to create apps for this OS. 

  - **[Andrew Ng: State of AI agents](https://www.youtube.com/watch?v=4pYzYmSdSH4)**

  - **[Knowledge Graphs + Semantic Search: Unlocking Smarter LLMs](https://www.youtube.com/watch?v=9_UWqdUnsTc)**
     - Presented by Alessandro Pireno from SurrealDB. Nice intro to GraphRAG, complemented by the MS paper "GraphRAG: Unlocking LLM discovery on narrative private data"
  
  - **[Anchoring Enterprise GenAI with Knowledge Graphs: Jonathan Lowe (Pfizer), Stephen Chin (Neo4j)](https://www.youtube.com/watch?v=OpVkWc3YnFc)**

  - **[GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)**
     - This is the MS paper that introduced GraphRAG
     - Use an LLM to generate the graph with entities and relationships. When a question is posed, the graphRAG system uses both vector search and graph retrieval to provide context that is inserted into a prompt for the answering LLM.
     - The structure of the graph helps improve higher-level reasoning about the private dataset - e.g. "what are the top 5 themes in the data?"
        - This is facilitated by "bottom-up clustering" done on the LLM-generated graph, which lets the answering LLM reason at different levels of hierarchy.
      
  - **[GraphGeeks -A visual cheat sheet for Graphs + LLMs](https://www.youtube.com/watch?v=UdF-ODQFwLk)**
     - Speaker: Veronique Gendner
     - p13, 14, 15, and at 22:31, 28:49 in the video are nice visuals of RAG and graphRAG 
  

    

## Card reviews 
- 409 (review): attention output in matrix form 
- 423: logits
- 186, 222: conditional independence
- 308: kernel function as a generalization of inner product
- 64: overview of logistic regression
- 199 to ~217: notes from book "Euler's Gem"
- 424: nonparametric conditional survival
- 425, 308: kernel functions as local neighborhoods
- 429, 430: cross-validation for prediction error and hyperparameter selection - ESL, p242
- 439, 409: attention mechanism and dimensions of matrices
- 255: deviance and AIC
- 191: motivation for MCM algorithms
- 385: basics of EM algorithm
- 439: attention mechanism and dimensions of matrices
- 98: point estimators, unbiasedness, sampling distributions
- 55: probability as a special case of expectation
- 227, 242, 261: expected value, expected utility and prospect theory
- 263, 269: instrumental variables and simple regression for ATE
- 153, 153A: multilevel models for modeling both individual- and group-level variation - SAT scores of children in schools example 
- 194: no-pooling vs full-pooling models (compared to soft constraint of partial pooling)
- 288: ARMA(0, 1, 0) same as random walk
- 272: overview of causal inf methods
- 264: effect of covariate imbalance on ATE_hat
- 356: BBG drug and subclassification for confounding variables (_not_ for mediating variables)
- 438: OHE vs BOW encoding  


## Cards 
- 426: ~~Brier score without censoring~~
- 427: ~~Brier score, with censoring~~
- 428: ~~time-dependent AUC and Brier score~~
- 431: ~~two uses of cross-validation - ISL, p175 (first page of chapter on resampling methods)~~ 
- 402: cross-entropy loss for multi-class classification
- 178: random forest training and variable importance 
- 432: mixture of experts transformers (cf. card 406 on main sub-layers in standard transformer). 
- 433: nested cross validation: https://inria.github.io/scikit-learn-mooc/python_scripts/cross_validation_nested.html (done) 
- 434: ~~how machine learning differs from optimization: Goodfellow et. al. book, p268~~ (done) 
- 435: ~~inference vs prediction: Gerds & Kattan *Medical Risk Prediction Models*, p27~~ (done) 
- 436: in `lifelines`, how do you get a conditional survival curve, and conditional prediction of e.g. median lifetime. See [here](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#prediction-on-censored-subjects)
- 437: should a non-significant variable be dropped from a _predictive_ regression model? See [this discussion by Frank Harrell](https://discourse.datamethods.org/t/model-selection-and-assessment-of-model-fit/321/5)
- 438: ~~one-hot encoding vs BOW for text - see j29, p89~~
- 439: ~~visual representation of K, Q, V matrices in transformers - [Francois Fleuret slides, p5-](https://fleuret.org/public/EN_20220809-Transformers/transformers-slides.pdf) + S.Prince, _Understanding Deep Learning_, p212 & 215~~
- 442: two equivalent ways of writing ridge regression objective function 
- xx: determining whether to gather more data: Goodfellow et al book, p414; and 426 ("fit a tiny dataset")
- xx?: How to tell if local minima are a problem in training a NN - Goodfellow et al, p277
- xx?: variadic keyword args - see this [module]([url](https://github.com/nayefahmad/algorithms-and-design-patterns/blob/main/src/variadic-args-and-kwargs.py))
- xx?: prediction intervals - Cox, _Principles of Statistical Inference_, p161-162 + [wiki article]([url](https://en.wikipedia.org/wiki/Prediction_interval)) on prediction intervals
- xx?: What is the "empirical Bayes" approach? - see p3 of Micci & Barreca, _A preprocessing scheme for high-cardinality categorical attributes_
- xx?: Empirical Bayes approach for "smoothing" averages in a category vs global across categories. See eq 4, 5, 6, 7, 8 of Micci & Barreca, _A preprocessing scheme for high-cardinality categorical attributes_
- xx?: Encoding hierarchical categories - see p5 of Micci & Barreca, _A preprocessing scheme for high-cardinality categorical attributes_
- xx?: Setting up relational tables to capture many-to-one/one-to-many relationships and many-to-many relationships. See J36, p23 

