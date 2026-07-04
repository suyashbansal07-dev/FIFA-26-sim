# **SYSTEM ARCHITECTURE & SPECIFICATION HANDOFF**

**Project:** 2026 FIFA World Cup Predictive Engine **Core Framework:** Time-Weighted Dixon-Coles Bivariate Poisson Model **Version:** 1.0.0

## **1\. Executive Summary**

This document outlines the architecture and mathematical specification for a football match prediction system. Unlike basic metrics (e.g., Expected Goals/xG), this system predicts exact match scorelines and tournament progression probabilities. It relies on the **Dixon-Coles (1997)** statistical model, which extends a standard Poisson regression to account for the unique characteristics of football: the correlation between home and away scores in low-scoring matches, and the temporal decay of team form (momentum).

## **2\. Core Mathematical Specification**

### **2.1 The Base Poisson Framework**

The foundational assumption is that the number of goals scored by a team in a 90-minute match follows a Poisson distribution.  
For a match between Team i (Home) and Team j (Away), the model calculates the expected goals (\\lambda and \\mu):

* **Home Expected Goals:** \\lambda \= \\exp(\\alpha\_i \+ \\beta\_j \+ \\gamma)  
* **Away Expected Goals:** \\mu \= \\exp(\\alpha\_j \+ \\beta\_i)

**Parameters to estimate:**

* \\alpha\_x: Attack strength of team x. (Higher is better).  
* \\beta\_x: Defense strength of team x. (Lower/more negative is better).  
* \\gamma: Home-field advantage multiplier. (Crucial for tournament hosts).

*Constraint:* To prevent the model from suffering from over-parameterization (infinite solutions), an average attack strength constraint is enforced: \\frac{1}{N} \\sum \\alpha\_i \= 1\.

### **2.2 The Bivariate Dependence Adjustment (\\rho)**

Standard Poisson assumes Team i's goals are entirely independent of Team j's goals. In football, this is false (e.g., a 0-0 game in the 80th minute changes both teams' tactical behavior).  
Dixon-Coles introduces a dependence parameter (\\rho) and a correction function (\\tau) to adjust the joint probability P(X=x, Y=y) of low-scoring outcomes:  
P(X=x, Y=y) \= \\tau\_{\\lambda, \\mu}(x, y) \\times \\frac{e^{-\\lambda} \\lambda^x}{x\!} \\times \\frac{e^{-\\mu} \\mu^y}{y\!}  
The correction matrix \\tau\_{\\lambda, \\mu}(x, y) is defined strictly for specific low-scoring permutations:

* **If x=0, y=0:** \\tau \= 1 \- \\lambda\\mu\\rho  
* **If x=0, y=1:** \\tau \= 1 \+ \\lambda\\rho  
* **If x=1, y=0:** \\tau \= 1 \+ \\mu\\rho  
* **If x=1, y=1:** \\tau \= 1 \- \\rho  
* **All other scores:** \\tau \= 1 (Returns to standard independent Poisson)

### **2.3 Temporal Decay (Momentum)**

To capture "momentum," the model does not treat all historical data equally. We apply an exponential time-decay function to the historical match data when training the model.  
Weight of a match t days ago:  
W(t) \= e^{-\\xi t}

* **\\xi (xi):** The decay rate parameter. A standard half-life for international football (where matches are sparse) is often set to roughly 1.5 to 2 years, though for a localized tournament phase, an aggressive decay rate (half-life of 45-60 days) is used to prioritize immediate group-stage form.

## **3\. System Architecture & Data Pipeline**

The software implementation of this model requires a distinct multi-stage pipeline:

### **Phase 1: Data Ingestion Layer**

* **Inputs:** Match data spanning back 2-4 years. Required fields: Date, Home\_Team, Away\_Team, Home\_Goals, Away\_Goals, Tournament\_Type (Friendly vs. Competitive).  
* **Preprocessing:** Map strings to integer IDs. Apply a confederation multiplier (to adjust for the disparity in difficulty between UEFA/CONMEBOL qualifiers and others).

### **Phase 2: Maximum Likelihood Estimation (MLE) Solver**

This is the core training engine. Because we cannot use a standard Generalized Linear Model (GLM) out-of-the-box due to the \\tau correction and time weights, we must manually maximize the Log-Likelihood function.

* **Objective Function:** The weighted log-likelihood of all historical matches given current parameters (\\alpha, \\beta, \\gamma, \\rho).  
* **Optimizer:** Use the L-BFGS-B algorithm (via scipy.optimize in Python) to find the parameter set that maximizes the log-likelihood.  
* **Output:** A trained dictionary mapping every nation to their current \\alpha and \\beta values, plus the global \\gamma and \\rho.

### **Phase 3: The Inference & Simulation Engine**

Once parameters are locked, the engine handles future predictions.

1. **Bivariate Grid Generation:** For a specific matchup (e.g., France vs. Spain), calculate \\lambda and \\mu. Generate a 10x10 matrix (covering scores 0-0 through 9-9).  
2. **Probability Assignment:** Apply the Dixon-Coles formula (including \\tau) to fill the matrix with probabilities summing to \~1.0.  
3. **Derived Markets:** From this grid, calculate macro-probabilities:  
   * *Home Win:* Sum of all matrix cells where x \> y.  
   * *Away Win:* Sum of all matrix cells where y \> x.  
   * *Draw:* Sum of the diagonal (x \= y).  
   * *Over 2.5 Goals:* Sum of all cells where x \+ y \> 2\.  
4. **Monte Carlo Knockout Simulator:** \* For the knockout bracket, games cannot end in a draw.  
   * If the simulation lands on the diagonal (draw), the engine queries a secondary probability parameter (historically weighted penalty shootout win rate) to forcibly advance one team.  
   * Run 10,000 iterations of the remaining bracket to generate % chances of teams reaching the Semifinal or Final.

## **4\. Technical Stack Recommendations**

* **Data Processing:** pandas, numpy (Python).  
* **Optimization/Math:** scipy.optimize for the MLE solver.  
* **Hosting/Execution:** An AWS Lambda or Google Cloud Function triggered daily to scrape overnight results, re-run the MLE optimizer to update \\alpha and \\beta ratings, and push the updated probability grids to a database (e.g., PostgreSQL or Redis).

## **5\. Known Limitations & Guardrails**

When communicating these predictions to end-users or stakeholders, flag the following constraints:

1. **The "Red Card / Injury" Blindspot:** The model does not consume lineup data. If a star player (e.g., Kylian Mbappé) breaks their leg, the model will overestimate France's Attack Strength (\\alpha) until enough matches pass for the MLE to decay the rating.  
2. **Tournament Strategy:** In the final match of a group stage, teams may play for a mutually beneficial draw (e.g., "The Disgrace of Gijón"). The model assumes both teams are attempting to maximize goals scored, which breaks down under specific game-theory tournament scenarios.