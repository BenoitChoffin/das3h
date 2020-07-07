% DAS3H: Modeling Student Learning and Forgetting for Optimally Scheduling Distributed Practice of Skills
% \alert{Benoît Choffin}, Fabrice Popineau, Yolaine Bourda & Jill-Jênn Vie\newline\newline\scriptsize LRI/CentraleSupélec - University of Paris-Saclay | Inria Lille
% LIP6, Cyberspace | April 28, 2020
---
theme: Frankfurt
institute: \includegraphics[height=1.2cm]{figures/logo_lri.jpeg} \quad \includegraphics[height=1.3cm]{figures/LogoCS1.png} \quad \includegraphics[height=1.3cm]{figures/logo_UP_saclay_final.png} \quad \includegraphics[height=1cm]{figures/logo_inria.png}
section-titles: false
handout: true
biblio-style: authoryear
header-includes:
    - \usepackage{booktabs}
    - \usepackage{makecell}
    - \usepackage{multicol}
    - \usepackage{multirow}
    - \usepackage{subfig}
    - \usepackage{bm}
    - \DeclareMathOperator\logit{logit}
biblatexoptions:
    - maxbibnames=99
    - maxcitenames=5
---

# Introduction

## Mitigating human forgetting with spaced repetition

* Human learners face a constant trade-off between **acquiring new knowledge** and **reviewing old knowledge** \bigskip
* Cognitive science provides simple + robust learning strategies for improving LT memory
	* \alert{Spaced repetition}
	* \alert{Testing} \bigskip
* Can we do better? **Yes**, by providing students with an _adaptive_ and _personalized_ spacing scheduler.

## Mitigating human forgetting with spaced repetition

\raisebox{.5cm}{\includegraphics[width=0.5\textwidth]{figures/leitner.png}}\includegraphics[width=0.5\textwidth]{figures/anki.png}

\begin{block}{Model-based}
Ex. select the item whose memory strength is closest to a threshold $\theta$ [\cite{lindsey2014improving}] $\rightarrow$ "almost forgotten"
\end{block}

\begin{block}{Model-free}
Ex. Reinforcement Learning methods such as [\cite{reddy2017accelerating}]
\end{block}

## Beyond flashcard memorization

**Problem**: these algorithms are designed for optimizing _pure memorization_ (of facts, vocabulary,...)

* In real-world educational settings, students also need to learn to master and remember a set of **skills**

* In that case, specific items are the only way to practice one or multiple skills because _we do not have to memorize the content directly_

* Traditional adaptive spacing schedulers are **not applicable for learning skills**

## Extension to skill practice and review

\begin{minipage}{0.4\linewidth}
\textcolor{blue!80}{Item}-\textcolor{green!50!black}{skill} relationships require expert labor and are synthesized inside a binary q-matrix $\rightarrow$
\end{minipage}\begin{minipage}{0.6\linewidth}
\scriptsize
\input{tables/dummy_qmat.tex}
\end{minipage}

\centering
\includegraphics[width=10cm]{figures/item_skills_relations.pdf}

## Limitations of student models

We need to be able to infer skill memory strength and dynamics, however in the student modeling literature:

* some models leverage item-skills relationships
* some others incorporate forgetting

But none does both!

## Our contribution

We take a model-based approach for this task.

1. Traditional adaptive spacing algorithms can be extended to review and practice skills (not only flashcards).\bigskip
2. We developed a new student _learning_ and _forgetting_ model that leverages item-skill relationships: \alert{\textbf{DAS3H}}.
	* DAS3H outperforms 4 SOTA student models on 3 datasets.
	* Incorporating skill info + forgetting effect improves over models that consider one or the other.
	* Using precise temporal information on past skill practice + assuming different learning/forgetting curves \alert{for different skills} improves performance.

## Outline

1. DASH\bigskip
2. Our model DAS3H\bigskip
3. Experiments\bigskip
4. Conclusion

# DASH

## Knowledge tracing

Predict future student performance given their history

\centering
\includegraphics[width=10cm]{figures/dkt_knowledge_tracing.png}
\raggedright

Given $(q_t,a_t)_{t \leq T}$ for former students

$q_t$ is the question index, $a_t \in \{0,1\}$ is the correctness

For new students, given $(q_t,a_t)_{t \leq T}$ and $q_{T+1}$, guess $a_{T+1}$

## DASH

$\rightarrow$ DASH = item **D**ifficulty, student **A**bility, and **S**tudent **H**istory

DASH [\cite{lindsey2014improving}] bridges the gap between _Factor Analysis models_ and _memory models_:

$$\mathbb{P}\left(Y_{s,j,t}=1\right)=\sigma(\alpha_s - \delta_j + h_{\theta}(\mathrm{t}_{s,j,1:l},\mathrm{y}_{s,j,1:l-1}))$$

where:

* $Y_{s,j,t}$ binary correctness of student $s$ answering item $j$ at time $t$;
* $\sigma$ logistic function;
* $\alpha_s$ ability of student $s$;
* $\delta_j$ difficulty of item $j$;
* $h_{\theta}$ summarizes the effect of the $l-1$ previous attempts of $s$ on $j$ at times $\mathrm{t}_{s,j,1:l-1}$ + the binary outcomes $\mathrm{y}_{s,j,1:l-1}$.

## DASH

Lindsey et al. chose:
\begin{align*}
    h_{\theta}(\mathrm{t}_{s,j,1:l},\mathrm{y}_{s,j,1:l-1}) = \sum_{w=0}^{W-1} & \theta_{2w+1}\log(1+c_{s,j,w}) \\
    &- \theta_{2w+2}\log(1+a_{s,j,w})
\end{align*}

where:

* $w$ indexes a set of expanding \alert{time windows};
* $c_{s,j,w}$ number of correct answers of $s$ on $j$ in time window $w$;
* $a_{s,j,w}$ number of attempts of $s$ on $j$ in time window $w$;
* $\theta$ is _learned_ by DASH.

## DASH

Assuming that the set of time windows is \{1, 7, 14, $+\infty$\}:

\centering
\includegraphics[width=10cm]{figures/time_windows.pdf}

## DASH

DASH:

* accounts for both _learning_ and _forgetting_ processes;

* induces diminishing returns of practice inside a time window (log-counts);

* has a time module $h_{\theta}$ inspired by ACT-R [\cite{anderson1997act}] and MCM [\cite{pashler2009predicting}].

# DAS3H

## From DASH to DAS3H

* DASH
	* outperforms a hierarchical Bayesian IRT on Lindsey et al. experimental data (vocabulary learning).
	* was successfully used to adaptively personalize item review in a real-world cognitive psychology experiment.
\bigskip
* However, DASH
	* does not handle multiple skill item tagging $\rightarrow$ useful to account for knowledge transfer from one item to another.
	* assumes that memory decays at the same rate for every KC.

## Our model DAS3H

We extend DASH in **3 ways**:
\begin{enumerate}
    \item Extension to handle multiple skills tagging: new temporal module $h_{\theta}$ that also takes the multiple skills into account.
    	\begin{itemize}
    	\item Influence of the temporal distribution of past attempts and outcomes can differ from one skill to another.
    	\end{itemize}
    \item Estimation of easiness parameters for \textit{each} item $j$ and skill $k$;
    \item Use of KTMs [\cite{Vie2019}] instead of mere logistic regression for multidimensional feature embeddings and pairwise interactions.
\end{enumerate}

## Knowledge Tracing Machines (KTMs)

Just pick features (ex. \textcolor{blue!80}{user}, \textcolor{orange}{item}, \textcolor{green!50!black}{skill}) and you get a student model

Each feature $k$ is modeled by bias $\alert{w_k}$ and embedding $\alert{\bm{v_k}}$.\vspace{2mm}
\begin{columns}
\begin{column}{0.47\linewidth}
\includegraphics[width=\linewidth]{figures/fm.pdf}
\end{column}
\begin{column}{0.53\linewidth}
\includegraphics[width=\linewidth]{figures/fm2.pdf}
\end{column}
\end{columns}\vspace{-2mm}

\hfill $\logit p(\bm{x}) = \mu + \underbrace{\sum_{k = 1}^N \alert{w_k} x_k}_\textnormal{logistic regression} + \underbrace{\sum_{1 \leq k < l \leq N} x_k x_l \langle \alert{\bm{v_k}}, \alert{\bm{v_l}} \rangle}_\textnormal{pairwise relationships}$

\small
\fullcite{Vie2019}

## Our model DAS3H

$\rightarrow$ DAS3H = item **D**ifficulty, student **A**bility, **S**kill and **S**tudent **S**kill practice **H**istory

For an embedding dimension of $d=0$, DAS3H is:

$\mathbb{P}\left(Y_{s,j,t}=1\right)=\sigma (\alpha_s - \delta_j + \underbrace{\alert{\sum_{k \in KC(j)} \beta_k}}_{\text{skill easiness biases}} +h_{\theta}\left(\mathrm{t}_{s,j,1:l},\mathrm{y}_{s,j,1:l-1}\right))$.

We choose:
\begin{align*}
    h_{\theta}(\mathrm{t}_{s,j,1:l},\mathrm{y}_{s,j,1:l-1}) = \alert{\sum_{k \in KC(j)}}&\sum_{w=0}^{W-1}\theta_{\alert{k},2w+1}\log(1+c_{s,\alert{k},w})\\
    &- \theta_{\alert{k},2w+2}\log(1+a_{s,\alert{k},w}).
\end{align*}

$\rightarrow$ Now, $h_{\theta}$ can be seen as a sum of _skill_ memory strengths!

## Example DAS3H forgetting curves

\begin{figure}%
    \centering
    \subfloat[Slow forgetting]{{\includegraphics[width=5cm]{figures/example_das3h_curve_1.pdf} }}%
    \:
    \subfloat[Rapid forgetting]{{\includegraphics[width=5cm]{figures/example_das3h_curve_2.pdf} }}%
    \caption{Example forgetting curves randomly generated from our DAS3H model}%
    \label{example_curves}%
\end{figure}
\vspace{-5mm}

# Experiments

## Experiments

1. Experimental setting

2. Contenders

3. Datasets

4. Main results

5. Further analyses

## Experimental setting

\begin{block}{How to compare ML models?}
Train the models on one part of the dataset

Test on the other part

Gather prediction metrics, compare the models
\end{block}

* **5-fold cross-validation** at the student level: predicting binary outcomes on \alert{unseen} students (_strong generalization_)
* Distributional assumptions to \alert{avoid overfitting}:
	* When $d=0$: L2 regularization/$\mathcal{N}(0,1)$ prior
	* When $d > 0$: hierarchical distributional scheme
* Same time windows as Lindsey et al.: {1/24,1,7,30,+$\infty$}

## Contenders

5 contenders:

* \alert{DAS3H}
* DASH [\cite{lindsey2014improving}]
* IRT/MIRT [\cite{van2013handbook}]
* PFA [\cite{pavlik2009performance}]
* AFM [\cite{cen2006learning}]

Every model was cast within the KTM framework $\rightarrow$ 3 embedding dimensions (0, 5 \& 20) + sparse feature encoding.

\tiny
|   | users | items | skills | wins | fails | attempts | tw [KC] | tw [items] |
|:-:|:-----:|:-----:|:------:|:----:|:-----:|:--------:|:-----:|:--------:|
| **DAS3H** | x | x | x | x | | x | x | |
| DASH | x | x | | x | | x | | x |
| IRT/MIRT | x | x | | | | | | |
| PFA | | | x | x | x | | | |
| AFM | | | x | | | x | | |

## Datasets

* 3 datasets: ASSISTments 2012-2013, Bridge to Algebra 2006-2007 \& Algebra I 2005-2006 (KDD Cup 2010)
	* Data consists of logs of student-item interactions on 2 ITS
	* Selected because they contain _both_ timestamps and items with multiple skills $\rightarrow$ rare species in the EDM datasets fauna
* Preprocessing scheme: removed users with < 10 interactions, interactions with \texttt{NaN} skills, duplicates

\tiny
\input{tables/datasets_caracs.tex}

## Main results

\input{tables/exp_results.tex}
$\rightarrow$ On every dataset, **DAS3H outperforms** the other models (between +0.04 and +0.05 AUC compared to DASH).

## Main results

\begin{figure}%
    \centering
    \subfloat[DAS3H]{{\includegraphics[width=5cm]{figures/comp_dim_das3h.pdf} }}%
    \:
    \subfloat[IRT]{{\includegraphics[width=5cm]{figures/comp_dim_irt.pdf} }}%
    \caption{AUC comparison on two models for $d=0, 5$ and 20 (all datasets, 5-fold cross-validation).}%
    \label{dim_com}%
\end{figure}
\vspace{-5mm}
\raggedright
$\rightarrow$ The impact of the multidim feature embeddings is small and not consistent across datasets and models (+ unstable sometimes).

## Importance of time windows

\centering
\begin{figure}
\includegraphics[width=5.5cm]{figures/pairwise_comp_all_datasets.pdf}
\caption{AUC comparison on DAS3H \textit{with} and \textit{without} time windows features (all datasets, 5-fold cross-validation).}
\end{figure}
\vspace{-3mm}
\raggedright
Without time windows, $h_{\theta}$ counts past wins and attempts in DAS3H.
$\rightarrow$ Using \alert{temporal distribution of past skill practice} instead of simple win/fail counters improves AUC performance: the _**when**_ matters.

## Importance of different learning/forgetting curves per skill
\scriptsize
\input{tables/comp_DAS3H_multiparams.tex}

\normalsize
$\rightarrow$ Assuming **different learning and forgetting curves for different skills** in DAS3H consistently yields better predictive power: some skills are easier to learn and slower to forget.

# Conclusion

## In a nutshell

* Human forgetting is _ubiquitous_ but luckily:
	* \alert{Cognitive science} gives us efficient and simple learning strategies
	* \alert{ML} can build us tools to **personalize these strategies** and further improve LT memory retention

* Adaptive spacing algorithms have been focusing on _pure memorization_ (e.g. vocabulary learning)
	* They can be used for \alert{optimizing practice and retention of skills}

* Our student model **DAS3H**
	* incorporates information on _skills_ **and** _forgetting_ to predict learner performance
	* shows higher predictive power than other SOTA student models
	* fits our model-based approach for optimally scheduling skill review

## Thanks for your attention!

Our paper is available at:

\centering
`https://arxiv.org/abs/1905.06873`

\raggedright
Python code is freely available on my GitHub page:

\centering
`https://github.com/BenoitChoffin/das3h` !

\raggedright
To send me questions about our paper or my research work:

\centering
`benoit.choffin@lri.fr`

