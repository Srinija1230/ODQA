# Telugu Open-Domain Question Answering Pipeline (IIIT Hyderabad Internship): Project Facts

**Project ID:** `TELUGU-ODQA-IIITH`
**Resume section:** IIIT HYDERABAD, Summer Intern and Research Intern (May 2020 – Jul 2021). This is the **only** project in that section.
**Also referred to as:** Telugu QA, Telugu ODQA, ODQA repo, IIIT Hyderabad internship, IIIT-H project, low-resource-language QA, Bing + SVM + spaCy QA pipeline, answer-type classification project, 1,342-question SVM project.
**Not to be confused with:** ScholarRAG (arXiv RAG assistant), the Accenture Banking Regulatory GenAI Assistant (multi-agent RAG), Escape Enterprises RAG, the Insurance Risk-Scoring Platform (XGBoost) or the USF face-recognition fairness thesis.

**Role:** Summer Intern and Research Intern, **IIIT Hyderabad**, **May 2020 – Jul 2021**
**Code:** GitHub `srinija1230/odqa`. Python 3.7, run as scripts and in Colab.

---

## Overview
- A **prototype open-domain question-answering (ODQA) pipeline** for **Telugu**, a **low-resource language**.
- **Input:** a factoid question in Telugu (who / where / how many / when).
- **Output:** a short answer, such as a person, place, number or date.
- **Core idea:** Telugu has few NLP tools, so the pipeline **translates the query to English**, retrieves evidence from the **live web (Bing)**, and extracts the answer with English NLP tools (spaCy NER).
- **Open-domain:** there is no fixed document collection. Evidence is fetched fresh from the web for every question.
- **Classic pipeline:** retrieval → sentence ranking → answer-type classification → NER → frequency voting. There is no neural reader and no generative model.
- **Data scale:**
  - **1,342 labeled Telugu questions** for answer-type classification (**7 classes**).
  - An end-to-end question–answer test file of **846** Telugu Q–A pairs (the file is named `updated_file_847_que`).

## Resume Claims
- Prototyped an **open-domain question-answering pipeline** for **Telugu**, a low-resource language, with **query translation to English**.
- Built web retrieval via **Bing** search results; tokenized sentences into **bag-of-words count vectors** and ranked them by **cosine similarity** to the query.
- Implemented a **multiclass SVM** on **TF-IDF** features for **7-class** answer-type prediction on **1,342 questions**, reaching **~80% accuracy**.
- Ranked type-matched **spaCy NER** answer candidates by frequency in the **top 10 sentences**, with **NLTK tokenization and stop-word removal**.

## Architecture / Pipeline

```
 Telugu question
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 [1] Translate Telugu → English (googletrans)  [5] Answer-type classifier (Telugu text)
        │                                          TF-IDF → multiclass SVM
        ▼                                          → 1 of 7 types:
 [2] Bing search → top ~10 result URLs               PERS / LOCA / NUMB / DATE /
        │                                            TIME / ORGA / PERC
        ▼                                              │
 [3] Fetch pages → <p> paragraphs (BeautifulSoup)      │
        │  split into sentences                        │
        ▼                                              │
 [4] Bag-of-words count vectors                        │
        │  cosine(query, sentence) → top 10 sentences  │
        ▼                                              ▼
 [6] spaCy NER on top 10 sentences ──────► keep entities matching predicted type
        │
        ▼
 [7] Frequency vote → drop entities already in the question
     → tie-break by sentence rank → fallback: any type → else "NO ANSWER"
        │
        ▼
 Final answer (English)
```

| # | Component | Implementation |
|---|---|---|
| 1 | Query translation | `googletrans` Translator, `src='te'`, `dest='en'` |
| 2 | Web retrieval | HTTP GET to Bing search; parse result links (`li.b_algo > h2 > a`) with BeautifulSoup |
| 3 | Passage extraction | `urllib3` to download each page; BeautifulSoup to collect all `<p>` text |
| 4 | Sentence ranking | Bag-of-words term-count vectors (`Counter` over `\w+` tokens); cosine similarity to the English query; keep top **10** |
| 5 | Answer-type classification | `TfidfVectorizer` features + **multiclass SVM** (Crammer–Singer, dual coordinate descent); trained on **1,342** questions, **7** classes |
| 6 | NER | spaCy `en_core_web_sm`; spaCy labels mapped to the 7 answer types |
| 7 | Answer selection | Count entity frequency (pandas), filter out question entities, tie-break, fallback |
| 8 | Evaluation | Answer normalization (NLTK tokenization + stop-word removal, number/date/time normalization); Exact Match and Partial Match |

## End-to-End Flow
1. **Input:** a Telugu question, typed in or read from the test file.
2. **Translate** it to English with googletrans.
3. **Search Bing** with the English query and collect the top ~10 result URLs.
4. **Download** each page and extract all `<p>` paragraph text.
5. **Split into sentences:** split paragraphs on `.` and discard pieces containing `?`, because questions are not useful as evidence.
6. **Vectorize:** turn the query and each sentence into **bag-of-words count vectors**.
7. **Rank:** compute the **cosine similarity** between the query and every sentence, then keep the **top 10 sentences**.
8. **Classify the question:** the **TF-IDF + multiclass SVM** model takes the original Telugu question and predicts its answer type (one of 7).
9. **Run NER:** spaCy tags entities in the top 10 sentences. The labels are mapped as follows:
   - `PERSON`→**PERS**
   - `GPE`/`LOC`→**LOCA**
   - `CARDINAL`→**NUMB**
   - `ORG`→**ORGA**
   - `PERCENT`→**PERC**
   - `DATE`→**DATE**
   - `TIME`→**TIME**
10. **Keep matching candidates:** only entities whose type matches the predicted answer type remain.
11. **Filter:** remove any candidate that already appears in the question itself (e.g. for "Who is the president of Canada?" the answer cannot be "Canada").
12. **Vote:** the most frequent remaining candidate across the top 10 sentences wins. Ties go to the candidate found in the highest-ranked sentence.
13. **Fallback:** if no candidate of the predicted type survives, repeat the vote over entities of **all** types. If that also fails, return `"NO ANSWER"`.
14. **Output:** the English answer string.

## Data

**Answer-type classification set:** `predefined.txt`, format `LABEL: <Telugu question>`. **1,342 questions** in **7 classes**:

| Class | Answer type | Count |
|---|---|---|
| NUMB | Number / count | 437 |
| PERS | Person | 322 |
| LOCA | Location | 314 |
| DATE | Date | 136 |
| PERC | Percentage | 50 |
| ORGA | Organization | 48 |
| TIME | Time / duration | 35 |

- The classes are **imbalanced**. NUMB, PERS and LOCA make up about 80% of the data.
- **Split:** `train_test_split(test_size=0.2, random_state=1250)`, which gives 80% train and 20% held out.
- **Features:** `TfidfVectorizer()` with default settings on the raw Telugu question text.

**End-to-end QA test set:** `updated_file_847_que`, format `LABEL@@@<gold answer>:::<Telugu question>`. **846 Q–A pairs**:
- LOCA 300
- PERS 265
- NUMB 136
- DATE 100
- TIME 27
- QUANT 12
- ORGA 4
- CUR 1
- PERC 1

## Models / Methods

**1. Answer-type classifier: multiclass SVM on TF-IDF**
- **Features:** TF-IDF vectors of the Telugu question.
- **Model:** a **Crammer–Singer multiclass SVM**, implemented by hand. It is trained in the dual with **coordinate descent**, and each per-sample update is a **projection onto the simplex**.
- **Hyperparameters:** `C=0.5`, `tol=0.001`, `max_iter=100`, `random_state=0`.
- **Prediction:** the class with the highest score, `argmax_k (x · w_k)`.
- **Result:** **~80% accuracy** on 7-class answer-type prediction.
- **Alternative tried:** an `MLPClassifier` (`solver='lbfgs'`, `alpha=1e-5`, one hidden layer of **14** units, `random_state=1`) on the same features, in the Colab notebook `ODQA_MLP_PRED.ipynb`.

**2. Sentence retrieval: bag-of-words + cosine similarity**
- Tokens are matched with the regex `\w+` and counted with `Counter`.
- Cosine similarity is computed between the query's and each sentence's count vectors.
- Top **10** sentences are kept as evidence.

**3. NER: spaCy `en_core_web_sm`**
- A pretrained English model trained on OntoNotes labels, used without fine-tuning.
- **Alternatives considered** (present in the code): a Stanford CoreNLP server and the Stanford NER tagger (MUC 7-class model).

**4. Answer selection: type-filtered frequency voting**
- This relies on **redundancy**: the correct answer tends to appear repeatedly across many top-ranked sentences from different web pages.

**5. Text normalization: NLTK**
- `word_tokenize` and the English stop-word list are used to normalize text (lowercasing, punctuation and stop-word removal).
- The project also has a `preprocess()` function with digit-to-word conversion (`inflect`) and lemmatization (`WordNetLemmatizer`).

## Evaluation & Results
- **Answer-type classification:** **~80% accuracy** across **7 classes** on **1,342 questions** (multiclass SVM, TF-IDF features).
- **End-to-end QA evaluation (`findingacc`):**
  - **Exact Match %:** the normalized prediction equals the normalized gold answer.
  - **Partial Match %:** exact match, **or** a `difflib.SequenceMatcher` similarity **≥ 0.7**, **or** (for PERS questions) any predicted name token appears in the gold answer.
  - **Normalization steps:**
    - lowercase, strip punctuation, remove NLTK stop words;
    - number words → digits (`word2number`);
    - dates → month / day / year;
    - durations → total seconds (regex patterns for "12 hours 30 min", "12:30 hours", "one hour", etc.);
    - drop filler words ("approximately", "nearly", "percent", "times", "days", "place").
- **Example run** (notebook):
  - "Who is the winner of Women's Singles World Badminton Championship 2017?" → **Nozomi Okuhara**
  - "Who was the first Indian astronaut to go to space?" → **Rakesh Sharma**
  - Total time for the two questions: **15.38 s**. Web requests dominate the latency.

## Key Design Decisions & Trade-offs

| Decision | Why | Cost | At larger scale |
|---|---|---|---|
| Translate Telugu → English first | Telugu is low-resource; English has mature NER and far more web content | Translation errors carry through every later stage; answers come out in English | Multilingual models (mBERT / XLM-R), Telugu NER, back-translate the answer |
| Live Bing search for retrieval | Open-domain coverage with no index to build or maintain | Slow (seconds per question), not reproducible, fragile to HTML changes | Search API, or a local BM25 / dense index over Wikipedia |
| Bag-of-words cosine for ranking | Simple, fast, needs no training data | No term weighting (unlike TF-IDF/BM25), no synonyms or paraphrases | BM25 → dense retriever → cross-encoder reranker |
| Answer-type classification before NER | Shrinks the candidate set to the right kind of entity | A wrong type prediction usually produces a wrong answer | Joint models, or an extractive reader that doesn't need type labels |
| TF-IDF + SVM classifier | Strong, cheap baseline for short-text classification with ~1.3K examples | Bag-of-words ignores word order; class imbalance | Fine-tuned multilingual transformer classifier |
| Frequency voting over NER candidates | Uses redundancy across many sources; no QA training data needed | Only works for entity-type answers; depends on NER quality | Trained extractive reader (span prediction), as in DrQA / DPR-style systems |

## Deployment / Production
- A **research prototype**, run as Python scripts (`OdqaFinalModel.py` for one question, `OdqaFinalModelFindingAccuracy.py` for batch evaluation) and in a Colab notebook.
- Not deployed as a service.
- Failure handling:
  - retry on translation errors;
  - re-query when Bing returns no results;
  - skip pages that fail to download.

## Challenges & Lessons
- **Error compounding in a multi-stage pipeline:** translation, retrieval, classification and NER each add error. Final accuracy is bounded by the weakest stage.
- **Live web retrieval is fragile:** Bing sometimes returned no results, which needed retries, and some pages failed to download, so those URLs were skipped.
- **Answer-format mismatch:** answers like numbers, dates and durations appear in many surface forms ("twice" vs. "2", "12:30 hours" vs. "12 hours 30 min"). This needed an extensive normalization layer before scoring.
- **Language mismatch in evaluation:** many gold answers are written in Telugu script, while the pipeline answers in English. String-match metrics therefore need the two sides in the same language.
- **Class imbalance:** NUMB, PERS and LOCA dominate the training data, so rare types (TIME, ORGA, PERC) are harder to classify.
- **Lesson:** for low-resource languages, a translation pivot gets a working system fast, but accuracy depends heavily on translation quality. Multilingual models avoid that bottleneck.

## Concepts to Know
- **Open-domain QA (ODQA):** answering questions without a given passage. The system must first retrieve evidence from a large corpus or the web, then extract the answer.
- **Retriever–reader pipeline:** stage 1 finds relevant text; stage 2 extracts the answer. Here the "reader" is type-filtered NER plus frequency voting.
- **Answer-type classification:** predicting the expected answer category (person, location, number…) so that only matching entities become candidates.
- **Low-resource language:** a language with little labeled data or few NLP tools. Common approaches are a translation pivot, multilingual pretrained models and transfer learning.
- **Bag-of-words:** text as unordered word counts; word order is ignored.
- **TF-IDF:** term frequency × log(N / document frequency). It down-weights words common across documents.
- **Cosine similarity:** dot(a, b) / (‖a‖‖b‖). It measures vector direction and ignores document length.
- **Multiclass SVM (Crammer–Singer):** one joint optimization that requires the true class's score to beat every other class's score by a margin. It differs from one-vs-rest.
- **SVM C parameter:** the regularization strength. A smaller C gives a wider margin and allows more training errors.
- **Dual coordinate descent:** optimizes the SVM dual one sample at a time. It is efficient for linear SVMs.
- **MLP with L-BFGS:** a feed-forward neural net trained with a quasi-Newton optimizer. It suits small datasets; `alpha` is the L2 penalty.
- **Named Entity Recognition (NER):** tagging text spans as PERSON, GPE, ORG, DATE, CARDINAL, etc.
- **GPE vs. LOC (spaCy):** GPE means countries, cities and states; LOC means other locations such as mountains and rivers. The project merges both into LOCA.
- **Redundancy-based answer extraction:** the correct answer appears more often across independent sources, so frequency works as a confidence signal.
- **Exact Match (EM):** the normalized prediction is identical to the gold answer.
- **Fuzzy / partial match:** credit for near-matches. difflib's ratio = 2 × matching characters / total characters; the threshold here is 0.7.
- **Stop-word removal / tokenization (NLTK):** splitting text into tokens and dropping very common words ("the", "is") that carry little meaning.
- **Class imbalance:** some classes have far more examples than others. Accuracy can then hide poor performance on the rare classes, so check per-class precision and recall.
