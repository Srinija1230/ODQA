# ODQA: Telugu Open-Domain Question Answering (Project Facts)

**Role:** Summer Intern and Research Intern, **IIIT Hyderabad**, **May 2020 – Jul 2021**

> Source of these notes: the GitHub repo `srinija1230/odqa` (`README.md`, `OdqaFinalModel.py`, `OdqaFinalModelFindingAccuracy.py`, `ODQA_MLP_PRED.ipynb`, `predefined.txt`, `updated_file_847_que`) and your resume.

---

## Overview
- **What it is:** a **factoid question-answering system for Telugu questions**. The README calls it "Telugu-Question-Answering".
- **Open-domain:** it has no fixed document collection. At query time it gets evidence from **live Bing web search results**.
- **Problem it solves:** answering short factual questions in Telugu (who / where / how many / when). Telugu has few NLP resources, so the system translates each question to English and uses English web pages and English NLP tools.
- **Approach:** a classic **retrieve → rank sentences → classify answer type → NER → vote** pipeline. It uses **no neural reader** and **no generative model**.
- **Output:** a single short answer string in **English**, printed to the console.
- **Users:** a research **prototype** (the resume says "prototyped"). [FILL IN: whether anyone other than you or your lab ran it.]
- **Scale (from the repo):**
  - Question-classification training file `predefined.txt`: **1,342 labeled Telugu questions**, **7 classes**.
  - Evaluation file `updated_file_847_que`: **846 Telugu question–answer pairs**. The file name says 847. [CHECK: the file has 846 records (845 newline characters, because the last line has no trailing newline). Use 846 unless an earlier version had 847.]
  - Per question: the **top ~10 Bing results**, all `<p>` paragraphs from those pages, and the **top 10 sentences** kept as evidence.

## Resume Claims
- Prototyped an **open-domain question-answering pipeline** for **Telugu**, a low-resource language, with **query translation to English**.
- Built web retrieval via **Bing** search results; tokenized sentences into **bag-of-words count vectors** and ranked them by **cosine similarity** to the query.
- Implemented a **multiclass SVM** on **TF-IDF** features for **7-class** answer-type prediction on **1,342 questions**, reaching **~80% accuracy**.
- Ranked type-matched **spaCy NER** answer candidates by frequency in the **top 10 sentences**, with **NLTK tokenization and stop-word removal**.

Claim checks against the repo:
- Translation, Bing, bag-of-words cosine, 7 classes, 1,342 questions, multiclass SVM on TF-IDF, spaCy NER, frequency ranking and top 10 sentences all **match the code**.
- [CHECK: **~80% accuracy** appears nowhere in the repo. The code never scores the 20% test split, and the custom SVM as written would crash (see Models). Find the run or notebook that produced ~80%, and confirm it was the **SVM**, not the MLP in `ODQA_MLP_PRED.ipynb`.]
- [CHECK: **NLTK tokenization and stop-word removal** is not part of the answer pipeline. The NLTK `preprocess()` function is never called; sentences are tokenized with the regex `\w+`, and spaCy handles NER tokenization. NLTK `word_tokenize` and the stop-word list are used only in the **evaluation** normalization (`findingacc`). Reword the bullet or be ready to explain this.]
- [CHECK: the resume describes **no end-to-end QA accuracy**. The repo has an Exact/Partial Match evaluation, but no results are saved.]

## Architecture / Pipeline

```
 Telugu question (input() or eval file)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 [1] Translate te→en (googletrans)         [5] Question Classifier (on Telugu text)
        │                                      TF-IDF → Multiclass SVM (.py)
        ▼                                             or MLP (notebook)
 [2] Bing search scrape → top ~10 URLs              │  → one of 7 answer types
        │                                              │  PERS/LOCA/NUMB/DATE/TIME/ORGA/PERC
        ▼                                              │
 [3] Fetch pages (urllib3) → <p> text (BeautifulSoup)  │
        │  split on "." ; drop pieces with "?"         │
        ▼                                              │
 [4] Rank sentences: cosine(word-count vectors)        │
        │  query vs each sentence → top 10             │
        ▼                                              ▼
 [6] spaCy NER (en_core_web_sm) on top-10 sentences ──► keep entities of predicted type
        │
        ▼
 [7] Answer selection: frequency vote, minus entities already in the question,
     ties → earliest-ranked sentence; fallback → all entity types; else "NO ANSWER"
        │
        ▼
 English answer string (console / list `final_ans` / `myfile.txt` in notebook)
```

| # | Component | Implementation | Function |
|---|---|---|---|
| 1 | Translation | `googletrans.Translator().translate(src='te', dest='en')`. An alternative, `goslate`, is defined in `translate2` but never called. | `translate` |
| 2 | Retrieval | GET `http://www.bing.com/search?q=…`, parse `li.b_algo > h2 > a[href]` | `qquery` |
| 3 | Passage extraction | `urllib3.PoolManager` GET + `BeautifulSoup.findAll('p')` | `Sentence_Extraction` |
| 4 | Sentence ranking | Bag-of-words cosine similarity on regex `\w+` tokens, keep top **10** | `Finding_Similarity_Matrix`, `Most_Relative_Sentence` |
| 5 | Question classification | `TfidfVectorizer()` + `MulticlassSVM` (custom Crammer–Singer) or `MLPClassifier` | `que_classification`, `predict_ques` |
| 6 | NER | spaCy `en_core_web_sm`, with spaCy labels mapped to the class labels | `NER_other_way`, `que_ner` |
| 7 | Answer extraction | Frequency count with pandas, question-entity filter, tie-break | main script, `find_max_freq`, `findingmostsimilar` |
| 8 | Evaluation | Normalization + exact match / fuzzy partial match | `findingacc` (accuracy script only) |

## End-to-End Flow
1. **Input:** one Telugu question. `OdqaFinalModel.py` reads it with `input()`. `OdqaFinalModelFindingAccuracy.py` and the notebook loop over the eval file.
2. **Reset state:** `cleanup_funct()` returns fresh empty global lists for each question.
3. **Translate** the Telugu question to English with googletrans. On any exception it retries recursively, with no limit.
4. **Search:** send the English question to Bing and scrape the result-title links. If no `b_algo` results come back, it calls itself again.
5. **Extract text:** download each URL and collect the text of every `<p>` tag into `whole_sent`. If a URL fails with a generic error, the code removes that URL and restarts extraction for the whole list.
6. **Sentence split:** split each paragraph on `"."` and keep only the pieces that contain no `"?"`. The result is the list `all_sen`.
7. **Rank:** compute the cosine similarity between the English query and every sentence, using raw term-count vectors. Keep the top **10**.
8. **Classify the question:** the classifier runs on the **original Telugu question**, not the English translation. The output is one of 7 answer types.
9. **NER:** run spaCy on the top-10 sentences and group entities by label. The labels are mapped as follows:
   - `PERSON`→`PERS`
   - `GPE`+`LOC`→`LOCA`
   - `ORG`→`ORGA`
   - `CARDINAL`→`NUMB`
   - `PERCENT`→`PERC`
   - `MONEY`→`CUR`
   - `QUANTITY`→`QUANT`
   - `DATE` and `TIME` keep their names.
10. **Candidate answers:** keep the entities whose label matches the predicted class.
11. **Filter:** run NER on the English question as well, and drop any candidate that already appears in the question under the same label (lowercased). This stops the system from answering with an entity from the question itself.
12. **Vote:** sort candidates by frequency and walk down the list, collecting those tied at the highest count.
    - **One winner:** return it.
    - **Tie:** `findingmostsimilar` picks the candidate found in the earliest-ranked sentence. See the bug note under Challenges.
13. **Fallback:** if no typed candidate survives, `find_max_freq` repeats the vote over entities of **all** types. If that also fails, it returns `"NO ANSWER"` (`" SORRY NO ANSWER"` in the accuracy script).
14. **Output:** the answer is printed. In batch mode it is appended to `final_ans`, and the notebook also writes it to `myfile.txt`. The total wall-clock time is printed.
15. **Score (accuracy script only):** `findingacc(actual_ans, final_ans, "updated_file_847_que")`.

## Data
**Question-classification data: `predefined.txt`**
- Format: `LABEL: <Telugu question>`, one per line. The code takes the label from the first token and the question from `line[5:]`.
- **1,342 questions** and **7 classes**:

| Class | Meaning | Count |
|---|---|---|
| NUMB | number / count | 437 |
| PERS | person | 322 |
| LOCA | location | 314 |
| DATE | date | 136 |
| PERC | percentage | 50 |
| ORGA | organization | 48 |
| TIME | time duration | 35 |

- **Imbalanced:** NUMB is about 9× the size of ORGA and about 12× the size of TIME.
- **Split:** `train_test_split(test_size=0.2, random_state=1250)`, which gives about 80/20. **The 20% test split is never scored in the code.**
- **Preprocessing:** none before TF-IDF (default `TfidfVectorizer` settings on the raw Telugu text). A `preprocess()` function (lowercase, digits→words, punctuation and stop-word removal, lemmatization) exists for English but is **never called**.
- [FILL IN: where the 1,342 questions came from (scraped quiz sites? translated? hand-written?) and who labeled them.]

**End-to-end evaluation data: `updated_file_847_que`**
- Format: `LABEL@@@<gold answer>:::<Telugu question>`
- **846 records.** Label counts:
  - LOCA 300
  - PERS 265
  - NUMB 136
  - DATE 100
  - TIME 27
  - QUANT 12
  - ORGA 4
  - CUR 1
  - PERC 1
- **13 questions (QUANT + CUR) have labels the classifier cannot predict.** It was trained on only 7 classes.
- [CHECK: **318 of the 846 eval questions appear word-for-word in `predefined.txt`**, the classifier's training file. Any question-classification accuracy measured on this file is inflated.]
- [CHECK: **693 of the 846 gold answers are written in Telugu script** (for example `రాకేష్ శర్మ`), but the pipeline outputs **English** (`Rakesh Sharma`). No back-translation step exists in the repo, so exact match fails on these even when the answer is correct. Confirm whether a translated gold file or back-translation step existed outside this repo.]
- [FILL IN: source of the eval Q–A pairs and how the gold answers were written.]

## Models / Methods

**1. Question classifier, version A: custom Multiclass SVM (`OdqaFinalModel.py`, `OdqaFinalModelFindingAccuracy.py`)**
- **Features:** `TfidfVectorizer()` fit on all 1,342 questions. The vectorizer is fit **before** the train/test split, so the vocabulary and IDF include the test portion.
- **Model:** a hand-written **Crammer–Singer multiclass SVM**, trained in the dual by **coordinate descent**. Each per-sample subproblem is solved by **projection onto the simplex** (`projection_simplex`).
- **Settings:** `C=0.5`, `tol=0.001`, `max_iter=100`, `random_state=0`.
- **Prediction:** `argmax_k (x · w_k)`. There is no bias term.
- [CHECK: `self.tol = tol,` has a trailing comma, so `tol` becomes a tuple. In Python 3, `vratio < self.tol` then raises a `TypeError` at the end of the first epoch. Confirm whether this version actually ran, or whether this bug is why the MLP version exists.]
- [CHECK: `predict_ques` starts its score at `maxx=0` and uses `>=`. If every class score is negative, it returns class 0, whichever label that is.]

**2. Question classifier, version B: MLP (`ODQA_MLP_PRED.ipynb`, run on Colab)**
- `MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(14,), random_state=1)` on the same TF-IDF features and the same split.
- `LogisticRegression` is imported but never used.
- [FILL IN: which classifier is the "final" one, and the held-out accuracy of the SVM vs. the MLP (and LR, if you tried it). The resume names the **SVM**.]

**3. Sentence retrieval:** cosine similarity on raw term counts. There is no TF-IDF weighting, no stemming, no lowercasing and no stop-word removal. The top-k is **k = 10**.

**4. NER**
- **Used:** spaCy `en_core_web_sm`, which is pretrained on OntoNotes labels. No fine-tuning.
- **Considered:**
  - `NER_With_corenlp`, which calls a Stanford CoreNLP server at `localhost:9000` (defined in the notebook, not used).
  - `StanfordNERTagger` with `english.muc.7class` (commented out).
- [CHECK: the README lists **StanfordCoreNLP** as a prerequisite, but the final code uses **spaCy**. The README also leaves out spaCy, googletrans, pandas, inflect and word2number.]

**5. Answer selection:** answer-type-filtered **entity frequency voting** (redundancy across top sentences), plus an exclusion filter for entities that appear in the question.

- [FILL IN: why each choice was made, e.g. why translate-then-English rather than Telugu NER, why Bing rather than Wikipedia dumps, why 10 sentences, and why 14 hidden units.]

## Evaluation & Results
- **Metrics (`findingacc`):**
  - **Exact Match %:** normalized prediction == normalized gold.
  - **Partial Match %:** exact match, **or** `difflib.SequenceMatcher` ratio **≥ 0.7**, **or** (for PERS questions only) any predicted token is a substring of the gold answer.
- **Normalization applied before matching:**
  - Lowercase, remove `. , ? ! ( )` and newlines, remove English stop words.
  - If the gold answer is numeric, convert number words in the prediction to digits (`word2number`).
  - Otherwise, remove the words `times, percent, %, approximately, place, nearly, days`.
  - **Dates:** reduced to month + day + 4-digit year.
  - **Times:** many regex patterns (`12 hours 30 min`, `12:30 hours`, `one hour`, …) converted to **total seconds**.
- **Test set:** `updated_file_847_que` (846 questions). There is **no baseline** and **no ablation** in the repo.
- **Only recorded run:** notebook, questions 845–846.
  - Outputs: `['Nozomi Okuhara', 'Rakesh Sharma']`, both correct in meaning.
  - Time: **15.38 s** total for 2 questions (about 7.7 s per question), dominated by web requests.
- A loop comment says `#running from 0 to 186`, which suggests the evaluation was run in batches. [FILL IN: batch sizes and whether results were merged.]
- [FILL IN: the actual Exact Match % and Partial Match % on the full set, which set they were computed on, and any comparison (SVM vs. MLP classifier, spaCy vs. CoreNLP NER, top-k values).]
- **Resume ~80% accuracy:** this is answer-type **classification** accuracy, not end-to-end QA accuracy. [FILL IN: how the ~80% was measured. Was it on the 20% split from `random_state=1250` (about **269** questions)? Was it the SVM or the MLP? Was it plain accuracy or a per-class measure? Note that the TF-IDF was fit before the split, which leaks the test vocabulary.]
- [FILL IN: per-class results. With NUMB at 437 vs. TIME at 35, the overall ~80% may hide weak small classes.]

## Key Design Decisions & Trade-offs

| Decision | Why (from code/context) | Cost | At larger scale |
|---|---|---|---|
| Translate Telugu → English, then do QA in English | Mature English NER and web content; few Telugu tools | Translation errors spread to every later step; answers come out in English and don't match Telugu gold | Multilingual retriever/reader (e.g. mBERT/XLM-R) or a Telugu-trained NER; back-translate answers |
| Live Bing scraping for retrieval | Open-domain coverage with no index to build | Slow (seconds per question), not reproducible, breaks when the HTML changes, rate limits / ToS | Search API, or a local index (BM25 / dense retrieval over Wikipedia) |
| Bag-of-words cosine for sentence ranking | Simple, no training needed | No IDF weighting, no synonyms, case-sensitive | BM25 → dense retriever → cross-encoder reranker |
| Classify on the Telugu question, not the English one | Training data is Telugu | Two inputs can disagree (translation vs. class) | Classify the translated text, or train a multilingual classifier |
| Answer type + NER + frequency vote | Uses redundancy across sources; no reader model needed | Only works for entity-type answers; fails when NER misses or mislabels the entity; ties are broken arbitrarily | Extractive reader (span prediction) trained on QA data |
| Custom SVM → MLP | [FILL IN: reason for the switch] | Custom code had a bug (tol tuple) | Use the `sklearn` `LinearSVC(multi_class='crammer_singer')` |
| Lenient partial-match metric | Answers vary in surface form | Inflates scores (a 0.7 fuzzy ratio, or any token overlap for PERS) | Standard EM/F1 (SQuAD-style) plus human review |

## Deployment / Production
- Not deployed. The system runs as a **Python 3.7 script / Colab notebook** with console input and output.
- Environment: Windows Java path hard-coded (`JAVAHOME`, for the Stanford tools); Google Drive paths in the notebook.
- Failure handling, all ad hoc:
  - recursive retries on translation errors (no limit);
  - a Bing retry when no results come back;
  - URL removal plus a restart of extraction when a page fails.
- No CI/CD, tests, monitoring, caching or `requirements.txt`.

## Challenges & Lessons
Code-level issues visible in the repo (good to know before an interviewer spots them):
- **Unbounded recursion:** `translate()` and `qquery()` retry by calling themselves with no limit. `qquery` also ignores the return value of the recursive call (it works only because `Top_10_urls` is global).
- **Duplicate paragraphs:** after a URL fails, `Sentence_Extraction` restarts the whole loop without clearing `whole_sent`, so earlier pages are added twice.
- **Top-k bugs:** `Most_Relative_Sentence` caps k by the number of *paragraphs*, not sentences. When several sentences share the same cosine score, it returns the same sentence repeatedly (`l[0]`).
- **Tie-break bug:** `findingmostsimilar` appends a candidate whenever its sentence index is ≤ the best so far, then returns `best[0]`, the *first* candidate appended, not the best one. The `.py` versions match by substring; the notebook matches by whole token.
- **Double counting:** `find_max_freq` adds every entity to `possibles` twice. All counts double, so the ranking is unchanged.
- **Label-id instability:** label ids come from `list(set(classes))`, and string-hash randomization reorders them between runs. The code also hard-codes `range(7)`.
- **Ordinal mapping:** the notebook adds a map from ordinals and multiplicatives to numbers (`thrice→three`, `first→one`, …). This suggests NUMB answers were failing on surface form.
- [FILL IN: problems you actually hit and debugged (e.g. Bing blocking/empty results, translation errors, NER mislabels), how you diagnosed them, and what you learned.]

## Concepts to Know
- **Open-domain QA (ODQA):** answering questions without a given context passage. The system must first retrieve evidence from a large corpus or the web, then extract or generate the answer.
- **Retriever–reader pipeline:** stage 1 retrieves candidate documents or passages; stage 2 extracts the answer. Here the "reader" is answer-type-filtered NER plus voting.
- **Answer-type (question) classification:** predicting the expected answer category (person, location, number…) so that only entities of that type become candidates.
- **TF-IDF:** term weight = term frequency × log(N / document frequency). It down-weights words common across documents. Vectors are usually L2-normalized.
- **Cosine similarity:** dot(a, b) / (‖a‖‖b‖). It measures the angle between vectors and ignores their length.
- **Bag-of-words:** a text representation as unordered term counts.
- **Crammer–Singer multiclass SVM:** a single joint optimization that requires the true class's score to exceed every other class's score by a margin. It differs from one-vs-rest SVMs.
- **Dual coordinate descent:** optimizes the SVM dual one sample's variables at a time. Each step here is a projection onto a simplex.
- **Projection onto the simplex:** finds the closest point to v whose components are ≥ 0 and sum to z. It is computed by sorting plus a threshold.
- **C (SVM):** the regularization trade-off. A smaller C gives a wider margin and allows more training errors.
- **MLP with L-BFGS:** a feed-forward neural net trained with a quasi-Newton optimizer. It works well on small datasets. `alpha` is the L2 penalty.
- **Named Entity Recognition (NER):** tagging spans as PERSON, GPE, ORG, DATE, CARDINAL, etc. spaCy `en_core_web_sm` is a small CNN-based pipeline trained on OntoNotes 5.
- **GPE vs. LOC:** GPE means geopolitical entities (countries, cities); LOC means other locations (mountains, rivers). The project merges both into LOCA.
- **Redundancy-based answer extraction:** the correct answer tends to appear more often across many retrieved sources, so frequency voting works as a ranking signal.
- **Exact Match (EM):** the normalized prediction equals the gold answer exactly.
- **Fuzzy / partial match:** credit for near-matches. Here, difflib's ratio = 2·matches / total characters, with a threshold of 0.7.
- **Machine translation as a pivot:** translating a low-resource language into English to reuse English tools. The main risk is that errors compound across steps.
- **Low-resource NLP:** languages with little labeled data or few tools. The usual options are translation pivots, multilingual pretrained models and transfer learning.
- **Data leakage:** test examples (or statistics computed from them) influencing training, which inflates scores. Examples here: the TF-IDF fit before the split, and the 318 eval questions that also appear in the training file.

## Open Gaps
- [FILL IN] Whether anyone besides you or your lab used the prototype.
- [FILL IN] Source and labeling of the 1,342 classification questions.
- [FILL IN] Source of the 846 eval Q–A pairs and how the gold answers were written.
- [FILL IN] Reasons for the key choices (translation pivot, Bing, k=10, 14 hidden units).
- [FILL IN] Which classifier is final; held-out accuracy of SVM vs. MLP (vs. LR).
- [FILL IN] Reason for switching from the custom SVM to the MLP.
- [FILL IN] Full-set Exact Match % / Partial Match %, plus any baseline or ablation.
- [FILL IN] How the ~80% was measured (split, SVM vs. MLP, metric); per-class results.
- [FILL IN] Batch sizes for the evaluation run ("0 to 186" comment).
- [FILL IN] Real challenges you hit, how you diagnosed them, and what you learned.
- [CHECK] The resume's ~80% accuracy is not in the repo, and the 20% split is never scored.
- [CHECK] The resume says NLTK tokenization and stop-word removal are in the pipeline; in the code, NLTK is used only in the evaluation.
- [CHECK] The eval file is named 847 but contains 846 records.
- [CHECK] 318 of the 846 eval questions also appear in the classifier's training data.
- [CHECK] 693 of the 846 gold answers are in Telugu script, while the system outputs English.
- [CHECK] 13 eval questions are labeled QUANT/CUR, which the 7-class classifier cannot predict.
- [CHECK] The custom SVM's `tol` is a tuple, which would crash `fit` in Python 3.
- [CHECK] `predict_ques` defaults to class 0 when every class score is negative.
- [CHECK] The README lists StanfordCoreNLP, but the code uses spaCy; the README is also missing several dependencies.
