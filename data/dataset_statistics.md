# BNSER Dataset Statistics

This document summarizes the composition, recording-source distribution, speaker distribution, annotation summary, and speaker-independent partitioning of the **Bangla Noakhali Dialect Speech Emotion Recognition (BNSER)** dataset used in the manuscript.

## 1. Dataset Overview

The BNSER dataset contains **2,300 WAV samples** collected from **25 native Noakhali speakers**. The speaker pool comprises **11 male and 14 female speakers**, with reported ages ranging from **20 to 70 years**.

The dataset contains six emotion classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad

The corpus was developed from two complementary recording sources:

1. **Scripted recordings:** 1,800 samples
2. **Drama-derived recordings:** 500 samples

Thus, the complete corpus contains **1,800 scripted samples (78.26%)** and **500 drama-derived samples (21.74%)**.

| Dataset property      |       Value |
| --------------------- | ----------: |
| Total WAV samples     |       2,300 |
| Speakers              |          25 |
| Male speakers         |          11 |
| Female speakers       |          14 |
| Reported age range    | 20–70 years |
| Emotion classes       |           6 |
| Scripted samples      |       1,800 |
| Drama-derived samples |         500 |

---

## 2. Recording Sources

### 2.1 Scripted recordings

The scripted subset contains **1,800 recordings** collected in everyday indoor settings using three consumer-grade recording devices:

* iPhone 16 Pro Max
* iPhone XR
* Vivo Y04

The use of consumer-grade devices rather than studio equipment was intended to retain realistic channel variability.

The scripted design used **10 phonetically diverse, semantically emotion-neutral sentences** representative of Noakhali conversational usage. Six scripted speakers recorded all ten sentences across the six emotion categories with five trials per sentence-emotion combination:

$$
10 \times 6 \times 5 \times 6 = 1,800
$$

### 2.2 Drama-derived recordings

The drama-derived subset contains **500 recordings** extracted from publicly available Noakhali-dialect theatrical productions. These recordings underwent FFmpeg-based conversion and Demucs vocal separation during corpus development.

Neutral was not included in the drama-derived subset because a consistently identifiable neutral register was not obtained from the theatrically expressive material.

---

## 3. Emotion-wise Dataset Distribution

The complete corpus distribution across emotion classes and recording sources is shown below.

| Emotion   |  Scripted | Drama-derived |     Total |
| --------- | --------: | ------------: | --------: |
| Angry     |       300 |           100 |       400 |
| Disgust   |       300 |           100 |       400 |
| Fear      |       300 |           100 |       400 |
| Happy     |       300 |           100 |       400 |
| Neutral   |       300 |             0 |       300 |
| Sad       |       300 |           100 |       400 |
| **Total** | **1,800** |       **500** | **2,300** |

Five emotion classes—Angry, Disgust, Fear, Happy, and Sad—contain both scripted and drama-derived recordings. **Neutral is represented exclusively by scripted recordings.**

This source restriction is an inherent property of the corpus design and should be considered when interpreting Neutral-class performance.

---

## 4. Dataset Partitioning

A strict **speaker-independent** partition was used for all model experiments. No speaker appears in more than one dataset split.

The resulting partition contains **18 training speakers, 3 validation speakers, and 4 test speakers**, with 1,610, 345, and 345 samples, respectively.

| Split      | Speakers |   Samples |  Proportion |
| ---------- | -------: | --------: | ----------: |
| Training   |       18 |     1,610 |      70.00% |
| Validation |        3 |       345 |      15.00% |
| Test       |        4 |       345 |      15.00% |
| **Total**  |   **25** | **2,300** | **100.00%** |

All four test speakers are unseen during model training.

The speaker-independent design prevents recordings from the same speaker from appearing across different partitions. However, the test set contains only four held-out speakers; therefore, the evaluation should be interpreted as an initial speaker-independent assessment on unseen speakers rather than as a population-level estimate for the broader Noakhali-speaking population.

---

## 5. Emotion-wise Partition Distribution

The number of samples from each emotion class in the training, validation, and test partitions is shown below.

| Emotion   |     Total |  Training | Validation |    Test |
| --------- | --------: | --------: | ---------: | ------: |
| Angry     |       400 |       281 |         59 |      60 |
| Disgust   |       400 |       281 |         59 |      60 |
| Fear      |       400 |       287 |         58 |      55 |
| Happy     |       400 |       281 |         59 |      60 |
| Neutral   |       300 |       200 |         50 |      50 |
| Sad       |       400 |       280 |         60 |      60 |
| **Total** | **2,300** | **1,610** |    **345** | **345** |

The partition is speaker-independent rather than a purely random sample-level split.

---

## 6. Test-set Composition

The fixed speaker-independent test set contains **345 samples** from **four held-out speakers**.

| Emotion   | Test samples |
| --------- | -----------: |
| Angry     |           60 |
| Disgust   |           60 |
| Fear      |           55 |
| Happy     |           60 |
| Neutral   |           50 |
| Sad       |           60 |
| **Total** |      **345** |

All **50 Neutral test samples** originate from **one held-out speaker** and from the **scripted source**.

This characteristic is important when interpreting Neutral-class results: the Neutral test subset does not provide variation across multiple speakers or recording sources.

---

## 7. Test-set Source Composition

For source-stratified evaluation, the five non-Neutral emotion classes provide both scripted and drama-derived test samples.

| Test subset        | Scripted | Drama-derived |   Total |
| ------------------ | -------: | ------------: | ------: |
| Non-Neutral        |      250 |            45 |     295 |
| Neutral            |       50 |             0 |      50 |
| **Total test set** |  **300** |        **45** | **345** |

Because no drama-derived Neutral samples are present, **Neutral is excluded from scripted-versus-drama-derived source-stratified comparisons**.

The resulting source distribution is therefore:

* Scripted: **300/345 (86.96%)**
* Drama-derived: **45/345 (13.04%)**

For the non-Neutral subset:

* Scripted: **250/295 (84.75%)**
* Drama-derived: **45/295 (15.25%)**

---

## 8. Dataset Development and Model-input Processing Summary

Corpus-level post-acquisition processing and model-input preprocessing are
described separately below.

### Scripted recordings

Scripted recordings were lightly denoised and manually trimmed using Audacity.

### Drama-derived recordings

Drama-derived clips underwent FFmpeg conversion and Demucs vocal separation before inclusion in the corpus.

### Common model-input standardization

Audio was standardized to:

| Property                 | Setting                            |
| ------------------------ | ---------------------------------- |
| Sampling rate            | 16 kHz                             |
| Channels                 | Mono                               |
| Target duration          | 4 seconds                          |
| Target samples per input | 64,000                             |
| Shorter recordings       | Zero-padded                        |
| Longer recordings        | Cropped during model preprocessing |
| Feature normalization    | Wav2Vec2 feature extractor         |

The resulting model input has the shape:

$$
1 \times 64,000
$$

The repository's training implementation applies random cropping to longer training recordings and center cropping during validation and test evaluation, while shorter recordings are zero-padded.

---

## 9. Annotation Summary

 The BNSER corpus comprises 2,300 speech samples. For the annotation-agreement analysis, samples were independently labelled by **three native Noakhali-dialect annotators** using Label Studio. The annotators comprised two male and one female annotator. A reproducible matrix of **2,200 samples with complete labels from all three annotators** was used to calculate the pairwise Cohen's $\kappa$ values.

Annotation was based on acoustic perception without visual or semantic context and followed a brief description of the six emotion categories.

Inter-annotator agreement was assessed pairwise using Cohen's \(\kappa\).

| Annotator pair               | Cohen's \(\kappa\) |
| ---------------------------- | -----------------: |
| Annotator 1 vs. Annotator 2  |              0.805 |
| Annotator 1 vs. Annotator 3  |              0.802 |
| Annotator 2 vs. Annotator 3  |              0.831 |
| **Mean pairwise \(\kappa\)** |          **0.813** |

The manuscript reports these values as the annotation-reliability evidence for the dataset.

---

## 10. File Naming and Speaker Identification

The dataset uses anonymized speaker identifiers. Speaker identifiers are represented using IDs rather than personally identifying information.

The standardized file-naming convention encodes information including:

* speaker gender,
* anonymized speaker ID,
* emotion,
* source sentence, and
* trial/sample number.

This organization supports programmatic association of recordings with their corresponding dataset metadata and experimental partitions.

---

## 11. Important Dataset Considerations

### 11.1 Neutral/source restriction

Neutral is represented only by scripted recordings. Consequently, Neutral-class performance cannot be interpreted as source-independent evidence of Neutral-affect recognition.

### 11.2 Source-associated differences

Scripted and drama-derived recordings differ in more than source label alone. Source is associated with factors including recording conditions, performance context, and speaker composition. Therefore, differences between the two source groups are treated as **source-associated performance differences**, rather than as isolated causal effects of recording or performance style.

### 11.3 Held-out speaker count

Although the dataset partition is strictly speaker-independent, the test set contains only four held-out speakers. The evaluation therefore provides evidence on unseen speakers within the constructed corpus but does not establish population-level generalization to the broader Noakhali-speaking population.

### 11.4 Test-set dependence

All reported model comparisons use the same fixed 345-sample speaker-independent test set. This common test set enables direct comparison among the evaluated model configurations while retaining the limitations imposed by its speaker and source composition.

---

## 12. Dataset Summary

The final BNSER corpus used in this study can be summarized as follows:

| Category                              |          Value |
| ------------------------------------- | -------------: |
| Total samples                         |          2,300 |
| Speakers                              |             25 |
| Male / Female speakers                |        11 / 14 |
| Emotion classes                       |              6 |
| Scripted samples                      |          1,800 |
| Drama-derived samples                 |            500 |
| Training samples                      |          1,610 |
| Validation samples                    |            345 |
| Test samples                          |            345 |
| Training / Validation / Test speakers |     18 / 3 / 4 |
| Test Neutral samples                  |             50 |
| Test Neutral speakers                 |              1 |
| Test scripted samples                 |            300 |
| Test drama-derived samples            |             45 |
| Annotators                            |              3 |
| Mean pairwise Cohen's \(\kappa\)      |          0.813 |
| Model sampling rate                   |         16 kHz |
| Model input duration                  |      4 seconds |
| Model input length                    | 64,000 samples |

---

## 13. Reproducibility Note

The statistics in this document describe the BNSER corpus configuration used for the manuscript revision and the associated speaker-independent experiments.

The audio corpus itself is not reproduced in this repository. Where access to the underlying dataset is required, the reported dataset statistics, partition definitions, and experimental scripts should be interpreted together with the applicable dataset-access and data-use conditions.

All dataset-level counts reported above are intended to remain consistent with the dataset composition and partitioning described in the manuscript.
