# Human Understanding AI Paper Challenge 2024 - Dataset Design
Se Won Oh, Hyuntae Jeong, Jeong Mook Lim, Seungeun Chung, Kyoung Ju Noh
`{sewonoh, htjeong, jmlim21, schung, kjnoh}@etri.re.kr`
Electronics and Telecommunications Research Institute
Daejeon, South Korea
March 2024

## Abstract
In 2024, we will hold a research paper competition (the third Human Understanding AI Paper Challenge) for the research and development of artificial intelligence technologies to understand human daily life. This document introduces the datasets that will be provided to participants in the competition, and summarizes the issues to consider in data processing and learning model development.

## 1 Introduction
In order to comprehend the multifaceted nature of human behavior in everyday contexts, it is essential to systematically collect and analyze extensive lifelogs obtained from multimodal sensors and user records. During the year 2020, we accumulated in excess of 10,000 hours of sensor data from a cohort consisting of 22 participants [1, 2], employing both smartphones and smartwatches for data acquisition. Further, we organized two paper competitions, held in 2022 [3] and 2023 [4] respectively, with the primary objective of stimulating diverse studies aimed at leveraging artificial intelligence methodologies for the analysis of the aforementioned dataset.

For this year's competition, we are focusing on studies exploring the impact of various daily experiences on an individual's sleep patterns, emotional states, and stress levels. We also collected longer-term lifelog data in 2023 using a similar experiment protocol to 2020, and plan to make it available for this competition. In essence, the objective is to analyze sensor data gathered from smartphones and smartwatches to develop a robust learning model capable of discerning and inferring a comprehensive set of seven indicators linked to sleep quality, emotional responses, and stress levels.

## 2 Dataset composition
The data provided for the 2024 paper competition comprises three distinct datasets, as summarized in Table 1. The train dataset incorporates information gathered in 2020 [1], while the validation and the test datasets encompass additional data collected separately in 2023. Consequently, the demographic makeup of the participants associated with the train dataset differs from that of the participants associated with the validation and the test datasets. On the other hand, the cohort involved in constructing both the validation and the test datasets remains consistent. However, distinctive random integer identifiers(i.e. 'subject_id') are assigned to each participant across the two datasets. Specifically, participant identifiers in the validation dataset range from 1 to 4, while those in the test dataset range from 5 to 8. Table 2 presents the demographics of each participant in the validation dataset. Each participant's age is presented in decade increments, while information regarding height and weight is rounded.

### Table 1: Dataset composition
| Dataset | Number of participants | Days | Year of collection |
|---|---|---|---|
| Train | 22 | 508 | 2020 |
| Validation | 4 | 105 | 2023 |
| Test | 4 | 115 | 2023 |

### Table 2: Demographics of the validation dataset
| subject_id | Gender | Age | Occupation | Height (cm) | Weight (kg) |
|---|---|---|---|---|---|
| 1 | male | 50s | Self-employed | 168 | 88 |
| 2 | female | 40s | Salaried | 163 | 62 |
| 3 | male | 20s | Unemployed | 178 | 74 |
| 4 | female | 50s | Salaried | 155 | 59 |

Each dataset comprises the following four data items, as depicted in Figure 1:
* **Daily survey records:** Self-recorded surveys completed either upon awakening or right before sleep each day.
* **Smartphone:** Sensor data obtained through an application installed on a participant-owned smartphone (Android OS version 10 or higher).
* **Smartwatch:** In the train dataset, acceleration and heart rate data were collected using an Empatica E4 device. For the validation and the test dataset, either a Galaxy Watch4 or Watch5 was employed to gather heart rate, step counts, and light intensity data.
* **Sleep sensor:** Sleep log data from a Withings Sleep Tracking Mat (https://www.withings.com/kr/en/sleep).

### Table 3: Key sensor data items and their corresponding data collection intervals
| Data items | Train Dataset | Validation and Test Dataset |
|---|---|---|
| Smartphone | 3-Axis acceleration (30Hz), Activity category (1/60Hz), GPS coordinates (1/5Hz), Application usage statistics (every 30 minutes) | 3-Axis acceleration (**50Hz**), Activity category (1/60Hz), GPS coordinates (1/5Hz), Application usage statistics (every **10** minutes), **Light intensity** (1/600Hz), **Ambient sound labels** (1/120Hz) |
| Smartwatch | 3-Axis acceleration (32Hz), Heart rate (1Hz) | **Step counts** (1/60Hz), Heart rate (**1/60Hz**), **Light intensity** (1/600Hz) |
| Daily survey records | Subjective sleep quality (5-point scale), Emotion states (5-point scale), Stress levels (5-point scale) | Subjective sleep quality (5-point scale), Emotion states (11-point scale), Stress levels (11-point scale) |
| Sleep sensor | Daily sleep log data (sleep duration, sleep efficiency, etc.) | |

Table 3 provides an overview of the principal sensor data items included in the datasets, along with their respective time intervals for data collection. It is notable that the validation and the test datasets include additional data items that were not present in the train dataset. Additionally, it is crucial to acknowledge that the validation and the test dataset may contain a considerable amount of missing data. Comprehensive descriptions of the data configuration and format for the train dataset are available on the corresponding website and within the reference documentation[1, 2]. Table 4 describes the detailed data configuration for the validation and the test datasets, emphasizing the data items collected from smartphones and smartwatches.

Then we set up a series of classification problem wherein data collected from smartphones and smartwatches serve as the independent variables (X) for training a machine learning model, while metrics derived from either daily survey records or sleep sensor constitute the dependent variables (Y).

### Table 4: The detailed composition of the validation and the test datasets
| Name | Column | DataType | Note |
|---|---|---|---|
| mAcc | x<br>y<br>z | float<br>float<br>float | 3-Axis acceleration(m/s^2) |
| mActivity | activity | string | Detected activity(a), classified by the Google ActivityRecognitionApi(b), represented as an integer between 0 and 8 |
| mAmbience | ambience_labels | string | List of pairs of the top 10 labels and their probabilities (derived from AudioSet Ontology(c)) |
| mGps | altitude<br>latitude<br>longitude<br>speed | float<br>float<br>float<br>float | Missing and unmeasured data are denoted by -1. The unit of speed is m/s. |
| mLight | m_light | float | Ambient light in lx unit |
| mUsageStats | m_usage_stats | string | List of app names and their respective usage times (in milliseconds unit) |
| wHr | heart_rate | integer | Heart rate in beats per minute |
| wPedo | burned_calories<br>distance<br>running_steps<br>speed<br>steps<br>step_frequency<br>walking_steps | float<br>float<br>integer<br>float<br>integer<br>float<br>integer | Number of calories<br>Distance in meters<br>Number of running steps<br>Speed in m/s unit<br>Number of all steps<br>Step frequency in a minute<br>Number of walking steps |
| wLight | w_light | float | Ambient light in lx unit |

*(a) 0 (IN_VEHICLE), 1 (ON_BICYCLE), 2 (ON_FOOT), 3 (STILL), 4 (UNKNOWN), 5 (TILTING), 7 (WALKING), 8 (RUNNING)*
*(b) https://developers.google.com/location-context/activity-recognition*
*(c) https://research.google.com/audioset/*

From the aforementioned data items of Table 3, we derive a total of seven metrics (Q1, Q2, Q3, S1, S2, S3, S4) based on the daily survey records and sleep sensor data, as described in Table 5.

### Table 5: Metrics to be identified
| Metric | Explanation | Values |
|---|---|---|
| Q1 | Overall sleep quality as perceived by a participant immediately after waking up | 0: below average,<br>1: above average |
| Q2 | Emotional state of a participant just before sleep | 0: below average,<br>1: above average |
| Q3 | Stress levels experienced by a participant just before sleep | 0: **above** average,<br>1: **below** average |
| S1 | Total sleep time (TST) | 0: below average,<br>1: above average |
| S2 | Sleep efficiency (SE) | 0: below average,<br>1: above average |
| S3 | Sleep onset latency (SOL, SL) | 0: below average,<br>1: above average |
| S4 | Wake after sleep onset (WASO) | 0: below average,<br>1: above average |

seven metrics per participant, per day, derived from the validation dataset.

In particular, the three questionnaire metrics (Q1, Q2, and Q3) related to the daily survey records are computed based on the average value of the questionnaire responses over the entire experimental period for each participant. For the first two questionnaire metrics (Q1 and Q2), a value of 1 is assigned to days with questionnaire responses above the average, while days with responses below the average receive a value of 0. The stress status indicator (Q3), conversely, assigns a value of 0 to days when each participant's stress level exceeds the average and a value of 1 to days when the stress level falls below the average.

On the other hand, the four sleep metrics (S1, S2, S3, S4) derived from the sleep sensor data are calibrated to comply with the sleep health guidelines set forth by the National Sleep Foundation (NSF)[5]. These metrics assign a value of 1 to sleep records that adhere to the recommended guidelines, while sleep records not meeting the guidelines are assigned a value of 0.
All datasets described so far will be made available for registered participants in this competition to access and analyze.

## 3 Submission format and evaluation method
The inference results of the learning model should be submitted in a comma-separated value (CSV) file format, including the participant IDs, dates, and the binary values (i.e., 0 or 1) for the seven metrics (Q1, Q2, Q3, S1, S2, S3, S4). As described in section 2, each participant's identifier must be represented as an integer value from 5 to 8, and the total data consists of 115 rows.

Each metric necessitates binary classification, and the resulting assessment involves computing a macro F1 score for each metric followed by averaging these scores. However, it is important to consult the pertinent announcements regarding this year's competition, as the weighting of each of the seven metrics in the submitted work may vary, influencing the final ranking. Here, the F1 score serves as a means to consolidate the precision and recall of the model, calculated as the harmonic mean of the model's precision and recall values.

## 4 Conclusion
This document introduces and details the designated datasets required for participants in the third Human Understanding AI Paper Challenge to analyze and develop learning models. We anticipate that this year's challenge would yield the development of best-performing learning models and techniques tailored to recognize various indicators of a person's daily life (i.e., sleep quality, emotional responses, and stress levels).

## References
[1] ETRI_Lifelog_Dataset_2020, 2021. URL https://nanum.etri.re.kr/share/schung/ETRILifelogDataset2020.
[2] Seungeun Chung, Chi Yoon Jeong, Jeong Mook Lim, Jiyoun Lim, Kyoung Ju Noh, Gague Kim, and Hyuntae Jeong. Real-world multimodal lifelog dataset for human behavior study. ETRI Journal, 44(3):426-437, 2022.
[3] The first human understanding AI paper challenge, 2022. URL https://aifactory.space/task/2006/overview.
[4] The second human understanding AI paper challenge, 2023. URL https://aifactory.space/task/2234/overview.
[5] National Sleep Foundation. Nsf's guidelines, 2024. URL https://www.thensf.org/guidelines.