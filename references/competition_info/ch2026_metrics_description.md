# Seven metrics in the 2026 challenge
March 2026

In this year's competition, seven metrics (i.e., Q1, Q2, Q3, S1, S2, S3, and S4), derived from sleep sensor data and daily questionnaire responses, are provided following a format similar to that of last year [1], as summarized in Table 1.

### Table 1: Metrics to be identified
| Metric | Explanation | Values |
|---|---|---|
| Q1 | Overall sleep quality as perceived by a subject immediately after waking up | 0: Below individual average,<br>1: Above individual average |
| Q2 | Physical fatigue of a subject just before sleep | 0: High level of fatigue,<br>1: Low level of fatigue |
| Q3 | Stress level experienced by a subject just before sleep | 0: High level of stress,<br>1: Low level of stress |
| S1 | Adherence to sleep guidelines for total sleep time (TST) | 0: Not recommended,<br>1: Recommended |
| S2 | Adherence to sleep guidelines for sleep efficiency (SE) | 0: Inappropriate,<br>1: Recommended |
| S3 | Adherence to sleep guidelines for sleep onset latency (SOL) | 0: Inappropriate,<br>1: Recommended |
| S4 | Adherence to sleep guidelines for wakefulness after sleep onset (WASO) | 0: Inappropriate,<br>1: Recommended |

Specifically, the three survey-based metrics (i.e., Q1, Q2, and Q3) were derived from the individual's pre- and post-sleep questionnaire responses averaged over the entire experimental period. Each questionnaire was originally recorded using a 5-point Likert scale but was converted into a binary format for this competition. For instance, the first questionnaire metric (Q1) is assigned a value of 1 on days when an individual's self-reported sleep quality exceeds their average over the experimental period, and 0 when it falls below that average. Similarly, the second and third metrics (Q2 and Q3) are assigned a value of 0 on days when the participant's fatigue and stress levels, respectively, exceed their average, and a value of 1 when these levels are below average. In summary, the survey-based metrics are labeled as 1 when they indicate a subjectively positive outcome based on the individual participant's self-assessment.

Meanwhile, the four sleep metrics (i.e., S1, S2, S3, and S4), derived from the sleep sensor [2], are designed to reflect the sleep health recommendations established by the National Sleep Foundation [3]. These metrics assign a value of 0 for sleep records that do not meet the recommended guidelines.

## References
[1] Oh, Se Won, Hyuntae Jeong, Seungeun Chung, Jeong Mook Lim, Kyoung Ju Noh, Sunkyung Lee, and Gyuwon Jung. "Understanding Human Daily Experience Through Continuous Sensing: ETRI Lifelog Dataset 2024," 2025 16th International Conference on Information and Communication Technology Convergence (ICTC), pp. 1675-1680. IEEE, 2025. https://doi.org/10.1109/ICTC66702.2025.11388616 or https://arxiv.org/abs/2508.03698
[2] Withings, Withings Sleep Analyzer, https://www.withings.com/eu/en/sleep-analyzer
[3] National Sleep Foundation, NSF's Guidelines, https://www.thensf.org/guidelines