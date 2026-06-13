# National Sleep Foundation (NSF) Sleep Guidelines & Quality Indicators

This document details the official **National Sleep Foundation (NSF) Sleep Guidelines** [3], capturing clinical benchmarks for healthy sleep duration and quality. These thresholds are directly mapped to the target binarization criteria (`S1`–`S4` variables) in the ETRI Lifelog dataset.

---

## 1. Age-Based Sleep Duration Recommendations
The National Sleep Foundation reviews and updates evidence-based sleep duration recommendations across the lifespan.

| Age Group | Recommended Duration (Daily) | May Be Appropriate Range | Not Recommended Range |
| :--- | :--- | :--- | :--- |
| **Newborns (0–3 months)** | **14 to 17 hours** | 11 to 13 / 18 to 19 hours | < 11 hours or > 19 hours |
| **Infants (4–11 months)** | **12 to 15 hours** | 10 to 11 / 16 to 18 hours | < 10 hours or > 18 hours |
| **Toddlers (1–2 years)** | **11 to 14 hours** | 9 to 10 / 15 to 16 hours | < 9 hours or > 16 hours |
| **Preschoolers (3–5 years)** | **10 to 13 hours** | 8 to 9 / 14 hours | < 8 hours or > 14 hours |
| **School-aged Children (6–13 years)** | **9 to 11 hours** | 7 to 8 / 12 hours | < 7 hours or > 12 hours |
| **Teenagers (14–17 years)** | **8 to 10 hours** | 7 hours / 11 hours | < 7 hours or > 11 hours |
| **Young Adults (18–25 years)** | **7 to 9 hours** | 6 hours / 10 to 11 hours | < 6 hours or > 11 hours |
| **Adults (26–64 years)** | **7 to 9 hours** | 6 hours / 10 hours | < 6 hours or > 10 hours |
| **Older Adults (65+ years)** | **7 to 8 hours** | 5 to 6 / 9 hours | < 5 hours or > 9 hours |

---

## 2. Standardized Sleep Quality Guidelines
The NSF consensus guidelines define the key objective parameters that indicate "good sleep quality." These directly correlate with target parameters `S1` to `S4` in the ETRI dataset.

### A. Sleep Duration (TST Compliance) ➡️ Target `S1`
- **Metric**: Total Sleep Time (TST).
- **Consensus Standard**: Sleep duration falling within the **7 to 9 hours (420 to 540 mins)** recommended window for adults/young adults.
- **DACON Target `S1`**: Binarized compliance marker.
  - `1`: Sleep duration satisfies the NSF guidelines.
  - `0`: Sleep duration is sub-optimal (insufficient or excessive).

### B. Sleep Efficiency (SE) ➡️ Target `S2`
- **Metric**: Sleep Efficiency (SE).
  $$\text{Sleep Efficiency} = \left( \frac{\text{Total Sleep Time (TST)}}{\text{Total Time in Bed (TIB)}} \right) \times 100\%$$
- **Consensus Standard**: A sleep efficiency of **$\ge 85\%$** is widely recognized as the indicator of healthy sleep continuity.
- **DACON Target `S2`**: Binarized sleep efficiency benchmark.
  - `1`: Sleep Efficiency $\ge 85\%$.
  - `0`: Sleep Efficiency $< 85\%$ (characterized by sleep fragmentation, nocturnal wake cycles).

### C. Sleep Onset Latency (SOL) ➡️ Target `S3`
- **Metric**: Sleep Onset Latency (SOL).
- **Consensus Standard**: The time it takes to fall asleep. Excellent sleep latency is **$\le 15$ minutes** (or up to **$\le 30$ minutes** as appropriate).
- **DACON Target `S3`**: Binarized sleep onset compliance.
  - `1`: Sleep Onset Latency $\le 30$ minutes (normal sleep initiation).
  - `0`: Sleep Onset Latency $> 30$ minutes (delayed sleep onset/insomnia).

### D. Wake After Sleep Onset (WASO) ➡️ Target `S4`
- **Metric**: Wake After Sleep Onset (WASO).
- **Consensus Standard**: The sum of wake times after initial sleep onset until final wake-up.
- **Adult Target**: Keeping WASO **$< 20$ minutes** represents optimal sleep continuity with minimal sleep micro-arousals.
- **DACON Target `S4`**: Binarized WASO continuity.
  - `1`: WASO $< 20$ minutes (or healthy continuity).
  - `0`: WASO $\ge 20$ minutes (excessive mid-night wakefulness).
