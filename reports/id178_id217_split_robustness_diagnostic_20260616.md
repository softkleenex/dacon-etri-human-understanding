# Split robustness diagnostic after ID217

Clean branch: ID212 `0.534469` / robust `0.566994` -> ID215 `0.534135` / robust `0.565863`.

Bridge branch: ID213 `0.523287` / robust `0.555222` -> ID216 `0.522875` / robust `0.554029`; ID214 `0.519982` / robust `0.551765` -> ID217 `0.519556` / robust `0.550573`.

Lowest robustness-score rows:
- ID189: overall `0.492595`, robust `0.526584`, worst `S4`/`id10`/`q2`
- ID172: overall `0.492762`, robust `0.527028`, worst `S4`/`id10`/`q2`
- ID217: overall `0.519556`, robust `0.550573`, worst `S4`/`id10`/`q4`
- ID214: overall `0.519982`, robust `0.551765`, worst `S4`/`id10`/`q4`
- ID211: overall `0.520106`, robust `0.551928`, worst `S4`/`id10`/`q4`
- ID208: overall `0.521177`, robust `0.553419`, worst `S4`/`id10`/`q4`
- ID205: overall `0.521424`, robust `0.553610`, worst `S4`/`id10`/`q4`
- ID216: overall `0.522875`, robust `0.554029`, worst `S4`/`id10`/`q4`

This is OOF evidence only, not public proof. ID216/217 inherit ID172 non-nested calibration risk.
