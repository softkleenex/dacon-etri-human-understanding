# Split robustness diagnostic after ID220

Clean branch: ID215 `0.534135` / robust `0.565863` -> ID218 `0.534103` / robust `0.565633`.

Bridge branch: ID216 `0.522875` / robust `0.554029` -> ID219 `0.522874` / robust `0.553879`; ID217 `0.519556` / robust `0.550573` -> ID220 `0.519561` / robust `0.550442`.

Lowest robustness-score rows:
- ID189: overall `0.492595`, robust `0.526584`, worst `S4`/`id10`/`q2`
- ID172: overall `0.492762`, robust `0.527028`, worst `S4`/`id10`/`q2`
- ID220: overall `0.519561`, robust `0.550442`, worst `S4`/`id10`/`q4`
- ID217: overall `0.519556`, robust `0.550573`, worst `S4`/`id10`/`q4`
- ID214: overall `0.519982`, robust `0.551765`, worst `S4`/`id10`/`q4`
- ID211: overall `0.520106`, robust `0.551928`, worst `S4`/`id10`/`q4`
- ID208: overall `0.521177`, robust `0.553419`, worst `S4`/`id10`/`q4`
- ID205: overall `0.521424`, robust `0.553610`, worst `S4`/`id10`/`q4`
- ID219: overall `0.522874`, robust `0.553879`, worst `S4`/`id10`/`q4`
- ID216: overall `0.522875`, robust `0.554029`, worst `S4`/`id10`/`q4`

This is OOF evidence only, not public proof. ID219/220 inherit ID172 non-nested calibration risk.
