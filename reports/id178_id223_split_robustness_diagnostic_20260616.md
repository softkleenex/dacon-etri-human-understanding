# Split robustness diagnostic after ID223

Clean branch: ID218 `0.534103` / robust `0.565633` -> ID221 `0.532990` / robust `0.564811`.
Moderate-low bridge: ID219 `0.522874` / robust `0.553879` -> ID222 `0.521966` / robust `0.553200`.
Moderate bridge: ID220 `0.519561` / robust `0.550442` -> ID223 `0.518724` / robust `0.549816`.

Deltas:
- ID221-ID218: overall `-0.001113`, robust `-0.000822`
- ID222-ID219: overall `-0.000908`, robust `-0.000679`
- ID223-ID220: overall `-0.000837`, robust `-0.000626`

Lowest robustness-score rows:
- ID189: overall `0.492595`, robust `0.526584`, worst `S4`/`id10`/`q2`
- ID172: overall `0.492762`, robust `0.527028`, worst `S4`/`id10`/`q2`
- ID223: overall `0.518724`, robust `0.549816`, worst `S4`/`id10`/`q4`
- ID220: overall `0.519561`, robust `0.550442`, worst `S4`/`id10`/`q4`
- ID217: overall `0.519556`, robust `0.550573`, worst `S4`/`id10`/`q4`
- ID214: overall `0.519982`, robust `0.551765`, worst `S4`/`id10`/`q4`
- ID211: overall `0.520106`, robust `0.551928`, worst `S4`/`id10`/`q4`
- ID222: overall `0.521966`, robust `0.553200`, worst `S4`/`id10`/`q4`
- ID208: overall `0.521177`, robust `0.553419`, worst `S4`/`id10`/`q4`
- ID205: overall `0.521424`, robust `0.553610`, worst `S4`/`id10`/`q4`
- ID219: overall `0.522874`, robust `0.553879`, worst `S4`/`id10`/`q4`
- ID216: overall `0.522875`, robust `0.554029`, worst `S4`/`id10`/`q4`
- ID202: overall `0.521667`, robust `0.554138`, worst `S4`/`id10`/`q4`
- ID213: overall `0.523287`, robust `0.555222`, worst `S4`/`id10`/`q4`

This is OOF evidence only, not public proof. ID222/ID223 inherit ID172 non-nested calibration risk.
