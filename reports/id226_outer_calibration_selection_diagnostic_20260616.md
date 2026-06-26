# ID226 outer calibration selection diagnostic

Base ID226 mean: `0.491771`.
Full-OOF fixed ID228 mean: `0.482245`.
Outer/nested narrow guard0 mean: `0.495039`.
Outer/nested wide guard0 mean: `0.495039`.

Target-level outer support:
- Q1: base `0.497013`, fixed `0.489696`, outer `0.500906`, improvement_vs_base `-0.003893`, chosen {'temp_t0.85_s0.08': 3, 'temp_t0.85_s0': 1, 'temp_t0.75_s-0.15': 1}
- Q2: base `0.491685`, fixed `0.485084`, outer `0.490435`, improvement_vs_base `+0.001250`, chosen {'temp_t0.85_s-0.08': 2, 'temp_t0.85_s0': 2, 'temp_t0.85_s0.08': 1}
- Q3: base `0.516014`, fixed `0.512239`, outer `0.516150`, improvement_vs_base `-0.000136`, chosen {'temp_t0.85_s0': 3, 'temp_t0.85_s0.08': 1, 'temp_t0.95_s0': 1}
- S1: base `0.433412`, fixed `0.424289`, outer `0.439550`, improvement_vs_base `-0.006138`, chosen {'temp_t0.85_s0.15': 1, 'temp_t0.75_s0.08': 1, 'temp_t0.75_s-0.15': 1, 'temp_t0.85_s-0.08': 1, 'temp_t0.85_s0': 1}
- S2: base `0.477167`, fixed `0.469514`, outer `0.482735`, improvement_vs_base `-0.005567`, chosen {'temp_t0.75_s0': 1, 'temp_t0.85_s0.15': 1, 'temp_t0.85_s0': 1, 'temp_t0.85_s0.08': 1, 'temp_t0.85_s-0.15': 1}
- S3: base `0.465115`, fixed `0.465115`, outer `0.471565`, improvement_vs_base `-0.006450`, chosen {'temp_t0.95_s-0.08': 1, 'temp_t1.05_s0.15': 1, 'temp_t0.95_s-0.15': 1, 'temp_t1.05_s0': 1, 'temp_t0.95_s0': 1}
- S4: base `0.561987`, fixed `0.529776`, outer `0.563932`, improvement_vs_base `-0.001944`, chosen {'temp_t0.75_s0': 2, 'temp_t0.85_s-0.08': 1, 'temp_t0.75_s0.08': 1, 'temp_t0.75_s-0.15': 1}

Interpretation: the full-OOF second-stage gain mostly does not survive outer selection. Q2 is the only target with positive outer support, so ID231 promotes only Q2 temp_t0.85_s0 on ID226.
