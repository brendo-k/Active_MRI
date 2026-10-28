# fastMRI local subset v1 (seed 1234)

Per class: train L 300 (S 80 inside L), val 40, test 40. Sizes: raw = h5 on CoreHPC; cache = complex64 k-space on a 320x320 grid (knee train every other slice, before the energy filter); maps = fp16 BART maps.

| tier | class | volumes | slices | raw GB | cache GB | maps GB |
|---|---|---|---|---|---|---|
| S | knee | 80 | 1447 | 82 | 18 | 9 |
| S | T2 | 80 | 1266 | 44 | 16 | 8 |
| S | FLAIR | 80 | 1262 | 29 | 16 | 8 |
| L | knee | 300 | 5381 | 307 | 66 | 33 |
| L | T2 | 300 | 4762 | 162 | 59 | 29 |
| L | FLAIR | 300 | 4762 | 110 | 59 | 29 |
| val | knee | 40 | 1419 | 41 | 17 | 9 |
| val | T2 | 40 | 640 | 23 | 8 | 4 |
| val | FLAIR | 40 | 634 | 16 | 8 | 4 |
| test | knee | 40 | 1423 | 41 | 17 | 9 |
| test | T2 | 40 | 638 | 22 | 8 | 4 |
| test | FLAIR | 40 | 634 | 15 | 8 | 4 |

| stage | what | volumes | raw GB to copy |
|---|---|---|---|
| 0 | knee train already on /export/scratch (batch2) | 184 | 0 |
| 1 | S train + val + test | 400 | 231 |
| 2 | rest of L | 556 | 316 |

Cache total (L + val + test): 252 GB k-space + 126 GB maps.

## Strata

### knee

| protocol | field | L-only | S | test | val |
|---|---|---|---|---|---|
| PD | 1.5T | 55 | 20 | 10 | 10 |
| PD | 3T | 55 | 20 | 10 | 10 |
| PDFS | 1.5T | 55 | 20 | 10 | 10 |
| PDFS | 3T | 55 | 20 | 10 | 10 |

### T2

| field | model | L-only | S | test | val |
|---|---|---|---|---|---|
| 1.5T | Aera | 50 | 19 | 11 | 10 |
| 3T | Biograph_mMR | 15 | 5 | 3 | 2 |
| 3T | Prisma_fit | 31 | 11 | 6 | 3 |
| 3T | Skyra | 87 | 32 | 15 | 21 |
| 3T | TrioTim | 37 | 13 | 5 | 4 |

### FLAIR

| field | model | L-only | S | test | val |
|---|---|---|---|---|---|
| 1.5T | Aera | 50 | 19 | 11 | 10 |
| 3T | Biograph_mMR | 15 | 5 | 3 | 2 |
| 3T | Prisma_fit | 31 | 11 | 6 | 3 |
| 3T | Skyra | 87 | 32 | 15 | 21 |
| 3T | TrioTim | 37 | 13 | 5 | 4 |

## QC exclusions (whole census)

| cls | qc | volumes |
|---|---|---|
| FLAIR | coils | 5 |
| T2 | artifact | 1 |
| T2 | coils | 5 |
| T2 | under_grid | 16 |
| knee | under_grid | 44 |

## Coils in the selection

| cls | 4 | 6 | 8 | 10 | 12 | 14 | 15 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| FLAIR | 57 | 2 | 1 | 0 | 20 | 13 | 0 | 155 | 5 | 127 |
| T2 | 59 | 0 | 1 | 1 | 19 | 10 | 0 | 156 | 3 | 131 |
| knee | 0 | 0 | 0 | 0 | 0 | 0 | 380 | 0 | 0 | 0 |

## FLAIR acquired phase lines and readout (the 320 grid is only partly filled)

| readout | acq_lines | volumes |
|---|---|---|
| 512 | 160 | 5 |
| 512 | 170 | 16 |
| 512 | 176 | 13 |
| 512 | 187 | 2 |
| 512 | 192 | 1 |
| 512 | 221 | 41 |
| 512 | 256 | 4 |
| 640 | 184 | 6 |
| 640 | 192 | 3 |
| 640 | 198 | 58 |
| 640 | 208 | 1 |
| 640 | 224 | 203 |
| 640 | 234 | 2 |
| 640 | 256 | 25 |
