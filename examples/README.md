# examples/

`example_5_spectra.mgf`: five positive-mode MS/MS spectra of flavonoid standards
(daidzein, genistein, luteolin, quercetin, myricetin), for a first test after installation:

```bash
annomass headless --mgf examples/example_5_spectra.mgf --out test_result.xlsx --top-n 3
```

All five are annotated at level A with the correct compound at rank 1.
