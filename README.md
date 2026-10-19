# WMU–PMU Equivalent: Synthetic Closed-Loop Inertia Decomposition

**Status:** Research prototype / synthetic regression test; **not** an IEEE 39-bus or actual converter EMT simulation.

## 1. Execution

```bash
pip install numpy scipy pandas matplotlib
python wmu_multires_validation.py
```

The script generates six figures and five CSV files in `results/`.

## 2. System setup

- System frequency `F0=60 Hz`, common system base `SBASE=500 MVA`.
- SG physical inertia: `H_SG=2.8 s` on the system base (SG 350 MVA × H=4 s / 500 MVA).
- ESS-VSG rated 100 MVA, PV-VSG rated 50 MVA.
- Three 120-s stages. Local settings: ESS `[3.0, 5.0, 3.0] s`, PV `[2.2, 2.2, 0.8] s`.
- Resulting **true VSG parameter sums on the system base**: `[0.82, 1.22, 0.68] s`.
- True ideal SG+GFM total H set: `[3.62, 4.02, 3.48] s`.
- Load disturbance is natural-style stochastic ambient variation with spectral support at 0.15–4 Hz, not a deliberate probing signal.
- A simplified **closed-loop** system is simulated at 240 Hz. SG swing dynamics link load perturbations and GFM output power to bus frequency.

### Aggregate swing model

The differential equation for measured frequency deviation Δf in Hz is

```
M_s * d(Δf)/dt = -ΔP_load - D_s*Δf + ΔP_ESS + ΔP_PV
M_s = 2*Sbase*Hsg/f0
```

Each GFM **external input–output behavior** (not a disclosed internal controller) is

```
ΔP_GFM(jΩ) = -[D/(1+jΩτ_d) + jΩ*M/(1+jΩτ_i)] * ΔF(jΩ)
M = 2*Sbase*H_gfm_on_system_base/f0
```

The 2 lags have independent time constants. In the generator simulation `H_gfm` is a known control parameter. Estimators are NOT given these parameter values or time constants.

### Waveform sampling / processing

- Synthesized **three-phase voltage/current at 15,360 samples/s**, 256 samples per 60-Hz cycle.
- A positive-sequence voltage phase estimate from `Vabc` yields measured frequency.
- Instantaneous three-phase power is computed as `va*ia+vb*ib+vc*ic`.
- WMU feature path: 1-cycle moving average, evaluated at **240 fps**.
- PMU-**like**, not IEEE C37.118 certified: 6-cycle moving average, evaluated at **60 fps**.
- No GNSS timestamp error simulation. No real PLL or proprietary GFM controller model. Voltage/current harmonic and instrument-noise distortions are simplified.
- The frequency features and source powers originate from the SAME synthetic three-phase waveforms.

## 3. Estimators

1. **WMU − PMU (invalid shortcut under test)**: subtract two estimates of *system-wide* apparent inertia. Both include GFM effects.
2. **WMU system-wide apparent inertia − known SG inertia**: subtract static SG from system-level frequency response. Bias remains if droop/filter dynamics create an imaginary spectral component.
3. **Raw terminal GFM spectral inertia**: use the imaginary part of `ΔP/Δf` at each GFM PCC. Delayed droop is mistaken for inertia; used as diagnostic, not a validated physical estimator.
4. **PMU-like terminal black-box fit**: fit an external **two-branch low-order transfer-function form** to P/F spectra; estimate H, droop, and both lag constants jointly.
5. **WMU terminal black-box fit**: same transfer-function model and identical fitting approach, using higher-rate 1-cycle features.

The last two methods are **control-parameter agnostic but NOT model-agnostic**. They assume the specific input-output structure used to synthesize the data. Near-perfect synthetic fit does **not** prove identifiability for a real GFM.

Each stage is estimated independently using approximately 105 s of stationary ambient data (12 s after the stage switch and 3 s before the end). **This is window-by-window offline identification, not yet continuous real-time estimation.** The spectral estimation band is 0.34–3.05 Hz; Welch analysis uses a ~17-s window.

## 4. Output figures and files

- `Fig1_ambient_true_vs_extracted.png`: **true** frequency and ESS power traces versus WMU and PMU-like measurements.
- `Fig2_true_and_decomposition.png`: **true** ideal system + GFM setting inertias versus both system estimators and different separation strategies.
- `Fig3_ESS_PV_true_vs_fit.png`: separate **true** ESS/PV parameter sums vs. fitted values.
- `Fig4_true_Hobs_vs_apparent_spectrum.png`: frequency-dependent **true** inertial transfer component vs raw spectral apparent inertia and fitted **full-response apparent** inertia. The latter includes delayed droop contamination and should NOT be interpreted as pure inertial component.
- `Fig5_method_errors.png`: algorithm comparisons and errors against **true H-set**.
- `Fig6_raw_abc_waveform.png`: example three-phase 15.36-kHz source voltage/current signals. By construction these are generated signals, not measured field recordings.
- `metrics_true_vs_estimates.csv`: per-stage numeric values **and signed errors** against true H-set.
- `summary_method_errors.csv`: mean absolute error, RMSE, bias and worst error by method.
- `component_metrics.csv`: true vs fitted ESS-VSG and PV-VSG contributions.
- `signal_quality_vs_truth.csv`: waveform-extraction RMSE compared to generating synthetic state traces.
- `features_60fps.csv`: processed synchronized features (stored at 60 fps for convenient inspection, 240-fps analysis is performed in RAM).
- `wmU_abc_120ms.csv`: a 120-ms sample of high-rate three-phase waveforms.

## 5. Numerical outcome (seed 20261019)

| Stage | GFM H-set true | WMU-PMU | Known SG subtraction | WMU terminal fit | PMU-like terminal fit |
|---|---:|---:|---:|---:|---:|
| 1 | 0.820 | 0.018 | 0.528 | 0.819 | 0.817 |
| 2 | 1.220 | 0.021 | 0.854 | 1.217 | 1.214 |
| 3 | 0.680 | 0.021 | 0.438 | 0.680 | 0.676 |

**Interpretation:** PMU-like system-wide estimates vary with GFM inertia, so “PMU = SG-only” is incorrect. SG-anchored subtraction removes the known SG contribution but suffers from external droop/inertia lag. An assumed-correct reduced external model can recover the *set parameters* extremely accurately on the synthetic system; the PMU-like estimate is also highly accurate, so this experiment does **not** demonstrate an inherent WMU superiority. The WMU short measurement window does track high-speed waveforms more closely, but an independent physically justified WMU-only feature and unmatched converter dynamics are needed before making a paper claim.

### Identifiability caveats

- The arbitrary internal inertia constant of an unknown GFM cannot always be uniquely determined from PCC voltage/current alone. The model form and sufficient ambient excitation matter.
- PMU equivalence means matched raw-data generation with different filters/report rates. This is not conformance testing of a PMU device.
- Time/frequency-dependent **observed inertia** is distinct from the sum of configured `H_set`. Figure 4 explicitly shows the distinction.
- Closed-loop identifiability with unobserved stochastic input or correlated errors can be biased; the simple synthetic system only uses one controlled stochastic load excitation and small independent measurement noise.
- Changing the external transfer structure, unmodeled control action, current saturation, finite reserve, spatial modal coupling and timestamp jitter may invalidate the estimator. These are necessary follow-up stress tests.
