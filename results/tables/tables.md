## table1_datasets

| dataset | split | chases | animals | frames | informative_frames | median_duration_s | fps |
|---|---|---|---|---|---|---|---|
| mouse and cricket (primary) | dev | 210 | 4 | 7713 | 748 | 0.970 | 30.000 |
| mouse and cricket (primary) | held | 275 | 3 | 9571 | 1022 | 0.930 | 30.000 |
| mouse and cricket (replication) | replication | 177 | 3 | 8245 | 3125 | 1.290 | 28.600 |
| mouse and mouse | dev | 5485 | 1987 | 286947 | 193415 | 1.470 | 30.000 |
| mouse and mouse | held | 7748 | 2767 | 400499 | 261207 | 1.470 | 30.000 |

## table2_model_comparison

| dataset | model | per_frame | diff_vs_best_single | diff_lo | diff_hi | diff_informative | diff_informative_lo | diff_informative_hi |
|---|---|---|---|---|---|---|---|---|
| mouse and cricket (primary) | null | 1.424 | -0.009 | -0.012 | -0.005 | -0.015 | -0.024 | -0.005 |
| mouse and cricket (primary) | single:pure_pursuit | 1.432 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mouse and cricket (primary) | single:lead_pursuit | 1.432 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mouse and cricket (primary) | single:proportional_navigation | 1.425 | -0.007 | -0.012 | -0.003 | -0.020 | -0.045 | -0.002 |
| mouse and cricket (primary) | switching:plain | 1.455 | 0.023 | 0.017 | 0.029 | 0.031 | 0.012 | 0.053 |
| mouse and cricket (primary) | switching:input | 1.463 | 0.031 | 0.023 | 0.038 | 0.032 | 0.003 | 0.058 |
| mouse and cricket (primary) | switching_ar | 1.461 | 0.028 | 0.020 | 0.037 | 0.039 | 0.014 | 0.065 |
| mouse and cricket (replication) | null | 0.904 | -0.012 | -0.017 | -0.008 | -0.010 | -0.016 | -0.004 |
| mouse and cricket (replication) | single:pure_pursuit | 0.917 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mouse and cricket (replication) | single:lead_pursuit | 0.917 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mouse and cricket (replication) | single:proportional_navigation | 0.907 | -0.010 | -0.014 | -0.005 | -0.009 | -0.015 | -0.001 |
| mouse and cricket (replication) | switching:plain | 0.942 | 0.026 | 0.018 | 0.033 | 0.027 | 0.015 | 0.039 |
| mouse and cricket (replication) | switching:input | 0.952 | 0.035 | 0.027 | 0.043 | 0.036 | 0.024 | 0.050 |
| mouse and cricket (replication) | switching_ar | 0.923 | 0.006 | -0.001 | 0.013 | 0.008 | -0.002 | 0.018 |
| mouse and mouse | null | 2.627 | -0.002 | -0.002 | -0.001 | -0.001 | -0.002 | -0.001 |
| mouse and mouse | single:pure_pursuit | 2.629 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mouse and mouse | single:lead_pursuit | 2.628 | -0.001 | -0.001 | 0.000 | -0.001 | -0.001 | 0.000 |
| mouse and mouse | single:proportional_navigation | 2.627 | -0.002 | -0.002 | -0.001 | -0.001 | -0.001 | -0.001 |
| mouse and mouse | switching:plain | 2.705 | 0.077 | 0.075 | 0.079 | 0.076 | 0.074 | 0.079 |
| mouse and mouse | switching:input | 2.707 | 0.079 | 0.077 | 0.081 | 0.078 | 0.076 | 0.080 |
| mouse and mouse | switching_ar | 2.924 | 0.296 | 0.293 | 0.298 | 0.304 | 0.301 | 0.307 |

## table3_parameters

| dataset | delay_frames | delay_ms | gain | gain_lo | gain_hi | tau | tau_lo | tau_hi | N | N_lo | N_hi | sigma | sigma_lo | sigma_hi | rho | outlier_fraction | stay_probability_mean | switching_gap_observed | switching_gap_null_q95 | switching_p_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mouse and cricket (primary) | 1 | 33.300 | 0.188 | 0.152 | 0.292 | 0.001 | 0.000 | 0.344 | -0.017 | -0.067 | 0.112 | 0.048 | 0.045 | 0.051 | 0.538 | 0.032 | 0.937 | 0.023 | -0.001 | 0.005 |
| mouse and cricket (replication) | 1 | 35.000 | 0.238 | 0.215 | 0.260 | 0.037 | 0.000 | 0.189 | 0.046 | -0.014 | 0.114 | 0.078 | 0.075 | 0.081 | 0.223 | 0.040 | 0.777 | 0.026 | 0.000 | 0.005 |
| mouse and mouse | 2 | 66.700 | 0.221 | 0.217 | 0.227 | 0.225 | 0.224 | 0.237 | -0.003 | -0.004 | -0.001 | 0.014 | 0.014 | 0.014 | 0.950 | 0.017 | 0.824 | 0.077 | -0.001 | 0.005 |

## table4_posteriors

| dataset | p_steering_all_frames | p_steering_target_moving | frac_steering_path | frac_pure | frac_lead | frac_pn | switches_per_chase | chases_with_switch | state_roles |
|---|---|---|---|---|---|---|---|---|---|
| mouse and cricket (primary) | 0.404 | 0.388 | 0.226 | 0.280 | 0.002 | 0.718 | 0.131 | 0.105 | steering steering non-steering |
| mouse and cricket (replication) | 0.250 | 0.243 | 0.146 | 0.044 | 0.068 | 0.888 | 1.316 | 0.508 | steering steering non-steering |
| mouse and mouse | 0.292 | 0.272 | 0.229 | 0.242 | 0.006 | 0.751 | 0.832 | 0.500 | steering steering non-steering |

## table5_game

| dataset | distance_from_optimal_deg_median | distance_q25_deg | distance_q75_deg | distance_target_moving_deg | distance_to_pure_pursuit_deg | contact_chases | capture_time_ratio_median | manoeuvre_coef_steering | manoeuvre_coef_lo | manoeuvre_coef_hi | manoeuvre_null_mean | manoeuvre_within_chase_effect | manoeuvre_permutation_p | n_manoeuvres | cross_correlation_zero_lag | cross_correlation_peak | cross_correlation_peak_lag_frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mouse and cricket (primary) | 22.700 | 13.600 | 41.200 | 31.800 | 20.200 | 5 | 1.206 | -0.217 | -1.875 | 1.198 | 0.335 | -0.552 | 0.275 | 33 | nan | nan | nan |
| mouse and cricket (replication) | 30.700 | 20.000 | 44.900 | 36.000 | 25.300 | 55 | 1.154 | 0.084 | -0.579 | 0.598 | 0.229 | -0.146 | 0.555 | 268 | nan | nan | nan |
| mouse and mouse | 63.600 | 45.000 | 87.900 | 75.200 | 44.000 | 451 | 1.125 | 0.334 | 0.229 | 0.438 | 0.318 | 0.016 | 0.693 | 5491 | 0.011 | 0.011 | 1.000 |

## table6_followup

| dataset | library | delay_frames | gain | tau | N | N_lo | N_hi | switching_gap | switching_gap_lo | switching_gap_hi | comparator_gap | null_p | p_steering | frac_pure | frac_pn | frac_persistence | state_roles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mouse and cricket (primary) | three laws | 1 | 0.188 | 0.001 | -0.017 | -0.067 | 0.112 | 0.023 | 0.017 | 0.029 | 0.028 | 0.005 | 0.404 | 0.280 | 0.718 | 0.000 | steering steering non-steering |
| mouse and cricket (primary) | three laws and persistence | 1 | 0.280 | 0.000 | -0.025 | -0.073 | 0.117 | 0.028 | 0.020 | 0.035 | 0.030 | 0.032 | 0.257 | 0.005 | 0.569 | 0.325 | steering steering non-steering non-steering |
| mouse and cricket (replication) | three laws | 1 | 0.238 | 0.037 | 0.046 | -0.014 | 0.114 | 0.026 | 0.018 | 0.033 | 0.006 | 0.005 | 0.250 | 0.044 | 0.888 | 0.000 | steering steering non-steering |
| mouse and cricket (replication) | three laws and persistence | 1 | 0.248 | 0.001 | 0.158 | 0.029 | 0.262 | 0.029 | 0.022 | 0.037 | 0.012 | 0.032 | 0.543 | 0.062 | 0.384 | 0.526 | steering steering steering non-steering |
| mouse and mouse | three laws | 2 | 0.221 | 0.225 | -0.003 | -0.004 | -0.001 | 0.077 | 0.075 | 0.079 | 0.296 | 0.005 | 0.292 | 0.242 | 0.751 | 0.000 | steering steering non-steering |
| mouse and mouse | three laws and persistence | 2 | 0.227 | 0.228 | -0.547 | -1.143 | -0.539 | 0.083 | 0.081 | 0.085 | 0.271 | 0.048 | 0.367 | 0.276 | 0.027 | 0.689 | steering steering steering non-steering |

## table7_leave_one_animal_out

| frames | model | per_frame | diff_vs_best_single_per_frame | diff_lo | diff_hi | n_chases | n_frames |
|---|---|---|---|---|---|---|---|
| all | null | 0.902 | -0.012 | -0.017 | -0.007 | 177 | 7537 |
| all | single:pure_pursuit | 0.914 | 0.000 | 0.000 | 0.000 | 177 | 7537 |
| all | single:lead_pursuit | 0.914 | 0.000 | -0.001 | 0.001 | 177 | 7537 |
| all | single:proportional_navigation | 0.905 | -0.010 | -0.015 | -0.005 | 177 | 7537 |
| all | switching:plain | 0.936 | 0.022 | 0.014 | 0.030 | 177 | 7537 |
| all | switching:input | 0.943 | 0.028 | 0.019 | 0.037 | 177 | 7537 |
| all | switching_ar | 0.919 | 0.005 | -0.002 | 0.012 | 177 | 7537 |
| informative | null | 0.880 | -0.010 | -0.017 | -0.004 | 151 | 2831 |
| informative | single:pure_pursuit | 0.890 | 0.000 | 0.000 | 0.000 | 151 | 2831 |
| informative | single:lead_pursuit | 0.889 | 0.000 | -0.002 | 0.001 | 151 | 2831 |
| informative | single:proportional_navigation | 0.881 | -0.009 | -0.017 | -0.001 | 151 | 2831 |
| informative | switching:plain | 0.912 | 0.023 | 0.011 | 0.035 | 151 | 2831 |
| informative | switching:input | 0.918 | 0.028 | 0.014 | 0.044 | 151 | 2831 |
| informative | switching_ar | 0.898 | 0.008 | -0.002 | 0.019 | 151 | 2831 |

## table8_scale_sensitivity

| scale | dev_chases | held_chases | held_informative_frames | delay_frames | gain | gain_lo | gain_hi | N | switching_gap | switching_gap_lo | switching_gap_hi | comparator_gap | null_p | p_steering |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deposit scale | 210 | 275 | 1022 | 1 | 0.188 | 0.152 | 0.292 | -0.017 | 0.023 | 0.017 | 0.029 | 0.028 | 0.005 | 0.404 |
| arena-fit scale | 227 | 311 | 1312 | 1 | 0.168 | 0.142 | 0.326 | -0.025 | 0.021 | 0.016 | 0.026 | 0.033 | 0.032 | 0.400 |

## table9_switch_predictors

| dataset | input_minus_plain | input_minus_plain_lo | input_minus_plain_hi | switching_gap | share_of_switching_gap | event_chases | steer_before | steer_after | phase_only_before | phase_only_after | steer_change | phase_only_change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mouse and cricket (primary) | 0.008 | 0.003 | 0.012 | 0.023 | 0.339 | 17 | 0.476 | 0.469 | 0.345 | 0.361 | -0.006 | 0.016 |
| mouse and cricket (replication) | 0.010 | 0.007 | 0.013 | 0.026 | 0.377 | 114 | 0.288 | 0.273 | 0.227 | 0.229 | -0.015 | 0.003 |
| mouse and mouse | 0.002 | 0.002 | 0.002 | 0.077 | 0.027 | 3635 | 0.362 | 0.239 | 0.317 | 0.238 | -0.123 | -0.079 |

