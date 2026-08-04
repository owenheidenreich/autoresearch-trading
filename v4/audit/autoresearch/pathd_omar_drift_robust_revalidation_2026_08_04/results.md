# omar drift-robust re-validation -- results

verdict: OMAR_NULL_FRAGILE_CLOSED   DIAGNOSTIC-ONLY   smoke: False
sessions: 213/215   surrogates/session: 200

known-answer gates (pooled 60m fixed-bin surrogate profile):
     zero_drift_wild: spread 0.10786332992481779   rho 0.40606060606060607   passes True
  permuted_time_wild: spread -0.006679374350314661   rho -0.4303030303030303   passes True
      reference_wild: spread 10.121460665405928   rho 1.0   passes False
                real: spread -0.5068678965943532   rho 0.2   passes True

              null                            member   raw IC   excess   maxT p  folds  passes
------------------------------------------------------------------------------------------------
   zero_drift_wild      omar_clipped_neg3_pos3|15m|+  -0.3246  -0.0182   1.0000    4/5  False
   zero_drift_wild      omar_clipped_neg3_pos3|15m|-  +0.3246  +0.0182   0.1883    4/5  False
   zero_drift_wild      omar_clipped_neg3_pos3|30m|+  -0.4188  -0.0176   1.0000    3/5  False
   zero_drift_wild      omar_clipped_neg3_pos3|30m|-  +0.4188  +0.0176   0.3633    3/5  False
   zero_drift_wild      omar_clipped_neg3_pos3|60m|+  -0.5217  -0.0187   1.0000    5/5  False
   zero_drift_wild      omar_clipped_neg3_pos3|60m|-  +0.5217  +0.0187   0.4033    5/5  False
permuted_time_wild      omar_clipped_neg3_pos3|15m|+  -0.3246  -0.0582   1.0000    5/5  False
permuted_time_wild      omar_clipped_neg3_pos3|15m|-  +0.3246  +0.0582   0.0000    5/5  True
permuted_time_wild      omar_clipped_neg3_pos3|30m|+  -0.4188  -0.0671   1.0000    5/5  False
permuted_time_wild      omar_clipped_neg3_pos3|30m|-  +0.4188  +0.0671   0.0000    5/5  True
permuted_time_wild      omar_clipped_neg3_pos3|60m|+  -0.5217  -0.0754   1.0000    5/5  False
permuted_time_wild      omar_clipped_neg3_pos3|60m|-  +0.5217  +0.0754   0.0000    5/5  True

survival by null: {'zero_drift_wild': False, 'permuted_time_wild': True}
valid nulls (KA pass): ['zero_drift_wild', 'permuted_time_wild']
control breaches: NONE
