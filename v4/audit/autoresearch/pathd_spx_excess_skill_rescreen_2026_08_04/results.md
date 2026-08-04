# SPX excess-skill re-screen (Option 0) -- results

verdict: EXCESS_SKILL_CANDIDATE   DIAGNOSTIC-ONLY
sessions: 213/215   family: 60   surrogates/session: 200
primary null: drift-preserving Rademacher wild bootstrap   secondary: circular block bootstrap 30m

member                                           raw IC   excess   maxT p  blk exc   blk p  folds  status
--------------------------------------------------------------------------------------------------------------
omar_clipped_neg3_pos3|60m|-                    +0.5217  +0.0780   0.0000  +0.1068  0.0000    5/5  EXCESS_SURVIVES
omar_clipped_neg3_pos3|15m|-                    +0.3246  +0.0521   0.0001  +0.0622  0.0000    5/5  EXCESS_SURVIVES
omar_clipped_neg3_pos3|30m|-                    +0.4188  +0.0634   0.0002  +0.0840  0.0000    5/5  EXCESS_SURVIVES
momentum_5m_bps|30m|-                           +0.0812  +0.0242   0.0624  +0.0019  0.3986    5/5  NO_EXCESS
momentum_5m_bps|15m|-                           +0.0604  +0.0228   0.0893  -0.0041  0.7731    5/5  NO_EXCESS
momentum_5m_over_session_range|30m|-            +0.0711  +0.0234   0.0925  +0.0019  0.3980    5/5  NO_EXCESS
momentum_5m_over_session_range|15m|-            +0.0523  +0.0218   0.1276  -0.0044  0.7814    5/5  NO_EXCESS
spx_vwap_gap_bps|15m|-                          +0.2909  +0.0258   0.2080  +0.0283  0.0044    5/5  NO_EXCESS
spx_vwap_gap_points|15m|-                       +0.2908  +0.0257   0.2108  +0.0282  0.0044    5/5  NO_EXCESS
spx_vwap_gap_bps|60m|-                          +0.4787  +0.0374   0.2687  +0.0542  0.0010    5/5  NO_EXCESS
spx_vwap_gap_points|60m|-                       +0.4787  +0.0374   0.2687  +0.0542  0.0010    5/5  NO_EXCESS
spx_vwap_gap_over_session_range|15m|-           +0.2258  +0.0226   0.4252  +0.0155  0.0784    4/5  NO_EXCESS
momentum_15m_over_session_range|15m|-           +0.0694  +0.0228   0.4709  -0.0042  0.6781    5/5  NO_EXCESS
momentum_15m_bps|15m|-                          +0.0822  +0.0223   0.4798  -0.0062  0.7553    5/5  NO_EXCESS
spx_vwap_gap_bps|30m|-                          +0.3742  +0.0264   0.4903  +0.0369  0.0054    5/5  NO_EXCESS
spx_vwap_gap_points|30m|-                       +0.3742  +0.0264   0.4924  +0.0369  0.0055    5/5  NO_EXCESS
momentum_15m_over_session_range|30m|-           +0.0879  +0.0171   0.8291  -0.0048  0.6638    4/5  NO_EXCESS
momentum_15m_bps|30m|-                          +0.1044  +0.0163   0.8493  -0.0078  0.7589    4/5  NO_EXCESS
minute_of_session|60m|-                         +0.0378  +0.0379   0.8728  +0.0390  0.1042    5/5  NO_EXCESS
minute_of_session|30m|-                         +0.0245  +0.0245   0.9360  +0.0256  0.1302    4/5  NO_EXCESS
spx_vwap_gap_over_session_range|30m|-           +0.2867  +0.0164   0.9441  +0.0168  0.1324    2/5  NO_EXCESS
minute_of_session|15m|-                         +0.0161  +0.0159   0.9508  +0.0174  0.1294    4/5  NO_EXCESS
session_range_bps|15m|+                         +0.1288  +0.0158   0.9858  +0.0341  0.0313    1/5  NO_EXCESS
spx_vwap_gap_over_session_range|60m|-           +0.3637  +0.0111   0.9993  +0.0195  0.1373    3/5  NO_EXCESS
session_range_bps|30m|+                         +0.1342  +0.0129   1.0000  +0.0378  0.0639    1/5  NO_EXCESS
session_range_bps|60m|+                         +0.1267  +0.0037   1.0000  +0.0343  0.1434    2/5  NO_EXCESS
momentum_5m_bps|60m|-                           +0.0861  +0.0032   1.0000  -0.0134  0.9722    1/5  NO_EXCESS
momentum_15m_over_session_range|60m|-           +0.1095  +0.0029   1.0000  -0.0117  0.8320    2/5  NO_EXCESS
momentum_5m_over_session_range|60m|-            +0.0732  +0.0023   1.0000  -0.0131  0.9632    1/5  NO_EXCESS
momentum_15m_bps|60m|-                          +0.1303  +0.0013   1.0000  -0.0164  0.9203    1/5  NO_EXCESS

survivors (both nulls agree): ['omar_clipped_neg3_pos3|60m|-', 'omar_clipped_neg3_pos3|15m|-', 'omar_clipped_neg3_pos3|30m|-']
null-dependent (primary only): NONE
control breaches: NONE
mean surrogate estimation noise: 0.01524
