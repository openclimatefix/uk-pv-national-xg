Additionally, here are the more raw results as well:
```bash
/home/jacob/anaconda3/envs/national_xg/bin/python /home/jacob/Development/gradboost_pv/scripts/models/train/region_filtered_model.py
/home/jacob/anaconda3/envs/national_xg/lib/python3.10/site-packages/nowcasting_datamodel/models/forecast.py:206: SAWarning: This declarative base already contains a class with the same class name and module name as nowcasting_datamodel.models.forecast.ForecastValueYearMonth, and will be replaced in the string-lookup table.
  class ForecastValueYearMonth(ForecastValueSQLMixin, Base_Forecast):
0

---------All-----------
Median MAE: 0.00758
Mean MAE: 0.00774
Std MAE: 0.00051
Mean Pinball Median: 0.49991
Std Pinball Median: 0.02385
Mean Pinball 10th percentile: 0.21619
Std Pinball 10th percentile: 0.01451
Mean Pinball 90th percentile: 0.72409
Std Pinball 90th percentile: 0.0279
Pinball Medians: [0.48077505298213746, 0.4874340286504147, 0.5429756220155818, 0.485530140275496, 0.4712945590994371, 0.5220264317180616, 0.5093668040464594]
Pinball 10th percentiles: [0.22176808961550107, 0.2119879366675044, 0.24918321186227696, 0.20510552255781625, 0.20537836147592245, 0.2131057268722467, 0.2068190333458224]
Pinball 90th percentiles: [0.6874053890402664, 0.7046996732847449, 0.7552148781100779, 0.7195753822823202, 0.6959349593495935, 0.7648678414096917, 0.7409142000749345]
Pinball Ranges: [0.4656373  0.49271174 0.50603167 0.51446986 0.4905566  0.55176211
 0.53409517]
Mean Ranges: 0.5078949201649341
Std Ranges: 0.02665745192155385
---------All-----------

[INFO][2023-07-06 12:31:53] : Trained model for 0 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0010550639840426152, pinball_test_loss=0.0037893298269305238, pinball_train_10_percentile_loss=0.001077463850161541, pinball_test_10_percentile_loss=0.0022186065315161377, pinball_train_90_percentile_loss=0.0005756842648535435, pinball_test_90_percentile_loss=0.0023311790724128223, mae_train_loss=0.0021101279680852303, mae_test_loss=0.0075786596538610475, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
1

---------All-----------
Median MAE: 0.0121
Mean MAE: 0.01219
Std MAE: 0.00094
Mean Pinball Median: 0.49761
Std Pinball Median: 0.03574
Mean Pinball 10th percentile: 0.24395
Std Pinball 10th percentile: 0.03457
Mean Pinball 90th percentile: 0.69554
Std Pinball 90th percentile: 0.03127
Pinball Medians: [0.5205935796486978, 0.528273435536567, 0.5364413169137975, 0.4578648136449779, 0.43477173233270794, 0.5165863730213351, 0.48875983514424876]
Pinball 10th percentiles: [0.29421562689279224, 0.2764513696908771, 0.27783362653933147, 0.2169298799747315, 0.2046278924327705, 0.22573984858912594, 0.21187710753091046]
Pinball 90th percentiles: [0.6883706844336765, 0.713998492083438, 0.7158833877858759, 0.6732785849652558, 0.6350218886804253, 0.7380591878871301, 0.7041963282128137]
Pinball Ranges: [0.39415506 0.43754712 0.43804976 0.4563487  0.430394   0.51231934
 0.49231922]
Mean Ranges: 0.4515904574854394
Std Ranges: 0.03683583466447259
---------All-----------

[INFO][2023-07-06 01:01:43] : Trained model for 1 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012045550234201239, pinball_test_loss=0.005886998386154272, pinball_train_10_percentile_loss=0.0013648913485722958, pinball_test_10_percentile_loss=0.0032549798597562565, pinball_train_90_percentile_loss=0.0007372956891999191, pinball_test_90_percentile_loss=0.003809915106989967, mae_train_loss=0.0024091100468402477, mae_test_loss=0.011773996772308544, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
2

---------All-----------
Median MAE: 0.01724
Mean MAE: 0.01742
Std MAE: 0.00147
Mean Pinball Median: 0.488
Std Pinball Median: 0.07414
Mean Pinball 10th percentile: 0.2563
Std Pinball 10th percentile: 0.06107
Mean Pinball 90th percentile: 0.66742
Std Pinball 90th percentile: 0.05387
Pinball Medians: [0.5631626779763708, 0.5996481528022116, 0.5118120130686102, 0.42238221548566374, 0.36910569105691055, 0.4974535443909153, 0.4524166354439865]
Pinball 10th percentiles: [0.3352014541048167, 0.34757476752953004, 0.2824830359386781, 0.20272830617658205, 0.1856160100062539, 0.23441156228492774, 0.20606968902210565]
Pinball 90th percentiles: [0.7043320205998183, 0.740512691631063, 0.6725307866298065, 0.623468485537451, 0.5661038148843027, 0.7040605643496215, 0.6609216935181716]
Pinball Ranges: [0.36913057 0.39293792 0.39004775 0.42074018 0.3804878  0.469649
 0.454852  ]
Mean Ranges: 0.4111207474410486
Std Ranges: 0.035670619414563225
---------All-----------

[INFO][2023-07-06 01:29:49] : Trained model for 2 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012566411983616936, pinball_test_loss=0.008501757542043954, pinball_train_10_percentile_loss=0.0016690717023773577, pinball_test_10_percentile_loss=0.004541239745485791, pinball_train_90_percentile_loss=0.0008389615227801574, pinball_test_90_percentile_loss=0.006025412010102769, mae_train_loss=0.002513282396723387, mae_test_loss=0.017003515084087907, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
3

---------All-----------
Median MAE: 0.01707
Mean MAE: 0.01748
Std MAE: 0.00148
Mean Pinball Median: 0.49109
Std Pinball Median: 0.06793
Mean Pinball 10th percentile: 0.25588
Std Pinball 10th percentile: 0.05882
Mean Pinball 90th percentile: 0.66645
Std Pinball 90th percentile: 0.05058
Pinball Medians: [0.5503030303030303, 0.5983915556672531, 0.5159587836139734, 0.4326303826240687, 0.3811131957473421, 0.5006194081211287, 0.4585987261146497]
Pinball 10th percentiles: [0.32696969696969697, 0.34367931641115856, 0.28901734104046245, 0.2057077913878015, 0.1878674171357098, 0.23468685478320717, 0.20325964780816785]
Pinball 90th percentiles: [0.6953030303030303, 0.7374968585071626, 0.6744156823322442, 0.6286147240813234, 0.5702313946216385, 0.7011699931176876, 0.6579243162233046]
Pinball Ranges: [0.36833333 0.39381754 0.38539834 0.42290693 0.38236398 0.46648314
 0.45466467]
Mean Ranges: 0.4105668476643124
Std Ranges: 0.03528777940003578
---------All-----------

[INFO][2023-07-06 02:01:31] : Trained model for 3 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012689549908944994, pinball_test_loss=0.008507723029261281, pinball_train_10_percentile_loss=0.0016784937869522307, pinball_test_10_percentile_loss=0.004494430521513261, pinball_train_90_percentile_loss=0.000847069912164344, pinball_test_90_percentile_loss=0.006066183673298961, mae_train_loss=0.0025379099817889987, mae_test_loss=0.017015446058522562, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
4

---------All-----------
Median MAE: 0.01735
Mean MAE: 0.01752
Std MAE: 0.00139
Mean Pinball Median: 0.48897
Std Pinball Median: 0.06672
Mean Pinball 10th percentile: 0.25631
Std Pinball 10th percentile: 0.05519
Mean Pinball 90th percentile: 0.66709
Std Pinball 90th percentile: 0.04684
Pinball Medians: [0.5488710410668283, 0.590726313144006, 0.5126916310630811, 0.4311324327736397, 0.3771106941838649, 0.499518238128011, 0.4627201198950918]
Pinball 10th percentiles: [0.319745415972117, 0.33990952500628296, 0.28914300075395827, 0.20792829188233808, 0.19111944965603503, 0.23799036476256022, 0.2083177219932559]
Pinball 90th percentiles: [0.6872253371722988, 0.7338527268157828, 0.6691379743654184, 0.636535790935488, 0.5779862414008755, 0.7064005505849965, 0.6584863244660921]
Pinball Ranges: [0.36747992 0.3939432  0.37999497 0.4286075  0.38686679 0.46841019
 0.4501686 ]
Mean Ranges: 0.4107815965306293
Std Ranges: 0.035593847790062895
---------All-----------

[INFO][2023-07-06 02:38:18] : Trained model for 4 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012600357333270877, pinball_test_loss=0.008674636532452047, pinball_train_10_percentile_loss=0.0016875834758432317, pinball_test_10_percentile_loss=0.00461961522857021, pinball_train_90_percentile_loss=0.0008385207358480372, pinball_test_90_percentile_loss=0.006020705398326387, mae_train_loss=0.0025200714666541754, mae_test_loss=0.017349273064904094, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
5

---------All-----------
Median MAE: 0.01722
Mean MAE: 0.01746
Std MAE: 0.00131
Mean Pinball Median: 0.49024
Std Pinball Median: 0.06373
Mean Pinball 10th percentile: 0.25536
Std Pinball 10th percentile: 0.05188
Mean Pinball 90th percentile: 0.66989
Std Pinball 90th percentile: 0.04642
Pinball Medians: [0.5396272162448856, 0.5859512440311636, 0.5141995476250314, 0.437263317344105, 0.3758599124452783, 0.5029593943565038, 0.47583364556013485]
Pinball 10th percentiles: [0.3135323533868768, 0.3331239004775069, 0.2877607439055039, 0.2085331986872002, 0.1886178861788618, 0.23826565726083965, 0.21768452603971525]
Pinball 90th percentiles: [0.6840430368237612, 0.72606182457904, 0.6742900226187484, 0.6437768240343348, 0.575234521575985, 0.7153475567790778, 0.6704758336455602]
Pinball Ranges: [0.37051068 0.39293792 0.38652928 0.43524363 0.38661664 0.4770819
 0.45279131]
Mean Ranges: 0.41453019344571473
Std Ranges: 0.03736468224737096
---------All-----------

[INFO][2023-07-06 03:17:02] : Trained model for 5 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012591772777983904, pinball_test_loss=0.008657398107276362, pinball_train_10_percentile_loss=0.0016868559946522192, pinball_test_10_percentile_loss=0.0046995857295653405, pinball_train_90_percentile_loss=0.0008298732496404436, pinball_test_90_percentile_loss=0.005925644820824493, mae_train_loss=0.002518354555596781, mae_test_loss=0.017314796214552723, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
6

---------All-----------
Median MAE: 0.01725
Mean MAE: 0.01738
Std MAE: 0.00118
Mean Pinball Median: 0.49079
Std Pinball Median: 0.05844
Mean Pinball 10th percentile: 0.25666
Std Pinball 10th percentile: 0.04768
Mean Pinball 90th percentile: 0.669
Std Pinball 90th percentile: 0.04436
Pinball Medians: [0.5296257008637673, 0.5810505152048253, 0.5163357627544609, 0.44041908608937136, 0.38649155722326456, 0.5003441156228493, 0.4812663919070813]
Pinball 10th percentiles: [0.3026216093347477, 0.3292284493591355, 0.29140487559688366, 0.21194142893208787, 0.19161976235146966, 0.24349621472814867, 0.22630198576245786]
Pinball 90th percentiles: [0.6760115168964995, 0.7168886654938427, 0.6801960291530535, 0.6428932087856601, 0.5764853033145716, 0.713833448038541, 0.6766579243162233]
Pinball Ranges: [0.37338991 0.38766022 0.38879115 0.43095178 0.38486554 0.47033723
 0.45035594]
Mean Ranges: 0.4123359671333516
Std Ranges: 0.03503240212807978
---------All-----------

[INFO][2023-07-06 03:51:43] : Trained model for 6 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012649109885043397, pinball_test_loss=0.008622888443444393, pinball_train_10_percentile_loss=0.0016861222899890171, pinball_test_10_percentile_loss=0.004686736469494854, pinball_train_90_percentile_loss=0.0008336435085191027, pinball_test_90_percentile_loss=0.005722216085552032, mae_train_loss=0.0025298219770086795, mae_test_loss=0.017245776886888785, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
7

---------All-----------
Median MAE: 0.0174
Mean MAE: 0.01737
Std MAE: 0.00108
Mean Pinball Median: 0.48862
Std Pinball Median: 0.05685
Mean Pinball 10th percentile: 0.25537
Std Pinball 10th percentile: 0.04477
Mean Pinball 90th percentile: 0.66705
Std Pinball 90th percentile: 0.04274
Pinball Medians: [0.520988028489165, 0.5760241266649913, 0.5144508670520231, 0.4327190103509215, 0.3889931207004378, 0.5026841018582243, 0.4844511052828775]
Pinball 10th percentiles: [0.29625700863767235, 0.326212616235235, 0.28813772304599145, 0.21610704367583944, 0.1961225766103815, 0.24129387474191327, 0.22349194454852003]
Pinball 90th percentiles: [0.6699499924230944, 0.7183965820557929, 0.6749183211862277, 0.6377177480434234, 0.5816135084427767, 0.7097040605643496, 0.6770325964780817]
Pinball Ranges: [0.37369298 0.39218397 0.3867806  0.4216107  0.38549093 0.46841019
 0.45354065]
Mean Ranges: 0.4116728602425991
Std Ranges: 0.03422238813970741
---------All-----------

[INFO][2023-07-06 04:25:46] : Trained model for 7 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012482301233079098, pinball_test_loss=0.008704918244465376, pinball_train_10_percentile_loss=0.0016749797908853167, pinball_test_10_percentile_loss=0.004772928317106886, pinball_train_90_percentile_loss=0.0008245152943541549, pinball_test_90_percentile_loss=0.005734638911555522, mae_train_loss=0.0024964602466158196, mae_test_loss=0.01740983648893075, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
8

---------All-----------
Median MAE: 0.01715
Mean MAE: 0.01739
Std MAE: 0.00099
Mean Pinball Median: 0.48715
Std Pinball Median: 0.05295
Mean Pinball 10th percentile: 0.25444
Std Pinball 10th percentile: 0.04409
Mean Pinball 90th percentile: 0.66836
Std Pinball 90th percentile: 0.03998
Pinball Medians: [0.505379602970147, 0.569364161849711, 0.517089721035436, 0.434036106552203, 0.3944965603502189, 0.5024088093599449, 0.4872611464968153]
Pinball 10th percentiles: [0.2876193362630702, 0.32445338024629305, 0.2896456396079417, 0.21461936624163616, 0.1944965603502189, 0.24996558843771507, 0.22030723117272386]
Pinball 90th percentiles: [0.6640400060615245, 0.7122392560944961, 0.6823322442824831, 0.6374195177376594, 0.5902439024390244, 0.7110805230557468, 0.6811539902585237]
Pinball Ranges: [0.37642067 0.38778588 0.3926866  0.42280015 0.39574734 0.46111493
 0.46084676]
Mean Ranges: 0.41391461965855125
Std Ranges: 0.032466761559661494
---------All-----------

9
[INFO][2023-07-06 05:07:18] : Trained model for 8 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012586437719674258, pinball_test_loss=0.008577267065011888, pinball_train_10_percentile_loss=0.0016860567623376726, pinball_test_10_percentile_loss=0.004750062878693247, pinball_train_90_percentile_loss=0.0008281783249143637, pinball_test_90_percentile_loss=0.005655593532143917, mae_train_loss=0.0025172875439348516, mae_test_loss=0.017154534130023776, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
/home/jacob/anaconda3/envs/national_xg/bin/python /home/jacob/Development/gradboost_pv/scripts/models/train/region_filtered_model.py
/home/jacob/anaconda3/envs/national_xg/lib/python3.10/site-packages/nowcasting_datamodel/models/forecast.py:206: SAWarning: This declarative base already contains a class with the same class name and module name as nowcasting_datamodel.models.forecast.ForecastValueYearMonth, and will be replaced in the string-lookup table.
  class ForecastValueYearMonth(ForecastValueSQLMixin, Base_Forecast):
9

---------All-----------
Median MAE: 0.01713
Mean MAE: 0.0174
Std MAE: 0.00102
Mean Pinball Median: 0.48759
Std Pinball Median: 0.04895
Mean Pinball 10th percentile: 0.25524
Std Pinball 10th percentile: 0.04225
Mean Pinball 90th percentile: 0.66886
Std Pinball 90th percentile: 0.03787
Pinball Medians: [0.5073495984240036, 0.5606936416184971, 0.5194772555918572, 0.4388334806211337, 0.40150093808630394, 0.5021335168616655, 0.48313975271637316]
Pinball 10th percentiles: [0.282618578572511, 0.32445338024629305, 0.29328977129932143, 0.21436687286958717, 0.20512820512820512, 0.24707501720578115, 0.2197452229299363]
Pinball 90th percentiles: [0.6572207910289438, 0.7087207841166122, 0.6819552651419954, 0.6423431384926146, 0.5968730456535335, 0.7150722642807984, 0.6798426376920195]
Pinball Ranges: [0.37460221 0.3842674  0.38866549 0.42797627 0.39174484 0.46799725
 0.46009741]
Mean Ranges: 0.413621554022126
Std Ranges: 0.03547302808795848
---------All-----------

10
[INFO][2023-07-06 06:59:45] : Trained model for 9 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.001276911017321958, pinball_test_loss=0.008566433111946945, pinball_train_10_percentile_loss=0.001689221029266051, pinball_test_10_percentile_loss=0.004718696386915341, pinball_train_90_percentile_loss=0.0008432420674470898, pinball_test_90_percentile_loss=0.005596480445033437, mae_train_loss=0.002553822034643916, mae_test_loss=0.01713286622389389, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.0172
Mean MAE: 0.01746
Std MAE: 0.00105
Mean Pinball Median: 0.48719
Std Pinball Median: 0.04847
Mean Pinball 10th percentile: 0.25443
Std Pinball 10th percentile: 0.03996
Mean Pinball 90th percentile: 0.66661
Std Pinball 90th percentile: 0.0381
Pinball Medians: [0.49742424242424244, 0.5618090452261306, 0.5125659713495854, 0.4429852254072484, 0.400200100050025, 0.5115638766519823, 0.48379239272999813]
Pinball 10th percentiles: [0.2781818181818182, 0.3185929648241206, 0.2920331741643629, 0.2194721555751989, 0.20210105052526264, 0.24931167400881057, 0.22128536631066142]
Pinball 90th percentiles: [0.6468181818181818, 0.703643216080402, 0.6809499874340287, 0.6417476954160879, 0.5966733366683342, 0.7166850220264317, 0.6797826494285178]
Pinball Ranges: [0.36863636 0.38505025 0.38891681 0.42227554 0.39457229 0.46737335
 0.45849728]
Mean Ranges: 0.4121888407545355
Std Ranges: 0.03540943122382602
---------All-----------

11
[INFO][2023-07-06 07:29:24] : Trained model for 10 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012650131625157681, pinball_test_loss=0.008599458078603668, pinball_train_10_percentile_loss=0.0016870140442079547, pinball_test_10_percentile_loss=0.00481138729144192, pinball_train_90_percentile_loss=0.0008371546305378639, pinball_test_90_percentile_loss=0.0055735914304872475, mae_train_loss=0.0025300263250315363, mae_test_loss=0.017198916157207336, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01707
Mean MAE: 0.01747
Std MAE: 0.00102
Mean Pinball Median: 0.48774
Std Pinball Median: 0.04806
Mean Pinball 10th percentile: 0.25302
Std Pinball 10th percentile: 0.03664
Mean Pinball 90th percentile: 0.66849
Std Pinball 90th percentile: 0.03872
Pinball Medians: [0.4831869130566495, 0.5601758793969849, 0.5185976375973863, 0.4459322890348661, 0.4020760380190095, 0.5170751858992013, 0.48716026241799437]
Pinball 10th percentiles: [0.2682520448348985, 0.31532663316582915, 0.2863784870570495, 0.22637695805962607, 0.20485242621310656, 0.2506196640044065, 0.21930646672914714]
Pinball 90th percentiles: [0.6422296273856407, 0.7003768844221105, 0.6861020356873586, 0.6421172309247094, 0.6006753376688344, 0.7222528229137979, 0.6856607310215558]
Pinball Ranges: [0.37397758 0.38505025 0.39972355 0.41574027 0.39582291 0.47163316
 0.46635426]
Mean Ranges: 0.4154717128514206
Std Ranges: 0.03590567491271854
---------All-----------

[INFO][2023-07-06 07:57:37] : Trained model for 11 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.001260908606714204, pinball_test_loss=0.008534888684404896, pinball_train_10_percentile_loss=0.0017009786169671054, pinball_test_10_percentile_loss=0.004781940978714406, pinball_train_90_percentile_loss=0.0008362756935949136, pinball_test_90_percentile_loss=0.005604765126643843, mae_train_loss=0.002521817213428408, mae_test_loss=0.017069777368809793, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
12

---------All-----------
Median MAE: 0.01691
Mean MAE: 0.01746
Std MAE: 0.00109
Mean Pinball Median: 0.4895
Std Pinball Median: 0.04458
Mean Pinball 10th percentile: 0.25294
Std Pinball 10th percentile: 0.03501
Mean Pinball 90th percentile: 0.66953
Std Pinball 90th percentile: 0.03755
Pinball Medians: [0.47940642035130226, 0.5510050251256281, 0.5283990952500628, 0.44933030073287844, 0.4117058529264632, 0.518595041322314, 0.4880930058128633]
Pinball 10th percentiles: [0.2628709872804361, 0.30967336683417085, 0.2901482784619251, 0.23313115996967398, 0.2054777388694347, 0.2513774104683196, 0.2178886180386274]
Pinball 90th percentiles: [0.6364324651726226, 0.6938442211055277, 0.6871073133953255, 0.6474601971190296, 0.6071785892946473, 0.7252066115702479, 0.6894805925370335]
Pinball Ranges: [0.37356148 0.38417085 0.39695903 0.41432904 0.40170085 0.4738292
 0.47159197]
Mean Ranges: 0.4165917757531209
Std Ranges: 0.03744743279683297
---------All-----------

13
[INFO][2023-07-06 08:32:50] : Trained model for 12 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012774249067674674, pinball_test_loss=0.00845450839859749, pinball_train_10_percentile_loss=0.0016880324521668347, pinball_test_10_percentile_loss=0.004728071141670281, pinball_train_90_percentile_loss=0.0008396215046314899, pinball_test_90_percentile_loss=0.0054995647254289706, mae_train_loss=0.002554849813534935, mae_test_loss=0.01690901679719498, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01703
Mean MAE: 0.01749
Std MAE: 0.00113
Mean Pinball Median: 0.48953
Std Pinball Median: 0.04298
Mean Pinball 10th percentile: 0.25381
Std Pinball 10th percentile: 0.03517
Mean Pinball 90th percentile: 0.66902
Std Pinball 90th percentile: 0.03694
Pinball Medians: [0.4709355131698456, 0.5467336683417086, 0.5301583312390048, 0.45462588473205257, 0.41458229114557277, 0.518186828327363, 0.4914650159444757]
Pinball 10th percentiles: [0.2612776264002422, 0.3079145728643216, 0.29429504900728826, 0.23559150657229525, 0.20560280140070036, 0.25530449159548085, 0.21665728756330896]
Pinball 90th percentiles: [0.6259461095973358, 0.6962311557788945, 0.6884895702437799, 0.6504044489383215, 0.6141820910455228, 0.7237531000275558, 0.684111798912024]
Pinball Ranges: [0.36466848 0.38831658 0.39419452 0.41481294 0.40857929 0.46844861
 0.46745451]
Mean Ranges: 0.41521070559139955
Std Ranges: 0.0365030595897159
---------All-----------

14
[INFO][2023-07-06 09:00:13] : Trained model for 13 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012669980085401019, pinball_test_loss=0.00851252161129882, pinball_train_10_percentile_loss=0.0016722306311165382, pinball_test_10_percentile_loss=0.0047605460944471125, pinball_train_90_percentile_loss=0.0008310473475914547, pinball_test_90_percentile_loss=0.005593144151329621, mae_train_loss=0.0025339960170802037, mae_test_loss=0.01702504322259764, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01701
Mean MAE: 0.01748
Std MAE: 0.00116
Mean Pinball Median: 0.48751
Std Pinball Median: 0.04409
Mean Pinball 10th percentile: 0.25379
Std Pinball 10th percentile: 0.03324
Mean Pinball 90th percentile: 0.67067
Std Pinball 90th percentile: 0.03848
Pinball Medians: [0.4555084745762712, 0.5380653266331659, 0.5285247549635587, 0.45391326337084337, 0.4115807903951976, 0.5226019845644984, 0.5023456558453744]
Pinball 10th percentiles: [0.25408595641646486, 0.3056532663316583, 0.29215883387785874, 0.23416361107598938, 0.2064782391195598, 0.26033627342888643, 0.2236817414148996]
Pinball 90th percentiles: [0.6197033898305084, 0.6879396984924623, 0.6882382508167881, 0.6578581363004172, 0.6149324662331166, 0.7253307607497244, 0.7006943141302309]
Pinball Ranges: [0.36561743 0.38228643 0.39607942 0.42369453 0.40845423 0.46499449
 0.47701257]
Mean Ranges: 0.4168770135554187
Std Ranges: 0.03836961494343664
---------All-----------

15
[INFO][2023-07-06 09:27:35] : Trained model for 14 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012647000685084525, pinball_test_loss=0.0085026477497896, pinball_train_10_percentile_loss=0.0016847799094325737, pinball_test_10_percentile_loss=0.004739563187348705, pinball_train_90_percentile_loss=0.0008386702612803291, pinball_test_90_percentile_loss=0.005475037722416321, mae_train_loss=0.002529400137016905, mae_test_loss=0.0170052954995792, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01708
Mean MAE: 0.01748
Std MAE: 0.00119
Mean Pinball Median: 0.49091
Std Pinball Median: 0.04508
Mean Pinball 10th percentile: 0.25463
Std Pinball 10th percentile: 0.03452
Mean Pinball 90th percentile: 0.66932
Std Pinball 90th percentile: 0.04336
Pinball Medians: [0.4518910741301059, 0.5419597989949749, 0.5339281226438803, 0.4661606578115117, 0.411455727863932, 0.5230217810862973, 0.5079782241411677]
Pinball 10th percentiles: [0.24326777609682299, 0.30917085427135677, 0.29668258356370947, 0.23908918406072105, 0.20485242621310656, 0.2612351805900193, 0.22808334897691007]
Pinball 90th percentiles: [0.6086232980332829, 0.6881909547738694, 0.6951495350590601, 0.6603415559772297, 0.607303651825913, 0.7321477805348773, 0.6934484700581941]
Pinball Ranges: [0.36535552 0.3790201  0.39846695 0.42125237 0.40245123 0.4709126
 0.46536512]
Mean Ranges: 0.41468912749854
Std Ranges: 0.03759031166642299
---------All-----------

16
[INFO][2023-07-06 09:55:19] : Trained model for 15 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012732621961271504, pinball_test_loss=0.00853815747613123, pinball_train_10_percentile_loss=0.001680214000215345, pinball_test_10_percentile_loss=0.004736485355835561, pinball_train_90_percentile_loss=0.0008384536084033679, pinball_test_90_percentile_loss=0.00549177710325261, mae_train_loss=0.0025465243922543008, mae_test_loss=0.01707631495226246, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01724
Mean MAE: 0.01748
Std MAE: 0.00119
Mean Pinball Median: 0.48787
Std Pinball Median: 0.04431
Mean Pinball 10th percentile: 0.25517
Std Pinball 10th percentile: 0.03534
Mean Pinball 90th percentile: 0.66843
Std Pinball 90th percentile: 0.04623
Pinball Medians: [0.4489487218272576, 0.5382585751978892, 0.528657616892911, 0.461781827385472, 0.4113306653326663, 0.5235797021511307, 0.5025352112676056]
Pinball 10th percentiles: [0.24081076992890638, 0.30795326045985677, 0.30002513826043237, 0.2457605669450772, 0.20135067533766884, 0.2617209045780474, 0.22854460093896714]
Pinball 90th percentiles: [0.6003630313114506, 0.6890312853373539, 0.6954499748617395, 0.6599595039230575, 0.6044272136068034, 0.7334528405956977, 0.6963380281690141]
Pinball Ranges: [0.35955226 0.38107802 0.39542484 0.41419894 0.40307654 0.47173194
 0.46779343]
Mean Ranges: 0.4132651373365944
Std Ranges: 0.039162190332510526
---------All-----------

[INFO][2023-07-06 10:24:20] : Trained model for 16 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012698797343801917, pinball_test_loss=0.008620782713796968, pinball_train_10_percentile_loss=0.0016710149684008723, pinball_test_10_percentile_loss=0.00478120810379803, pinball_train_90_percentile_loss=0.0008293242311460279, pinball_test_90_percentile_loss=0.005573138166152395, mae_train_loss=0.0025397594687603833, mae_test_loss=0.017241565427593936, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
17

---------All-----------
Median MAE: 0.01714
Mean MAE: 0.01757
Std MAE: 0.00119
Mean Pinball Median: 0.48815
Std Pinball Median: 0.0457
Mean Pinball 10th percentile: 0.25394
Std Pinball 10th percentile: 0.0355
Mean Pinball 90th percentile: 0.66976
Std Pinball 90th percentile: 0.04753
Pinball Medians: [0.43918305597579427, 0.5337353938937053, 0.5266465560583208, 0.4669620253164557, 0.4107053526763382, 0.5297848869277441, 0.5100469483568075]
Pinball 10th percentiles: [0.23524962178517397, 0.30782761653474056, 0.298139768728004, 0.24430379746835443, 0.204352176088044, 0.2646166574738003, 0.2230985915492958]
Pinball 90th percentiles: [0.5939485627836611, 0.6823721573061943, 0.6945701357466063, 0.6624050632911392, 0.6105552776388194, 0.7329012686155544, 0.7115492957746479]
Pinball Ranges: [0.35869894 0.37454454 0.39643037 0.41810127 0.4062031  0.46828461
 0.4884507 ]
Mean Ranges: 0.4158162187898871
Std Ranges: 0.04386322933052418
---------All-----------

[INFO][2023-07-06 10:54:06] : Trained model for 17 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012658686559484413, pinball_test_loss=0.008543226636660023, pinball_train_10_percentile_loss=0.001690819139025288, pinball_test_10_percentile_loss=0.004785913458655618, pinball_train_90_percentile_loss=0.0008321267661905012, pinball_test_90_percentile_loss=0.005381196188324619, mae_train_loss=0.0025317373118968825, mae_test_loss=0.017086453273320045, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
18

---------All-----------
Median MAE: 0.01731
Mean MAE: 0.01768
Std MAE: 0.00121
Mean Pinball Median: 0.48961
Std Pinball Median: 0.04782
Mean Pinball 10th percentile: 0.25636
Std Pinball 10th percentile: 0.03673
Mean Pinball 90th percentile: 0.67008
Std Pinball 90th percentile: 0.05013
Pinball Medians: [0.4328087167070218, 0.5339866817439377, 0.5325540472599296, 0.4711392405063291, 0.4103301650825413, 0.5296388199614006, 0.5168075117370892]
Pinball 10th percentiles: [0.22790556900726391, 0.3119738660635758, 0.30417295123177474, 0.24772151898734177, 0.2061030515257629, 0.264543700027571, 0.23211267605633804]
Pinball 90th percentiles: [0.5838377723970944, 0.6807387862796834, 0.7021116138763197, 0.6643037974683544, 0.6141820910455228, 0.7353184449958643, 0.7100469483568075]
Pinball Ranges: [0.3559322  0.36876492 0.39793866 0.41658228 0.40807904 0.47077474
 0.47793427]
Mean Ranges: 0.41371516021714555
Std Ranges: 0.0431412445080359
---------All-----------

[INFO][2023-07-06 11:23:43] : Trained model for 18 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012919508728866224, pinball_test_loss=0.008656735151307405, pinball_train_10_percentile_loss=0.0016863978077136306, pinball_test_10_percentile_loss=0.004791495154049992, pinball_train_90_percentile_loss=0.000842435028896126, pinball_test_90_percentile_loss=0.00536829961327684, mae_train_loss=0.002583901745773245, mae_test_loss=0.01731347030261481, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
19

---------All-----------
Median MAE: 0.01743
Mean MAE: 0.01773
Std MAE: 0.00122
Mean Pinball Median: 0.48977
Std Pinball Median: 0.05046
Mean Pinball 10th percentile: 0.25522
Std Pinball 10th percentile: 0.03748
Mean Pinball 90th percentile: 0.66885
Std Pinball 90th percentile: 0.05256
Pinball Medians: [0.4288525582803512, 0.5382585751978892, 0.5360734037204625, 0.4687341772151899, 0.40857928964482243, 0.5350055126791621, 0.5128638497652582]
Pinball 10th percentiles: [0.22721768089615502, 0.3162457595175273, 0.2980140774258421, 0.25037974683544306, 0.20147573786893447, 0.2613009922822492, 0.231924882629108]
Pinball 90th percentiles: [0.577051165607024, 0.6853876115089835, 0.6998491704374057, 0.6608860759493671, 0.6118059029514757, 0.7374586549062845, 0.7094835680751174]
Pinball Ranges: [0.34983348 0.36914185 0.40183509 0.41050633 0.41033017 0.47615766
 0.47755869]
Mean Ranges: 0.4136233245686284
Std Ranges: 0.04504845302809219
---------All-----------

[INFO][2023-07-06 11:52:42] : Trained model for 19 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.00127526215181906, pinball_test_loss=0.008715613899405063, pinball_train_10_percentile_loss=0.0016778850467503283, pinball_test_10_percentile_loss=0.0048602709625879415, pinball_train_90_percentile_loss=0.0008371653829621742, pinball_test_90_percentile_loss=0.005415860572442875, mae_train_loss=0.00255052430363812, mae_test_loss=0.017431227798810126, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
20

---------All-----------
Median MAE: 0.01729
Mean MAE: 0.01772
Std MAE: 0.00128
Mean Pinball Median: 0.48906
Std Pinball Median: 0.05052
Mean Pinball 10th percentile: 0.25505
Std Pinball 10th percentile: 0.03659
Mean Pinball 90th percentile: 0.66857
Std Pinball 90th percentile: 0.0526
Pinball Medians: [0.4239854633555421, 0.5337353938937053, 0.5356963298139769, 0.47037974683544304, 0.4095797898949475, 0.5351336456324056, 0.5149295774647887]
Pinball 10th percentiles: [0.22244094488188976, 0.30895841186078654, 0.2997737556561086, 0.24936708860759493, 0.19984992496248125, 0.2649490217690824, 0.24]
Pinball 90th percentiles: [0.5760145366444579, 0.6812413619801483, 0.7009803921568627, 0.6616455696202531, 0.6119309654827414, 0.7332598511986773, 0.7149295774647887]
Pinball Ranges: [0.35357359 0.37228295 0.40120664 0.41227848 0.41208104 0.46831083
 0.47492958]
Mean Ranges: 0.41352330097285517
Std Ranges: 0.041785965100367306
---------All-----------

21
[INFO][2023-07-07 12:21:26] : Trained model for 20 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012847043268713798, pinball_test_loss=0.008644220878087975, pinball_train_10_percentile_loss=0.0016892926949156211, pinball_test_10_percentile_loss=0.0048805092640344045, pinball_train_90_percentile_loss=0.0008416362061357607, pinball_test_90_percentile_loss=0.00531101084314585, mae_train_loss=0.0025694086537427596, mae_test_loss=0.01728844175617595, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01746
Mean MAE: 0.01782
Std MAE: 0.00135
Mean Pinball Median: 0.49072
Std Pinball Median: 0.05139
Mean Pinball 10th percentile: 0.25536
Std Pinball 10th percentile: 0.03604
Mean Pinball 90th percentile: 0.671
Std Pinball 90th percentile: 0.05401
Pinball Medians: [0.4150257497727961, 0.5346149013695188, 0.5395927601809954, 0.4701341432548722, 0.4193346673336668, 0.5359504132231405, 0.5203755868544601]
Pinball 10th percentiles: [0.21705543774613753, 0.3095866314863676, 0.30153343388637505, 0.24791192103264997, 0.2073536768384192, 0.2612947658402204, 0.2428169014084507]
Pinball 90th percentiles: [0.5769463798848834, 0.6812413619801483, 0.7044997486173957, 0.6650215135408757, 0.6126813406703352, 0.7414600550964188, 0.7151173708920188]
Pinball Ranges: [0.35989094 0.37165473 0.40296631 0.41710959 0.40532766 0.48016529
 0.47230047]
Mean Ranges: 0.4156307146347794
Std Ranges: 0.04257046060994731
---------All-----------

22
[INFO][2023-07-07 12:48:58] : Trained model for 21 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012986811551767075, pinball_test_loss=0.008730679895254008, pinball_train_10_percentile_loss=0.0016905778176565204, pinball_test_10_percentile_loss=0.00490367850380755, pinball_train_90_percentile_loss=0.0008539198678374681, pinball_test_90_percentile_loss=0.00524868195926644, mae_train_loss=0.002597362310353415, mae_test_loss=0.017461359790508016, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01769
Mean MAE: 0.01787
Std MAE: 0.00143
Mean Pinball Median: 0.48986
Std Pinball Median: 0.05083
Mean Pinball 10th percentile: 0.25627
Std Pinball 10th percentile: 0.03406
Mean Pinball 90th percentile: 0.66942
Std Pinball 90th percentile: 0.05467
Pinball Medians: [0.4116666666666667, 0.5287096368890564, 0.5385872297637003, 0.47393724696356276, 0.4200850425212606, 0.5331864500137703, 0.5228169014084507]
Pinball 10th percentiles: [0.22075757575757576, 0.3082045483100892, 0.30027652086475615, 0.24949392712550608, 0.21048024012006003, 0.2573671164968328, 0.24732394366197183]
Pinball 90th percentiles: [0.5677272727272727, 0.6802362105792185, 0.7062594268476622, 0.6651062753036437, 0.6174337168584292, 0.7368493527953732, 0.7123004694835681]
Pinball Ranges: [0.3469697  0.37203166 0.40598291 0.41561235 0.40695348 0.47948224
 0.46497653]
Mean Ranges: 0.4131441217511965
Std Ranges: 0.04351080590232759
---------All-----------

[INFO][2023-07-07 01:16:28] : Trained model for 22 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012987797890023941, pinball_test_loss=0.008845683290537575, pinball_train_10_percentile_loss=0.0016895254192384945, pinball_test_10_percentile_loss=0.0049633475517156765, pinball_train_90_percentile_loss=0.0008495955172837783, pinball_test_90_percentile_loss=0.0053355587499376465, mae_train_loss=0.0025975595780047883, mae_test_loss=0.01769136658107515, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
23

---------All-----------
Median MAE: 0.01741
Mean MAE: 0.01792
Std MAE: 0.00151
Mean Pinball Median: 0.49085
Std Pinball Median: 0.05234
Mean Pinball 10th percentile: 0.25585
Std Pinball 10th percentile: 0.03326
Mean Pinball 90th percentile: 0.67046
Std Pinball 90th percentile: 0.05437
Pinball Medians: [0.4127008184298272, 0.5290865686644051, 0.542357968828557, 0.47438330170777987, 0.4169584792396198, 0.5379955947136564, 0.5224413145539906]
Pinball 10th percentiles: [0.22476508032737194, 0.30355572308078904, 0.3011563599798894, 0.2480708412397217, 0.20872936468234118, 0.26073788546255505, 0.24394366197183098]
Pinball 90th percentiles: [0.5691118520763868, 0.6762156049754995, 0.7012317747611865, 0.6659076533839342, 0.6218109054527263, 0.7359581497797357, 0.7230046948356808]
Pinball Ranges: [0.34434677 0.37265988 0.40007541 0.41783681 0.41308154 0.47522026
 0.47906103]
Mean Ranges: 0.41461167407437866
Std Ranges: 0.04590094316367743
---------All-----------

24
[INFO][2023-07-07 01:44:10] : Trained model for 23 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.001303632885697362, pinball_test_loss=0.008702512974211271, pinball_train_10_percentile_loss=0.0017075155605252013, pinball_test_10_percentile_loss=0.004946837215621098, pinball_train_90_percentile_loss=0.0008531797831767166, pinball_test_90_percentile_loss=0.005193451508992686, mae_train_loss=0.002607265771394724, mae_test_loss=0.017405025948422543, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01768
Mean MAE: 0.01808
Std MAE: 0.00155
Mean Pinball Median: 0.4914
Std Pinball Median: 0.05257
Mean Pinball 10th percentile: 0.25764
Std Pinball 10th percentile: 0.03301
Mean Pinball 90th percentile: 0.67382
Std Pinball 90th percentile: 0.05624
Pinball Medians: [0.40888417222559126, 0.5229300163337103, 0.5456259426847662, 0.4759074238016947, 0.4210855427713857, 0.5348197082301128, 0.5305164319248826]
Pinball 10th percentiles: [0.22377198302001214, 0.3028018595300917, 0.30165912518853694, 0.24939926647274566, 0.2088544272136068, 0.2636939168731076, 0.25333333333333335]
Pinball 90th percentiles: [0.5708004851425106, 0.6802362105792185, 0.705379587732529, 0.6699127355507778, 0.6198099049524762, 0.7347921827690613, 0.7357746478873239]
Pinball Ranges: [0.3470285  0.37743435 0.40372046 0.42051347 0.41095548 0.47109827
 0.48244131]
Mean Ranges: 0.41617026328320905
Std Ranges: 0.044547000386217055
---------All-----------

25
[INFO][2023-07-07 02:11:45] : Trained model for 24 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0013182253728295837, pinball_test_loss=0.008839582194506672, pinball_train_10_percentile_loss=0.0017034921036401117, pinball_test_10_percentile_loss=0.0050669811571025115, pinball_train_90_percentile_loss=0.0008656499083700535, pinball_test_90_percentile_loss=0.005267883046415519, mae_train_loss=0.0026364507456591675, mae_test_loss=0.017679164389013343, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01797
Mean MAE: 0.01827
Std MAE: 0.0016
Mean Pinball Median: 0.49036
Std Pinball Median: 0.05222
Mean Pinball 10th percentile: 0.25694
Std Pinball 10th percentile: 0.03419
Mean Pinball 90th percentile: 0.66981
Std Pinball 90th percentile: 0.05769
Pinball Medians: [0.4105247194419169, 0.5259454705364995, 0.5405982905982906, 0.4775572133013023, 0.4165832916458229, 0.5245630934360809, 0.5367136150234741]
Pinball 10th percentiles: [0.22474977252047315, 0.30858148008543784, 0.29876822523881347, 0.24971551397142497, 0.20560280140070036, 0.26007981285262144, 0.25107981220657277]
Pinball 90th percentiles: [0.5585380649074917, 0.6772207563764292, 0.7023629964806435, 0.6721456568466304, 0.6189344672336168, 0.7334525939177102, 0.7260093896713615]
Pinball Ranges: [0.33378829 0.36863928 0.40359477 0.42243014 0.41333167 0.47337278
 0.47492958]
Mean Ranges: 0.4128695010225485
Std Ranges: 0.047659342367675384
---------All-----------

26
[INFO][2023-07-07 02:39:20] : Trained model for 25 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0013030482581678085, pinball_test_loss=0.008985214631422224, pinball_train_10_percentile_loss=0.0017247333785609111, pinball_test_10_percentile_loss=0.005140689048928028, pinball_train_90_percentile_loss=0.0008583387101152229, pinball_test_90_percentile_loss=0.005343117188407871, mae_train_loss=0.002606096516335617, mae_test_loss=0.017970429262844447, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01784
Mean MAE: 0.0185
Std MAE: 0.00164
Mean Pinball Median: 0.49247
Std Pinball Median: 0.05261
Mean Pinball 10th percentile: 0.25876
Std Pinball 10th percentile: 0.03352
Mean Pinball 90th percentile: 0.67047
Std Pinball 90th percentile: 0.05773
Pinball Medians: [0.41535194174757284, 0.5290865686644051, 0.5335595776772247, 0.48047023132347366, 0.41420710355177587, 0.529792211366451, 0.5447887323943662]
Pinball 10th percentiles: [0.22800364077669902, 0.30732504083427564, 0.2996480643539467, 0.25774238402224753, 0.20522761380690346, 0.2592541626530893, 0.2540845070422535]
Pinball 90th percentiles: [0.5597694174757282, 0.6782259077773589, 0.7008547008547008, 0.6785488560232588, 0.6160580290145072, 0.7309756433191138, 0.7288262910798122]
Pinball Ranges: [0.33176578 0.37090087 0.40120664 0.42080647 0.41083042 0.47172148
 0.47474178]
Mean Ranges: 0.41171049029358064
Std Ranges: 0.04754650040618202
---------All-----------

[INFO][2023-07-07 03:06:57] : Trained model for 26 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012980675723370477, pinball_test_loss=0.008920236421295797, pinball_train_10_percentile_loss=0.0017618581959236077, pinball_test_10_percentile_loss=0.005150908260646434, pinball_train_90_percentile_loss=0.0008644861619017658, pinball_test_90_percentile_loss=0.005291997578052626, mae_train_loss=0.0025961351446740955, mae_test_loss=0.017840472842591593, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
27

---------All-----------
Median MAE: 0.01819
Mean MAE: 0.01869
Std MAE: 0.00159
Mean Pinball Median: 0.49424
Std Pinball Median: 0.05456
Mean Pinball 10th percentile: 0.26142
Std Pinball 10th percentile: 0.03462
Mean Pinball 90th percentile: 0.67243
Std Pinball 90th percentile: 0.05793
Pinball Medians: [0.4139605462822458, 0.5298404322151024, 0.5353192559074912, 0.48351231838281744, 0.4129564782391196, 0.5335076372643457, 0.5506103286384977]
Pinball 10th percentiles: [0.2267071320182094, 0.31285337353938936, 0.3017848164906988, 0.25647504737839544, 0.20835417708854428, 0.267785881381588, 0.255962441314554]
Pinball 90th percentiles: [0.5625189681335356, 0.6798592788038699, 0.6997234791352438, 0.6778269109286166, 0.6184342171085543, 0.7349662859501858, 0.7337089201877934]
Pinball Ranges: [0.33581184 0.36700591 0.39793866 0.42135186 0.41008004 0.4671804
 0.47774648]
Mean Ranges: 0.4110164558623457
Std Ranges: 0.04702381570269305
---------All-----------

28
[INFO][2023-07-07 03:34:38] : Trained model for 27 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0013076377145214697, pinball_test_loss=0.009097436120233832, pinball_train_10_percentile_loss=0.001765176031112239, pinball_test_10_percentile_loss=0.005153501837098349, pinball_train_90_percentile_loss=0.0008735743636610195, pinball_test_90_percentile_loss=0.005362809661932802, mae_train_loss=0.0026152754290429394, mae_test_loss=0.018194872240467664, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01843
Mean MAE: 0.0188
Std MAE: 0.00154
Mean Pinball Median: 0.4941
Std Pinball Median: 0.05269
Mean Pinball 10th percentile: 0.25904
Std Pinball 10th percentile: 0.03575
Mean Pinball 90th percentile: 0.6705
Std Pinball 90th percentile: 0.05801
Pinball Medians: [0.4146304446805282, 0.5321020228671943, 0.5340623428858723, 0.48099027409372236, 0.4189594797398699, 0.5282785193339755, 0.5496713615023474]
Pinball 10th percentiles: [0.22082258309303385, 0.31561753989194624, 0.29386626445449976, 0.25906277630415564, 0.2039769884942471, 0.2643456722168708, 0.2555868544600939]
Pinball 90th percentiles: [0.5604795871907725, 0.681115718055032, 0.697209653092006, 0.6742452949349501, 0.6159329664832416, 0.734003027384065, 0.7305164319248826]
Pinball Ranges: [0.339657   0.36549818 0.40334339 0.41518252 0.41195598 0.46965736
 0.47492958]
Mean Ranges: 0.4114605714500147
Std Ranges: 0.0459245673686284
---------All-----------

29
[INFO][2023-07-07 04:02:14] : Trained model for 28 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012954575882129715, pinball_test_loss=0.009215504415729813, pinball_train_10_percentile_loss=0.001764406335686039, pinball_test_10_percentile_loss=0.005256930262445614, pinball_train_90_percentile_loss=0.0008659928656951278, pinball_test_90_percentile_loss=0.005541367442411047, mae_train_loss=0.002590915176425943, mae_test_loss=0.018431008831459626, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.0184
Mean MAE: 0.01885
Std MAE: 0.0015
Mean Pinball Median: 0.49409
Std Pinball Median: 0.05571
Mean Pinball 10th percentile: 0.26
Std Pinball 10th percentile: 0.0341
Mean Pinball 90th percentile: 0.67048
Std Pinball 90th percentile: 0.05798
Pinball Medians: [0.4159963575656397, 0.5307199396909159, 0.5365761689291101, 0.48200530369996214, 0.4080790395197599, 0.53199394523187, 0.5532394366197183]
Pinball 10th percentiles: [0.2270450751252087, 0.3075763286845081, 0.2986425339366516, 0.25823967672685944, 0.20285142571285641, 0.2651713224164029, 0.26046948356807514]
Pinball 90th percentiles: [0.5630596448626499, 0.6796079909536374, 0.7007290095525389, 0.6709180452077282, 0.6136818409204602, 0.7286363010871061, 0.7367136150234742]
Pinball Ranges: [0.33601457 0.37203166 0.40208648 0.41267837 0.41083042 0.46346498
 0.47624413]
Mean Ranges: 0.4104786573481475
Std Ranges: 0.04508897676980494
---------All-----------

[INFO][2023-07-07 04:29:56] : Trained model for 29 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012895139433024073, pinball_test_loss=0.009199003116964324, pinball_train_10_percentile_loss=0.0017906353245660902, pinball_test_10_percentile_loss=0.005320346273509434, pinball_train_90_percentile_loss=0.0008597386619097377, pinball_test_90_percentile_loss=0.005524520070077412, mae_train_loss=0.0025790278866048145, mae_test_loss=0.01839800623392865, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
30

---------All-----------
Median MAE: 0.01866
Mean MAE: 0.01892
Std MAE: 0.00149
Mean Pinball Median: 0.49636
Std Pinball Median: 0.05573
Mean Pinball 10th percentile: 0.25909
Std Pinball 10th percentile: 0.03424
Mean Pinball 90th percentile: 0.67121
Std Pinball 90th percentile: 0.05629
Pinball Medians: [0.41554105327060253, 0.5338610378188214, 0.5346907993966817, 0.48516226796312667, 0.412456228114057, 0.5374982798954177, 0.5553051643192488]
Pinball 10th percentiles: [0.22249203217483685, 0.30895841186078654, 0.2961287078934138, 0.2564717767394873, 0.2046023011505753, 0.2644832805834595, 0.26046948356807514]
Pinball 90th percentiles: [0.5641220215510699, 0.6789797713280563, 0.6997234791352438, 0.6745801237529991, 0.6193096548274137, 0.7327645520847668, 0.7290140845070423]
Pinball Ranges: [0.34162999 0.37002136 0.40359477 0.41810835 0.41470735 0.46828127
 0.4685446 ]
Mean Ranges: 0.41212681331656537
Std Ranges: 0.04341526235856544
---------All-----------

31
[INFO][2023-07-07 04:57:37] : Trained model for 30 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012955736178779339, pinball_test_loss=0.009328795956185111, pinball_train_10_percentile_loss=0.001788058576759438, pinball_test_10_percentile_loss=0.0054302836770548, pinball_train_90_percentile_loss=0.0008646832482125305, pinball_test_90_percentile_loss=0.005537810198316984, mae_train_loss=0.0025911472357558678, mae_test_loss=0.018657591912370222, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01878
Mean MAE: 0.01904
Std MAE: 0.00152
Mean Pinball Median: 0.49471
Std Pinball Median: 0.05561
Mean Pinball 10th percentile: 0.26019
Std Pinball 10th percentile: 0.03249
Mean Pinball 90th percentile: 0.67071
Std Pinball 90th percentile: 0.05912
Pinball Medians: [0.41402337228714525, 0.532478954642543, 0.5345651080945198, 0.48591993938628614, 0.4094547273636818, 0.537773496628595, 0.5487323943661971]
Pinball 10th percentiles: [0.23068750948550615, 0.31021485111194874, 0.29374057315233787, 0.2571031695921202, 0.2066033016508254, 0.26310719691757256, 0.25990610328638497]
Pinball 90th percentiles: [0.5606313552891182, 0.6775976881517779, 0.6984665661136249, 0.6766005808814244, 0.6129314657328664, 0.7395073620476125, 0.7292018779342723]
Pinball Ranges: [0.32994385 0.36738284 0.40472599 0.41949741 0.40632816 0.47640017
 0.46929577]
Mean Ranges: 0.4105105987077144
Std Ranges: 0.04822544467253771
---------All-----------

32
[INFO][2023-07-07 05:25:14] : Trained model for 31 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.001290009389558908, pinball_test_loss=0.009390295423821181, pinball_train_10_percentile_loss=0.0017811929460957217, pinball_test_10_percentile_loss=0.0055581915637342785, pinball_train_90_percentile_loss=0.0008593586014283763, pinball_test_90_percentile_loss=0.00561921732520536, mae_train_loss=0.002580018779117816, mae_test_loss=0.018780590847642362, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.0186
Mean MAE: 0.01906
Std MAE: 0.00152
Mean Pinball Median: 0.49492
Std Pinball Median: 0.05605
Mean Pinball 10th percentile: 0.25914
Std Pinball 10th percentile: 0.03255
Mean Pinball 90th percentile: 0.67213
Std Pinball 90th percentile: 0.05893
Pinball Medians: [0.4159963575656397, 0.5326045985676593, 0.5301659125188537, 0.4827020202020202, 0.40857928964482243, 0.5434154396587313, 0.5509859154929577]
Pinball 10th percentiles: [0.23174988617392625, 0.30719939690915943, 0.29499748617395677, 0.2556818181818182, 0.2046023011505753, 0.2664097977157011, 0.25333333333333335]
Pinball 90th percentiles: [0.5627561086659584, 0.6775976881517779, 0.7001005530417295, 0.6797979797979798, 0.6135567783891946, 0.7360671528828953, 0.7350234741784037]
Pinball Ranges: [0.33100622 0.37039829 0.40510307 0.42411616 0.40895448 0.46965736
 0.48169014]
Mean Ranges: 0.4129893879242098
Std Ranges: 0.048797793496666804
---------All-----------

[INFO][2023-07-07 05:52:51] : Trained model for 32 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012960914146342636, pinball_test_loss=0.009299732322450785, pinball_train_10_percentile_loss=0.0018057087757206067, pinball_test_10_percentile_loss=0.005556968719797479, pinball_train_90_percentile_loss=0.0008630667659619875, pinball_test_90_percentile_loss=0.00555033343373616, mae_train_loss=0.0025921828292685273, mae_test_loss=0.01859946464490157, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))
33

---------All-----------
Median MAE: 0.01868
Mean MAE: 0.01914
Std MAE: 0.00147
Mean Pinball Median: 0.49671
Std Pinball Median: 0.05544
Mean Pinball 10th percentile: 0.26286
Std Pinball 10th percentile: 0.03257
Mean Pinball 90th percentile: 0.6728
Std Pinball 90th percentile: 0.0591
Pinball Medians: [0.41387160418879954, 0.5333584621183566, 0.5311714429361488, 0.4872442535993938, 0.41483241620810407, 0.5449291316912068, 0.5515492957746478]
Pinball 10th percentiles: [0.23129458187888904, 0.30795326045985677, 0.29788838612368024, 0.25915635261429654, 0.20785392696348173, 0.27411586624466766, 0.26178403755868546]
Pinball 90th percentiles: [0.5609348914858097, 0.6814926498303807, 0.6968325791855203, 0.6818641070977519, 0.6163081540770385, 0.7346910692170084, 0.7374647887323944]
Pinball Ranges: [0.32964031 0.37353939 0.39894419 0.42270775 0.40845423 0.4605752
 0.47568075]
Mean Ranges: 0.4099345468260495
Std Ranges: 0.04620597598029042
---------All-----------

34
[INFO][2023-07-07 06:20:53] : Trained model for 33 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0013090735832662125, pinball_test_loss=0.00934038376072662, pinball_train_10_percentile_loss=0.0018012606971985, pinball_test_10_percentile_loss=0.00564998335007959, pinball_train_90_percentile_loss=0.0008672189146408471, pinball_test_90_percentile_loss=0.005519624378418749, mae_train_loss=0.002618147166532425, mae_test_loss=0.01868076752145324, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01889
Mean MAE: 0.01924
Std MAE: 0.00155
Mean Pinball Median: 0.49789
Std Pinball Median: 0.05794
Mean Pinball 10th percentile: 0.26168
Std Pinball 10th percentile: 0.03415
Mean Pinball 90th percentile: 0.67299
Std Pinball 90th percentile: 0.06185
Pinball Medians: [0.4071320182094082, 0.5374371859296483, 0.529907011812013, 0.4903372489579386, 0.4161560585219457, 0.5468685478320716, 0.5573924478677438]
Pinball 10th percentiles: [0.23110773899848255, 0.31293969849246234, 0.29404372958029656, 0.25982063913098397, 0.20320120045016882, 0.27253957329662765, 0.25812511741499156]
Pinball 90th percentiles: [0.5541729893778452, 0.6851758793969849, 0.6985423473234481, 0.6792977137804724, 0.616481180442666, 0.7408121128699243, 0.7364268269772685]
Pinball Ranges: [0.32306525 0.37223618 0.40449862 0.41947707 0.41327998 0.46827254
 0.47830171]
Mean Ranges: 0.4113044789720851
Std Ranges: 0.049516946425886436
---------All-----------

35
[INFO][2023-07-07 06:48:32] : Trained model for 34 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.001294774137447554, pinball_test_loss=0.009443563615123555, pinball_train_10_percentile_loss=0.0017957128932384813, pinball_test_10_percentile_loss=0.005684710938359013, pinball_train_90_percentile_loss=0.0008671634992713917, pinball_test_90_percentile_loss=0.005652420971268249, mae_train_loss=0.002589548274895108, mae_test_loss=0.01888712723024711, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01902
Mean MAE: 0.01924
Std MAE: 0.00153
Mean Pinball Median: 0.49521
Std Pinball Median: 0.05804
Mean Pinball 10th percentile: 0.26133
Std Pinball 10th percentile: 0.03275
Mean Pinball 90th percentile: 0.67172
Std Pinball 90th percentile: 0.06206
Pinball Medians: [0.4065533980582524, 0.5353015075376885, 0.5308455836160322, 0.48477574226152875, 0.4115293234963111, 0.5466060856395429, 0.5508363089644803]
Pinball 10th percentiles: [0.2287621359223301, 0.3051507537688442, 0.2960170875738158, 0.26064434617814275, 0.20582718519444793, 0.2761944100234063, 0.25671866190565684]
Pinball 90th percentiles: [0.5508191747572816, 0.6811557788944723, 0.6993340871968841, 0.680101073910297, 0.6173565086907591, 0.7408784248932948, 0.7323811313662846]
Pinball Ranges: [0.32205704 0.37600503 0.403317   0.41945673 0.41152932 0.46468401
 0.47566247]
Mean Ranges: 0.41038737130608993
Std Ranges: 0.04829508218270192
---------All-----------

36
[INFO][2023-07-07 07:16:48] : Trained model for 35 hour forecast. training_results=ExperimentSummary(pinball_train_loss=0.0012997304674958064, pinball_test_loss=0.00950992328807736, pinball_train_10_percentile_loss=0.001805503431029501, pinball_test_10_percentile_loss=0.005724750026593408, pinball_train_90_percentile_loss=0.0008668660562161332, pinball_test_90_percentile_loss=0.0057226792431076005, mae_train_loss=0.002599460934991613, mae_test_loss=0.01901984657615472, model=XGBRegressor(base_score=None, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.85,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.005, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=None, max_delta_step=None,
             max_depth=80, max_leaves=0, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=1250,
             n_jobs=-1, num_parallel_tree=1, objective='reg:quantileerror', ...))

---------All-----------
Median MAE: 0.01906
Mean MAE: 0.01927
Std MAE: 0.00153
Mean Pinball Median: 0.49714
Std Pinball Median: 0.05919
Mean Pinball 10th percentile: 0.26343
Std Pinball 10th percentile: 0.0324
Mean Pinball 90th percentile: 0.67256
Std Pinball 90th percentile: 0.06084
Pinball Medians: [0.40279041552926903, 0.5341708542713568, 0.5307199396909159, 0.4893213698976368, 0.41553082405902214, 0.5528164164715604, 0.5546155292348186]
Pinball 10th percentiles: [0.22596299666363362, 0.30263819095477384, 0.3004146249528835, 0.26260583849361807, 0.21007877954232837, 0.2772345406968737, 0.2650874224478285]
Pinball 90th percentiles: [0.5573248407643312, 0.6804020100502512, 0.7008418142982786, 0.6786301023631998, 0.615605852194573, 0.741908827985126, 0.7332205301748449]
Pinball Ranges: [0.33136184 0.37776382 0.40042719 0.41602426 0.40552707 0.46467429
 0.46813311]
Mean Ranges: 0.40913022629695217
Std Ranges: 0.04429528968996802
---------All-----------

```
