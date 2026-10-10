# Results (results/main)

## Camelyon17 / EffNet-B0 (camelyon17_seed_1)
- greedy ref in-sample k=2 0.9619 (full 0.8759); conf best 0.9520
- cross-fitted greedy k=[2, 2] score 0.9612 (full halves 0.8758); gain (0.08535438776016235, 0.08313142508268356, 0.08735761046409607); greedy-topk (0.0004668235778808594, -0.0003089725796598941, 0.0011736275628209114)
- search: k=2: greedy 0.9619 best 0.9619 (rand 0.9618873635346372, swap 0.9619, back 0.960630133948188); k=5: greedy 0.9615 best 0.9618 (rand 0.9615067625847836, swap 0.9618, back 0.9618088430597088); k=10: greedy 0.9601 best 0.9601 (rand 0.9600841800141605, swap 0.9601, back 0.9600841800141605); k=20: greedy 0.9528 best 0.9528 (rand 0.9527938013616115, swap 0.9528, back 0.9526703912343663); k=40: greedy 0.9409 best 0.9409 (rand 0.9409037091810913, swap 0.9409, back 0.9409037091810913)
- gain cf +0.0758 (k [2, 2]), best member share cf 0.7562902569770813, certificate B(2) = 0.1602; B by k {'1': 0.0814071074128151, '2': 0.1601923406124115, '5': 0.1282283365726471, '10': 0.14667387306690216, '20': 0.13181743025779724, '40': 0.10993075370788574}; mean D on H 1.0
- soft cf: vote 0.9612 k [2, 2] full 0.8794; wvote 0.9612 k [2, 2] full 0.8780; avg 0.9612 k [2, 2] full 0.8758
- P7 oracle cf: k [5, 3] 0.9606
- gain in-sample k=2 +0.0765 bound 0.1566 (ties as errors 0.0783) headroom 0.1070 minority-right 0.1047 best member 0.7563
- q bar 1.0000 q tilde 1.0000 corrected 1.0 ece 0.08167824746815258
- mu<0 split 0.06198429688811302 plain 0.0625; err align wrong 0.7535851155909229
- decider held-out greedy k=5 eps 0.0158 -> 0.9486 (full 0.8759); diff (0.07277309894561768, 0.07078424841165543, 0.07465410232543945); uniform prior held-out greedy k=5 0.9486; conservative 0.9486232697356612
- transfer decider {'reference': 'full pool', 'dis_V': 0.029910610988736153, 'a_V': -0.5977011530042371, 'dis_T': 0.07257741689682007, 'a_T': -0.9222420373063787, 'gamma': 0.32454088430214156, 'regret': -0.06693394482135773}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.029910610988736153, 'a_V': -0.5977011530042371, 'dis_T': 0.07257741689682007, 'a_T': -0.9222420373063787, 'gamma': 0.32454088430214156, 'regret': -0.06693394482135773} test 0.9486
  - ablation P1-P7: held-out greedy k=5 0.9486
  - ablation P1-P8: held-out greedy k=5 0.9486
  - ablation P1-P7,P9-P10: held-out greedy k=5 0.9486
  - ablation P1-P10: held-out greedy k=5 0.9486
  - ablation P1-P12: held-out greedy k=5 0.9486
  - ablation P9-P12: held-out greedy k=5 0.9486
  - ablation P11-P12: held-out greedy k=5 0.9486
  - ablation P7,P11-P12: held-out greedy k=5 0.9486
  - ablation P9-P12 no search: P10 consensus geo alpha low k=10 0.9287
  - ablation P11-P12 no search: P11 signals mean alpha low k=42 0.9291
  - ablation P7,P11-P12 no search: P11 signals mean alpha low k=42 0.9291
  - ablation menu, held-out greedy only: held-out greedy k=5 0.9486
  - ablation menu, held-out top-k only: held-out top-k k=4 0.9329
  - ablation full pool vs held-out greedy: held-out greedy k=5 0.9486
  - budget 5: held-out greedy k=5 0.9486
  - budget 10: held-out greedy k=5 0.9486
  - budget 20: held-out greedy k=5 0.9486
  - baseline best member on V: k=1  0.9504 diff (0.07450687885284424, 0.07233042269945145, 0.07650158554315567)
  - baseline top-k on V: k=4  0.9329 diff (0.05703139305114746, 0.055296607315540314, 0.05877051502466202)
  - baseline greedy on V: k=5  0.9486 diff (0.07277309894561768, 0.07078424841165543, 0.07465410232543945)
  - baseline Caruana (bagged): k=46  0.9336 diff (0.057745933532714844, 0.055955301970243454, 0.05940934643149376)
  - baseline log-odds weighted vote: k=100  0.8951 diff (0.019205987453460693, 0.018141595646739006, 0.020128898322582245)
  - baseline linear-pool stacking: k=25  0.9388 diff (0.06298327445983887, 0.0611339807510376, 0.06466317176818848)
  - baseline proposed paths by balanced accuracy: k=42 P11 signals mean alpha low 0.9291 diff (0.05322545766830444, 0.05152742564678192, 0.05476350337266922)
  - baseline proposed paths by NLL: k=19 P11 signals mean alpha low 0.9312 diff (0.05532616376876831, 0.05354505032300949, 0.056927818804979324)
  - baseline proposed paths by BRIER: k=42 P12 signals geo alpha low 0.9305 diff (0.05460953712463379, 0.05280708894133568, 0.05619918555021286)
- inductive: held-out greedy k=5 0.9480; transductive 0.9480; full 0.8751 (eval half, one domain, random halves)
- vote decider held-out greedy k=6 0.9418 (full 0.8794)

## Camelyon17 / ConvNeXt-A (camelyon17_convnext_atto_seed_2)
- greedy ref in-sample k=5 0.9675 (full 0.8170); conf best 0.9494
- cross-fitted greedy k=[5, 5] score 0.9675 (full halves 0.8170); gain (0.1505386233329773, 0.14781443774700165, 0.15312688052654266); greedy-topk (0.001618027687072754, 0.0009960264433175325, 0.0022313788067549467)
- search: k=5: greedy 0.9675 best 0.9675 (rand 0.9675417609092429, swap 0.9675, back 0.9675417609092429); k=10: greedy 0.9658 best 0.9659 (rand 0.9658094227399407, swap 0.9659, back 0.9657894335309744); k=20: greedy 0.9627 best 0.9627 (rand 0.9626953712126687, swap 0.9627, back 0.9626953712126687); k=40: greedy 0.9539 best 0.9540 (rand 0.9538728306503936, swap 0.9540, back 0.9538955026821766)
- gain cf +0.1405 (k [4, 6]), best member share cf 0.8840790688991547, certificate B(6) = 0.2896; B by k {'1': 0.14776495099067688, '2': 0.294095516204834, '5': 0.2421833872795105, '10': 0.2844216823577881, '20': 0.2747631072998047, '40': 0.2446945607662201}; mean D on H 1.0
- soft cf: vote 0.9667 k [4, 6] full 0.8101; wvote 0.9669 k [5, 6] full 0.8123; avg 0.9675 k [5, 5] full 0.8170
- P7 oracle cf: k [3, 5] 0.9662
- gain in-sample k=6 +0.1408 bound 0.2854 (ties as errors 0.2141) headroom 0.1671 minority-right 0.1661 best member 0.8841
- q bar 1.0000 q tilde 1.0000 corrected 1.0 ece 0.1350522667722491
- mu<0 split 0.08009110763669014 plain 0.0807; err align wrong 0.6712051471544679
- decider held-out greedy k=5 eps 0.0158 -> 0.9616 (full 0.8170); diff (0.14461636543273926, 0.14200487732887268, 0.14716720581054688); uniform prior held-out greedy k=5 0.9616; conservative 0.9596136133974107
- transfer decider {'reference': 'full pool', 'dis_V': 0.048876918852329254, 'a_V': -0.720984691048938, 'dis_T': 0.1407223641872406, 'a_T': -0.9226334373813364, 'gamma': 0.20164874633239838, 'regret': -0.12983515858650208}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.049650467932224274, 'a_V': -0.7207155551200463, 'dis_T': 0.14241540431976318, 'a_T': -0.9192603461885195, 'gamma': 0.19854479106847323, 'regret': -0.13091683387756348} test 0.9627
  - ablation P1-P7: held-out greedy k=5 0.9616
  - ablation P1-P8: held-out greedy k=5 0.9616
  - ablation P1-P7,P9-P10: held-out greedy k=5 0.9616
  - ablation P1-P10: held-out greedy k=5 0.9616
  - ablation P1-P12: held-out greedy k=5 0.9616
  - ablation P9-P12: held-out greedy k=5 0.9616
  - ablation P11-P12: held-out greedy k=5 0.9616
  - ablation P7,P11-P12: held-out greedy k=5 0.9616
  - ablation P9-P12 no search: P12 signals geo alpha high k=1 0.9015
  - ablation P11-P12 no search: P12 signals geo alpha high k=1 0.9015
  - ablation P7,P11-P12 no search: P12 signals geo alpha high k=1 0.9015
  - ablation menu, held-out greedy only: held-out greedy k=5 0.9616
  - ablation menu, held-out top-k only: held-out top-k k=10 0.9574
  - ablation full pool vs held-out greedy: held-out greedy k=5 0.9616
  - budget 5: held-out greedy k=5 0.9616
  - budget 10: held-out greedy k=5 0.9616
  - budget 20: held-out greedy k=5 0.9616
  - baseline best member on V: k=1  0.9529 diff (0.1359248161315918, 0.13328945636749268, 0.13857656717300415)
  - baseline top-k on V: k=7  0.9570 diff (0.13997739553451538, 0.13738849759101868, 0.14252212643623352)
  - baseline greedy on V: k=6  0.9627 diff (0.14573466777801514, 0.14302116632461548, 0.14830754697322845)
  - baseline Caruana (bagged): k=36  0.9595 diff (0.14244234561920166, 0.13970786333084106, 0.14511215686798096)
  - baseline log-odds weighted vote: k=100  0.8615 diff (0.0444483757019043, 0.0428415909409523, 0.04597432538866997)
  - baseline linear-pool stacking: k=29  0.9593 diff (0.14226043224334717, 0.1397208571434021, 0.1447877138853073)
  - baseline proposed paths by balanced accuracy: k=1 P12 signals geo alpha high 0.9015 diff (0.08448207378387451, 0.08185042440891266, 0.08703561127185822)
  - baseline proposed paths by NLL: k=14 P12 signals geo alpha high 0.8925 diff (0.0754619836807251, 0.07337508350610733, 0.0775393545627594)
  - baseline proposed paths by BRIER: k=14 P12 signals geo alpha high 0.8925 diff (0.0754619836807251, 0.07337508350610733, 0.0775393545627594)
- inductive: held-out greedy k=5 0.9616; transductive 0.9616; full 0.8162 (eval half, one domain, random halves)
- vote decider held-out greedy k=8 0.9618 (full 0.8101)

## Camelyon17 / mixed (camelyon17_mixed_seed_3)
- greedy ref in-sample k=5 0.9661 (full 0.8470); conf best 0.9583
- cross-fitted greedy k=[6, 5] score 0.9657 (full halves 0.8470); gain (0.11869561672210693, 0.11633802950382233, 0.12111988663673401); greedy-topk (0.002125263214111328, 0.0014762431383132935, 0.002782395575195551)
- search: k=5: greedy 0.9661 best 0.9661 (rand 0.9657315080030302, swap 0.9661, back 0.9660533367712295); k=10: greedy 0.9651 best 0.9651 (rand 0.9649445717180448, swap 0.9651, back 0.964731608487441); k=20: greedy 0.9625 best 0.9626 (rand 0.9625121621650896, swap 0.9626, back 0.9625948015739118); k=40: greedy 0.9553 best 0.9553 (rand 0.9553289729994684, swap 0.9553, back 0.9553289729994684)
- gain cf +0.1120 (k [6, 4]), best member share cf 0.802753210067749, certificate B(4) = 0.2306; B by k {'1': 0.1170080155134201, '2': 0.23369857668876648, '5': 0.19118835031986237, '10': 0.22386249899864197, '20': 0.21593694388866425, '40': 0.18936851620674133}; mean D on H 1.0
- soft cf: vote 0.9656 k [6, 4] full 0.8428; wvote 0.9656 k [6, 4] full 0.8443; avg 0.9657 k [6, 5] full 0.8470
- P7 oracle cf: k [5, 7] 0.9645
- gain in-sample k=4 +0.1123 bound 0.2306 (ties as errors 0.1538) headroom 0.1452 minority-right 0.1440 best member 0.8049
- q bar 1.0000 q tilde 1.0000 corrected 1.0 ece 0.11764228208145709
- mu<0 split 0.07506302744150162 plain 0.0745; err align wrong 0.660183155484517
- decider held-out greedy k=7 eps 0.0158 -> 0.9588 (full 0.8470); diff (0.11179977655410767, 0.10954638570547104, 0.114236980676651); uniform prior held-out greedy k=7 0.9588; conservative 0.9511580087401513
- transfer decider {'reference': 'full pool', 'dis_V': 0.03999541327357292, 'a_V': -0.7063037370716437, 'dis_T': 0.11119993776082993, 'a_T': -0.916684304854076, 'gamma': 0.2103805677824323, 'regret': -0.1019352376461029}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.03979486599564552, 'a_V': -0.7120229971804901, 'dis_T': 0.1088014617562294, 'a_T': -0.9165766685601022, 'gamma': 0.2045536713796121, 'regret': -0.09972488135099411} test 0.9565
  - ablation P1-P7: held-out greedy k=7 0.9588
  - ablation P1-P8: held-out greedy k=7 0.9588
  - ablation P1-P7,P9-P10: held-out greedy k=7 0.9588
  - ablation P1-P10: held-out greedy k=7 0.9588
  - ablation P1-P12: held-out greedy k=7 0.9588
  - ablation P9-P12: held-out greedy k=7 0.9588
  - ablation P11-P12: held-out greedy k=7 0.9588
  - ablation P7,P11-P12: held-out greedy k=7 0.9588
  - ablation P9-P12 no search: P11 signals mean k=4 0.9118
  - ablation P11-P12 no search: P11 signals mean k=4 0.9118
  - ablation P7,P11-P12 no search: P11 signals mean k=4 0.9118
  - ablation menu, held-out greedy only: held-out greedy k=7 0.9588
  - ablation menu, held-out top-k only: held-out top-k k=6 0.9429
  - ablation full pool vs held-out greedy: held-out greedy k=7 0.9588
  - budget 5: held-out greedy k=5 0.9528
  - budget 10: held-out greedy k=7 0.9588
  - budget 20: held-out greedy k=7 0.9588
  - baseline best member on V: k=1  0.9092 diff (0.06213223934173584, 0.060039639472961426, 0.06403496861457825)
  - baseline top-k on V: k=12  0.9446 diff (0.09760582447052002, 0.09534908086061478, 0.09985648840665817)
  - baseline greedy on V: k=6  0.9565 diff (0.10953086614608765, 0.1072860136628151, 0.11188232898712158)
  - baseline Caruana (bagged): k=41  0.9505 diff (0.10352998971939087, 0.1013362854719162, 0.10589655488729477)
  - baseline log-odds weighted vote: k=100  0.8856 diff (0.03861600160598755, 0.03717745468020439, 0.04005831107497215)
  - baseline linear-pool stacking: k=29  0.9421 diff (0.09506767988204956, 0.09292978048324585, 0.09722410142421722)
  - baseline proposed paths by balanced accuracy: k=14 P12 signals geo 0.9259 diff (0.07888847589492798, 0.0767330750823021, 0.08096607029438019)
  - baseline proposed paths by NLL: k=14 P12 signals geo 0.9259 diff (0.07888847589492798, 0.0767330750823021, 0.08096607029438019)
  - baseline proposed paths by BRIER: k=14 P12 signals geo 0.9259 diff (0.07888847589492798, 0.0767330750823021, 0.08096607029438019)
- inductive: held-out greedy k=7 0.9585; transductive 0.9585; full 0.8463 (eval half, one domain, random halves)
- vote decider held-out greedy k=4 0.9584 (full 0.8428)

## Kather / EffNet-B0 (kather_seed_1)
- greedy ref in-sample k=22 0.9759 (full 0.9666); conf best 0.9668
- cross-fitted greedy k=[10, 23] score 0.9709 (full halves 0.9666); gain (0.004363536834716797, 0.0012888461351394653, 0.007385385688394308); greedy-topk (0.004669874906539917, 0.0022200928069651127, 0.00743379769846797)
- search: k=5: greedy 0.9734 best 0.9741 (rand 0.9740939534885718, swap 0.9734, back 0.9734493983730098); k=10: greedy 0.9746 best 0.9758 (rand 0.9757822462344964, swap 0.9753, back 0.9745809559449863); k=20: greedy 0.9755 best 0.9768 (rand 0.9765688780583931, swap 0.9768, back 0.9726353190986873); k=22: greedy 0.9759 best 0.9765 (rand 0.9765290584015566, swap 0.9763, back 0.972993377479945); k=40: greedy 0.9745 best 0.9754 (rand 0.9753869640417616, swap 0.9750, back 0.9714802540145802)
- gain cf +0.0021 (k [13, 15]), best member share cf 0.3448708653450012, certificate B(20) = 0.0213; B by k {'1': 0.008913649246096611, '2': 0.017688022926449776, '5': 0.020032497122883797, '10': 0.021044569090008736, '20': 0.02131582424044609, '40': 0.018492672592401505}; mean D on H 1.8670886754989624
- soft cf: vote 0.9689 k [11, 23] full 0.9657; wvote 0.9687 k [8, 16] full 0.9655; avg 0.9709 k [10, 23] full 0.9666
- P7 oracle cf: k [17, 23] 0.9688
- gain in-sample k=20 +0.0058 bound 0.0426 (ties as errors 0.0320) headroom 0.0212 minority-right 0.0209 best member 0.4200
- q bar 0.8183 q tilde 0.7250 corrected None ece 0.048597015805895295
- mu<0 split 0.021239555440843105 plain 0.0217; err align wrong 0.5416783656373795

## Kather / ConvNeXt-A (kather_convnext_atto_seed_2)
- greedy ref in-sample k=15 0.9741 (full 0.9692); conf best 0.9692
- cross-fitted greedy k=[25, 17] score 0.9704 (full halves 0.9693); gain (0.0011390745639801025, -0.00045161991147324443, 0.002823159797117114); greedy-topk (0.000460892915725708, -0.001775273703970015, 0.0028217348735779524)
- search: k=5: greedy 0.9726 best 0.9728 (rand 0.9727668886263977, swap 0.9726, back 0.9700878114694147); k=10: greedy 0.9731 best 0.9746 (rand 0.9746309255616769, swap 0.9744, back 0.9729410941941353); k=15: greedy 0.9741 best 0.9743 (rand 0.9741897621590575, swap 0.9743, back 0.9726253724781); k=20: greedy 0.9731 best 0.9743 (rand 0.9740451963494003, swap 0.9743, back 0.9727580266718107); k=40: greedy 0.9725 best 0.9731 (rand 0.9730833236746026, swap 0.9731, back 0.9714255259323541)
- gain cf +0.0006 (k [12, 21]), best member share cf 0.3652016669511795, certificate B(14) = 0.0162; B by k {'1': 0.007520891726016998, '2': 0.01420612819492817, '5': 0.014995358884334564, '10': 0.016067780554294586, '20': 0.016476323828101158, '40': 0.015477020293474197}; mean D on H 1.865771770477295
- soft cf: vote 0.9707 k [14, 15] full 0.9692; wvote 0.9684 k [19, 9] full 0.9688; avg 0.9704 k [25, 17] full 0.9693
- P7 oracle cf: k [13, 25] 0.9692
- gain in-sample k=14 +0.0032 bound 0.0372 (ties as errors 0.0248) headroom 0.0203 minority-right 0.0202 best member 0.3655
- q bar 0.8225 q tilde 0.7243 corrected None ece 0.04721801196765102
- mu<0 split 0.02110027987509966 plain 0.0206; err align wrong 0.5192524693704479

## Kather / mixed (kather_mixed_seed_3)
- greedy ref in-sample k=13 0.9781 (full 0.9715); conf best 0.9731
- cross-fitted greedy k=[12, 11] score 0.9720 (full halves 0.9715); gain (0.00041103363037109375, -0.001931843115016818, 0.002732162596657872); greedy-topk (0.0022265613079071045, -0.0007573813199996948, 0.005600310396403074)
- search: k=5: greedy 0.9757 best 0.9772 (rand 0.9758986076508589, swap 0.9772, back 0.9745558976039254); k=10: greedy 0.9777 best 0.9787 (rand 0.9777095647739543, swap 0.9787, back 0.9761028662161428); k=13: greedy 0.9781 best 0.9783 (rand 0.9783231636853447, swap 0.9783, back 0.9761034820272496); k=20: greedy 0.9773 best 0.9782 (rand 0.9777463811254478, swap 0.9782, back 0.9768732355353729); k=40: greedy 0.9768 best 0.9773 (rand 0.9772350569067795, swap 0.9773, back 0.9761573839244205)
- gain cf +0.0001 (k [27, 18]), best member share cf 0.3309384137392044, certificate B(26) = 0.0153; B by k {'1': 0.006685236934572458, '2': 0.012952646240592003, '5': 0.015529247932136059, '10': 0.01639275997877121, '20': 0.016026992350816727, '40': 0.014940808527171612}; mean D on H 2.031007766723633
- soft cf: vote 0.9730 k [16, 20] full 0.9716; wvote 0.9729 k [21, 22] full 0.9706; avg 0.9720 k [12, 11] full 0.9715
- P7 oracle cf: k [29, 19] 0.9698
- gain in-sample k=26 +0.0032 bound 0.0396 (ties as errors 0.0297) headroom 0.0166 minority-right 0.0163 best member 0.4017
- q bar 0.7986 q tilde 0.7058 corrected None ece 0.05018871985305319
- mu<0 split 0.018245126120746136 plain 0.0178; err align wrong 0.4884893411809968

## iWildCam / EffNet-B0 (iwildcam_seed_1)
- greedy ref in-sample k=27 0.1884 (full 0.1713); conf best 0.1792
- cross-fitted greedy k=[10, 9] score 0.1896 (full halves 0.1946); gain (-0.0050357431173324585, -0.01573897711932659, 0.0044045173563063145); greedy-topk (-0.011437661945819855, -0.02323349192738533, 0.007261369377374649)
- search: k=5: greedy 0.1882 best 0.1957 (rand 0.19565932385781507, swap 0.1898, back 0.18882236242689512); k=10: greedy 0.1871 best 0.1939 (rand 0.19392420117792125, swap 0.1880, back 0.18830145151025915); k=20: greedy 0.1861 best 0.1921 (rand 0.19209098657233115, swap 0.1881, back 0.18819064616411416); k=27: greedy 0.1884 best 0.1911 (rand 0.19112029458627322, swap 0.1889, back 0.18625963576640223); k=40: greedy 0.1858 best 0.1882 (rand 0.188192223321653, swap 0.1881, back 0.18472004298670747)
- gain cf -0.0147 (k [2, 10]), best member share cf 0.11435611918568611, certificate B(6) = 0.2633; B by k {'1': 0.06022294610738754, '2': 0.1117524802684784, '5': 0.23687615990638733, '10': 0.2553578317165375, '20': 0.25965601205825806, '40': 0.24536018073558807}; mean D on H 7.037649631500244
- soft cf: vote 0.1910 k [19, 7] full 0.1928; wvote 0.1940 k [14, 6] full 0.1939; avg 0.1896 k [10, 9] full 0.1946
- P7 oracle cf: k [1, 27] 0.1763
- gain in-sample k=6 +0.0148 bound 0.2701 (ties as errors 0.1351) headroom 0.2978 minority-right 0.2969 best member 0.2022
- q bar 0.4255 q tilde 0.3248 corrected 0.4020767856803471 ece 0.20583585004874944
- mu<0 split 0.5258004367351532 plain 0.5366; err align wrong 0.4932818945709677
- decider held-out top-k k=51 eps 0.1027 -> 0.1729 (full 0.1713); diff (0.0016566067934036255, -0.0028167101554572582, 0.004479842726141214); uniform prior held-out top-k k=51 0.1729; conservative 0.1707014629830534
- transfer decider {'reference': 'full pool', 'dis_V': 0.07145244628190994, 'a_V': -0.05144994596037511, 'dis_T': 0.043046437203884125, 'a_T': 0.020629777615080322, 'gamma': 0.07207972357545543, 'regret': 0.0008880384266376494}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.1664995700120926, 'a_V': -0.04696911836000529, 'dis_T': 0.09889930486679077, 'a_T': 0.042296768730544734, 'gamma': 0.08926588709055003, 'regret': 0.004183121025562286} test 0.1725
  - ablation P1-P7: held-out top-k k=51 0.1729
  - ablation P1-P8: held-out top-k k=51 0.1729
  - ablation P1-P7,P9-P10: held-out top-k k=51 0.1729
  - ablation P1-P10: held-out top-k k=51 0.1729
  - ablation P1-P12: held-out top-k k=51 0.1729
  - ablation P9-P12: held-out top-k k=51 0.1729
  - ablation P11-P12: held-out top-k k=51 0.1729
  - ablation P7,P11-P12: held-out top-k k=51 0.1729
  - ablation P9-P12 no search: P12 signals geo alpha high k=13 0.1784
  - ablation P11-P12 no search: P12 signals geo alpha high k=13 0.1784
  - ablation P7,P11-P12 no search: P12 signals geo alpha high k=13 0.1784
  - ablation menu, held-out greedy only: held-out greedy k=28 0.1709
  - ablation menu, held-out top-k only: held-out top-k k=51 0.1729
  - ablation full pool vs held-out greedy: held-out greedy k=28 0.1709
  - budget 5: held-out top-k k=5 0.1671
  - budget 10: held-out top-k k=10 0.1657
  - budget 20: held-out top-k k=10 0.1657
  - baseline best member on V: k=1  0.1557 diff (-0.015598013997077942, -0.03634233772754669, -0.0006465497426688671)
  - baseline top-k on V: k=11  0.1693 diff (-0.0020067989826202393, -0.01799727976322174, 0.005878360942006111)
  - baseline greedy on V: k=9  0.1725 diff (0.0012366771697998047, -0.012123968452215195, 0.011291942559182644)
  - baseline Caruana (bagged): k=85  0.1705 diff (-0.0007473528385162354, -0.005394995212554932, 0.003771803341805935)
  - baseline log-odds weighted vote: k=100  0.1692 diff (-0.0020941197872161865, -0.004815604072064161, -9.83577047009021e-05)
  - baseline linear-pool stacking: k=43  0.1647 diff (-0.006537914276123047, -0.014338810928165913, -0.0009855631506070495)
  - baseline proposed paths by balanced accuracy: k=13 P12 signals geo alpha high 0.1784 diff (0.007122337818145752, -0.007686160504817963, 0.018633056432008743)
  - baseline proposed paths by NLL: k=73 P7 collision-aware 0.1707 diff (-0.0005372315645217896, -0.004076275043189526, 0.0023573506623506546)
  - baseline proposed paths by BRIER: k=97 P7 collision-aware 0.1710 diff (-0.00029417872428894043, -0.0012696478515863419, 0.00035771558759734035)
- inductive: held-out top-k k=51 0.2080; transductive 0.2080; full 0.2062 (eval half, 48 domains, 24+24)
- vote decider held-out greedy k=10 0.1742 (full 0.1695)

## iWildCam / ConvNeXt-A (iwildcam_convnext_atto_seed_2)
- greedy ref in-sample k=7 0.1864 (full 0.1660); conf best 0.1719
- cross-fitted greedy k=[9, 10] score 0.1769 (full halves 0.1882); gain (-0.011295042932033539, -0.01993965543806553, -0.0001411398989148438); greedy-topk (-0.005822107195854187, -0.017500633373856544, 0.008740251883864403)
- search: k=5: greedy 0.1810 best 0.1861 (rand 0.1831144262952569, swap 0.1861, back 0.17724055600279465); k=7: greedy 0.1864 best 0.1864 (rand 0.18597666356093678, swap 0.1864, back 0.17858804411402837); k=10: greedy 0.1855 best 0.1892 (rand 0.18755233816301276, swap 0.1892, back 0.17766695098555796); k=20: greedy 0.1797 best 0.1844 (rand 0.1843848079736781, swap 0.1817, back 0.17856759062885752); k=40: greedy 0.1754 best 0.1817 (rand 0.18173426511317264, swap 0.1784, back 0.1730280586057524)
- gain cf -0.0169 (k [4, 14]), best member share cf 0.12402951717376709, certificate B(6) = 0.4106; B by k {'1': 0.10548947751522064, '2': 0.187212273478508, '5': 0.3788491487503052, '10': 0.3862292170524597, '20': 0.3805600106716156, '40': 0.34390515089035034}; mean D on H 7.522610187530518
- soft cf: vote 0.1834 k [20, 28] full 0.1866; wvote 0.1821 k [17, 31] full 0.1882; avg 0.1769 k [9, 10] full 0.1882
- P7 oracle cf: k [4, 31] 0.1779
- gain in-sample k=6 +0.0513 bound 0.4285 (ties as errors 0.2143) headroom 0.3330 minority-right 0.3319 best member 0.3169
- q bar 0.3813 q tilde 0.2859 corrected 0.3791975646989943 ece 0.20820537724305166
- mu<0 split 0.5052469670772552 plain 0.5191; err align wrong 0.48399017986921405
- decider held-out greedy k=5 eps 0.0987 -> 0.1544 (full 0.1660); diff (-0.011574521660804749, -0.025238679721951485, 0.008973963558673859); uniform prior held-out greedy k=5 0.1544; conservative 0.1560673281277717
- transfer decider {'reference': 'full pool', 'dis_V': 0.21114899218082428, 'a_V': -0.043684710954554666, 'dis_T': 0.1490967720746994, 'a_T': 0.08605012235507407, 'gamma': 0.12973483330962873, 'regret': 0.012829795479774475}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.1878885179758072, 'a_V': -0.045535386075881604, 'dis_T': 0.13692131638526917, 'a_T': 0.10564943603151458, 'gamma': 0.1511848221073962, 'regret': 0.014465659856796265} test 0.1547
  - ablation P1-P7: held-out greedy k=5 0.1544
  - ablation P1-P8: held-out greedy k=5 0.1544
  - ablation P1-P7,P9-P10: held-out greedy k=5 0.1544
  - ablation P1-P10: held-out greedy k=5 0.1544
  - ablation P1-P12: held-out greedy k=5 0.1544
  - ablation P9-P12: held-out greedy k=5 0.1544
  - ablation P11-P12: held-out greedy k=5 0.1544
  - ablation P7,P11-P12: held-out greedy k=5 0.1544
  - ablation P9-P12 no search: P9 consensus mean alpha high k=10 0.1624
  - ablation P11-P12 no search: P12 signals geo alpha low k=9 0.1647
  - ablation P7,P11-P12 no search: P12 signals geo alpha low k=9 0.1647
  - ablation menu, held-out greedy only: held-out greedy k=5 0.1544
  - ablation menu, held-out top-k only: held-out top-k k=9 0.1572
  - ablation full pool vs held-out greedy: held-out greedy k=5 0.1544
  - budget 5: held-out greedy k=5 0.1544
  - budget 10: held-out greedy k=5 0.1544
  - budget 20: held-out greedy k=5 0.1544
  - baseline best member on V: k=1  0.1492 diff (-0.016802310943603516, -0.034353043884038925, -0.0039099338464438915)
  - baseline top-k on V: k=10  0.1565 diff (-0.009517297148704529, -0.02393708936870098, 0.009586118161678314)
  - baseline greedy on V: k=7  0.1547 diff (-0.0112781822681427, -0.02708611451089382, 0.009829331189393997)
  - baseline Caruana (bagged): k=77  0.1581 diff (-0.007866859436035156, -0.0177572350949049, 0.007063642609864473)
  - baseline log-odds weighted vote: k=100  0.1662 diff (0.00020831823348999023, -0.003257370088249445, 0.003365653334185481)
  - baseline linear-pool stacking: k=35  0.1459 diff (-0.020057201385498047, -0.029387596994638443, -0.0003432741214055568)
  - baseline proposed paths by balanced accuracy: k=9 P12 signals geo alpha low 0.1647 diff (-0.001305803656578064, -0.014473775401711464, 0.012811039574444294)
  - baseline proposed paths by NLL: k=79 P7 collision-aware 0.1622 diff (-0.0038019269704818726, -0.011655023321509361, 0.0024531525559723377)
  - baseline proposed paths by BRIER: k=92 P7 collision-aware 0.1654 diff (-0.0005671679973602295, -0.001839947304688394, 0.0005461839609779418)
- inductive: held-out greedy k=5 0.1967; transductive 0.1967; full 0.2057 (eval half, 48 domains, 24+24)
- vote decider held-out top-k k=9 0.1585 (full 0.1660)

## iWildCam / mixed (iwildcam_mixed_seed_3)
- greedy ref in-sample k=5 0.1932 (full 0.1600); conf best 0.1737
- cross-fitted greedy k=[8, 5] score 0.1871 (full halves 0.1913); gain (-0.00417531281709671, -0.012194402515888214, 0.006508102174848318); greedy-topk (-8.040666580200195e-05, -0.00944456271827221, 0.01480273436754942)
- search: k=5: greedy 0.1932 best 0.1932 (rand 0.18771172720628268, swap 0.1932, back 0.18005406252905976); k=10: greedy 0.1895 best 0.1907 (rand 0.19070218980651646, swap 0.1895, back 0.18456093107897004); k=20: greedy 0.1810 best 0.1907 (rand 0.190710649573485, swap 0.1869, back 0.18421649524063696); k=40: greedy 0.1790 best 0.1837 (rand 0.18366541692259208, swap 0.1818, back 0.1788422814399966)
- gain cf -0.0031 (k [5, 19]), best member share cf 0.09346157684922218, certificate B(5) = 0.2974; B by k {'1': 0.06527073681354523, '2': 0.12972354888916016, '5': 0.29739511013031006, '10': 0.34513723850250244, '20': 0.338468074798584, '40': 0.31314969062805176}; mean D on H 8.027247428894043
- soft cf: vote 0.1895 k [10, 8] full 0.1932; wvote 0.1871 k [5, 8] full 0.1916; avg 0.1871 k [8, 5] full 0.1913
- P7 oracle cf: k [4, 33] 0.1775
- gain in-sample k=5 +0.0348 bound 0.2882 (ties as errors 0.1441) headroom 0.3256 minority-right 0.3242 best member 0.1999
- q bar 0.3598 q tilde 0.2769 corrected 0.3561775907632703 ece 0.18352948761218113
- mu<0 split 0.49652431905269623 plain 0.5107; err align wrong 0.4339595242613388
- decider held-out top-k k=55 eps 0.1025 -> 0.1633 (full 0.1600); diff (0.0032709240913391113, -0.004516025073826313, 0.008802591823041439); uniform prior held-out top-k k=55 0.1633; conservative 0.16364044276051443
- transfer decider {'reference': 'full pool', 'dis_V': 0.055009692907333374, 'a_V': -0.02065614234115046, 'dis_T': 0.038115493953228, 'a_T': -0.023298522706725763, 'gamma': 0.0026423803655753024, 'regret': -0.000888034701347351}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.09170509874820709, 'a_V': -0.08017492048145786, 'dis_T': 0.05746535584330559, 'a_T': 0.0479870040586062, 'gamma': 0.12816192454006406, 'regret': 0.002757590264081955} test 0.1582
  - ablation P1-P7: held-out top-k k=55 0.1633
  - ablation P1-P8: held-out top-k k=55 0.1633
  - ablation P1-P7,P9-P10: held-out top-k k=55 0.1633
  - ablation P1-P10: held-out top-k k=55 0.1633
  - ablation P1-P12: held-out top-k k=55 0.1633
  - ablation P9-P12: held-out top-k k=55 0.1633
  - ablation P11-P12: held-out top-k k=55 0.1633
  - ablation P7,P11-P12: held-out top-k k=55 0.1633
  - ablation P9-P12 no search: P10 consensus geo k=10 0.1740
  - ablation P11-P12 no search: P11 signals mean k=10 0.1687
  - ablation P7,P11-P12 no search: P11 signals mean k=10 0.1687
  - ablation menu, held-out greedy only: held-out greedy k=7 0.1723
  - ablation menu, held-out top-k only: held-out top-k k=55 0.1633
  - ablation full pool vs held-out greedy: held-out greedy k=7 0.1723
  - budget 5: held-out greedy k=5 0.1624
  - budget 10: held-out greedy k=7 0.1723
  - budget 20: held-out greedy k=7 0.1723
  - baseline best member on V: k=1  0.1511 diff (-0.008888795971870422, -0.0379047729074955, 0.012247283011674881)
  - baseline top-k on V: k=19  0.1670 diff (0.0069964975118637085, -0.006505317986011505, 0.015191545709967613)
  - baseline greedy on V: k=31  0.1582 diff (-0.001841023564338684, -0.00801536999642849, 0.002111593959853053)
  - baseline Caruana (bagged): k=86  0.1608 diff (0.0007823556661605835, -0.004890915937721729, 0.0057573746889829636)
  - baseline log-odds weighted vote: k=100  0.1632 diff (0.003171667456626892, -0.00545090576633811, 0.009919595904648304)
  - baseline linear-pool stacking: k=36  0.1551 diff (-0.0048897117376327515, -0.01760716922581196, 0.0014857283094897866)
  - baseline proposed paths by balanced accuracy: k=25 P11 signals mean 0.1656 diff (0.0055532306432724, -0.005486717913299799, 0.014265012927353382)
  - baseline proposed paths by NLL: k=59 P7 collision-aware 0.1548 diff (-0.005211547017097473, -0.01785803586244583, 0.0009968666126951575)
  - baseline proposed paths by BRIER: k=88 P7 collision-aware 0.1611 diff (0.0011366605758666992, -0.006869285833090544, 0.009336316958069801)
- inductive: held-out top-k k=55 0.2054; transductive 0.2054; full 0.2066 (eval half, 48 domains, 24+24)
- vote decider held-out top-k k=17 0.1641 (full 0.1637)

## RxRx1 / EffNet-B0 (rxrx1_seed_1)
- greedy ref in-sample k=31 0.2856 (full 0.2765); conf best 0.2806
- cross-fitted greedy k=[34, 29] score 0.2819 (full halves 0.2770); gain (0.004882603883743286, 0.0006927076610736549, 0.008777951821684837); greedy-topk (0.0035522431135177612, -0.0018495291005820036, 0.008510000072419643)
- search: k=5: greedy 0.2578 best 0.2616 (rand 0.26160225062504955, swap 0.2616, back 0.2549007404600366); k=10: greedy 0.2713 best 0.2761 (rand 0.276069443461835, swap 0.2753, back 0.26916926569053135); k=20: greedy 0.2821 best 0.2864 (rand 0.2864209837354504, swap 0.2835, back 0.28249605132843797); k=31: greedy 0.2856 best 0.2880 (rand 0.2875642671334854, swap 0.2880, back 0.28531195904128004); k=40: greedy 0.2842 best 0.2877 (rand 0.2877072604140816, swap 0.2862, back 0.28529706326879933)
- gain cf +0.0039 (k [37, 26]), best member share cf 0.0933246836066246, certificate B(33) = 0.6585; B by k {'1': 0.039178669452667236, '2': 0.07827021181583405, '5': 0.18844389915466309, '10': 0.35297009348869324, '20': 0.5924333333969116, '40': 0.6520824432373047}; mean D on H 32.883750915527344
- soft cf: vote 0.2784 k [41, 25] full 0.2746; wvote 0.2749 k [29, 40] full 0.2739; avg 0.2819 k [34, 29] full 0.2770
- P7 oracle cf: k [59, 59] 0.2784
- gain in-sample k=33 +0.0116 bound 1.0590 (ties as errors 0.5295) headroom 0.3989 minority-right 0.3960 best member 0.0975
- q bar 0.1074 q tilde 0.0659 corrected 0.08843911775832014 ece 0.13769816013184621
- mu<0 split 0.7993200719356537 plain 0.8471; err align wrong 0.1957545306095707
- decider P12 signals geo alpha low k=52 eps 0.0238 -> 0.2803 (full 0.2765); diff (0.003779083490371704, 0.0003351364575792104, 0.007227287162095308); uniform prior P12 signals geo alpha low k=52 0.2803; conservative None
- transfer decider {'reference': 'full pool', 'dis_V': 0.20133955776691437, 'a_V': -0.026209715468903213, 'dis_T': 0.14428438246250153, 'a_T': -0.027375249683432235, 'gamma': 0.0011655342145290223, 'regret': -0.0039498209953308105}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.3435153067111969, 'a_V': -0.04401776052510827, 'dis_T': 0.24785083532333374, 'a_T': -0.004921484943054128, 'gamma': 0.03909627558205414, 'regret': -0.0012197941541671753} test 0.2764
  - ablation P1-P7: P2 alpha low k=66 0.2772
  - ablation P1-P8: P8 noise 0.02 k=60 0.2690
  - ablation P1-P7,P9-P10: P2 alpha low k=66 0.2772
  - ablation P1-P10: P8 noise 0.02 k=60 0.2690
  - ablation P1-P12: P12 signals geo alpha low k=52 0.2803
  - ablation P9-P12: P12 signals geo alpha low k=52 0.2803
  - ablation P11-P12: P12 signals geo alpha low k=52 0.2803
  - ablation P7,P11-P12: P12 signals geo alpha low k=52 0.2803
  - ablation P9-P12 no search: P12 signals geo alpha low k=52 0.2803
  - ablation P11-P12 no search: P12 signals geo alpha low k=52 0.2803
  - ablation P7,P11-P12 no search: P12 signals geo alpha low k=52 0.2803
  - ablation menu, held-out greedy only: P12 signals geo alpha low k=52 0.2803
  - ablation menu, held-out top-k only: P12 signals geo alpha low k=52 0.2803
  - ablation full pool vs held-out greedy: full pool k=100 0.2765
  - budget 5: held-out top-k k=5 0.2460
  - budget 10: P12 signals geo alpha low k=10 0.2689
  - budget 20: P11 signals mean alpha low k=18 0.2768
  - baseline best member on V: k=1  0.2180 diff (-0.0585714727640152, -0.06851280480623245, -0.047918930649757385)
  - baseline top-k on V: k=40  0.2745 diff (-0.0019993185997009277, -0.006943578366190195, 0.0021749427542090416)
  - baseline greedy on V: k=29  0.2764 diff (-0.00016033649444580078, -0.005579311866313219, 0.004942294675856829)
  - baseline Caruana (bagged): k=89  0.2771 diff (0.0005904734134674072, -0.0035485432017594576, 0.004128171131014824)
  - baseline log-odds weighted vote: k=100  0.2736 diff (-0.0029712319374084473, -0.005869650281965733, -0.0006263660034164786)
  - baseline linear-pool stacking: k=60  0.2720 diff (-0.00451698899269104, -0.010099699720740318, 0.00029153950163163245)
  - baseline proposed paths by balanced accuracy: k=52 P12 signals geo alpha low 0.2803 diff (0.003779083490371704, 0.0003351364575792104, 0.007227287162095308)
  - baseline proposed paths by NLL: k=99 P11 signals mean alpha high 0.2763 diff (-0.0002734661102294922, -0.0009327970328740776, 0.0003372112405486405)
  - baseline proposed paths by BRIER: k=49 P12 signals geo alpha low 0.2810 diff (0.004422008991241455, 0.0007200032705441117, 0.008040162734687328)
- inductive: P12 signals geo alpha low k=36 0.3031; transductive 0.3029; full 0.2980 (eval half, 14 domains, 7+7)
- vote decider P11 signals mean alpha low k=52 0.2768 (full 0.2742)

## RxRx1 / ConvNeXt-A (rxrx1_convnext_atto_seed_2)
- greedy ref in-sample k=12 0.2223 (full 0.2026); conf best 0.2058
- cross-fitted greedy k=[15, 20] score 0.2203 (full halves 0.2031); gain (0.017150163650512695, 0.011350368149578571, 0.02304130606353283); greedy-topk (0.008043147623538971, 0.0017525379080325365, 0.01361638493835926)
- search: k=5: greedy 0.2032 best 0.2068 (rand 0.20683259774744672, swap 0.2042, back 0.20591022350300656); k=10: greedy 0.2174 best 0.2208 (rand 0.2207513007968132, swap 0.2192, back 0.21971177048866375); k=12: greedy 0.2213 best 0.2234 (rand 0.22341491402942149, swap 0.2213, back 0.22146724964748826); k=20: greedy 0.2201 best 0.2262 (rand 0.22620260119122435, swap 0.2239, back 0.2242061088156668); k=40: greedy 0.2187 best 0.2210 (rand 0.22104250908888604, swap 0.2199, back 0.21848689640868457)
- gain cf +0.0195 (k [24, 27]), best member share cf 0.10400062426924706, certificate B(25) = 0.6845; B by k {'1': 0.042954228818416595, '2': 0.0849790871143341, '5': 0.2070796638727188, '10': 0.3938139081001282, '20': 0.6477788686752319, '40': 0.6515547633171082}; mean D on H 31.455564498901367
- soft cf: vote 0.2178 k [24, 34] full 0.1977; wvote 0.2134 k [16, 18] full 0.2005; avg 0.2203 k [15, 20] full 0.2031
- P7 oracle cf: k [35, 44] 0.2135
- gain in-sample k=25 +0.0254 bound 0.8699 (ties as errors 0.4349) headroom 0.3894 minority-right 0.3870 best member 0.1096
- q bar 0.1193 q tilde 0.0697 corrected 0.09819229297651744 ece 0.09910494284835891
- mu<0 split 0.8632027506828308 plain 0.8959; err align wrong 0.25254288743998626
- decider held-out top-k k=7 eps 0.0337 -> 0.2041 (full 0.2026); diff (0.0014899671077728271, -0.012190510518848896, 0.01248579565435648); uniform prior held-out top-k k=7 0.2041; conservative 0.20545455495558784
- transfer decider {'reference': 'full pool', 'dis_V': 0.7154455184936523, 'a_V': -0.0341843464200832, 'dis_T': 0.5270388126373291, 'a_T': -0.003251210891714818, 'gamma': 0.030933135528368384, 'regret': -0.0017135143280029297}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.6679520606994629, 'a_V': -0.04390759332008849, 'dis_T': 0.47513940930366516, 'a_T': -0.02267724089889295, 'gamma': 0.02123035242119554, 'regret': -0.010774850845336914} test 0.2133
  - ablation P1-P7: held-out top-k k=7 0.2041
  - ablation P1-P8: P8 noise 0.02 k=16 0.2127
  - ablation P1-P7,P9-P10: held-out top-k k=7 0.2041
  - ablation P1-P10: P8 noise 0.02 k=16 0.2127
  - ablation P1-P12: P8 noise 0.02 k=16 0.2127
  - ablation P9-P12: held-out top-k k=7 0.2041
  - ablation P11-P12: held-out top-k k=7 0.2041
  - ablation P7,P11-P12: held-out top-k k=7 0.2041
  - ablation P9-P12 no search: P12 signals geo alpha high k=13 0.2116
  - ablation P11-P12 no search: P12 signals geo alpha high k=13 0.2116
  - ablation P7,P11-P12 no search: P12 signals geo alpha high k=13 0.2116
  - ablation menu, held-out greedy only: held-out greedy k=7 0.2141
  - ablation menu, held-out top-k only: held-out top-k k=7 0.2041
  - ablation full pool vs held-out greedy: held-out greedy k=7 0.2141
  - budget 5: held-out top-k k=5 0.1946
  - budget 10: held-out top-k k=7 0.2041
  - budget 20: held-out top-k k=7 0.2041
  - baseline best member on V: k=1  0.1406 diff (-0.061986491084098816, -0.08487202972173691, -0.040382593870162964)
  - baseline top-k on V: k=10  0.2148 diff (0.012188583612442017, 0.0020513669587671757, 0.021915405988693237)
  - baseline greedy on V: k=8  0.2133 diff (0.010640725493431091, 0.0014928955351933837, 0.0190828088670969)
  - baseline Caruana (bagged): k=75  0.2182 diff (0.015545710921287537, 0.008007770404219627, 0.021731631830334663)
  - baseline log-odds weighted vote: k=100  0.1997 diff (-0.0029438138008117676, -0.006888567935675383, -0.00037417953717522323)
  - baseline linear-pool stacking: k=46  0.2110 diff (0.00838981568813324, -0.0028087389655411243, 0.018447525799274445)
  - baseline proposed paths by balanced accuracy: k=13 P12 signals geo alpha high 0.2116 diff (0.009031802415847778, 0.004422888625413179, 0.013766325078904629)
  - baseline proposed paths by NLL: k=94 P12 signals geo alpha high 0.2035 diff (0.0008331984281539917, 0.0001769468217389658, 0.001546435640193522)
  - baseline proposed paths by BRIER: k=13 P12 signals geo alpha high 0.2116 diff (0.009031802415847778, 0.004422888625413179, 0.013766325078904629)
- inductive: held-out top-k k=7 0.2190; transductive 0.2190; full 0.2205 (eval half, 14 domains, 7+7)
- vote decider held-out top-k k=12 0.2109 (full 0.1976)

## RxRx1 / mixed (rxrx1_mixed_seed_3)
- greedy ref in-sample k=33 0.2879 (full 0.2750); conf best 0.2799
- cross-fitted greedy k=[30, 24] score 0.2824 (full halves 0.2753); gain (0.0070948004722595215, 0.0038636464159935713, 0.010467138141393661); greedy-topk (0.0025072842836380005, -0.0010693870717659593, 0.006324244197458029)
- search: k=5: greedy 0.2713 best 0.2713 (rand 0.27130561132850567, swap 0.2713, back 0.27130561132850567); k=10: greedy 0.2810 best 0.2843 (rand 0.2843291163087215, swap 0.2816, back 0.2810993665497254); k=20: greedy 0.2849 best 0.2901 (rand 0.290095123544427, swap 0.2856, back 0.2867649639174839); k=33: greedy 0.2876 best 0.2895 (rand 0.28892151920345677, swap 0.2895, back 0.2876780828833023); k=40: greedy 0.2866 best 0.2894 (rand 0.2894087451074416, swap 0.2879, back 0.2868626197329236)
- gain cf +0.0075 (k [28, 36]), best member share cf 0.10147694498300552, certificate B(26) = 0.6910; B by k {'1': 0.0436222106218338, '2': 0.08707016706466675, '5': 0.20717646181583405, '10': 0.387346088886261, '20': 0.6363697052001953, '40': 0.6796345710754395}; mean D on H 32.006595611572266
- soft cf: vote 0.2785 k [42, 21] full 0.2732; wvote 0.2780 k [47, 15] full 0.2732; avg 0.2824 k [30, 24] full 0.2753
- P7 oracle cf: k [51, 44] 0.2818
- gain in-sample k=26 +0.0139 bound 0.8487 (ties as errors 0.4243) headroom 0.3909 minority-right 0.3879 best member 0.1106
- q bar 0.1145 q tilde 0.0689 corrected 0.09392880072656867 ece 0.12590546171516734
- mu<0 split 0.8107874691486359 plain 0.8480; err align wrong 0.2095097609045143
- decider P12 signals geo k=15 eps 0.0233 -> 0.2752 (full 0.2750); diff (0.00017920136451721191, -0.0039216941222548485, 0.004174032248556614); uniform prior P12 signals geo k=15 0.2752; conservative None
- transfer decider {'reference': 'full pool', 'dis_V': 0.3647249639034271, 'a_V': -0.010294947622346829, 'dis_T': 0.27851998805999756, 'a_T': 0.008237690972630912, 'gamma': 0.01853263859497774, 'regret': 0.0022943615913391113}; in-sample greedy {'reference': 'full pool', 'dis_V': 0.42074283957481384, 'a_V': -0.033526359336629744, 'dis_T': 0.32246166467666626, 'a_T': 0.020084624628166053, 'gamma': 0.0536109839647958, 'regret': 0.0064765214920043945} test 0.2689
  - ablation P1-P7: P3 agreement k=30 0.2759
  - ablation P1-P8: P3 agreement k=30 0.2759
  - ablation P1-P7,P9-P10: P3 agreement k=30 0.2759
  - ablation P1-P10: P3 agreement k=30 0.2759
  - ablation P1-P12: P3 agreement k=30 0.2759
  - ablation P9-P12: P12 signals geo k=15 0.2752
  - ablation P11-P12: P12 signals geo k=15 0.2752
  - ablation P7,P11-P12: P12 signals geo k=15 0.2752
  - ablation P9-P12 no search: P12 signals geo k=15 0.2752
  - ablation P11-P12 no search: P12 signals geo k=15 0.2752
  - ablation P7,P11-P12 no search: P12 signals geo k=15 0.2752
  - ablation menu, held-out greedy only: P12 signals geo k=15 0.2752
  - ablation menu, held-out top-k only: P12 signals geo k=15 0.2752
  - ablation full pool vs held-out greedy: full pool k=100 0.2750
  - budget 5: P12 signals geo k=5 0.2521
  - budget 10: P11 signals mean k=10 0.2673
  - budget 20: P12 signals geo k=15 0.2752
  - baseline best member on V: k=1  0.2338 diff (-0.04120156168937683, -0.051333967596292496, -0.03191923350095749)
  - baseline top-k on V: k=34  0.2731 diff (-0.0019414424896240234, -0.008398047648370266, 0.003990215715020895)
  - baseline greedy on V: k=16  0.2689 diff (-0.006115317344665527, -0.013295742683112621, 0.0005105296731926501)
  - baseline Caruana (bagged): k=82  0.2781 diff (0.003117471933364868, -0.0021087229251861572, 0.007363998331129551)
  - baseline log-odds weighted vote: k=100  0.2733 diff (-0.0016703903675079346, -0.005174900405108929, 0.0011632872046902776)
  - baseline linear-pool stacking: k=54  0.2739 diff (-0.001135110855102539, -0.007959473878145218, 0.004687420558184385)
  - baseline proposed paths by balanced accuracy: k=15 P12 signals geo 0.2752 diff (0.00017920136451721191, -0.0039216941222548485, 0.004174032248556614)
  - baseline proposed paths by NLL: k=100 P7 collision-aware 0.2750 diff (0.0, 0.0, 0.0)
  - baseline proposed paths by BRIER: k=49 P12 signals geo 0.2796 diff (0.004653722047805786, 0.0016499608755111694, 0.0075066182762384415)
- inductive: P12 signals geo k=43 0.3032; transductive 0.2969; full 0.2975 (eval half, 14 domains, 7+7)
- vote decider P11 signals mean k=47 0.2804 (full 0.2730)

## regime map
{
 "main": {
  "n_cells": 84,
  "all": {
   "share-recovered": 0.2762377240052648,
   "q-share": 0.7171526970745091,
   "share-k": -0.4504080355231845,
   "q-k": -0.7718417123079266,
   "K-k": 0.5987131116105047
  },
  "kather": {
   "share-recovered": {
    "est": 0.5494505494505494,
    "lo": 0.4000000000000001,
    "hi": 0.7972350230414745,
    "per_pool": [
     0.7,
     0.39999999999999997
    ]
   },
   "q-share": {
    "est": 0.1043956043956044,
    "lo": -0.1472972972972972,
    "hi": 0.5,
    "per_pool": [
     -0.09999999999999999,
     -0.19999999999999998
    ]
   },
   "share-k": {
    "est": 0.008253102718210942,
    "lo": -0.5,
    "hi": 0.30000000000000004,
    "per_pool": [
     0.09999999999999999,
     0.3
    ]
   },
   "q-k": {
    "est": -0.7207709707237556,
    "lo": -1.0,
    "hi": -0.6540222751870066,
    "per_pool": [
     -0.9999999999999999,
     -0.7
    ]
   },
   "K-k": {
    "est": 0.7100469468046932,
    "lo": 0.5270462766947299,
    "hi": 1.0,
    "per_pool": [
     0.9486832980505137,
     0.5270462766947298
    ]
   }
  },
  "iwildcam": {
   "share-recovered": {
    "est": -0.13207134259765838,
    "lo": -0.4545454545454545,
    "hi": -0.027472527472527476,
    "per_pool": [
     -0.4545454545454546,
     -0.18131868131868134,
     -0.027472527472527472
    ]
   },
   "q-share": {
    "est": 0.616293703390043,
    "lo": 0.5714285714285715,
    "hi": 0.8531468531468531,
    "per_pool": [
     0.8531468531468532,
     0.5714285714285714,
     0.5714285714285714
    ]
   },
   "share-k": {
    "est": -0.20422458626494108,
    "lo": -0.2802197802197803,
    "hi": -0.016483516483516484,
    "per_pool": [
     -0.14010529366766927,
     -0.2802197802197803,
     -0.016483516483516484
    ]
   },
   "q-k": {
    "est": -0.41295901612747477,
    "lo": -0.5111280160510597,
    "hi": -0.2626974256268798,
    "per_pool": [
     -0.26269742562687987,
     -0.510989010989011,
     -0.33516483516483514
    ]
   },
   "K-k": {
    "est": 0.10912379729639557,
    "lo": 0.03684853277181696,
    "hi": 0.32173150483511703,
    "per_pool": [
     0.03943564347417965,
     0.3217315048351169,
     0.05875097044815179
    ]
   }
  },
  "rxrx1": {
   "share-recovered": {
    "est": -0.26585094549499444,
    "lo": -0.305685618729097,
    "hi": 0.0787878787878788,
    "per_pool": [
     -0.29696969696969694,
     -0.19999999999999998,
     0.07878787878787878
    ]
   },
   "q-share": {
    "est": 0.9176863181312569,
    "lo": 0.8424242424242426,
    "hi": 1.0,
    "per_pool": [
     0.9515151515151514,
     0.8424242424242423,
     0.9999999999999999
    ]
   },
   "share-k": {
    "est": -0.5201913481961158,
    "lo": -0.7333333333333334,
    "hi": -0.4181818181818182,
    "per_pool": [
     -0.41818181818181815,
     -0.6,
     -0.7333333333333332
    ]
   },
   "q-k": {
    "est": -0.47947491675048315,
    "lo": -0.7333333333333334,
    "hi": -0.3454545454545455,
    "per_pool": [
     -0.4666666666666666,
     -0.34545454545454546,
     -0.7333333333333332
    ]
   },
   "K-k": {
    "est": 0.478336873145101,
    "lo": 0.37133366509093585,
    "hi": 0.6365441169421684,
    "per_pool": [
     0.5349722293682974,
     0.3713336650909358,
     0.7615487029831055
    ]
   }
  }
 },
 "alt": {
  "n_cells": 84,
  "all": {
   "share-recovered": 0.3092639465424724,
   "q-share": 0.6821845133425186,
   "share-k": -0.3385814817278581,
   "q-k": -0.701932910421743,
   "K-k": 0.4605701675745749
  },
  "kather": {
   "share-recovered": {
    "est": 0.38461538461538464,
    "lo": -0.5,
    "hi": 0.45585585585585575,
    "per_pool": [
     0.3,
     0.39999999999999997
    ]
   },
   "q-share": {
    "est": 0.32967032967032966,
    "lo": -0.30000000000000004,
    "hi": 1.0,
    "per_pool": [
     -0.3,
     0.8999999999999998
    ]
   },
   "share-k": {
    "est": 0.20329670329670327,
    "lo": -0.30000000000000004,
    "hi": 0.6000000000000001,
    "per_pool": [
     -0.3,
     0.6
    ]
   },
   "q-k": {
    "est": 0.15934065934065936,
    "lo": -0.20000000000000004,
    "hi": 0.7000000000000002,
    "per_pool": [
     -0.19999999999999998,
     0.7
    ]
   },
   "K-k": {
    "est": 0.8614332868494854,
    "lo": 0.785589258975166,
    "hi": 1.0,
    "per_pool": [
     0.7905694150420949,
     0.9486832980505137
    ]
   }
  },
  "iwildcam": {
   "share-recovered": {
    "est": -0.040376408797461426,
    "lo": -0.15645773979107308,
    "hi": 0.3956043956043957,
    "per_pool": [
     -0.04195804195804196,
     -0.03296703296703297,
     0.39560439560439564
    ]
   },
   "q-share": {
    "est": 0.2417113469745049,
    "lo": -0.06593406593406594,
    "hi": 0.5051501205347358,
    "per_pool": [
     0.27272727272727276,
     0.46153846153846156,
     -0.06593406593406594
    ]
   },
   "share-k": {
    "est": 0.15366935397146084,
    "lo": -0.20329670329670335,
    "hi": 0.4951861630926565,
    "per_pool": [
     0.22377622377622378,
     -0.20329670329670327,
     0.49518616309265656
    ]
   },
   "q-k": {
    "est": -0.40617305739892534,
    "lo": -0.533218467212833,
    "hi": 0.027972027972027972,
    "per_pool": [
     0.027972027972027972,
     -0.4450549450549451,
     -0.2861075608979793
    ]
   },
   "K-k": {
    "est": -0.17315256939546803,
    "lo": -0.4650516796635759,
    "hi": 0.14268092823122583,
    "per_pool": [
     -0.14673020240898965,
     0.1426809282312258,
     -0.4650516796635759
    ]
   }
  },
  "rxrx1": {
   "share-recovered": {
    "est": 0.24449388209121245,
    "lo": -0.04242424242424243,
    "hi": 0.4024526198439242,
    "per_pool": [
     0.1515151515151515,
     -0.04242424242424241,
     -0.01818181818181818
    ]
   },
   "q-share": {
    "est": 0.2921023359288098,
    "lo": -0.44242424242424244,
    "hi": 0.3128205128205129,
    "per_pool": [
     -0.4424242424242424,
     0.24848484848484845,
     -0.32121212121212117
    ]
   },
   "share-k": {
    "est": -0.2413793103448276,
    "lo": -0.2486064659977704,
    "hi": 0.04242424242424243,
    "per_pool": [
     -0.0303030303030303,
     0.0303030303030303,
     0.04242424242424241
    ]
   },
   "q-k": {
    "est": -0.5265850945494994,
    "lo": -0.5375696767001116,
    "hi": 0.0787878787878788,
    "per_pool": [
     0.07878787878787878,
     -0.28484848484848485,
     0.0303030303030303
    ]
   },
   "K-k": {
    "est": 0.2889195341304711,
    "lo": 0.16518126189048865,
    "hi": 0.5349722293682974,
    "per_pool": [
     0.40909641069340386,
     0.5349722293682974,
     0.18251993707859557
    ]
   }
  }
 }
}