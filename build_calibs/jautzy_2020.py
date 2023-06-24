
jautzy_2020 = D47calib(
	samples = ['CA5_05', 'CA5_07', 'CA25', 'CA50', 'CA70', 'CA90', 'CA120', 'CA170', 'CA200', 'CA250_06', 'CA250_09', 'CA726'],
	T = [5.0, 5.0, 25.0, 50.0, 70.0, 90.0, 120.0, 170.0, 200.0, 250.0, 250.0, 726.85],
	D47 = [0.6615571376944879, 0.6803947266163454, 0.6084615459291493, 0.5437495489279841, 0.4883903706975946, 0.4453680582992847, 0.40622015185829574, 0.3447184594574459, 0.32363679899349485, 0.28513110576847217, 0.2960435511889537, 0.19314439156331145],
	sT = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]],
	sD47 = [[0.00015228711253293062, 5.8155388481008025e-05, 4.2810457515007255e-05, 4.679508053850865e-05, 0.0, 4.121182414165165e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5.815538848100824e-05, 0.00016471503927422172, 6.641774522381374e-05, 4.2119019520729754e-05, 0.0, 1.0922868020471207e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.281045751500747e-05, 6.6417745223814e-05, 0.00013011568228814613, 3.592588681695319e-05, 0.0, 1.2215630176775208e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.679508053850872e-05, 4.211901952072967e-05, 3.59258868169531e-05, 0.00011416546796168953, 0.0, 6.446997892028058e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 7.804518496793538e-05, 0.0, 3.429141843267716e-08, 1.2630202963022846e-06, 1.2717350685030508e-05, 8.729523789681191e-06, 7.754991298370811e-06, 0.0], [4.121182414165219e-06, 1.0922868020471305e-05, 1.2215630176775252e-05, 6.446997892028069e-06, 0.0, 0.00012363336233255616, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.4291418432677195e-08, 0.0, 9.838356008255752e-05, 2.963450658143165e-06, 3.0073903019192047e-07, 2.0886325129735805e-08, 8.345484658212321e-09, 0.0], [0.0, 0.0, 0.0, 0.0, 1.263020296302286e-06, 0.0, 2.963450658143165e-06, 0.00010914970856844108, 1.1076809195524627e-05, 7.69284379583632e-07, 3.0738059221712363e-07, 0.0], [0.0, 0.0, 0.0, 0.0, 1.2717350685030518e-05, 0.0, 3.007390301919204e-07, 1.1076809195524627e-05, 0.00011153238583977817, 7.745924004802925e-06, 3.09501501792331e-06, 0.0], [0.0, 0.0, 0.0, 0.0, 8.729523789681176e-06, 0.0, 2.0886325129735792e-08, 7.692843795836314e-07, 7.745924004802921e-06, 8.786793428553773e-05, 2.60322423926731e-05, 0.0], [0.0, 0.0, 0.0, 0.0, 7.754991298370832e-06, 0.0, 8.345484658212311e-09, 3.073805922171233e-07, 3.095015017923307e-06, 2.6032242392673032e-05, 9.934869933087751e-05, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0993090435641564e-05]],
	degrees = [0, 1, 2],
	description = 'Jautzy et al. (2020) synthetics, reprocessed in I-CDES',
	label = 'Jautzy et al. (2020)',
	bfp = {'a0': 0.169653713997062, 'a1': -22.79086032625165, 'a2': 45285.43587564845},
	bfp_CM = [[0.0005433353263790602, -0.4972680025692366, 103.6912515684891], [-0.49726800256923687, 493.3613007289042, -108145.82006604537], [103.69125156848916, -108145.8200660454, 24646712.416698255]],
	chisq = 4.319253342413334,
	cholesky_residuals = [-1.4088274316539071, 0.2457407175136937, 0.2976333157608978, 0.9524670124847976, 0.08372061014281754, -0.42239968777311254, 0.16176067233059732, -0.38743994888283584, 0.03472737567197525, -0.8359346871948038, 0.44771121946295034, 0.12706736662051882],
	aic = -6.261889198492895,
	bic = -4.8071692491288935,
	ks_pvalue = 0.40772853967983036,
	)
