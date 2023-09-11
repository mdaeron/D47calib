
fiebig_2021 = D47calib(
	samples = ['CA-120', 'CA-170', 'CA-200', 'CA-250A', 'CA-250B', 'CM-351', 'DHC2-8', 'DVH-2', 'ETH-1-1100', 'ETH-2-1100', 'LGB-2'],
	T = [120.0, 170.0, 200.0, 250.0, 250.0, 726.85, 33.7, 33.7, 1100.0, 1100.0, 7.9],
	D47 = [0.39883243587910994, 0.3400534383476339, 0.32226474895341267, 0.2922115683405872, 0.29266668453870354, 0.20231837163774677, 0.5687265043550249, 0.5669559645368026, 0.1807813135466912, 0.18345001629731184, 0.6481174772171734],
	sT = [[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04000000000000001, 0.04000000000000001, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04000000000000001, 0.04000000000000001, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04000000000000001]],
	sD47 = [[2.4578745312908135e-05, 3.947700897668114e-06, 4.08857422750312e-06, 4.220537448599231e-06, 4.117114349538624e-06, 8.003134601449757e-07, 0.0, 0.0, 0.0, 0.0, 0.0], [3.947700897668114e-06, 2.0803084846723228e-05, 4.391958855992876e-06, 4.596817776703999e-06, 4.464553102168207e-06, 7.097994221875508e-07, 0.0, 0.0, 0.0, 0.0, 0.0], [4.088574227503116e-06, 4.391958855992875e-06, 1.8473901953248506e-05, 4.877946023928402e-06, 4.725892971855741e-06, 6.468611759860959e-07, 0.0, 0.0, 0.0, 0.0, 0.0], [4.220537448599227e-06, 4.596817776703995e-06, 4.8779460239284e-06, 2.1788245136796487e-05, 5.002151504013243e-06, 5.781396086974895e-07, 0.0, 0.0, 0.0, 0.0, 0.0], [4.11711434953862e-06, 4.464553102168203e-06, 4.725892971855741e-06, 5.002151504013243e-06, 1.8683484270007545e-05, 6.151545223139814e-07, 0.0, 0.0, 0.0, 0.0, 0.0], [8.003134601449745e-07, 7.097994221875492e-07, 6.468611759860949e-07, 5.781396086974899e-07, 6.151545223139813e-07, 6.143911193320136e-06, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.874903216489722e-06, 1.537242890923763e-06, 8.147654783321116e-08, 1.6216860102893624e-07, 1.6373448967404438e-06], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5372428909237627e-06, 1.288953806399494e-05, 1.9726627574400908e-07, 3.926331787286959e-07, 3.977062522013425e-06], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.147654783321128e-08, 1.972662757440094e-07, 1.1564350857095249e-05, 1.7713435889227819e-06, 1.173842554678134e-07], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6216860102893624e-07, 3.9263317872869586e-07, 1.7713435889227785e-06, 1.1091835733087883e-05, 2.336382799502845e-07], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6373448967404438e-06, 3.977062522013428e-06, 1.1738425546781319e-07, 2.336382799502845e-07, 1.4250289035965501e-05]],
	degrees = [0, 1, 2],
	description = 'Fiebig et al. (2021) in I-CDES with additional constraints from 100 °C and 25 °C equilibrated CO2',
	name = '',
	label = 'Fiebig et al. (2021)',
	bfp = {'a0': 0.18037085040734538, 'a1': -26.077332386262295, 'a2': 44427.281642756694},
	bfp_CM = [[3.784481663390472e-05, -0.045772850616032486, 10.531093448849356], [-0.04577285061603251, 60.81615769881044, -14456.276763941229], [10.531093448849365, -14456.276763941232, 3503086.4192256723]],
	chisq = 7.322417423430449,
	cholesky_residuals = [-0.3540505604683415, -1.626248807595709, -0.3539953498076503, -0.1782298452194647, -0.11827129485538806, 1.406976116901386, 0.7749863075352663, 0.09301820013896483, -1.1628499459650996, -0.4381554590839405, -0.49827261227729314],
	aic = 1.5234977473285216,
	bic = 2.7171835657236336,
	ks_pvalue = 0.29156667618600707,
	)
