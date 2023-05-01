with open('../src/D47calib/calibs.py', 'w') as fid:

	fid.write('from . import D47calib\n')

	for file in [
		'breitenbach_2018.py',
		'peral_2018.py',
		'jautzy_2020.py',
		'anderson_2021_mit.py',
		'anderson_2021_lsce.py',
		'fiebig_2021.py',
		'huyghe_2022.py',
		]:
	
		with open(file) as gid:
			fid.write(gid.read())