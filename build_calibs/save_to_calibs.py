with open('../src/D47calib/calibs.py', 'w') as fid:

	fid.write('from . import D47calib\n')

	for file in [
		'peral_2018.py',
		'jautzy_2020.py',
		'anderson_2021_mit.py',
		'anderson_2021_lsce.py',
		]:
	
		with open(file) as gid:
			fid.write(gid.read())