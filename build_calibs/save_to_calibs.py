with open('../src/D47calib/calibs.py', 'w') as fid:

	fid.write("""'''
.. include:: ../../doc/calibs.md
'''

from . import D47calib
""")

	for file in [
		'breitenbach_2018.py',
		'peral_2018.py',
		'jautzy_2020.py',
		'anderson_2021_mit.py',
		'anderson_2021_lsce.py',
		'fiebig_2021.py',
		'huyghe_2022.py',
		'devils_laghetto_2023.py',
		]:
	
		with open(file) as gid:
			fid.write(gid.read())