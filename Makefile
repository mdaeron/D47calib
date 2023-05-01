build:
	cd ./build_calibs/jautzy_2020; python build_jautzy_2020.py
	cd ./build_calibs/anderson_2021_mit; python build_anderson_2021_mit.py
	cd ./build_calibs/anderson_2021_lsce; python build_anderson_2021_lsce.py
	cd ./build_calibs/peral_2018; python build_peral_2018.py
	cd ./build_calibs; python save_to_calibs.py