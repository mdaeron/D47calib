all: calibs combine doc
	@echo "All done!"

combine: savecalibs
	@echo "Building Devils Laghetto (2023)..."
	@cd ./build_calibs/build_devils_laghetto_2023; python build_devils_laghetto_2023.py
	@cd ./build_calibs; python save_to_calibs.py
	@echo "Combining calibrations..."
	@cd ./build_calibs; python combine_calibs.py

calibs:
	@echo "Processing Breitenbach et al. (2018)..."
	@cd ./build_calibs/build_breitenbach_2018; python build_breitenbach_2018.py
	@echo "Processing Peral et al. (2018)..."
	@cd ./build_calibs/build_peral_2018; python build_peral_2018.py
	@echo "Processing Jautzy et al. (2020)..."
	@cd ./build_calibs/build_jautzy_2020; python build_jautzy_2020.py
	@echo "Processing Anderson et al. (2021)..."
	@cd ./build_calibs/build_anderson_2021_mit; python build_anderson_2021_mit.py
	@cd ./build_calibs/build_anderson_2021_lsce; python build_anderson_2021_lsce.py
	@echo "Processing Fiebig et al. (2021)..."
	@cd ./build_calibs/build_fiebig_2021; python build_fiebig_2021.py
	@echo "Processing Huyghe et al. (2022)..."
	@cd ./build_calibs/build_huyghe_2022; python build_huyghe_2022.py

savecalibs:
	@echo "Saving calibrations..."
	@cd ./build_calibs; python save_to_calibs.py

doc:
	@echo "Building documentation..."
	@python build_doc.py
