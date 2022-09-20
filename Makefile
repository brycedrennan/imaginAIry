SHELL := /bin/bash
python_version = 3.10.6
venv_prefix = imaginairy
venv_name = $(venv_prefix)-$(python_version)
pyenv_instructions=https://github.com/pyenv/pyenv#installation
pyenv_virt_instructions=https://github.com/pyenv/pyenv-virtualenv#pyenv-virtualenv


init: require_pyenv  ## Setup a dev environment for local development.
	@pyenv install $(python_version) -s
	@echo -e "\033[0;32m ✔️  🐍 $(python_version) installed \033[0m"
	@if ! [ -d "$$(pyenv root)/versions/$(venv_name)" ]; then\
		pyenv virtualenv $(python_version) $(venv_name);\
	fi;
	@pyenv local $(venv_name)
	@echo -e "\033[0;32m ✔️  🐍 $(venv_name) virtualenv activated \033[0m"
	pip install --upgrade pip pip-tools
	pip-sync requirements-dev.txt
	pip install -e . --no-deps
	@echo -e "\nEnvironment setup! ✨ 🍰 ✨ 🐍 \n\nCopy this path to tell PyCharm where your virtualenv is. You may have to click the refresh button in the pycharm file explorer.\n"
	@echo -e "\033[0;32m"
	@pyenv which python
	@echo -e "\n\033[0m"
	@echo -e "The following commands are available to run in the Makefile\n"
	@make -s help

af: autoformat  ## Alias for `autoformat`
autoformat:  ## Run the autoformatter.
	@pycln . --all
	@isort --atomic --profile black .
	@black .

test:  ## Run the tests.
	@pytest
	@echo -e "The tests pass! ✨ 🍰 ✨"

lint:  ## Run the code linter.
	@pylama
	@echo -e "No linting errors - well done! ✨ 🍰 ✨"

deploy:  ## Deploy the package to pypi.org
	pip install twine wheel
	-git tag $$(python setup.py -V)
	git push --tags
	rm -rf dist
	python setup.py bdist_wheel
	#python setup.py sdist
	@echo 'pypi.org Username: '
	@read username && twine upload dist/* -u $$username;
	rm -rf build
	rm -rf dist
	@echo "Deploy successful! ✨ 🍰 ✨"

requirements:  ## Freeze the requirements.txt file
	pip-compile setup.py requirements-dev.in --output-file=requirements-dev.txt --upgrade

require_pyenv:
	@if ! [ -x "$$(command -v pyenv)" ]; then\
	  echo -e '\n\033[0;31m ❌ pyenv is not installed.  Follow instructions here: $(pyenv_instructions)\n\033[0m';\
	  exit 1;\
	else\
	  echo -e "\033[0;32m ✔️  pyenv installed\033[0m";\
	fi
	@if ! [[ "$$(pyenv virtualenv --version)" == *"pyenv-virtualenv"* ]]; then\
	  echo -e '\n\033[0;31m ❌ pyenv virtualenv is not installed.  Follow instructions here: $(pyenv_virt_instructions) \n\033[0m';\
	  exit 1;\
	else\
	  echo -e "\033[0;32m ✔️  pyenv-virtualenv installed\033[0m";\
	fi

vendor_openai_clip:
	mkdir -p ./downloads
	-cd ./downloads && git clone git@github.com:openai/CLIP.git
	cd ./downloads/CLIP && git pull
	rm -rf ./imaginairy/vendored/clip
	cp -R ./downloads/CLIP/clip imaginairy/vendored/
	git --git-dir ./downloads/CLIP/.git rev-parse HEAD | tee ./imaginairy/vendored/clip/clip-commit-hash.txt
	echo "vendored from git@github.com:openai/CLIP.git" | tee ./imaginairy/vendored/clip/readme.txt

revendorize: vendorize_kdiffusion
	make vendorize REPO=git@github.com:openai/CLIP.git PKG=clip COMMIT=d50d76daa670286dd6cacf3bcd80b5e4823fc8e1

	make af

vendorize_clipseg:
	make download_repo REPO=git@github.com:timojl/clipseg.git PKG=clipseg COMMIT=664ee94393491cdd7ad422f67eb1ce670d3d00e6
	rm -rf ./imaginairy/vendored/clipseg
	mkdir -p ./imaginairy/vendored/clipseg
	cp -R ./downloads/clipseg/models/* ./imaginairy/vendored/clipseg/
	sed -i '' -e 's#import clip#from imaginairy.vendored import clip#g' ./imaginairy/vendored/clipseg/clipseg.py
	rm ./imaginairy/vendored/clipseg/vitseg.py
	mv ./imaginairy/vendored/clipseg/clipseg.py ./imaginairy/vendored/clipseg/__init__.py
	wget https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth -P ./imaginairy/vendored/clipseg

vendorize_blip:
	make download_repo REPO=git@github.com:salesforce/BLIP.git PKG=blip COMMIT=48211a1594f1321b00f14c9f7a5b4813144b2fb9
	rm -rf ./imaginairy/vendored/blip
	mkdir -p ./imaginairy/vendored/blip
	cp -R ./downloads/blip/models/* ./imaginairy/vendored/blip/
	cp -R ./downloads/blip/configs ./imaginairy/vendored/blip/
	sed -i '' -e 's#from models\.#from imaginairy.vendored.blip.#g' ./imaginairy/vendored/blip/blip.py
	sed -i '' -e 's#print(#\# print(#g' ./imaginairy/vendored/blip/blip.py

vendorize_kdiffusion:
	make vendorize REPO=git@github.com:crowsonkb/k-diffusion.git PKG=k_diffusion COMMIT=1a0703dfb7d24d8806267c3e7ccc4caf67fd1331
	#sed -i '' -e 's/import\sclip/from\simaginairy.vendored\simport\sclip/g' imaginairy/vendored/k_diffusion/evaluation.py
	rm imaginairy/vendored/k_diffusion/evaluation.py
	touch imaginairy/vendored/k_diffusion/evaluation.py
	rm imaginairy/vendored/k_diffusion/config.py
	touch imaginairy/vendored/k_diffusion/config.py
	# without this most of the k-diffusion samplers didn't work
	sed -i '' -e 's#return (x - denoised) / utils.append_dims(sigma, x.ndim)#return (x - denoised) / sigma#g' imaginairy/vendored/k_diffusion/sampling.py
	sed -i '' -e 's#x = x + torch.randn_like(x) \* sigma_up#x = x + torch.randn_like(x, device="cpu").to(x.device) \* sigma_up#g' imaginairy/vendored/k_diffusion/sampling.py
	make af

vendorize:  ## vendorize a github repo.  `make vendorize REPO=git@github.com:openai/CLIP.git PKG=clip`
	mkdir -p ./downloads
	-cd ./downloads && git clone $(REPO) $(PKG)
	cd ./downloads/$(PKG) && git fetch && git checkout $(COMMIT)
	rm -rf ./imaginairy/vendored/$(PKG)
	cp -R ./downloads/$(PKG)/$(PKG) imaginairy/vendored/
	git --git-dir ./downloads/$(PKG)/.git rev-parse HEAD | tee ./imaginairy/vendored/$(PKG)/clip-commit-hash.txt
	touch ./imaginairy/vendored/$(PKG)/version.py
	echo "vendored from $(REPO)" | tee ./imaginairy/vendored/$(PKG)/readme.txt

download_repo:
	mkdir -p ./downloads
	-cd ./downloads && git clone $(REPO) $(PKG)
	cd ./downloads/$(PKG) && git pull

vendorize_whole_repo:
	mkdir -p ./downloads
	-cd ./downloads && git clone $(REPO) $(PKG)
	cd ./downloads/$(PKG) && git pull
	rm -rf ./imaginairy/vendored/$(PKG)
	cp -R ./downloads/$(PKG) imaginairy/vendored/
	git --git-dir ./downloads/$(PKG)/.git rev-parse HEAD | tee ./imaginairy/vendored/$(PKG)/clip-commit-hash.txt
	touch ./imaginairy/vendored/$(PKG)/version.py
	echo "vendored from $(REPO)" | tee ./imaginairy/vendored/$(PKG)/readme.txt


help: ## Show this help message.
	@## https://gist.github.com/prwhite/8168133#gistcomment-1716694
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)" | sort