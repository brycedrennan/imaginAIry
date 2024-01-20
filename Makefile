SHELL := /bin/bash
python_version = 3.10.13
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
	pip install -e .
	# the compiled requirements don't included OS specific subdependencies so we trigger those this way
	#pip install `pip freeze | grep "^torch=="`
	@echo -e "\nEnvironment setup! ✨ 🍰 ✨ 🐍 \n\nCopy this path to tell PyCharm where your virtualenv is. You may have to click the refresh button in the pycharm file explorer.\n"
	@echo -e "\033[0;32m"
	@pyenv which python
	@echo -e "\n\033[0m"
	@echo -e "The following commands are available to run in the Makefile\n"
	@make -s help

af: autoformat  ## Alias for `autoformat`
autoformat:  ## Run the autoformatter.
	@-ruff check --config tests/ruff.toml . --fix-only
	@ruff format --config tests/ruff.toml .

test:  ## Run the tests.
	@pytest
	@echo -e "The tests pass! ✨ 🍰 ✨"

test-fast:  ## Run the fast tests.
	@pytest -m "not gputest"
	@echo -e "The non-gpu tests pass! ✨ 🍰 ✨"

lint:  ## Run the code linter.
	@ruff check --config tests/ruff.toml .
	@echo -e "No linting errors - well done! ✨ 🍰 ✨"

type-check: ## Run the type checker.
	@mypy --config-file tox.ini .

check-fast:  ## Run autoformatter, linter, typechecker, and fast tests
	@make autoformat
	@make lint
	@make type-check
	@make test-fast

build-pkg:  ## Build the package
	python setup.py sdist bdist_wheel
	python setup.py bdist_wheel --plat-name=win-amd64

deploy:  ## Deploy the package to pypi.org
	pip install twine wheel
	-git tag $$(python setup.py -V)
	git push --tags
	rm -rf dist
	make build-pkg
	#python setup.py sdist
	@twine upload --verbose dist/* -u __token__;
	rm -rf build
	rm -rf dist
	@echo "Deploy successful! ✨ 🍰 ✨"

build-dev-image:
	docker build -f tests/Dockerfile -t imaginairy-dev .

run-dev: build-dev-image
	docker run -it -v $$HOME/.cache/huggingface:/root/.cache/huggingface -v $$HOME/.cache/torch:/root/.cache/torch -v `pwd`/outputs:/outputs imaginairy-dev /bin/bash

requirements:  ## Freeze the requirements.txt file
	pip-compile setup.py requirements-dev.in --output-file=requirements-dev.txt --upgrade --resolver=backtracking

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

.PHONY: docs

docs:
	mkdocs serve

update-stablestudio:
	@echo "Updating stablestudio"
	cd ../imaginAIry-StableStudio && \
	yarn build && \
	yarn build:production
	rm -rf imaginairy/http/stablestudio/dist
	cp -R ../imaginAIry-StableStudio/packages/stablestudio-ui/dist imaginairy/http/stablestudio/dist
	rm -rf imaginairy/http/stablestudio/dist/examples
	rm -rf imaginairy/http/stablestudio/dist/media
	rm -rf imaginairy/http/stablestudio/dist/presets
	cp ../imaginAIry-StableStudio/LICENSE imaginairy/http/stablestudio/dist/LICENSE
	@echo "Updated stablestudio"

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
	make download_repo REPO=git@github.com:timojl/clipseg.git PKG=clipseg COMMIT=ea54753df1e444c4445bac6e023546b6a41951d8
	rm -rf ./imaginairy/vendored/clipseg
	mkdir -p ./imaginairy/vendored/clipseg
	cp -R ./downloads/clipseg/models/* ./imaginairy/vendored/clipseg/
	sed -i '' -e 's#import clip#from imaginairy.vendored import clip#g' ./imaginairy/vendored/clipseg/clipseg.py
	rm ./imaginairy/vendored/clipseg/vitseg.py
	mv ./imaginairy/vendored/clipseg/clipseg.py ./imaginairy/vendored/clipseg/__init__.py
	# download weights
	rm -rf ./downloads/clipseg-weights
	mkdir -p ./downloads/clipseg-weights
	wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O ./downloads/clipseg-weights/weights.tar
	cd downloads/clipseg-weights && unzip -d weights -j weights.tar
	cp ./downloads/clipseg-weights/weights/rd64-uni-refined.pth ./imaginairy/vendored/clipseg/

vendorize_blip:
	make download_repo REPO=git@github.com:salesforce/BLIP.git PKG=blip COMMIT=48211a1594f1321b00f14c9f7a5b4813144b2fb9
	rm -rf ./imaginairy/vendored/blip
	mkdir -p ./imaginairy/vendored/blip
	cp -R ./downloads/blip/models/* ./imaginairy/vendored/blip/
	cp -R ./downloads/blip/configs ./imaginairy/vendored/blip/
	sed -i '' -e 's#from models\.#from imaginairy.vendored.blip.#g' ./imaginairy/vendored/blip/blip.py
	sed -i '' -e 's#print(#\# print(#g' ./imaginairy/vendored/blip/blip.py

vendorize_kdiffusion:
	rm -rf ./imaginairy/vendored/k_diffusion
	rm -rf ./downloads/k_diffusion
    # version 0.0.9
	make vendorize REPO=git@github.com:crowsonkb/k-diffusion.git PKG=k_diffusion COMMIT=5b3af030dd83e0297272d861c19477735d0317ec
	#sed -i '' -e 's/import\sclip/from\simaginairy.vendored\simport\sclip/g' imaginairy/vendored/k_diffusion/evaluation.py
	mv ./downloads/k_diffusion/LICENSE ./imaginairy/vendored/k_diffusion/
	rm imaginairy/vendored/k_diffusion/evaluation.py
	touch imaginairy/vendored/k_diffusion/evaluation.py
	rm imaginairy/vendored/k_diffusion/config.py
	touch imaginairy/vendored/k_diffusion/config.py
	# without this most of the k-diffusion samplers didn't work
	sed -i '' -e 's#return (x - denoised) / utils.append_dims(sigma, x.ndim)#return (x - denoised) / sigma#g' imaginairy/vendored/k_diffusion/sampling.py
	sed -i '' -e 's#torch.randn_like(x)#torch.randn_like(x, device="cpu").to(x.device)#g' imaginairy/vendored/k_diffusion/sampling.py
 	# https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/4558#issuecomment-1310387114
	sed -i '' -e 's#t_fn = lambda sigma: sigma.log().neg()#t_fn = lambda sigma: sigma.to("cpu").log().neg().to(x.device)#g' imaginairy/vendored/k_diffusion/sampling.py
	sed -i '' -e 's#return (x - denoised) / sigma#return ((x - denoised) / sigma.to("cpu")).to(x.device)#g' imaginairy/vendored/k_diffusion/sampling.py
	sed -i '' -e 's#return t.neg().exp()#return t.to("cpu").neg().exp().to(self.model.device)#g' imaginairy/vendored/k_diffusion/sampling.py
	sed -i '' -e 's#import torchsde##g' imaginairy/vendored/k_diffusion/sampling.py
	sed -i '' -e 's#torch.randint(0, 2\*\*63 - 1, \[\])#torch.randint(0, 2**63 - 1, [], device="cpu")#g' imaginairy/vendored/k_diffusion/sampling.py
	sed -i '' -e 's#torch.randint_like(x, 2)#torch.randint_like(x, 2, device="cpu")#g' imaginairy/vendored/k_diffusion/sampling.py
	make af

vendorize_noodle_soup:
	make download_repo REPO=git@github.com:WASasquatch/noodle-soup-prompts.git PKG=noodle-soup-prompts COMMIT=5642feb4d0e1340b9d145f5ff64f2b57eab1ae71
	mkdir -p ./imaginairy/vendored/noodle_soup_prompts
	rm ./imaginairy/vendored/noodle_soup_prompts/*
	mv ./downloads/noodle-soup-prompts/LICENSE ./imaginairy/vendored/noodle_soup_prompts/
	python scripts/prep_vocab_lists.py
	make af

vendorize_controlnet_annotators:
	make download_repo REPO=git@github.com:lllyasviel/ControlNet-v1-1-nightly.git PKG=controlnet11 COMMIT=b9ae087ef56ca786d9a3ee1008f814bb171bb913
	mkdir -p ./imaginairy/vendored/controlnet_annotators
	rm -rf ./imaginairy/vendored/controlnet_annotators/*
	cp -R ./downloads/controlnet11/annotator/* ./imaginairy/vendored/controlnet_annotators/
	rm -rf ./imaginairy/vendored/controlnet_annotators/canny
	rm -rf ./imaginairy/vendored/controlnet_annotators/ckpts
	#black imaginairy/vendored/controlnet_annotators
	sed -i '' -e 's#from annotator.uniformer.mmseg#from .mmseg#g' imaginairy/vendored/controlnet_annotators/uniformer/__init__.py
	find imaginairy/vendored/controlnet_annotators -type f -name "__init__.py" -exec sed -i '' -e 's#checkpoint_file#remote_model_path#g' {} \;
	find imaginairy/vendored/controlnet_annotators -type f -name "__init__.py" -exec sed -i '' -e 's#modelpath#model_path#g' {} \;
	find imaginairy/vendored/controlnet_annotators -type f -name "__init__.py" -exec sed -i '' -e '/^ *model_path = os.path.join(annotator_ckpts_path, [^)]*/,/^ *load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)/c\'$$'\n''        model_path = get_cached_url_path(remote_model_path)' {} \;
	find imaginairy/vendored/controlnet_annotators -type f -name "__init__.py" -exec sed -i '' -e 's|^ *from annotator.util import annotator_ckpts_path|from imaginairy.model_manager import get_cached_url_path|' {} \;
	find imaginairy/vendored/controlnet_annotators -type f -name "__init__.py" -exec sed -i '' -e 's|^ *from annotator.util import|from imaginairy.vendored.controlnet_annotators.util import|' {} \;
	find imaginairy/vendored/controlnet_annotators -type f -name "*.py" -exec sed -i '' -e 's|^ *from annotator.|from imaginairy.vendored.controlnet_annotators.|' {} \;
	touch imaginairy/vendored/controlnet_annotators/__init__.py
	sed -i '' -e 's#from annotator.uniformer.mmseg#from .mmseg#g' imaginairy/vendored/controlnet_annotators/uniformer/__init__.py
	sed -i '' '11i\'$$'\n''annotator_ckpts_path = os.path.dirname(os.path.dirname(__file__))' imaginairy/vendored/controlnet_annotators/uniformer/__init__.py
	rm ./imaginairy/vendored/controlnet_annotators/oneformer/oneformer/data/bpe_simple_voc*
	rm -rf ./imaginairy/vendored/controlnet_annotators/zoe/zoedepth/models/base_models/midas_repo/mobile
	make af



vendorize_normal_map:
	make download_repo REPO=git@github.com:brycedrennan/imaginairy-normal-map.git PKG=imaginairy_normal_map COMMIT=6b3b1692cbdc21d55c84a01e0b7875df030b6d79
	mkdir -p ./imaginairy/vendored/imaginairy_normal_map
	rm -rf ./imaginairy/vendored/imaginairy_normal_map/*
	cp -R ./downloads/imaginairy_normal_map/imaginairy_normal_map/* ./imaginairy/vendored/imaginairy_normal_map/
	make af


vendorize_refiners:
	export REPO=git@github.com:finegrain-ai/refiners.git PKG=refiners COMMIT=91aea9b7ff63ddf93f99e2ce6a4452bd658b1948 && \
	make download_repo REPO=$$REPO PKG=$$PKG COMMIT=$$COMMIT && \
	mkdir -p ./imaginairy/vendored/$$PKG && \
	rm -rf ./imaginairy/vendored/$$PKG/* && \
	cp -R ./downloads/refiners/src/refiners/* ./imaginairy/vendored/$$PKG/ && \
	cp ./downloads/refiners/LICENSE ./imaginairy/vendored/$$PKG/ && \
	rm -rf ./imaginairy/vendored/$$PKG/training_utils && \
	echo "vendored from $$REPO @ $$COMMIT" | tee ./imaginairy/vendored/$$PKG/readme.txt
	find ./imaginairy/vendored/refiners/ -type f -name "*.py" -exec sed -i '' 's/from refiners/from imaginairy.vendored.refiners/g' {} + &&\
    find ./imaginairy/vendored/refiners/ -type f -name "*.py" -exec sed -i '' 's/import refiners/import imaginairy.vendored.refiners/g' {} + &&\
	make af

vendorize_facexlib:
	export REPO=git@github.com:xinntao/facexlib.git PKG=facexlib COMMIT=260620ae93990a300f4b16448df9bb459f1caba9 && \
	make download_repo REPO=$$REPO PKG=$$PKG COMMIT=$$COMMIT && \
	mkdir -p ./imaginairy/vendored/$$PKG && \
	rm -rf ./imaginairy/vendored/$$PKG/* && \
	cp -R ./downloads/$$PKG/facexlib/* ./imaginairy/vendored/$$PKG/ && \
	rm -rf ./imaginairy/vendored/$$PKG/weights && \
	cp ./downloads/$$PKG/LICENSE ./imaginairy/vendored/$$PKG/ && \
	echo "vendored from $$REPO @ $$COMMIT" | tee ./imaginairy/vendored/$$PKG/readme.txt
	find ./imaginairy/vendored/facexlib/ -type f -name "*.py" -exec sed -i '' 's/from facexlib/from imaginairy.vendored.facexlib/g' {} + &&\
	sed -i '' '/from \.version import __gitsha__, __version__/d' ./imaginairy/vendored/facexlib/__init__.py
	make af

vendorize:  ## vendorize a github repo.  `make vendorize REPO=git@github.com:openai/CLIP.git PKG=clip`
	mkdir -p ./downloads
	-cd ./downloads && git clone $(REPO) $(PKG)
	cd ./downloads/$(PKG) && git fetch && git checkout $(COMMIT)
	rm -rf ./imaginairy/vendored/$(PKG)
	cp -R ./downloads/$(PKG)/$(PKG) imaginairy/vendored/
	git --git-dir ./downloads/$(PKG)/.git rev-parse HEAD | tee ./imaginairy/vendored/$(PKG)/source-commit-hash.txt
	touch ./imaginairy/vendored/$(PKG)/version.py
	echo "vendored from $(REPO)" | tee ./imaginairy/vendored/$(PKG)/readme.txt

download_repo:
	mkdir -p ./downloads
	rm -rf ./downloads/$(PKG)
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