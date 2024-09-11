# Changelog

All notable changes to this project will be documented in this file.

## [1.2.3](https://github.com/x-pt/template/compare/v1.2.2..v1.2.3) - 2024-09-11

### 🚀 Features

- *(cuda)* Use generic for example - ([3f65611](https://github.com/x-pt/template/commit/3f656112b24010d7cd0d2dd691fcc183028b0ee3))

### 🐛 Bug Fixes

- *(cuda)* PTX JIT compiler library not found when using xmake - ([b41da2c](https://github.com/x-pt/template/commit/b41da2cac72bdaa478ae3bfd4f3f48a9d6474894))
- *(cuda)* PTX JIT compiler library not found - ([94b0e5d](https://github.com/x-pt/template/commit/94b0e5d9e599aa6685bea50bc05636a39c0bf42f))

### 🎨 Styling

- *(cuda)* Split the cuda and cxx settings for clarity - ([283a176](https://github.com/x-pt/template/commit/283a1762cff714be0180f83c10eaf9053950a8b4))
- *(cuda)* Add more comments - ([116437c](https://github.com/x-pt/template/commit/116437ce7682f16ba8e5dffda571aa524ce20d6f))
- *(cuda)* Reformat the function/variable name and add some comments - ([11949fe](https://github.com/x-pt/template/commit/11949fe9e4a44217d3bf256cbbd5cfca4b3b9eb7))

### 🧪 Testing

- *(cuda)* Update the case - ([a31beb6](https://github.com/x-pt/template/commit/a31beb65178329d4678e8957caf285989c63dedf))
- *(cuda)* Add more test cases - ([645975f](https://github.com/x-pt/template/commit/645975fab379a0fc9c9e20dde2a0da8f2b2ae438))
- *(cxx)* Add more test cases - ([652df3f](https://github.com/x-pt/template/commit/652df3fbed6b78024270b1b8fdf021ac0300971a))

### ⚙️ Miscellaneous Tasks

- Rename python to py at root cookiecutter.json - ([94165bb](https://github.com/x-pt/template/commit/94165bb8554162e4562fed386f24262d0cf70db4))


## [1.2.2](https://github.com/x-pt/template/compare/v1.2.1..v1.2.2) - 2024-09-09

### 🚀 Features

- *(cxx)* Add version for xmake and update Dockerfile - ([0327864](https://github.com/x-pt/template/commit/03278643b42d6763623b8a53b8e16726dcfdfb93))
- *(cxx)* Introduce google test - ([8b11c1f](https://github.com/x-pt/template/commit/8b11c1fdd917e3e4e886c777e5ef6df0dd7247c4))
- Introduce xmake for cuda - ([649f696](https://github.com/x-pt/template/commit/649f6964e75a8bfb1602a56df12be54fd4395acf))
- Introduce cuda boilerplate - ([0c993f7](https://github.com/x-pt/template/commit/0c993f7495b0d9753bafa8a610f5c54ed0244f9e))

### 🐛 Bug Fixes

- *(ci)* Cxx cannot build with xmake when auto deploying - ([b0cca27](https://github.com/x-pt/template/commit/b0cca27eae9eda56e694ff64495d21c148843f53))
- *(cxx)* Tests are not be executed - ([1a0af6b](https://github.com/x-pt/template/commit/1a0af6b1f5bf75044597238b23fbd205390d9e85))
- *(py)* One line 'module-not-imported' - ([763207c](https://github.com/x-pt/template/commit/763207c3352c5b0275deff8f12e778510b1cd1a9))

### 🎨 Styling

- *(cxx)* Align the cxx and cuda boilerplate CMakeLists.txt - ([9483e90](https://github.com/x-pt/template/commit/9483e9052944fb65816876393cc9e68d1ff68a67))

### ⚙️ Miscellaneous Tasks

- *(cuda)* Use the specified version for win and ubuntu - ([7609f73](https://github.com/x-pt/template/commit/7609f7347118a5ffc7f4f8389a2d4cc31c0788b5))
- *(cuda)* Cancel the test at post build in xmake - ([4e02d86](https://github.com/x-pt/template/commit/4e02d862845fe9df5b4d17092ccd2968893c6b12))
- *(cuda)* Fix the typos - ([2723dc2](https://github.com/x-pt/template/commit/2723dc29fba643f199c586ad330943da38811a24))
- *(cuda)* Enable log for installing cuda - ([4736dde](https://github.com/x-pt/template/commit/4736dde757848a930c175a4780046b638b4c6522))
- *(cuda)* Remove macos platform - ([b242314](https://github.com/x-pt/template/commit/b242314feced20f664d7cab58359551f89b53a85))
- *(cuda)* Install cuda - ([0a2cbc0](https://github.com/x-pt/template/commit/0a2cbc01e3747bab3d9693ed08ba0d696a4889b6))
- *(cxx)* Ignore build dir - ([deaf865](https://github.com/x-pt/template/commit/deaf8657ad789c18db885c81a68b0dedb8a71d4c))
- *(cxx)* Optimize the gitignore - ([2edca53](https://github.com/x-pt/template/commit/2edca53dec72ec079fdd774977cf8effb8b6f321))
- *(cxx)* Add test stage at github action - ([9c3c6b0](https://github.com/x-pt/template/commit/9c3c6b020b2f8250c77adba22b45c8a2110ddda5))
- *(cxx)* Use APP_NAME variable at Makefile run command - ([10ad86e](https://github.com/x-pt/template/commit/10ad86e6db72192f3017bf5c5ebcb846597e37dc))
- *(cxx)* Add init command at Makefile - ([dc18d7c](https://github.com/x-pt/template/commit/dc18d7cd6bae2612d69642ec4f54ed156fe3d8dd))
- *(go)* Update the golangci-lint - ([2819ce6](https://github.com/x-pt/template/commit/2819ce6dbfa41ffb00eadd34effc88efd658ef0b))
- *(py)* Skip coverage.lcov - ([1543693](https://github.com/x-pt/template/commit/15436937555d2af7197208eeda29f1a354fb13ba))
- Add example cuda deployment - ([28bf892](https://github.com/x-pt/template/commit/28bf892bde6d48c3b1da7e57b4b36546bafc6e32))

### Build

- Add fail-fast at pre-commit - ([7a98e66](https://github.com/x-pt/template/commit/7a98e662617692edae4ada3dbc4249d45a265f81))


## [1.2.1](https://github.com/x-pt/template/compare/v1.2.0..v1.2.1) - 2024-09-02

### 🚀 Features

- *(cxx)* Git init at post_gen_project - ([d41eb07](https://github.com/x-pt/template/commit/d41eb0749b47c4351e0590445eebe125174d6788))
- *(go)* Git init at post_gen_project - ([5b819d6](https://github.com/x-pt/template/commit/5b819d637e37bc1dac3b4f73fe438e4d995df448))
- *(py)* Git init at post_gen_project - ([6f311b4](https://github.com/x-pt/template/commit/6f311b445e7db74fbadaa854c04a65d4bc986354))
- *(py)* Use uv command in github action instead of make - ([9d77b0b](https://github.com/x-pt/template/commit/9d77b0b96510ce62004117629e68ce287c18c5a0))
- *(ts)* Introduce conventional commit check - ([ae1639f](https://github.com/x-pt/template/commit/ae1639f74e2fc384189f323f6ceb3ac057965ff4))
- *(ts)* Git init at post_gen_project - ([21a4cb5](https://github.com/x-pt/template/commit/21a4cb5705597bff450658a96d1b03a03dca11a3))
- *(ts)* Introduce auto release with release notes - ([7c26d6e](https://github.com/x-pt/template/commit/7c26d6edbd60b82ec374a0ae80fdd3fbde8d4e89))
- *(ts)* Update the startup code - ([41f6592](https://github.com/x-pt/template/commit/41f6592e9ca8b44889ca6d00c306387a83160b70))
- Introduce the pre-commit - ([106a949](https://github.com/x-pt/template/commit/106a949bc245398a69b33d8ae8e642adbb36faad))
- Introduce crate-ci/typos - ([1ccf903](https://github.com/x-pt/template/commit/1ccf903b8225250676503a2243d4219968b538fa))

### 🐛 Bug Fixes

- *(ts)* Pnpm build cannot find lib/index.js - ([511bd4b](https://github.com/x-pt/template/commit/511bd4b48a1286adb5dbe633b0eaef11ff47c0b1))

### 📚 Documentation

- Add ts example - ([9f95dae](https://github.com/x-pt/template/commit/9f95dae103821395e48786229b288c1d504b9b99))

### ⚙️ Miscellaneous Tasks

- *(ts)* Use pnpm - ([b5a0558](https://github.com/x-pt/template/commit/b5a05583a496f3b01f2269d0d22c6958dcc06dc7))
- *(ts)* Use npm command instead of make - ([c9af759](https://github.com/x-pt/template/commit/c9af759aea25a7018a4772ef4199a0564e47aba4))
- *(ts)* Disable the cache default - ([d521e88](https://github.com/x-pt/template/commit/d521e889e67f18393a1e78bcc0472ba98700da5c))
- Add ts example - ([c951cc1](https://github.com/x-pt/template/commit/c951cc1985f71f7848d0ebbe707eb21a98ef9d3a))
- Some minor changes - ([0d54e6a](https://github.com/x-pt/template/commit/0d54e6a6eeec013bbdb5f7606dc58c72baa0d619))

### Build

- *(ts)* Update the lint-staged - ([8a1b2bc](https://github.com/x-pt/template/commit/8a1b2bcc37a43caa6863e038de50672642eb32d3))
- *(ts)* Update the axios - ([d140771](https://github.com/x-pt/template/commit/d140771a274607b4c77044f2f599ccab67e208a0))
- Add init - ([2629087](https://github.com/x-pt/template/commit/262908798caef03cd787fb8c8e04b47c19976766))


## [1.2.0](https://github.com/x-pt/template/compare/v1.1.2..v1.2.0) - 2024-08-28

### 🚀 Features

- *(cxx)* Introduce auto release with release notes - ([e923548](https://github.com/x-pt/template/commit/e9235485857729a769a79b20c6c05570d039ab9c))
- *(go)* Introduce auto release with release notes - ([5e8e301](https://github.com/x-pt/template/commit/5e8e301d756cec946409072b4aa3efc2540b5a6f))
- *(python)* Introduce auto release with release notes - ([33bfd59](https://github.com/x-pt/template/commit/33bfd59411f4af46e83deb65c421b1e79db8a73d))
- *(ts)* Update the example - ([c614959](https://github.com/x-pt/template/commit/c614959666c6dfabffa04ef3fa1e4c19becda2f2))
- Add ts boilerplate - ([53350c9](https://github.com/x-pt/template/commit/53350c9f8e1a5f71d2a9eb94dd81e16400825b55))

### 🐛 Bug Fixes

- *(python)* Parse error when use {{ at github action - ([dbc1591](https://github.com/x-pt/template/commit/dbc1591139b256871a4fdfef3a100befed2a57ef))
- Loss husky hooks - ([0e3da2b](https://github.com/x-pt/template/commit/0e3da2b618c473fca214c6ee45824bc2978b01ac))

### 📚 Documentation

- *(ts)* Add development.md - ([7bfcc5e](https://github.com/x-pt/template/commit/7bfcc5e50582967a628501f4995fe2ca9427cefc))

### ⚙️ Miscellaneous Tasks

- *(cxx)* Introduce issue labeler bot - ([b716642](https://github.com/x-pt/template/commit/b71664205cd0cba27bf75e2ab18a2563a51e797d))
- *(go)* Introduce issue labeler bot - ([60687aa](https://github.com/x-pt/template/commit/60687aa01b18deff314d662d8a4e75d773cb856f))
- *(python)* Introduce issue labeler bot - ([7e2c3d3](https://github.com/x-pt/template/commit/7e2c3d342e3fa085fac3bf8a35d71d2492837b08))
- *(python)* Support make build for macOS, Ubuntu, Win - ([a827f4d](https://github.com/x-pt/template/commit/a827f4dbc3cf83d5958eea452b7d1deba5d4030c))
- *(ts)* Add automation test - ([20381e7](https://github.com/x-pt/template/commit/20381e7259216c67cda48ce79a842aaf8bece6ef))
- Update the git-cliff version - ([8f6a615](https://github.com/x-pt/template/commit/8f6a61563b658bca308dd8a1a845f9d67c84babf))
- Fix the typos - ([6543281](https://github.com/x-pt/template/commit/6543281ac84cee5e8c6769d083601118e41dcd75))
- Use srvaroa/labeler instead of github/issue-labeler - ([6812345](https://github.com/x-pt/template/commit/6812345e48cf1d4c8ec615cb28f196933ef79cbd))
- Fix the labeler 401 error - ([9423020](https://github.com/x-pt/template/commit/9423020fab8603975f5c0390ee5603c619602273))
- Use srvaroa/labeler - ([2b8603e](https://github.com/x-pt/template/commit/2b8603ea41929a063cb915a18fcf0030a46336ee))
- Add manual trigger for example deployments - ([e61e7ad](https://github.com/x-pt/template/commit/e61e7ad62f6e50ab376f3d389ef457f2c09a4f7e))
- Optimize the example deployment - ([aab00d4](https://github.com/x-pt/template/commit/aab00d44f10b09dec1082123fc971cea1907747b))
- Remove the deps for the release job - ([721be13](https://github.com/x-pt/template/commit/721be13bfe5fbb917d39a1f682d7722ff2ba4572))

### Build

- Update the cliff.toml without using raw and endraw - ([619d827](https://github.com/x-pt/template/commit/619d8270c0215b3ec85d53b034e2183676b2303d))


## [1.1.2](https://github.com/x-pt/template/compare/v1.1.0..v1.1.2) - 2024-08-24

### ⚙️ Miscellaneous Tasks

- Release automatically - ([abafc23](https://github.com/x-pt/template/commit/abafc23ee8f236c68948570a33d2f92b11befd2d))


## [1.1.0](https://github.com/x-pt/template/compare/v1.0.0..v1.1.0) - 2024-08-24

### 🚀 Features

- *(cxx)* Setup the output dir for binary and lib - ([7be7f88](https://github.com/x-pt/template/commit/7be7f88ac306871adb5d672b53b61d154120cbb6))
- *(cxx)* Introduce xmake option - ([ee9b41e](https://github.com/x-pt/template/commit/ee9b41eacbaad914a9d782b670a21cb0038eae43))
- *(cxx)* Optimize the CMakeLists - ([8ec7187](https://github.com/x-pt/template/commit/8ec718753894eef0940228708b0987acc562e76a))
- *(cxx)* Optimize the CMakeLists - ([796de0b](https://github.com/x-pt/template/commit/796de0b75c85cb7cdc068508985076952ea15cb3))

### 🐛 Bug Fixes

- *(cxx)* Align the output directory for all platforms - ([c41438f](https://github.com/x-pt/template/commit/c41438fa0047a4794be92fe9f97232516158800a))

### 🚜 Refactor

- *(cxx)* Rearrange the CMakeLists - ([a9ca556](https://github.com/x-pt/template/commit/a9ca55651defb9f302c6a51179fbc89396c06faf))

### 📚 Documentation

- *(cxx)* Update the README.md - ([3131ce2](https://github.com/x-pt/template/commit/3131ce2fe46e0e1abdce3f06042c73214f01a5d5))
- *(cxx)* Update the README.md - ([5cb84d3](https://github.com/x-pt/template/commit/5cb84d3ba0560085ca0bec4cb777e160e5f12e9f))
- *(cxx)* Remove the rust info - ([fab989b](https://github.com/x-pt/template/commit/fab989bd9de7b9e2564ba7e24153d0f745829f57))
- *(python)* Update the development.md - ([9e5c449](https://github.com/x-pt/template/commit/9e5c449d4aecd8520e8ca2b361220a19fe565a1b))
- *(python)* Require python version by the input - ([5b511a5](https://github.com/x-pt/template/commit/5b511a58aaffd5e226e305b4cf1cb524dccf8a37))
- *(python)* Split the development guide - ([58693be](https://github.com/x-pt/template/commit/58693be3ac1a7bb627e3f273429c553388dbec30))
- *(python)* Update the README.md - ([a338b8c](https://github.com/x-pt/template/commit/a338b8c334291badc0ade185a79ae302c53103e5))
- Add code_of_conduct - ([426209b](https://github.com/x-pt/template/commit/426209bf56c0c32a3afe70f302ed42651efcc3d9))
- Update some doc - ([38495fb](https://github.com/x-pt/template/commit/38495fb550dfb847bd3191ae9e90c5e8a95d52fc))
- Update CHANGELOG - ([dc6317a](https://github.com/x-pt/template/commit/dc6317a3b46113940934e8ceb24d2ba9b9b940db))

### 🎨 Styling

- Remove the redundant blank lines for CMakeLists.txt - ([a37bfc9](https://github.com/x-pt/template/commit/a37bfc9191e5fea87ef588222c26a3ad9aee3c7c))
- Remove the redundant blank lines - ([7fdc61b](https://github.com/x-pt/template/commit/7fdc61bcd318b7b1fd733cc362e79543bf56f1ad))
- Using dash instead of asterisk - ([4c9410b](https://github.com/x-pt/template/commit/4c9410b922f77179095492ab1ca476a73e4a0e75))

### ⚙️ Miscellaneous Tasks

- *(cxx)* Fix failed to run github action job for xmake project - ([a145866](https://github.com/x-pt/template/commit/a145866ab3750cad3cfbdad841e7d69728ec4377))
- *(cxx)* Update the ci config - ([f63e6df](https://github.com/x-pt/template/commit/f63e6dfab6059f144f203949c8283bd609a99654))
- Enable the typos as default - ([a0b0f17](https://github.com/x-pt/template/commit/a0b0f1746752a079f7d3195c05bbfbf3c35eb73c))
- Add much more info in changelog by cliff - ([23657ba](https://github.com/x-pt/template/commit/23657ba177cadc4c2bcbd018200f054a0b57c2a5))
- Build xmake and cmake example - ([d295be9](https://github.com/x-pt/template/commit/d295be905384ac19a5c723091c2dd2f40a5aea20))
- Fix the template input error - ([5d1a541](https://github.com/x-pt/template/commit/5d1a541752c931107f4ffa74e151cc9edc313b23))
- Update the pre-commit version - ([122e571](https://github.com/x-pt/template/commit/122e571d8e3f4922f354f8e0a0dd386c958a5499))
- Use local time for license date - ([370eaf1](https://github.com/x-pt/template/commit/370eaf1df3d5f5617341089f12914c98410c2900))
- Add year to license dynamically - ([c3081ba](https://github.com/x-pt/template/commit/c3081ba985442502809bb36043ce1051ec8e0631))
- Make some minor changes - ([26d35b9](https://github.com/x-pt/template/commit/26d35b99de48aca7a404ac3506214c925acb3ba9))
- Fix enable-versioned-regex not found - ([f451099](https://github.com/x-pt/template/commit/f451099efd8c4c2ac6a38423784a14efac694d7b))
- Update issue labeler match rules - ([d91420d](https://github.com/x-pt/template/commit/d91420d7aad951b3726cf98930ef1f83617df68a))
- Add labeler for issues - ([15a38a5](https://github.com/x-pt/template/commit/15a38a53427234564ea0195f29e7a86cbd843540))
- Update compose - ([1517b36](https://github.com/x-pt/template/commit/1517b361224b4b25e26999e38604c465852c768b))
- Rearrange the template inputs location - ([66a5ff6](https://github.com/x-pt/template/commit/66a5ff618014effb157fc1cf4d5354ea5c7051eb))

### Build

- *(go)* Update the package version - ([2229160](https://github.com/x-pt/template/commit/2229160d126ad142154ecdf114db3ecbc6376bd8))
- *(python)* Use debian bookworm - ([6a5560d](https://github.com/x-pt/template/commit/6a5560d02315bb46f235444286429158761fc1a4))
- *(python)* Add UV_INDEX_URL at Dockerfile - ([6b76681](https://github.com/x-pt/template/commit/6b766811155a9f02cdb7bbf0aec89573209cc2e5))
- *(python)* Only install uv at Dockerfile - ([4212835](https://github.com/x-pt/template/commit/4212835f33845b12876f002d80d7da9701814c6f))
- *(python)* Use uv for all scenario - ([c76a122](https://github.com/x-pt/template/commit/c76a12247d357d5f95ca1225015255685e973412))
- Add Makefile - ([016dd0d](https://github.com/x-pt/template/commit/016dd0d27a4ec26cfdc07f2aa08babc248e081d4))
- Introduce git-cliff to generate the changelog - ([f940d5e](https://github.com/x-pt/template/commit/f940d5eed5ce34251b29ff6c0e72085e18cab87f))
- Remove version from the docker compose - ([95d203e](https://github.com/x-pt/template/commit/95d203e616005394c3aeffd931d97940a45d0013))


## [1.0.0](https://github.com/x-pt/template/compare/v0.1.1..v1.0.0) - 2024-08-17

### 🚀 Features

- *(py)* Update README and refactor Dockerfile and compose - ([5c177cf](https://github.com/x-pt/template/commit/5c177cf0d2a054f489f83c6f6827872b4f391c63))
- *(py)* Update Dockerfile and compose - ([a587fd2](https://github.com/x-pt/template/commit/a587fd2e68f1df5c0d247b4f2548cd0464b7f782))
- Use the specified python version - ([6652c4b](https://github.com/x-pt/template/commit/6652c4b9cbe36694ee7051f6c43c417b21b44cdb))
- Update python Dockerfile - ([0bfeb9d](https://github.com/x-pt/template/commit/0bfeb9daac3b20176301e86a5a18bf2df703a961))
- Update python boilerplate - ([ed2c1b1](https://github.com/x-pt/template/commit/ed2c1b12a25d7882c55197702eab01c8d4f477c8))
- Introduce rhai for preprocessing - ([4536870](https://github.com/x-pt/template/commit/4536870d29e01a0bf6854ab326ac0a9afc47e66c))
- Update python template - ([d97fa75](https://github.com/x-pt/template/commit/d97fa75a0c5dddb5d4d129b4d77ffa5e4aac824c))
- Introduce hatch and uv for python - ([688c9ec](https://github.com/x-pt/template/commit/688c9ec28b17362df0027377b17eee20822ca04c))

### 🐛 Bug Fixes

- Msvc cannot recognize the header file with '-' - ([baeb2ea](https://github.com/x-pt/template/commit/baeb2eae9193e13e90a2de61953563c2786ea24b))

### 🚜 Refactor

- *(init)* Introduce cookiecutter for cxx and golang - ([2d3729a](https://github.com/x-pt/template/commit/2d3729a68268eb41c08cc2e0be52eb58ba9eb4cc))
- *(ongoing)* Change the location of py cookiecutter - ([b38b7c3](https://github.com/x-pt/template/commit/b38b7c30856f9224b84f708a37fd02523813a563))
- *(ongoing)* Introduce cookiecutter - ([8833bb6](https://github.com/x-pt/template/commit/8833bb61968787d1d51bc88a1f344c6e3c8e7f0f))
- *(py)* Introduce cookiecutter - ([d811568](https://github.com/x-pt/template/commit/d81156844ff1360047b52d54f895e37fd4b084db))
- Optimize the cmakelist - ([b4effcd](https://github.com/x-pt/template/commit/b4effcd0e300d98f1d30841a002f0df92b5746df))
- Minor changes - ([f4f52f9](https://github.com/x-pt/template/commit/f4f52f9942d6f45a52c424608a0394c9e6968d46))
- Introduce cookiecutter for cxx and golang - ([ab4ac14](https://github.com/x-pt/template/commit/ab4ac141424f677e0477039cc1ac2e7188663697))
- Use cookiecutter instead of cargo-generate - ([f745668](https://github.com/x-pt/template/commit/f74566883abc49e9716631a7caa100acc36e8370))

### 🎨 Styling

- Align the format - ([f3fdf8f](https://github.com/x-pt/template/commit/f3fdf8f63e40cb9f753952d2846a09a635066658))

### 🧪 Testing

- Add some test cases - ([d511992](https://github.com/x-pt/template/commit/d511992f40ab800bcc32523713a5d15f4f4f3381))

### ⚙️ Miscellaneous Tasks

- Fix workflow failure - ([e65c2ae](https://github.com/x-pt/template/commit/e65c2ae86c99b7106e721c730fcc13bafefa92c8))
- Update workflow - ([2c845dc](https://github.com/x-pt/template/commit/2c845dcb64c84d8db9e88ff9420e385707fc8918))
- Make some minor changes - ([e2ff497](https://github.com/x-pt/template/commit/e2ff49739176e32a676d470f17c09621f1626d3a))
- Make some minor changes - ([6f65d8d](https://github.com/x-pt/template/commit/6f65d8de796249cdcf77ec786808bb8b4d1ddb49))
- Make some minor changes - ([7a795ab](https://github.com/x-pt/template/commit/7a795ab816b871251f146fcda768375619038701))
- Make some minor changes - ([82e2ebc](https://github.com/x-pt/template/commit/82e2ebce32aa8ba6a3da180fc99850db86d7a73c))
- Update README - ([e78d168](https://github.com/x-pt/template/commit/e78d1683aa088f1c02225f7515e4aa4c6159f87e))
- Update README - ([03feb7b](https://github.com/x-pt/template/commit/03feb7b40225c8504381254502efbb6c4b1b4a50))
- Some minor changes - ([e5bd0ee](https://github.com/x-pt/template/commit/e5bd0ee0d040bdfe1e7aecc9dc1a2059a17adfd1))
- Update workflow - ([2614ae8](https://github.com/x-pt/template/commit/2614ae86990c8dfea2f9b066275cf96154319a39))


## [0.1.1](https://github.com/x-pt/template/compare/v0.1.0-beta..v0.1.1) - 2024-04-16

### 🚀 Features

- *(Makefile)* Add clean command - ([6bcb0f0](https://github.com/x-pt/template/commit/6bcb0f0e7ff4ddc78c680cd62134ec8bc89faf55))
- *(action)* Add release and renovate action - ([85d5115](https://github.com/x-pt/template/commit/85d5115eb95f2a08348d8c39fa02a8d823093765))
- *(action)* Add renovate for cpp - ([65d6f2a](https://github.com/x-pt/template/commit/65d6f2a8a1149ed5aea990e05a3a9ac23d789ade))
- *(action)* Change the logic of docker build - ([eb92044](https://github.com/x-pt/template/commit/eb920441a685d0163d1f2e9ca9cea0ea673b4793))
- *(action)* Remove audit - ([df278b6](https://github.com/x-pt/template/commit/df278b666eccfc4faf57d4fefe175e0c6b9a4e35))
- *(cargo-gen)* Add much more support for python - ([46e85a5](https://github.com/x-pt/template/commit/46e85a5f86dc9e328f92716b1473dd6a97f6cdbf))
- *(changelog)* Add some default info in CHANGELOG.md - ([96ca4c9](https://github.com/x-pt/template/commit/96ca4c9fb2aad4fdf2ff3a2acbce68217fd9397d))
- *(cmake)* DO NOT IGNORE Makefile - ([81e4cbd](https://github.com/x-pt/template/commit/81e4cbd574182846ea537c48eb42b0db6add676b))
- *(cmake)* Update the CMakeLists.txt and add Makefile - ([4de5ca5](https://github.com/x-pt/template/commit/4de5ca5038d0f788603a560de0bfd8d9986260b9))
- *(cxx)* Add some files to .dockerignore - ([45a4877](https://github.com/x-pt/template/commit/45a4877eabf4cd32d295130663ee27962f3d581e))
- *(cxx)* Replace the cmake build way - ([c7379b3](https://github.com/x-pt/template/commit/c7379b3ae89c732a3122149bbb9e44f7ceb1c3c9))
- *(docker)* Build image only when tagged like as v0.1.0 - ([79806af](https://github.com/x-pt/template/commit/79806af19acf68a1d916b936c20fc689b6197735))
- *(docker)* Add the prefix for github docker image - ([fd868c0](https://github.com/x-pt/template/commit/fd868c03971d7a4ddf399c9b8a0f4d209d30cf53))
- *(docker)* Add github action to build docker image - ([7f5f39f](https://github.com/x-pt/template/commit/7f5f39fd93dc041dd672af74bde51bb95be080d0))
- *(github)* Add contributing and issuing template - ([62b8df0](https://github.com/x-pt/template/commit/62b8df09777e5054434083a23720f413a7f11828))
- *(go)* Update Dockerfile and github action - ([18962a0](https://github.com/x-pt/template/commit/18962a07f9390aeb0b02871d903f775d94d9afeb))
- *(go)* Some minor changes - ([61dd638](https://github.com/x-pt/template/commit/61dd638e544cb58908612c9860eac98c15aee20c))
- *(go)* Add some options - ([0d685ec](https://github.com/x-pt/template/commit/0d685ec437e8f81657e6c9ece23aa44e53bc4668))
- *(go)* Add github actions - ([dcc7a19](https://github.com/x-pt/template/commit/dcc7a19aea0e0e2f6688e033010a82ddfce1f573))
- *(py)* Add src-layout - ([6bbb32b](https://github.com/x-pt/template/commit/6bbb32b1c60558aa4011eb698f430fed6e7e0fa0))
- *(py)* Add .dockerignore - ([3344352](https://github.com/x-pt/template/commit/33443521a8aa2aaa4c0e9836bfdeed05c123fa09))
- *(template)* Change {{project-name}} dir to src in python project - ([4d13df6](https://github.com/x-pt/template/commit/4d13df604b30777c6cbeaf4e411dda1c97ab9143))
- Update python template ci - ([2db86fe](https://github.com/x-pt/template/commit/2db86fec333d6f990577c5318a99fa3a915fb6ec))
- Introduce pre-commit for all templates - ([1108026](https://github.com/x-pt/template/commit/11080265c59d786fc09fa383fe174b89736279f8))
- Update the github action cd.yml in py template - ([78baf86](https://github.com/x-pt/template/commit/78baf862e78f2dcccf63eb7e70aab73198b3d166))
- Update the github action cd.yml in go and cxx template - ([5b6180b](https://github.com/x-pt/template/commit/5b6180b71f9a85ea356aa81253557693a9bbcaeb))
- Go template with cobra and viper - ([0c1e5e9](https://github.com/x-pt/template/commit/0c1e5e91d43510a8577be804770eaf22f42b98c7))
- Update go proj template - ([87826fb](https://github.com/x-pt/template/commit/87826fb243216ec3129d2e726ec39953dfebbc29))
- Update the docker github action - ([24b2e8d](https://github.com/x-pt/template/commit/24b2e8d29080a462db3605e7aea9ffddc05c2202))
- Update code of conduct - ([e91af05](https://github.com/x-pt/template/commit/e91af05570ef7df8f7b0fad421a57a269593cdf4))
- Update cxx, go, py template - ([b1de8e3](https://github.com/x-pt/template/commit/b1de8e3811ba3e90a7ceb02f70e796126a11c9b7))
- Remove macos verify from cxx - ([5916345](https://github.com/x-pt/template/commit/59163450459737b5446451ccd591f124005a03a0))
- Update the docker buildx version - ([876c0f6](https://github.com/x-pt/template/commit/876c0f692eeb86622864162263e22088b2e6498a))
- Update the docker buildx version - ([3a52b14](https://github.com/x-pt/template/commit/3a52b144ce2cfbf937bd19b4d28a252b3a26e8e7))
- Update the docker buildx version - ([4cb2a85](https://github.com/x-pt/template/commit/4cb2a856d5f72a3d65ab3bb6506879273d106e89))
- Add ccache to speed up the compile - ([ad75c34](https://github.com/x-pt/template/commit/ad75c340be07f6596b2936198c33a8f7e91732e0))
- Add .editorconfig - ([1891998](https://github.com/x-pt/template/commit/18919981a70d924bd62f65322de464b9adca6aa8))

### 🐛 Bug Fixes

- Failed to make build for example go on github ci - ([5e19476](https://github.com/x-pt/template/commit/5e1947613afad232a118bc650c2796ef9f430718))
- Failed to rye init - ([45bbca7](https://github.com/x-pt/template/commit/45bbca75626c955264f6fcce5b1337d03d525d40))

### ⚙️ Miscellaneous Tasks

- *(support)* Add jetbrains badge - ([8311244](https://github.com/x-pt/template/commit/831124491d57a71261b8c04826a6308e9268f833))
- Update liquid syntax - ([39170d9](https://github.com/x-pt/template/commit/39170d92c6af23d4669db044a645e3e32ba71cad))
- Continue to fix project name - ([4c705ae](https://github.com/x-pt/template/commit/4c705ae1f4e0472e51727575a015fa0a20567d2f))
- Gen go and py example - ([dda9ba5](https://github.com/x-pt/template/commit/dda9ba579e408c827ed82cc822e1137fcb8de2fe))
- Rename pre-commit file - ([a22919a](https://github.com/x-pt/template/commit/a22919a0b40b0b247fee448101314f291983c771))
- Some minor changes - ([e3b8108](https://github.com/x-pt/template/commit/e3b81087fd80570010cb9ad9ddde3aa70c0b70c0))
- Downgrade the cargo-generate-action - ([3f49101](https://github.com/x-pt/template/commit/3f491014ac9194693e8bc738c4017a91be833f21))
- Use make build and test - ([73d0ae5](https://github.com/x-pt/template/commit/73d0ae583319669ba722cbafa6dc075a2799f4e9))
- Some minor changes - ([91299b7](https://github.com/x-pt/template/commit/91299b73631c4448e20c53ee47a607706fea7561))
- Add Makefile tab rule in editorconfig - ([c25fa61](https://github.com/x-pt/template/commit/c25fa61ee8f6ccecc696d342ca7c1ed669ca0439))
- Change the typo - ([3bfd5a9](https://github.com/x-pt/template/commit/3bfd5a908b0f7cae11222cfbff8b8653978bffe4))
- Update checkout to v4 - ([73c995e](https://github.com/x-pt/template/commit/73c995e45d57463958e259ff3fdfb8b47aaa8a17))
- Some minor changes - ([d2eda3d](https://github.com/x-pt/template/commit/d2eda3dc2b3a59d0be6f33124fbc887322fa142a))
- Add missing "bin_type" - ([175dcb7](https://github.com/x-pt/template/commit/175dcb76f06fa2e3e1ba2a0d08f06c1c71414c9c))

### Build

- *(deps)* Bump cargo-generate/cargo-generate-action - ([7e8d3c2](https://github.com/x-pt/template/commit/7e8d3c2e1e32c5b1049e424e2757927965ed9484))
- *(deps)* Bump softprops/action-gh-release from 1 to 2 - ([0df101b](https://github.com/x-pt/template/commit/0df101bccd75cbacc6ab670867dfabfeef0043b3))
- *(deps)* Bump peaceiris/actions-gh-pages from 3 to 4 - ([84ef76f](https://github.com/x-pt/template/commit/84ef76f69dcb159a313240d79308464404819ce8))
- *(deps)* Bump cargo-generate/cargo-generate-action - ([bc6dc8b](https://github.com/x-pt/template/commit/bc6dc8b0640213e64907c2dfa189733ae2c8024e))
- *(deps)* Bump actions/checkout from 3 to 4 - ([89d48c8](https://github.com/x-pt/template/commit/89d48c8f463a4b2b85db2cd0aaa18adcdf676549))
- *(deps)* Bump cargo-generate/cargo-generate-action - ([bcbe0df](https://github.com/x-pt/template/commit/bcbe0df4f5e970084cb73610f91ed19e13079998))
- *(deps)* Bump cargo-generate/cargo-generate-action - ([80dee16](https://github.com/x-pt/template/commit/80dee16aab0b3a612b7ff5903cdafb9a055cbef0))
- *(deps)* Bump actions/checkout from 2 to 3 - ([27f585a](https://github.com/x-pt/template/commit/27f585a5ffe7bc20b2e9ffa09399bae5cd2bb35b))

## New Contributors ❤️

* @dependabot[bot] made their first contribution in [#12](https://github.com/x-pt/template/pull/12)

## [0.1.0-beta] - 2022-09-20

### 🚀 Features

- *(classify)* Bin and lib(shared or static) - ([db7b912](https://github.com/x-pt/template/commit/db7b912d3e511bd0fbbc0b88512677769ed04fc1))
- *(docker)* Distinguish static and dynamic binary on Dockerfile - ([8247b93](https://github.com/x-pt/template/commit/8247b93c62114b7c18c476a014304fba74a9ff73))
- *(docker)* Enable crb repo - ([3befc49](https://github.com/x-pt/template/commit/3befc490d670d2a4dcafe030674afd0bfa5d0722))
- *(docker)* Add static binary compile - ([491534c](https://github.com/x-pt/template/commit/491534cc3ce04338b8beb4b970f7893ac4103c04))
- *(docker)* Optimize the Dockerfile - ([18ab452](https://github.com/x-pt/template/commit/18ab452c16a0f2fdbda1d3c6489dfece200e1d56))
- *(docker)* Support Dockerfile - ([bb70f8f](https://github.com/x-pt/template/commit/bb70f8fa05bcec87e6c1d9cc4d1239f4452bc1b3))
- *(init)* Cargo generate cpp project - ([7439531](https://github.com/x-pt/template/commit/7439531c7fb7a8b29df081a34dc1cd1a06f7ba72))
- *(template)* Add Dockerfile - ([105bd73](https://github.com/x-pt/template/commit/105bd73bcc07c7e28edf237315ee1f4b0ba8ff46))
- *(template)* Update cxx CMakeLists.txt - ([4214016](https://github.com/x-pt/template/commit/421401667c606e2eec1e332bbf3f3e4a8c23f8f0))
- *(template)* Add golang and python support - ([0eca7f1](https://github.com/x-pt/template/commit/0eca7f1e606b954da84e8773847a114707a1479d))
- Add changelog - ([5c9088d](https://github.com/x-pt/template/commit/5c9088dc6e101e12ae1adf6d4f55905ea31f85a7))
- :art: optimize the control flow - ([bbd8c95](https://github.com/x-pt/template/commit/bbd8c9508ae615a4f8f6e84c87a58783e33d0fb7))
- Add .editorconfig for new project - ([00f2395](https://github.com/x-pt/template/commit/00f23957a0e160a3b49fd4793cb201a3157d4684))

### 🚜 Refactor

- *(dir)* Rearrange the hierarchy - ([e04f732](https://github.com/x-pt/template/commit/e04f732af8f154b882262cd9753d6351ec4c2288))
- *(rename)* Gh-proj to x-pt - ([d1d0e9f](https://github.com/x-pt/template/commit/d1d0e9f4820c03d7eb655b36d62fb4fd02e25d4a))
- *(rename)* Org name cxx-gh to gh-proj - ([6a893f0](https://github.com/x-pt/template/commit/6a893f0a79337d4057a7427ead4f0ea17527ff8c))

### 📚 Documentation

- Fix some legacy info - ([7a2fd26](https://github.com/x-pt/template/commit/7a2fd26fa37c4512beee148915c3953ee0296535))

### ⚙️ Miscellaneous Tasks

- *(init)* Remove .nojekyll file - ([e122a0e](https://github.com/x-pt/template/commit/e122a0e8e43b866876445ff40574924fe9bea6b5))
- *(init)* Include .github dir and exclude .nojekyll file - ([deaa493](https://github.com/x-pt/template/commit/deaa49378a240ece1756804c2ac0d5bb2804a397))
- *(init)* Remove the checkout of example repo - ([15b7852](https://github.com/x-pt/template/commit/15b78527291a5228d54a6d1235e64fef7f8ef697))
- *(init)* Update deployment - ([032273d](https://github.com/x-pt/template/commit/032273d0772e92674142adcaf7d64f3860a46e17))
- Remove build action - ([6038a3b](https://github.com/x-pt/template/commit/6038a3be8fbcd337e18c287157ed230fc878fbc1))
- Fix ci error - ([8eb22b5](https://github.com/x-pt/template/commit/8eb22b57324e8ae8cdbc5a5d6fd30d2042ec3b43))
- Update cd.yml for cpp project - ([f63f613](https://github.com/x-pt/template/commit/f63f613a7929a82f14109582cfa364161026597d))
- Update cd.yml for cpp project - ([8cd6e90](https://github.com/x-pt/template/commit/8cd6e9055b02a10c9e89125f03c770ed11bc875b))
- Update cd.yml for cpp project - ([a680933](https://github.com/x-pt/template/commit/a68093353f7e1bab48ded705f31a2cac815e8e4b))
- Update ci.yml for cpp project - ([bd70b5e](https://github.com/x-pt/template/commit/bd70b5e9926ca4babfbff4962ed373693e728047))
- Add missing fields for cd - ([cd0bb72](https://github.com/x-pt/template/commit/cd0bb729e479b47668e1bf17cf941cdeca0c8aa5))
- Fix ignore not work - ([62e0ac3](https://github.com/x-pt/template/commit/62e0ac346bd5cdff61816d3273fde962838c214a))
- Rename some variables - ([56aa4cd](https://github.com/x-pt/template/commit/56aa4cd75e94f5008b443395996ae1f0e8d677ff))
- Enable jekyll to remove .nojekyll - ([c388060](https://github.com/x-pt/template/commit/c388060f6f7c97ae8cf8836a4e91bd1a1a7b3901))
- Replace the commit message - ([3a68fcb](https://github.com/x-pt/template/commit/3a68fcbe680186276b796b65f18f4f2ffaba76bd))


<!-- generated by git-cliff -->
