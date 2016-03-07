# Repository for ViSP contrib modules

This repository is intended for development of "extra" modules that are contribution from the community, or that do not have stable API, that are not well-tested or that cannot be added to ViSP due to licensing. Thus, they shouldn't be released as a part of official ViSP distribution.

So, all the new modules should be developed separately, and published in the [visp_contrib](https://github.com/lagadic/visp_contrib) repository at first. Later, when the module matures, gains popularity and is compatible with ViSP licensing, it is moved to the central [ViSP](https://github.com/lagadic/visp) repository, and the development team provides production quality support for this module.

## How to build ViSP with extra modules

You can build ViSP, so it will include the modules from this repository. Here is the CMake command:

```
$ cd <visp_build_directory>
$ cmake -DVISP_CONTRIB_MODULES_PATH=<visp_contrib>/modules <visp_source_directory>
$ make -j4
```

As the result, ViSP will be built in the <visp_build_directory> with all modules from visp_contrib repository. If you don't want all of the modules, use CMake's BUILD_visp_* options. Like in this example:

```
$ cmake -DVISP_CONTRIB_MODULES_PATH=<visp_contrib>/modules -DBUILD_visp_ar=OFF <visp_source_directory>
```
