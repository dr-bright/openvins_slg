# openvins_lightglue

Extension workspace for OpenVINS that adds a custom visual frontend based on:

- SuperPoint keypoints + descriptors
- LightGlue matching

## Package layout

- `ov_lightglue`: catkin package that hosts tracking/frontend integration.

## Current status

Scaffold initialized with:

- package metadata (`package.xml`, `CMakeLists.txt`)
- tracker interface header `ov_lightglue/src/track/TrackSuperLightGlue.h`

No inference implementation (`.cpp`) is included yet.

## License

This repository currently uses the OpenVINS project license (GPLv3), inherited for compatibility.
