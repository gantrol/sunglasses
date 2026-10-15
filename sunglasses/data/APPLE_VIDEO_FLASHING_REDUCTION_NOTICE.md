# Apple VideoFlashingReduction algorithm ports and kernel data

Sunglasses uses Apple Inc.'s MIT-licensed VideoFlashingReduction reference.
This notice covers the copied kernel data and the reference-derived algorithm
ports described below. The Apple copyright and permission notice must remain
with copies or substantial portions of that material, including binary
distributions that embed it. Acknowledgements do not replace this notice.

## Algorithm sources and local adaptations

The reference is pinned to commit
`6e9604d92446c8e6763bc53c5e12ccc4e0046796`. Its MATLAB sources include:

- [VideoProcesser.m](https://github.com/apple-aiml-research/VideoFlashingReduction/blob/6e9604d92446c8e6763bc53c5e12ccc4e0046796/VideoFlashingReduction_MATLAB/VideoProcesser.m)
- [RiskComputePass0.m](https://github.com/apple-aiml-research/VideoFlashingReduction/blob/6e9604d92446c8e6763bc53c5e12ccc4e0046796/VideoFlashingReduction_MATLAB/RiskComputePass0.m)
- [RiskComputePass1.m](https://github.com/apple-aiml-research/VideoFlashingReduction/blob/6e9604d92446c8e6763bc53c5e12ccc4e0046796/VideoFlashingReduction_MATLAB/RiskComputePass1.m)
- [RiskComputePass2.m](https://github.com/apple-aiml-research/VideoFlashingReduction/blob/6e9604d92446c8e6763bc53c5e12ccc4e0046796/VideoFlashingReduction_MATLAB/RiskComputePass2.m)
- [GammaKernelFunction.m](https://github.com/apple-aiml-research/VideoFlashingReduction/blob/6e9604d92446c8e6763bc53c5e12ccc4e0046796/VideoFlashingReduction_MATLAB/GammaKernelFunction.m)

Local files, relative to the repository root:

- `sunglasses/apple_vfm.py`: Python port of the temporal reference core.
- `shaders/apple_vfm.hlsl`: HLSL port of the reference passes and RGB mitigation.
- `sunglasses/apple_vfm_screen.py` and `native/screen/apple_vfm.cpp` / `.h`:
  desktop sampling adapters, kernel preparation and execution support.
- `native/apple_vfm_source.h.in`: embeds the shader and kernel data in native
  builds.

The ports retain the reference's whole-frame luminance adaptation, UFM FIR
responses, gamma energy pooling, response mapping and mitigation release.
Project adaptations include a nonzero denominator floor for black inputs and
explicit desktop sampling, reset and presentation logic. The screen mode uses
fixed 60 Hz, 100 nit and 1265.625 square-degree assumptions; these are not
measurements of the user's display.

The optional "Reduce repeated flashes" screen mode uses this reference
(previously labeled "Apple VFM (replica)"). Neither it nor the offline
reference is endorsed by Apple. They are ports of the public research reference,
not Apple's operating-system feature, and do not establish medical protection
or standards conformance.

## Unmodified kernel data

`apple_ufm_kernels_6e9604d.txt` is copied without modification from:

- Repository: https://github.com/apple-aiml-research/VideoFlashingReduction
- Commit: `6e9604d92446c8e6763bc53c5e12ccc4e0046796`
- Upstream path: `VideoFlashingReduction_Xcode/VideoFlashingReduction/Resources/UMFKernel-Data.txt`
- Upstream Git blob SHA: `40681956be219b1d0ba11e3c3cb484d9331b6f6e`
- SHA-256: `209bc2e47a8a8dcaab6172787dc2721efe38a6b5d9d49fdeecd5938100a8dc79`
- Retrieved: 2026-10-12

## Upstream MIT license

[License at the pinned commit](https://github.com/apple-aiml-research/VideoFlashingReduction/blob/6e9604d92446c8e6763bc53c5e12ccc4e0046796/LICENSE.md),
checked 2026-10-14. The original notice follows unchanged:

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
