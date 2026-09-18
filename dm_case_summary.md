
# Overview 
---

| Skia version | Platform       | Baseline  | Case src | Case Count |
| ------------ | -------------- | --------- | -------- | ---------- |
| m144         | HO 7.0(api 23) | glesdmsaa | gm       | 1076       |


Classify types of difference between `grdawn_vk` vs baseline(`glesdmsaa`) as  
- **Match**:  structure similarity > 99.99
- **Trivial**: barely noticeable difference but similarity < 99.99
- **Noticeable Difference**: visual difference but without element missing. To be classified by pattern
	- Line alias pattern
	- Dot line pattern change
	- Stroked rectangle shifting
	- etc.
- **Failure**: no result or visually noticeable elements missing

# Summary

| Category                         | HO Count | Android Count |
| -------------------------------- | -------- | ------------- |
| Failure                          | 231      | 134           |
| Difference(Noticeable + Trivial) | 325      | 326           |
| Match                            | 517      | 613           |
| Graphite only                    | 3        | 3             |


## Category - Failure
---

| Type                 | Count on HO 7.0 | Count on Android 16 | Description                                                                                                                    |
| -------------------- | --------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| General errors       | 109             | 12                  | No results both with Ganesh and Graphite                                                                                       |
| Unsupported features | 44              | 44                  | No results with Graphite only.                                                                                                 |
| Missing elements     | 78              | 78                  | Shape elements missing in Graphite results<br>- HO Graphite vs Android Graphite: 7x out of 78 cases with noticeable difference |


### Failure - General errors

These cases fail due to shared reasons with both Ganesh and Graphite.

| Sub-category      | Count | Cases                                                                                                                                                                                                                                                                                                                    | Description                                                                          |
| ----------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Test font missing | 97    |                                                                                                                                                                                                                                                                                                                          | - HO: no render results <br>- Android: rendered (86x match + 11x trivial difference) |
| TBD               | 12    | rectangle_texture<br>paragraph_layout_<br>ycbcrimage<br>readpixelspicture<br>readpixelscodec<br>nearest_half_pixel_image<br>mirror_tile<br>hittestpath<br>encode-gray-color-types-webp-lossy<br>encode-opaque-color-types-webp-lossy<br>encode-gray-color-types-webp-lossless<br>encode-opaque-color-types-webp-lossless | No results both on HO and Android.                                                   |

### Failure - Unsupported features

These cases rely on features only exist in ganesh, therefore no render results on Graphite.

| Sub-category | Description                                                                                 | Count | Graphite support |
| ------------ | ------------------------------------------------------------------------------------------- | ----- | ---------------- |
| GPU features | no read to unpremultiply support <br>- see `graphite::Context::asyncRescaleAndReadPixels`   | 1     | ?                |
|              | no non-multiple-of-four texture support <br>- due to D3D combability of Dawn                | 1     | no               |
| API support  | `ganesh::GrTextureGenerator` support                                                        | 1     | ?                |
|              | Customized op insertion: `SurfaceDrawContext::addDrawOp`                                    | 26    | no               |
|              | Rectangle util API:  `SurfaceDrawContext::drawRect` or `SurfaceDrawContext::FillRectToRect` | 10    | ?                |
|              | texture util API: `SurfaceDrawContext::drawTexture`                                         | 1     | ?                |
|              | `ganesh::SurfaceFillContext` swizzle API                                                    | 2     | ?                |
|              | do `ShaderCaps::supportedSkSLVerion` check only in ganesh                                   | 2     | yes              |


###  Failure - Missing elements

These cases are rendered with shape elements missing in Graphite comparing to Ganesh. Classify them based on draw primitive types.

| Sub-category     | Total Count on HO | Type - Missing Support | Type - Improvement Candidate | Count on Android |
| ---------------- | ----------------- | ---------------------- | ---------------------------- | ---------------- |
| Rectangle        | 26                | 7                      | 6                            |                  |
| Vertices         | 15                | 15                     | 0                            |                  |
| Path Shape       | 20                | 0                      | 4                            |                  |
| Other Primitives | 9                 | 5                      | ?                            |                  |
| Non-Draw Op      | 9                 | ?                      | ?                            | 1                |


#### Rectangle
---

| Category                           | Cases                                                                                                        | Description                                                                                                                                                                          | Could be Improvement? | Graphite Support |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- | ---------------- |
| Large Rectangle size               | bigrect                                                                                                      | drawRect:<br>stroke style + (stroke width == 0) + (rect size>5e10f): some strokes disappear<br>fill style + noaa: less blur, stoke rectangle shifting                                |                       |                  |
|                                    | fast_constraint_red_is_allowed<br>strict_constraint_batch_no_red_allowed<br>strict_constraint_no_red_allowed | drawImageRect + bigRect(2K)<br> - graphite: primitive not drawn (ok with `SkTiledImageUtils::DrawImageRect`)                                                                         |                       |                  |
|                                    | verylarge_picture_image<br>verylargebitmap                                                                   | same<br>[graphite] WARNING - Couldn't convert SkImage to a Graphite-backed representation<br>[graphite] WARNING - Key context creation failed in Device::drawGeometry, draw dropped! |                       |                  |
| drawEdgeAAQuad + large matrix      | compositor_quads_filter                                                                                      | parameters:  drawEdgeAAQuad(required by maskfilter ) + perspective matrix<br>- graphite: texture position shift in depth??                                                           |                       |                  |
|                                    | crbug_1174186                                                                                                | drawEdgeAAQuad + large matrix + line quad<br>- ganesh:  disappear <br >- graphite:  draw something  (correct?)                                                                       | yes                   |                  |
| Gradient  Shader + large matrix    | hardstop_gradients_many                                                                                      | GradientShader + translate(y>1000) = effect direction shift                                                                                                                          |                       |                  |
| DrawRect + Clipping rounding issue | clipdrawdraw                                                                                                 | drawRect: clipping rounding error(0.5 vs 0.499), <br> - graphite: one pixel shifting                                                                                                 |                       |                  |
| Texture-related                    | bleed_downscale                                                                                              | drawImageRect + kFast_SrcRectConstraint + linear filter + linear mipmap <br>- ganesh: red-ish result <br>- graphite: still blue                                                      |                       |                  |
|                                    | makecolortypeandspace                                                                                        | drawImageRect + ColorType(kRGB_565_SkColorType or kGray_8_SkColorType)<br>- ganesh: expected failure ( no change )<br> - graphite: quantized or gray effect                          | yes                   |                  |
|                                    | workingspace_input_output                                                                                    | DrawRect + color/image shader + manual unpremul in shader<br>- ganesh:  grey<br>- graphite:  green(same as premul)                                                                   |                       |                  |
| Fp Effects                         | emboss<br>embossmaskfilter<br>smallemboss                                                                    | - ganesh: no emboss effect<br>- graphite: embossMaskFilter rendered                                                                                                                  | yes                   |                  |
|                                    | image_dither                                                                                                 | gradientShader + dither <br>- ganesh: no dither, = original<br>- graphite:  dither applied                                                                                           | yes                   |                  |
| Missing graphite API               | compositor_quads_image                                                                                       | graphite version of `sk_gpu_test::LazyYUVImage::refImage` defined but not used.                                                                                                      |                       | yes              |
|                                    | image-shader<br>image-surface<br>skbug_12212                                                                 | `SkSurfaces::RenderTarget` allocation <br>- ganesh: via `ganesh::GrRecordingContext`<br>- graphite: missing logic of `graphite::recorder`                                            |                       | yes              |
|                                    | scale-pixels                                                                                                 | `SkImage::scalePixels` requires `ganesh::GrDirectContext` support internally<br>- ganesh: correct<br>- graphite: scaled pixels disappear.                                            |                       | ?                |
|                                    | draw_quad_set                                                                                                | render color gradient effect<br>- ganesh: utilize `ganesh::SurfaceDrawContext::fillRectWithEdgeAA`<br>- graphite: standard routine without gradient effect                           |                       |                  |
|                                    | drawable                                                                                                     | - graphite: `graphite::device::drawDrawable` defined but not implemented.                                                                                                            |                       |                  |
| TBC                                | textureimage_and_shader                                                                                      | drawImageRect + draw image shader (image = green)<br>- ganesh: green<br>- graphite: red                                                                                              |                       |                  |

#### Vertices
---

| Category             | Cases                                                                                                                                                                                   | Description                                                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Missing Graphite API | imagefiltersbase<br>skbug_14554<br>compare_atlas_vertices<br>draw-atlas<br>draw-atlas-colors                                                                                            | - ganesh: `drawAtlas` implemented <br>- graphite:  `SkDevice::drawAtlas` -> `SkVertices::Builder` + `graphite::Device::drawVertices` = disappear |
|                      | custommesh<br>custommesh_cs_uniforms<br>custommesh_uniforms<br>mesh_updates<br>mesh_with_effects<br>mesh_with_image<br>mesh_with_paint_color<br>mesh_with_paint_image<br>mesh_zero_init | - ganesh: support `SkMesh` via `ganesh::device::drawMesh`<br>- graphite: `graphite::device::drawMesh` is defined but not implemented.            |
|                      | picture_mesh                                                                                                                                                                            | using ganesh version of `SkMeshes::CopyVertexBuffer` with `ganesh::GrDirectContext` parameter.<br>- graphite:  no draw.                          |


#### Path Shape
---

| Category                | Cases                                                                                                | Description                                                                                                              | Could be Improvement? |
| ----------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------- |
| Stroke style issue      | degeneratesegments                                                                                   | - stroke_and_fill + inverse(Even/Odd or winding) = disappear<br>- stroke + inverse(Even/Odd or winding) = inverse result |                       |
|                         | lineclosepath<br>linepath                                                                            | same                                                                                                                     |                       |
|                         | quadclosepath<br>quadpath                                                                            | same                                                                                                                     |                       |
|                         | cubicclosepath<br>cubicpath<br>cubicpath_shader<br>inverse_paths                                     | same                                                                                                                     |                       |
| RoundCap + Stroke style | zero_length_paths_aa<br>zero_length_paths_bw<br>zero_length_paths_dbl_aa<br>zero_length_paths_dbl_bw | less failure cells(marked as red) in Graphite                                                                            | yes                   |
| Large shape size        | path_huge_aa                                                                                         | draw path from big `SkPath::RRect` = disappear                                                                           |                       |
| Matrix related          | filltypespersp<br>persp_shaders_aa<br>persp_shaders_bw                                               | gradient shader + translate = no gradient effect                                                                         |                       |
|                         | perspective_clip                                                                                     | path + image shader + perspective matrix = no perspective mapping                                                        |                       |
| AA effect               | PlusMergesAA                                                                                         | aa path shape + srcOver blending<br>- ganesh: seam<br>- graphite: seam covered                                           |                       |
| Fp Effect Issues        | tablemaskfilter                                                                                      | path + mask filter<br>- ganesh: white outside shape<br>- graphite: half coverage outside shape                           |                       |

#### Other Primitive Types
---

| Category                         | Cases                                                                                      | Description                                                                                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arc Stroke Style                 | circular_arcs_stroke_butt                                                                  | parameters: kButt_Cap strokeCap + useCenter(true) + sweep(90)+start(10,30) or sweep(180)+start(30)<br> - ganesh: no overlap<br> - graphite: extra overlap area |
| Font stroke + Perspective Matrix | scaledemojiperspective_test                                                                | perspective + glyph font<br>- graphite: thick stroke of 2nd character(ok without perspective)                                                                  |
|                                  | glyph_pos_n_b                                                                              | Perspective Matrix + font (width = 1.2f) + strokeAndFill style<br>- ganesh: normal<br>- graphite: thick strokes                                                |
| Missing Graphite API             | lattice<br>lattice2<br>lattice_alpha<br>ninepatch-stretch<br>ninepatch_edge_case_349428795 | - ganesh: suport `LatticeOp`<br>- graphite: `drawImageLattice` defined but not implemented. (in TODO)                                                          |
| TBC                              | smallcircles                                                                               | drawArc + translate + image blending<br> - graphite: different blending result??                                                                               |

#### NonDraw Ops
---

| Category | Cases                                                                                                                                                                                                                                                 | Description |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| TBD      | wacky_yuv_formats<br>wacky_yuv_formats_cs<br>wacky_yuv_formats_cubic<br>wacky_yuv_formats_domain<br>wacky_yuv_formats_fromimages<br>wacky_yuv_formats_limited<br>wacky_yuv_formats_limited_cs<br>wacky_yuv_formats_limited_fromimages<br>yuv_splitter |             |


## Category - Difference
---

| Type           | Count on HO 7.0 | Count on Android 16 |
| -------------- | --------------- | ------------------- |
| Difference TBD |                 |                     |
| Trivial        | 164+8(??)       |                     |
| Total          | 325             | 326                 |

### Diff - Platform comparison

These cases only behave differently between Graphite and Ganesh on one platform.

| Catgory      | Count | Cases                                                                                                                                                                                                                                                 | Description                               | Improvement? | Trivial |
| ------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------ | ------- |
| HO only      | 11    | annotated_text<br>colorwheelnative<br>gradtext<br>highcontrastfilter<br>lcdoverlap (pixel shifting fixed)<br>mipmap<br>scaledemoji_rendering(more blur, no artifact)<br>skbug_5321<br>textblobblockreordering<br>textfilter_color<br>textfilter_image | Smoother text alias                       | Yes          |         |
|              | 4     | fontscaler (sharply clipped alias)<br>fontscalerdistortable<br>gammatext<br>macaa_colors                                                                                                                                                              | Text alias clipping artifact              |              |         |
|              | 1     | crbug_1073670                                                                                                                                                                                                                                         | Trivial text alias difference             | Yes          | Yes     |
|              | 1     | bmp_filter_quality_repeat<br>crbug_938592                                                                                                                                                                                                             | TBC - rectangle edge shifting one pixel   |              |         |
| Android only | 4     | crbug_10141204<br>crbug_224618<br>textblobmixedsizes<br>textblobmixedsizes_df                                                                                                                                                                         | alias pattern  change                     |              | Yes     |
|              | 1     | hugebitmapshader                                                                                                                                                                                                                                      | shape rendered only in Android grdawn_vk  |              |         |
|              | 1     | BlurDrawImage<br><br>                                                                                                                                                                                                                                 | TBC - noticeable blur pattern change      |              |         |
|              | 1     | slug                                                                                                                                                                                                                                                  | TBC - low text quality                    |              |         |
|              | 1     | xfermodes                                                                                                                                                                                                                                             | TBC - one pixel shifting of small rects   |              |         |
|              | 11    |                                                                                                                                                                                                                                                       | Result missing on HO due to font missing. |              | Yes     |
