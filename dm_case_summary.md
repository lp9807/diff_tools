
# Overview 
---

| Skia version | Platform       | Case src | Baseline  |
| ------------ | -------------- | -------- | --------- |
| m144         | HO 7.0(api 23) | gm       | glesdmsaa |


Classify types of difference between `grdawn_vk` vs baseline(`glesdmsaa`) as  
- **Match**:  structure similarity > 99.99
- **Trivial**: barely noticeable difference but similarity < 99.99
- **Noticeable Difference**: visual difference but without feature missing. To be classified by pattern
	- Line alias pattern
	- Dot line pattern change
	- Stroked rectangle shifting
	- etc.
- **Failure**: no result or visually noticeable elements missing

# Summary

## Category - Failure
---

| Type                 | Count on HO 7.0 | Count on Android 16 | Notes                                                        |
| -------------------- | --------------- | ------------------- | ------------------------------------------------------------ |
| unsupported features | 44              | 44                  |                                                              |
| missing elements     | 74              | 74                  | HO vs Android(grdawn_vk): 7 cases with noticeable difference |
| general errors       | 4               | 4                   |                                                              |


### Failure - General errors

These cases fail due to platform configuration, existing in both ganesh and graphite.

| Sub-category | Description                                                                                                                                                                      | Count |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| Font missing | missing font /system/fonts/NotoColorEmojiLegacy.ttf<br>scaledemojiperspective_svg<br>scaledemojiperspective_colrv0<br>scaledemojiperspective_sbix<br>scaledemojiperspective_cbdt | 4     |



### Failure - Unsupported features

These cases rely on features only exist in ganesh.

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


| Sub-category     | Total Count | Improvement Candidate Count |
| ---------------- | ----------- | --------------------------- |
| Rectangle        | 22          | 4                           |
| Vertices         | 18          |                             |
| Path Shape       | 19          |                             |
| Other Primitives | 6           |                             |
| Non-Draw Op      | 9           |                             |


#### Rectangle
---

| Category                     | Case                                                                                                         | Description                                                                                                                                                | Could be Improvement? |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| Rectangle size numeric error | bigrect                                                                                                      | drawRect:<br>stroke style + (stroke width == 0): numeric overflow?(rect size>5e10f)<br>fill style + noaa: less blur, stoke rectangle shifting              |                       |
|                              | clipdrawdraw                                                                                                 | drawRect: clipping rounding error(0.5 vs 0.499), <br> - graphite: one pixel shifting                                                                       |                       |
|                              | fast_constraint_red_is_allowed<br>strict_constraint_batch_no_red_allowed<br>strict_constraint_no_red_allowed | drawImageRect + bigRect(2K)<br> - graphite: primitive not drawn (ok with `SkTiledImageUtils::DrawImageRect`)                                               |                       |
|                              | verylarge_picture_image<br>verylargebitmap                                                                   | same                                                                                                                                                       |                       |
| drawImageRect                | bleed_downscale                                                                                              | kFast_SrcRectConstraint + linear filter + linear mipmap <br>- ganesh: red-ish result <br>- graphite: still blue                                            |                       |
|                              | makecolortypeandspace                                                                                        | ColorType issue:  kRGB_565_SkColorType or kGray_8_SkColorType<br>- ganesh: expected failure ( no change )<br> - graphite: quantized or gray effect         | yes                   |
| drawEdgeAAQuad               | compositor_quads_filter                                                                                      | parameters:  drawEdgeAAQuad(required by maskfilter ) + perspective matrix<br>- graphite: texture position shift                                            |                       |
|                              | crbug_1174186                                                                                                | drawEdgeAAQuad + large matrix + line quad<br>- ganesh:  disappear <br >- graphite:  draw something  (correct?)                                             | yes                   |
|                              | draw_quad_set                                                                                                | render color gradient effect<br>- ganesh: utilize `ganesh::SurfaceDrawContext::fillRectWithEdgeAA`<br>- graphite: standard routine without gradient effect |                       |
| Fp Effects                   | emboss                                                                                                       | - ganesh: no emboss effect<br>- graphite: embossMaskFilter rendered                                                                                        | yes                   |
|                              | hardstop_gradients_many                                                                                      | GradientShader + translate(y>1000) = effect direction shift                                                                                                |                       |
|                              | image_dither                                                                                                 | gradientShader + dither <br>- ganesh: no dither, = original<br>- graphite:  dither applied                                                                 | yes                   |
| graphite test API support    | compositor_quads_image                                                                                       | graphite version of `sk_gpu_test::LazyYUVImage::refImage` defined but not used.                                                                            |                       |
|                              | image-shader<br>image-surface<br>skbug_12212                                                                 | `SkSurfaces::RenderTarget` allocation <br>- ganesh: via `ganesh::GrRecordingContext`<br>- graphite: missing logic of `graphite::recorder`                  |                       |
|                              | scale-pixels                                                                                                 | `SkImage::scalePixels` requires internal GrDirectContext support<br>- ganesh: correct<br>- graphite: scaled pixels disappear.                              |                       |
| TBC                          | drawable                                                                                                     |                                                                                                                                                            |                       |
|                              | textureimage_and_shader                                                                                      | drawImageRect + draw image shader (image = green)<br>- ganesh: green<br>- graphite: red                                                                    |                       |

#### Vertices
---

| Category                  | Case                                                                                                                                                                               | Description                                                                                                     |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| drawAtlas support         | imagefiltersbase<br>skbug_14554                                                                                                                                                    | - graphite:  `SkDevice::drawAtlas` -> `SkVertices::Builder` + `graphite::Device::drawVertices` = disappear      |
| graphite test API support | picture_mesh                                                                                                                                                                       | using ganesh version of `SkMeshes::CopyVertexBuffer` with `GrDirectContext` parameter.<br>- graphite:  no draw. |
| TBD                       | compare_atlas_vertices                                                                                                                                                             |                                                                                                                 |
|                           | custommesh<br>custommesh_cs_uniforms<br>custommesh_uniforms                                                                                                                        |                                                                                                                 |
|                           | lattice<br>lattice2<br>lattice_alpha                                                                                                                                               |                                                                                                                 |
|                           | mesh_updates<br>mesh_with_effects<br>mesh_with_image<br>mesh_with_paint_color<br>mesh_with_paint_image<br>mesh_zero_init<br>ninepatch-stretch<br>ninepatch_edge_case_349428795<br> |                                                                                                                 |

#### Path Shape
---

| Category                     | Case                                                                                                                                                                     | Description                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Stroke style issue           | degeneratesegments                                                                                                                                                       | - stroke_and_fill + inverse(Even/Odd or winding) = disappear<br>- stroke + inverse(Even/Odd or winding) = inverse result |
|                              | lineclosepath<br>linepath                                                                                                                                                | same                                                                                                                     |
|                              | quadclosepath<br>quadpath                                                                                                                                                | same                                                                                                                     |
| Rectangle size numeric error | path_huge_aa                                                                                                                                                             | draw path from big `SkPath::RRect` = disappear                                                                           |
| Perspective matrix           |                                                                                                                                                                          |                                                                                                                          |
| AA effect                    | PlusMergesAA                                                                                                                                                             | aa path shape + srcOver blending<br>- ganesh: seam<br>- graphite: seam covered                                           |
| Fp Effect Issues             | filltypespersp<br>persp_shaders_aa<br>persp_shaders_bw                                                                                                                   | gradient shader + translate = no gradient effect                                                                         |
|                              | perspective_clip                                                                                                                                                         | path + image shader + perspective matrix = no perspective mapping                                                        |
| TBD                          | cubicclosepath<br>cubicpath<br>cubicpath_shader<br>inverse_paths<br>zero_length_paths_aa<br>zero_length_paths_bw<br>zero_length_paths_dbl_aa<br>zero_length_paths_dbl_bw |                                                                                                                          |
|                              |                                                                                                                                                                          |                                                                                                                          |

#### Other Primitive Types
---

| Category         | Case                            | Description                                                                                                                                                    |
| ---------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Arc Stroke Style | circular_arcs_stroke_butt       | parameters: kButt_Cap strokeCap + useCenter(true) + sweep(90)+start(10,30) or sweep(180)+start(30)<br> - ganesh: no overlap<br> - graphite: extra overlap area |
| Font stroke      | scaledemojiperspective_test     | perspective + glyph font<br>- graphite: thick stroke of 2nd character(ok without perspective)                                                                  |
| TBD              | draw-atlas<br>draw-atlas-colors |                                                                                                                                                                |
|                  | glyph_pos_n_b                   |                                                                                                                                                                |
|                  | smallcircles                    |                                                                                                                                                                |

#### NonDraw Ops
---

| Category | Case                                                                                                                                                                                                                                                   | Description |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- |
| TBD      | wacky_yuv_formats<br>wacky_yuv_formats_cs<br>wacky_yuv_formats_cubic<br>wacky_yuv_formats_domain<br>wacky_yuv_formats_fromimages<br>wacky_yuv_formats_limited<br>wacky_yuv_formats_limited_cs<br>wacky_yuv_formats_limited_fromimages<br>yuv_splittert |             |


## Category - Difference
---

| Type                  | Count     |
| --------------------- | --------- |
| Noticeable Difference |           |
| Trivial               | 164+3(??) |
| Total                 | 299       |
