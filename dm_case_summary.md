
# Overview 
---

| Skia version | Platform       | Suite src | Baseline  |
| ------------ | -------------- | --------- | --------- |
| m144         | HO 7.0(api 23) | gm        | glesdmsaa |


Classify types of difference between `grdawn_vk` vs baseline as  
- **Match**:  structure similarity > 99.99
- **Trivial**: barely noticeable difference but similarity < 99.99
- **Difference**: noticeable difference to be classified
	- Line alias pattern
	- Dot line pattern change
	- Stroked rectangle shifting
	- etc.
- **Failure**: no result or missing elements


# Summary

## Category - Failure
---

| Type                 | Count(HO 7.0) | Android Count(Pixel 9)                          |
| -------------------- | ------------- | ----------------------------------------------- |
| unsupported features | 44            | 44                                              |
| missing elements     | 74            | 74 (with 7 cases different results HO vs Pixel) |

### Failure - unsupported features
- no read to unpremultiply support - see `graphite::Context::asyncRescaleAndReadPixels`
- no cached texture - `ganesh::GrTextureGenerator` support
- no non-multiple-of-four texture support (due to D3D combability of Dawn)
- direct `ganesh::SurfaceDrawContext` test API
- direct `ganesh::SurfaceFillContext` test API
- direct `ganesh::GrDirectContext` view API

###  Failure - missing elements

#### Rectangle

| Category                     | Case                                                                                                                                                                                                                                                                                                                                                                                                                                        | Description                                                                                                                                 | Could be Improvement? |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| Rectangle size numeric error | bigrect                                                                                                                                                                                                                                                                                                                                                                                                                                     | big rect size: numeric overflow?(>5e10f), stroke width = 0<br>fill+ noaa: less blur, stoke rectangle shifting                               |                       |
|                              | clipdrawdraw                                                                                                                                                                                                                                                                                                                                                                                                                                | rect size: clipping rounding error(0.5 vs 0.499), one pixel shifting                                                                        |                       |
|                              | fast_constraint_red_is_allowed                                                                                                                                                                                                                                                                                                                                                                                                              | kFast_SrcRectConstraint  + drawImageRect + bigRect(2K)                                                                                      |                       |
| Linear mipmap filter         | bleed_downscale                                                                                                                                                                                                                                                                                                                                                                                                                             | parameters: kFast_SrcRectConstraint + linear filter + linear mipmap <br>- ganesh: red-ish result <br>- graphite: still blue                 |                       |
| drawEdgeAAQuad               | compositor_quads_filter                                                                                                                                                                                                                                                                                                                                                                                                                     | parameters:  drawEdgeAAQuad(required by maskfilter ) + perspective matrix                                                                   |                       |
|                              | crbug_1174186                                                                                                                                                                                                                                                                                                                                                                                                                               | drawEdgeAAQuad + large matrix + line quad<br>- ganesh: white <br >- graphite:  can draw  (correct?)                                         | yes                   |
|                              | draw_quad_set                                                                                                                                                                                                                                                                                                                                                                                                                               | no `ganesh::SurfaceDrawContext::fillRectWithEdgeAA` support - color gradient effect                                                         |                       |
| Fp Effects                   | emboss                                                                                                                                                                                                                                                                                                                                                                                                                                      | - ganesh: no emboss effect<br>- graphite: embossMaskFilter rendered                                                                         | yes                   |
|                              | hardstop_gradients_many                                                                                                                                                                                                                                                                                                                                                                                                                     | GradientShader + translate(y>1000) = effect direction shift                                                                                 |                       |
|                              | image_dither                                                                                                                                                                                                                                                                                                                                                                                                                                | gradientShader + dither <br>- ganesh: no dither, = original<br>- graphite:  dither applied                                                  | yes                   |
| graphite test API support    | compositor_quads_image                                                                                                                                                                                                                                                                                                                                                                                                                      | graphite version of `sk_gpu_test::LazyYUVImage::refImage` defined but not used.                                                             |                       |
|                              | image-shader<br>image-surface                                                                                                                                                                                                                                                                                                                                                                                                               | `SkSurfaces::RenderTarget` allocation <br>- ganesh: via `ganesh::GrRecordingContext`<br>- graphite: via `graphite::recorder`, missing logic |                       |
| TBC                          | drawable                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                                                                                                             |                       |
| TBD                          | lineclosepath<br>linepath<br>makecolortypeandspace<br>PlusMergesAA<br>path_huge_aa<br>persp_shaders_aa<br>persp_shaders_bw<br>perspective_clip<br>picture_mesh<br>quadclosepath<br>quadpath<br>scale-pixels<br>scaledemojiperspective_test<br>skbug_12212<br>skbug_14554<br><br>strict_constraint_batch_no_red_allowed<br>strict_constraint_no_red_allowed<br>textureimage_and_shader<br>verylarge_picture_image<br>verylargebitmap<br><br> |                                                                                                                                             |                       |

#### Vertices

| Category          | Case                                                                                                                                                                               | Description                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| drawAtlas support | imagefiltersbase                                                                                                                                                                   | - graphite:  `SkDevice::drawAtlas` -> `SkVertices::Builder` + `graphite::Device::drawVertices` = disappear |
| TBD               | compare_atlas_vertices                                                                                                                                                             |                                                                                                            |
|                   | custommesh<br>custommesh_cs_uniforms<br>custommesh_uniforms                                                                                                                        |                                                                                                            |
|                   | lattice<br>lattice2<br>lattice_alpha                                                                                                                                               |                                                                                                            |
|                   | mesh_updates<br>mesh_with_effects<br>mesh_with_image<br>mesh_with_paint_color<br>mesh_with_paint_image<br>mesh_zero_init<br>ninepatch-stretch<br>ninepatch_edge_case_349428795<br> |                                                                                                            |

#### Path Shape

| Category           | Case                                                                                                                                                                     | Description                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Stroke style issue | degeneratesegments                                                                                                                                                       | - stroke_and_fill + inverse Even/Odd = disappear<br>- stroke + inverse(Even/Odd or winding) = inverse result |
| Fp Effect Issues   | filltypespersp                                                                                                                                                           | gradient shader + translate = no gradient effect                                                             |
| TBD                | cubicclosepath<br>cubicpath<br>cubicpath_shader<br>inverse_paths<br>zero_length_paths_aa<br>zero_length_paths_bw<br>zero_length_paths_dbl_aa<br>zero_length_paths_dbl_bw |                                                                                                              |
|                    |                                                                                                                                                                          |                                                                                                              |

#### Other Prim Types

| Category         | Case                            | Description                                                                                                                        |
| ---------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Arc Stroke Style | circular_arcs_stroke_butt       | parameters: kButt_Cap strokeCap + useCenter(true) + sweep(90)+start(10,30) or sweep(180)+start(30)<br>graphite: extra overlap area |
| TBD              | draw-atlas<br>draw-atlas-colors |                                                                                                                                    |
|                  | glyph_pos_n_b                   |                                                                                                                                    |
|                  | smallcircles                    |                                                                                                                                    |

#### NonDraw Ops
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
