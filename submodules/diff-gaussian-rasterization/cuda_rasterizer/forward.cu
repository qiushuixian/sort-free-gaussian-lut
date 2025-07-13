/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __constant__ float pi = 3.14159265358979323846f;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
	const bool if_depth_correct,
	const float sigma,
	bool* valid_gs,
	int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* vi,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	float* alpha_weight,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();	
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	valid_gs[idx] = false;

	// Perform near culling, quit if outside.

    // filter 1: in_frustum and depth
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	if (if_depth_correct){
		if (p_view.z >= sigma)
			return;
	}

	// filter 2: opacity
	float opacity = opacities[idx];
	if(opacity <= 0.0f)
		return;

	// filter 3: vi
	float t_vi = vi[idx];
	if (t_vi <= 0.0f)
		return;
	float depth_correct;

	if (if_depth_correct){
		depth_correct = 1.0f - (p_view.z / sigma);
	}
	else{
		depth_correct = 1.0f;
	}

	alpha_weight[idx] = t_vi * depth_correct;	
	valid_gs[idx] = true;

	// end filter
	
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4

	// // // LC-WSR
	// w(d_i) = max(0, 1 - (d_i/sigma)) * v_i
	// modify_opacity = opacity * w(d_i)
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity};

	// conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx]};  // origin method

	// modified by lt
	// float det_bias = ((cov.x+0.3) * (cov.z+0.3) - cov.y * cov.y);
	// conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx]*sqrt(det/det_bias)};
	
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void filter_preprocessCUDA(int P, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float* cov3Ds,
	const dim3 grid,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;


	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	// float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// float my_radius = ceil(3.f * sqrt(lambda1));
	float my_radius = ceil(2.f * sqrt(lambda1));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;


	radii[idx] = my_radius;
	
}
// Forward version of 2D covariance matrix computation
__device__ float3 anchor_computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float& level_scale, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 cov = level_scale * level_scale * glm::transpose(T) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	// printf("Line 188, forward.cu");
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}
// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.

__device__ unsigned long long g_total_count = 0; 
__device__ unsigned long long g_total_count_1 = 0; 
__device__ unsigned long long g_total_count_2 = 0; 
__device__ unsigned long long g_total_count_3 = 0; 

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const float weight_background,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,  // opacity
	const float* __restrict__ alpha_weight,  // Vi * ( 1 - depth/sigma )
	float* __restrict__ weight_sum,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
    int* radii
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_alpha_weight[BLOCK_SIZE];

	// Initialize helper variables
	float alpha_sum_a = 0.0f;
    float alpha_sum_b = 0.0f;
    float alpha_sum_c = 0.0f;
	float Ca[CHANNELS] = { 0 };
    float Cb[CHANNELS] = { 0 };
    float Cc[CHANNELS] = { 0 }; 
    unsigned long long count = 0;
    unsigned long long count1 = 0;
    unsigned long long count2 = 0;
    unsigned long long count3 = 0;
    float x1=25.0f; 
    float x2=50.0f;
    float max_alpha_sum=0.0f;
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// // End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
		// block.sync();

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_alpha_weight[block.thread_rank()] = alpha_weight[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if(collected_conic_opacity[j].w <= 0.0)
				continue;
            
            count++;
            float4 con_o = collected_conic_opacity[j];
            float w=abs(collected_alpha_weight[j]*con_o.w);
            float w_new=min(255.0f*w,1.1f);
            float r=w;
            float k=abs(radii[collected_id[j]]/r);
            if (k<x1)
            {
                count1++;
            }
			if (k>=x1&&k<=x2)
            {
                count2++;
            }
            if (k>x2)
            {
                count3++;
            }
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float alpha = con_o.w * exp(power);
			if (abs(alpha) < 1.0f / 255.0f)
				continue;

			uint32_t gs_id = collected_id[j];

			
			alpha *= collected_alpha_weight[j];
            if (k<x1)
            {
			    for (int ch = 0; ch < CHANNELS; ch++)
				    Ca[ch] += features[gs_id * CHANNELS + ch] * alpha;

			    alpha_sum_a += alpha;
                //count1++;
            }
			if (k>=x1&&k<=x2)
            {
			    for (int ch = 0; ch < CHANNELS; ch++)
				    Cb[ch] += features[gs_id * CHANNELS + ch] * alpha;

			    alpha_sum_b += alpha;
                //count2++;
            }
            if (k>x2)
            {
			    for (int ch = 0; ch < CHANNELS; ch++)
				    Cc[ch] += features[gs_id * CHANNELS + ch] * alpha;

			    alpha_sum_c += alpha;
                //count3++;
            }

		}
	}

    block.sync();
    __syncthreads();

    atomicAdd(&g_total_count, count);
    atomicAdd(&g_total_count_1, count1);
    atomicAdd(&g_total_count_2, count2);
    atomicAdd(&g_total_count_3, count3);

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		alpha_sum_c += weight_background;
		if (alpha_sum_a == 0.0f)
			alpha_sum_a = 0.0001f;
        if (alpha_sum_b == 0.0f)
			alpha_sum_b = 0.0001f;
        if (alpha_sum_c == 0.0f)
			alpha_sum_c = 0.0001f;
    }

     block.sync();
    __syncthreads();

    if (inside)
    {
		for (int ch = 0; ch < CHANNELS; ch++)
			{
                out_color[ch * H * W + pix_id]=((threadIdx.x%2==0&&threadIdx.y%2==0)||(threadIdx.x==15||threadIdx.y==15)) ? Cb[ch] : 0;
                
            }  
        weight_sum[pix_id]=alpha_sum_b;
    }

     block.sync();
    __syncthreads();

    if (inside)
    {
        if ((threadIdx.x%2!=0||threadIdx.y%2!=0)&&threadIdx.x!=15&&threadIdx.y!=15)
        {
            if (threadIdx.x%2==1&&threadIdx.y%2==0)
            {
                for (int ch = 0; ch < CHANNELS; ch++)
			        out_color[ch * H * W + pix_id]=0.5f*(out_color[ch * H * W + pix_id-1]+out_color[ch * H * W + pix_id+1]);
                weight_sum[pix_id]=0.5f*(weight_sum[pix_id-1]+weight_sum[pix_id+1]);
            }
        else if (threadIdx.x%2==0&&threadIdx.y%2==1)
            {
                for (int ch = 0; ch < CHANNELS; ch++)
			        out_color[ch * H * W + pix_id]=0.5f*(out_color[ch * H * W + pix_id-W]+out_color[ch * H * W + pix_id+W]);
                weight_sum[pix_id]=0.5f*(weight_sum[pix_id-W]+weight_sum[pix_id+W]);
            }
        else if (threadIdx.x%2==1&&threadIdx.y%2==1)
            {
                for (int ch = 0; ch < CHANNELS; ch++)
			        out_color[ch * H * W + pix_id]=0.25f*(out_color[ch * H * W + pix_id-1-W]+
                            out_color[ch * H * W + pix_id+1+W]+out_color[ch * H * W + pix_id+1-W]+out_color[ch * H * W + pix_id-1+W]);
                weight_sum[pix_id]=0.25f*(weight_sum[pix_id-1-W]+
                        weight_sum[pix_id+1+W]+weight_sum[pix_id+1-W]+weight_sum[pix_id-1+W]);
            }
        }
    }

     block.sync();
    __syncthreads();

    if (inside)
    {
        alpha_sum_b=weight_sum[pix_id];
        for (int ch = 0; ch < CHANNELS; ch++)
			{
                Cb[ch]=out_color[ch * H * W + pix_id];
                
            }
    }

     block.sync();
    __syncthreads();

    if (inside)
    {
        for (int ch = 0; ch < CHANNELS; ch++)
			{
                out_color[ch * H * W + pix_id]=(threadIdx.x%3==0&&threadIdx.y%3==0) ? (Cc[ch]+bg_color[ch] * weight_background) : 0;
                
            }
        weight_sum[pix_id]=alpha_sum_c;
    }

     block.sync();
    __syncthreads();

    if (inside)
    {
        if (threadIdx.x%3!=0||threadIdx.y%3!=0)
        {
            for (int ch = 0; ch < CHANNELS; ch++)
            {
                
                //__syncthreads();
                if (threadIdx.x%3==0&&threadIdx.y%3==1) 
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id-W]*2.0f+out_color[ch * H * W + pix_id+2*W])/3.0f;}
                else if (threadIdx.x%3==0&&threadIdx.y%3==2) 
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id-2*W]+out_color[ch * H * W + pix_id+W]*2.0f)/3.0f;}
                else if (threadIdx.x%3==1&&threadIdx.y%3==0) 
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id-1]*2.0f+out_color[ch * H * W + pix_id+2])/3.0f;}
                else if (threadIdx.x%3==2&&threadIdx.y%3==0) 
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id-2]+out_color[ch * H * W + pix_id+1]*2.0f)/3.0f;}
                else if (threadIdx.x%3==1&&threadIdx.y%3==1) 
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id-W-1]*4.0f+out_color[ch * H * W + pix_id+2*W-1]*2.0f+
                                                out_color[ch * H * W + pix_id-W+2]*2.0f+out_color[ch * H * W + pix_id+2+2*W])/9.0f;}
                else if (threadIdx.x%3==1&&threadIdx.y%3==2) 
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id+W-1]*4.0f+out_color[ch * H * W + pix_id+W+2]*2.0f+
                                                out_color[ch * H * W + pix_id-1-2*W]*2.0f+out_color[ch * H * W + pix_id+2-2*W])/9.0f;}
                else if (threadIdx.x%3==2&&threadIdx.y%3==1) 
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id-W+1]*4.0f+out_color[ch * H * W + pix_id-W-2]*2.0f+
                                                out_color[ch * H * W + pix_id+1+2*W]*2.0f+out_color[ch * H * W + pix_id-2+2*W])/9.0f;}
                else if (threadIdx.x%3==2&&threadIdx.y%3==2)  
                    {out_color[ch * H * W + pix_id]=(out_color[ch * H * W + pix_id+W+1]*4.0f+out_color[ch * H * W + pix_id+W-2]*2.0f+
                                                out_color[ch * H * W + pix_id-2*W+1]*2.0f+out_color[ch * H * W + pix_id-2-2*W])/9.0f;}
            }
            //__syncthreads();
                if (threadIdx.x%3==0&&threadIdx.y%3==1) 
                    {weight_sum[pix_id]=(weight_sum[pix_id-W]*2.0f+weight_sum[pix_id+2*W])/3.0f;}
                else if (threadIdx.x%3==0&&threadIdx.y%3==2) 
                    {weight_sum[pix_id]=(weight_sum[pix_id-2*W]+weight_sum[pix_id+W]*2.0f)/3.0f;}
                else if (threadIdx.x%3==1&&threadIdx.y%3==0) 
                    {weight_sum[pix_id]=(weight_sum[pix_id-1]*2.0f+weight_sum[pix_id+2])/3.0f;}
                else if (threadIdx.x%3==2&&threadIdx.y%3==0) 
                    {weight_sum[pix_id]=(weight_sum[pix_id-2]+weight_sum[pix_id+1]*2.0f)/3.0f;}
                else if (threadIdx.x%3==1&&threadIdx.y%3==1) 
                    {weight_sum[pix_id]=(weight_sum[pix_id-W-1]*4.0f+weight_sum[pix_id+2*W-1]*2.0f+
                                                weight_sum[pix_id-W+2]*2.0f+weight_sum[pix_id+2+2*W])/9.0f;}
                else if (threadIdx.x%3==1&&threadIdx.y%3==2) 
                    {weight_sum[pix_id]=(weight_sum[pix_id+W-1]*4.0f+weight_sum[pix_id+W+2]*2.0f+
                                                weight_sum[pix_id-1-2*W]*2.0f+weight_sum[pix_id+2-2*W])/9.0f;}
                else if (threadIdx.x%3==2&&threadIdx.y%3==1) 
                    {weight_sum[pix_id]=(weight_sum[pix_id-W+1]*4.0f+weight_sum[pix_id-W-2]*2.0f+
                                                weight_sum[pix_id+1+2*W]*2.0f+weight_sum[pix_id-2+2*W])/9.0f;}
                else if (threadIdx.x%3==2&&threadIdx.y%3==2)  
                    {weight_sum[pix_id]=(weight_sum[pix_id+W+1]*4.0f+weight_sum[pix_id+W-2]*2.0f+
                                                weight_sum[pix_id-2*W+1]*2.0f+weight_sum[pix_id-2-2*W])/9.0f;}
        }
        
        
    }

     block.sync();
    __syncthreads();

    if (inside)
    {
        alpha_sum_c=weight_sum[pix_id];
        for (int ch = 0; ch < CHANNELS; ch++)
			{
                Cc[ch]=out_color[ch * H * W + pix_id];
                
            }
    }
    block.sync();
    __syncthreads();
    float weight_sum_float=0.0f;
    int weight_sum_int=0;
    float weight_sum_final=0.0f;
    if (inside)
    {
        weight_sum_float=(alpha_sum_a+alpha_sum_b+alpha_sum_c);
        
        if (weight_sum_float<=4.0f)
        {
            weight_sum_int=(int) (weight_sum_float*16.0f+0.5f);
            weight_sum_final=(float) weight_sum_int/16.0f; 
            //weight_sum_final=weight_sum_float;   
        }
        else if (weight_sum_float<=8.0f)
        {
            weight_sum_int=(int) (weight_sum_float*8.0f+0.5f);
            weight_sum_final=(float) weight_sum_int/8.0f; 
            //weight_sum_final=weight_sum_float;   
        }
        else if (weight_sum_float<=16.0f)
        {
            weight_sum_int=(int) (weight_sum_float*4.0f+0.5f);
            weight_sum_final=(float) weight_sum_int/4.0f;  
            //weight_sum_final=weight_sum_float;     
        }
        else if (weight_sum_float<=32.0f)
        {
            weight_sum_int=(int) (weight_sum_float*2.0f+0.5f);
            weight_sum_final=(float) weight_sum_int/2.0f;    
        }
        else if (weight_sum_float<=64.0f)
        {
            weight_sum_int=(int) (weight_sum_float*1.0f+0.5f);
            weight_sum_final=(float) weight_sum_int/1.0f;    
        }
        else if (weight_sum_float<=128.0f)
        {
            weight_sum_int=(int) (weight_sum_float*0.5f+0.5f);
            weight_sum_final=(float) weight_sum_int/0.5f;    
        }
        else if (weight_sum_float<=256.0f)
        {
            weight_sum_int=(int) (weight_sum_float*0.25f+0.5f);
            weight_sum_final=(float) weight_sum_int/0.25f;    
        }
        else if (weight_sum_float<=512.0f)
        {
            weight_sum_int=(int) (weight_sum_float*0.125f+0.5f);
            weight_sum_final=(float) weight_sum_int/0.125f;    
        }
        else
        {
            weight_sum_final=513.0f;
        }
        for (int ch = 0; ch < CHANNELS; ch++)
			{
                out_color[ch * H * W + pix_id]=(Ca[ch]+Cb[ch]+Cc[ch])/weight_sum_final;
                
                
            }
        weight_sum[pix_id]=alpha_sum_a+alpha_sum_b+alpha_sum_c;
        if (abs(alpha_sum_a+alpha_sum_b+alpha_sum_c)>max_alpha_sum)
        {
            max_alpha_sum=abs(alpha_sum_a+alpha_sum_b+alpha_sum_c);
        }
    }
    if (blockIdx.x == 20 && blockIdx.y == 20 && threadIdx.x == 0 && threadIdx.y == 0) 
        {
            printf("Total valid gaussians sampled: %llu\n", g_total_count);
            printf("Total valid gaussians_1 sampled: %llu\n", g_total_count_1);
            printf("Total valid gaussians_2 sampled: %llu\n", g_total_count_2);
            printf("Total valid gaussians_3 sampled: %llu\n", g_total_count_3);
            printf("Ratio: %f\n", (float) (g_total_count_1+g_total_count_2*0.37f+g_total_count_3*0.1112f)/g_total_count);
            printf("Max_sum: %f\n", (float) max_alpha_sum);
        }
	
}


void FORWARD::render(
	const float weight_background,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	const float* alpha_weight,
	float* weight_sum,
	const float* bg_color,
	float* out_color,
    int* radii
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		weight_background,
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		alpha_weight,
		weight_sum,
		bg_color,
		out_color,
        radii
		);
}


void FORWARD::preprocess(
	const bool if_depth_correct,
	const float sigma,
	bool* valid_gs,
	int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* vi,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	float* alpha_weight,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		if_depth_correct,
		sigma,
		valid_gs,
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		vi,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		alpha_weight,
		grid,
		tiles_touched,
		prefiltered
		);
}


void FORWARD::filter_preprocess(int P, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float* cov3Ds,
	const dim3 grid,
	bool prefiltered)
{

	filter_preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		cov3D_precomp,
		viewmatrix, 
		projmatrix,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		cov3Ds,
		grid,
		prefiltered
		);
}