// vim: set syntax=asm :
//
// Intel AMX int8 GEMM kernel, 16 M-rows x 16 N-cols i32 accumulator output.
//
// One `tdpbssd tmm0, tmm1, tmm2` instruction performs:
//   tmm0[m, n] += sum_{k=0..63} A[m, k] * B[k, n]
// for m=0..15, n=0..15: 16 * 16 * 64 = 16384 mul-adds per single instruction.
// That's 4x the work-per-instruction of the 8x8 sibling kernel, hitting the
// full AMX i8 tile geometry (max colsb=64, max rows=16, max bytes=1024).
//
// Tile geometry (palette 1):
//   tmm0 = C accumulator: 16 rows x 64 colsb = 16 M-rows x 16 N-cols of i32
//   tmm1 = A tile:        16 rows x 64 colsb = 16 M-rows x 64 K-bytes per iter
//   tmm2 = B tile:        16 rows x 64 colsb = 16 K-pair-rows x 16 N-cols * 4 K
//
// `tdpbssd` is signed-signed, so no +128 trick is needed; the i32 accumulators
// are bit-identical to the AVX2 / VNNI / 8x8-AMX reference paths.
//
// A is packed via PackedAmxA(16): per panel of 16 M-rows, row-major within the
// panel, K-bytes contiguous along the row, K_padded = ceil(K/64)*64.
// B reuses PackedI8K4(16): per K=4 block, 16 N-cols * 4 K-bytes = 64 bytes;
// 16 such K-blocks per tmm2 tile = 1024 bytes = one tileloadd.
//
// REGISTER LAYOUT
//   zmm0..zmm15 = accumulators, ROW-MAJOR: zmm{m} holds the 16 i32 lanes
//                 [C[m, 0], C[m, 1], ..., C[m, 15]] for row m.
// This matches the row-major i32 layout that `tilestored` writes directly,
// so the hot path (Clear -> AddMatMul -> Store) needs no transpose.



.intel_syntax noprefix
.text
.p2align 5
.globl avx512amx_mmm_i32_16x16_0_23_4
avx512amx_mmm_i32_16x16_0_23_4:
.cfi_startproc



    push        rbp
    mov         rbp, rsp



    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15

    sub         rsp, 8


.cfi_def_cfa_offset 64


    stmxcsr     [rsp + 4]

    mov         rax, 0x1FC0

    mov         [rsp], eax
    ldmxcsr     [rsp]

    // Reserve 64 bytes for the AMX tile-config block, zero it, populate
    // palette + dims (all three tiles are 16 rows x 64 colsb, the maximum
    // i8 tile geometry on Sapphire Rapids / Emerald Rapids / Granite Rapids).
    sub         rsp, 64
    vpxor       xmm15, xmm15, xmm15
    vmovdqu     [rsp     ], xmm15
    vmovdqu     [rsp + 16], xmm15
    vmovdqu     [rsp + 32], xmm15
    vmovdqu     [rsp + 48], xmm15
    mov         byte ptr  [rsp + 0 ], 1    // palette = 1
    mov         word ptr  [rsp + 16], 64   // colsb[0] = 64 (tmm0)
    mov         word ptr  [rsp + 18], 64   // colsb[1] = 64 (tmm1)
    mov         word ptr  [rsp + 20], 64   // colsb[2] = 64 (tmm2)
    mov         byte ptr  [rsp + 48], 16   // rows[0]  = 16 (tmm0)
    mov         byte ptr  [rsp + 49], 16   // rows[1]  = 16 (tmm1)
    mov         byte ptr  [rsp + 50], 16   // rows[2]  = 16 (tmm2)
    ldtilecfg   [rsp]

// vim: set syntax=asm :

.Lnon_linear:

.Lnon_linear_loop_enter:
    sub     rdi,    40
.Lnon_linear_loop:
    add     rdi,    40
    mov     rax,    [rdi]

    mov     r8, 29
    cmp     rax, 0
    cmovl   rax, r8
    cmp     rax, 29
    cmovg   rax, r8


    lea     r8, [ rip + .Ljmp_table ]

    movsxd  r9, dword ptr [ r8 + rax * 4 ]
    lea     r8, [ r8 + r9 ]
    jmp     r8

.Ljmp_table:

    .long      .Ldone-.Ljmp_table

    .long      .Lclear-.Ljmp_table

    .long      .Lload_tile-.Ljmp_table

    .long      .Lscalar_min-.Ljmp_table

    .long      .Lscalar_max-.Ljmp_table

    .long      .Lscalar_add-.Ljmp_table

    .long      .Lscalar_mul-.Ljmp_table

    .long      .Lscalar_sub-.Ljmp_table

    .long      .Lscalar_sub_flipped-.Ljmp_table

    .long      .Lleaky_relu-.Ljmp_table

    .long      .Lper_row_min-.Ljmp_table

    .long      .Lper_row_max-.Ljmp_table

    .long      .Lper_row_add-.Ljmp_table

    .long      .Lper_row_mul-.Ljmp_table

    .long      .Lper_row_sub-.Ljmp_table

    .long      .Lper_row_sub_flipped-.Ljmp_table

    .long      .Lper_col_min-.Ljmp_table

    .long      .Lper_col_max-.Ljmp_table

    .long      .Lper_col_add-.Ljmp_table

    .long      .Lper_col_mul-.Ljmp_table

    .long      .Lper_col_sub-.Ljmp_table

    .long      .Lper_col_sub_flipped-.Ljmp_table

    .long      .Lq_scale-.Ljmp_table

    .long      .Lq_shr-.Ljmp_table

    .long      .Lq_shl-.Ljmp_table

    .long      .Ladd_unicast-.Ljmp_table

    .long      .Ladd_row_col_products-.Ljmp_table

    .long      .Lstore-.Ljmp_table

    .long      .Ladd_mat_mul-.Ljmp_table

    .long      .Lunsupported-.Ljmp_table

.Lunsupported:
    mov     rax,    1
    jmp     .Lreturn


.Ldone:
    mov     rax, 0
    jmp     .Lreturn


.Lclear:
    // vzeroall only zeros lower-256 of zmm0..15; explicitly zero the full
    // accumulators (zmm0..zmm15) for AMX.
    
    vpxorq      zmm0, zmm0, zmm0
    
    vpxorq      zmm1, zmm1, zmm1
    
    vpxorq      zmm2, zmm2, zmm2
    
    vpxorq      zmm3, zmm3, zmm3
    
    vpxorq      zmm4, zmm4, zmm4
    
    vpxorq      zmm5, zmm5, zmm5
    
    vpxorq      zmm6, zmm6, zmm6
    
    vpxorq      zmm7, zmm7, zmm7
    
    vpxorq      zmm8, zmm8, zmm8
    
    vpxorq      zmm9, zmm9, zmm9
    
    vpxorq      zmm10, zmm10, zmm10
    
    vpxorq      zmm11, zmm11, zmm11
    
    vpxorq      zmm12, zmm12, zmm12
    
    vpxorq      zmm13, zmm13, zmm13
    
    vpxorq      zmm14, zmm14, zmm14
    
    vpxorq      zmm15, zmm15, zmm15
    
    jmp     .Lnon_linear_loop

.Ladd_mat_mul:
    mov     r12,    [rdi + 32]   // packing
    mov     rbx,    [rdi + 24]   // B
    mov     rax,    [rdi + 16]   // A

    mov     rcx,    [rdi + 8]    // k
    test    rcx,    rcx
    jz      .Lnon_linear_loop

    cmp     r12, 1
    je      .Lmain_loop_packed_packed_i8i8

.Lmain_loop_packed_packed:
    // Generic i32 x i32 fallback path (not AMX). For row-major accumulators
    // with zmm{m} = row m of C, accumulating C[m, n] += A[m, k] * B[k, n]:
    //   - load 16 B values for this K row into zmm16 (row of B)
    //   - for each m: broadcast A[m, k], multiply by zmm16, add to zmm{m}
    vmovups         zmm16, [rbx]               // 16 i32 of B at this K row

    
        vpbroadcastd    zmm17, dword ptr [rax + 0 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm0, zmm0, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 1 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm1, zmm1, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 2 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm2, zmm2, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 3 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm3, zmm3, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 4 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm4, zmm4, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 5 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm5, zmm5, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 6 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm6, zmm6, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 7 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm7, zmm7, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 8 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm8, zmm8, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 9 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm9, zmm9, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 10 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm10, zmm10, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 11 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm11, zmm11, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 12 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm12, zmm12, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 13 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm13, zmm13, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 14 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm14, zmm14, zmm18
    
        vpbroadcastd    zmm17, dword ptr [rax + 15 * 4]
        vpmulld         zmm18, zmm16, zmm17
        vpaddd          zmm15, zmm15, zmm18
    

    add             rax,    64                 // 16 i32 lanes per K step
    add             rbx,    64
    dec             rcx
    jnz             .Lmain_loop_packed_packed

    jmp             .Lnon_linear_loop

.Lmain_loop_packed_packed_i8i8:
    // AMX i8 layout:
    //   A panel: 16 M-rows x K_padded K-bytes ROW-major within the panel
    //            (PackedAmxA, K_padded = ceil(K/64)*64).
    //   B panel: PackedI8K4(16) -- 16 N-cols x 4 K-bytes per K=4 block, with
    //            16 K-blocks per tileloadd (16 K-pair-rows x 64 colsb).
    //
    // Per tdpbssd: tmm0[m, n] += sum_{k=0..63} A[m, k] * B[k, n].
    // Inner loop steps along K in 64-K chunks.

    // r8 <- K_padded = ceil(k/64) * 64 = byte-stride between A's M-rows.
    mov             r8,     rcx
    add             r8,     63
    and             r8,     -64

    // rcx <- ceil(k/64) = number of K=64 AMX inner iterations.
    add             rcx,    63
    shr             rcx,    6

    // r9 <- 64 = byte-stride between B's K-pair-rows (each row = 16 N-cols * 4 K).
    mov             r9,     64

    tilezero        tmm0

.Lloop_64k_amx_i8i8_16x16:
    // Cache strategy follows oneDNN's AMX BRGEMM heuristics (Intel-backed):
    //   - A is reused across N-tiles in tract's outer matmul loop, so we use
    //     `tileloadd` (cached, brings into L1) and `prefetcht0` (to L1) for A.
    //   - B streams through once per kernel call (one B-panel per N-tile), and
    //     for the AMX-typical large-matmul case the B working set exceeds the
    //     32 KB L1d on Sapphire Rapids+. We use `tileloaddt1` (non-temporal,
    //     bypasses L1) and `prefetcht1` (to L2) for B -- the same pattern
    //     oneDNN picks when its footprint heuristic crosses the L1 threshold.
    //   - Sapphire Rapids has 16 L1d Fill Buffers (LFBs); each in-flight
    //     prefetch/load consumes one. The previous version's 17 prefetches +
    //     2 active tileloadds overflowed the LFB budget. The reduced count
    //     below leaves headroom and lets the HW streaming prefetcher cover
    //     the remaining B-panel lines.
    //
    // A advances 64 B / iter (one cache line). B advances 1024 B / iter
    // (16 cache lines). We prime 6 of the next 16 B-lines at +1024..+1344,
    // then trust the HW stream prefetcher (very aggressive on SPR/EMR/GNR)
    // to cover lines +1408..+1984.
    prefetcht0      [rax + 64]                 // next A-row K-block (to L1)
    prefetcht1      [rbx + 1024]               // next B-panel head (to L2)
    prefetcht1      [rbx + 1088]
    prefetcht1      [rbx + 1152]
    prefetcht1      [rbx + 1216]
    prefetcht1      [rbx + 1280]
    prefetcht1      [rbx + 1344]
    tileloadd       tmm1,   [rax + r8 * 1]     // A tile (cached): stride = K_padded
    tileloaddt1     tmm2,   [rbx + r9 * 1]     // B tile (non-temporal): stride = 64
    tdpbssd         tmm0,   tmm1,   tmm2
    add             rax,    64                 // +64 K-bytes in A row 0
    add             rbx,    1024               // 16 K-pairs * 64 = 1024 bytes
    dec             rcx
    jnz             .Lloop_64k_amx_i8i8_16x16

    // tmm0 -> stack scratch (16 rows x 64 bytes = 1024 B row-major i32).
    // Then load each row into zmm0..zmm15.  Row m's 16 i32 are contiguous
    // in memory, so each load is a single 64-byte vmovdqu32.
    sub             rsp,    1024
    mov             r10,    rsp
    mov             r11,    64
    tilestored      [r10 + r11 * 1], tmm0

    
    vmovdqu32       zmm0, [r10 + 0]
    
    vmovdqu32       zmm1, [r10 + 64]
    
    vmovdqu32       zmm2, [r10 + 128]
    
    vmovdqu32       zmm3, [r10 + 192]
    
    vmovdqu32       zmm4, [r10 + 256]
    
    vmovdqu32       zmm5, [r10 + 320]
    
    vmovdqu32       zmm6, [r10 + 384]
    
    vmovdqu32       zmm7, [r10 + 448]
    
    vmovdqu32       zmm8, [r10 + 512]
    
    vmovdqu32       zmm9, [r10 + 576]
    
    vmovdqu32       zmm10, [r10 + 640]
    
    vmovdqu32       zmm11, [r10 + 704]
    
    vmovdqu32       zmm12, [r10 + 768]
    
    vmovdqu32       zmm13, [r10 + 832]
    
    vmovdqu32       zmm14, [r10 + 896]
    
    vmovdqu32       zmm15, [r10 + 960]
    

    add             rsp,    1024

    jmp             .Lnon_linear_loop

// ---- Scalar / per-row / per-col elementwise epilogues -------------------

.Lscalar_min:
    vpbroadcastd    zmm16, dword ptr [rdi + 8]
    vpminsd zmm0, zmm0, zmm16
    vpminsd zmm1, zmm1, zmm16
    vpminsd zmm2, zmm2, zmm16
    vpminsd zmm3, zmm3, zmm16
    vpminsd zmm4, zmm4, zmm16
    vpminsd zmm5, zmm5, zmm16
    vpminsd zmm6, zmm6, zmm16
    vpminsd zmm7, zmm7, zmm16
    vpminsd zmm8, zmm8, zmm16
    vpminsd zmm9, zmm9, zmm16
    vpminsd zmm10, zmm10, zmm16
    vpminsd zmm11, zmm11, zmm16
    vpminsd zmm12, zmm12, zmm16
    vpminsd zmm13, zmm13, zmm16
    vpminsd zmm14, zmm14, zmm16
    vpminsd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_max:
    vpbroadcastd    zmm16, dword ptr [rdi + 8]
    vpmaxsd zmm0, zmm0, zmm16
    vpmaxsd zmm1, zmm1, zmm16
    vpmaxsd zmm2, zmm2, zmm16
    vpmaxsd zmm3, zmm3, zmm16
    vpmaxsd zmm4, zmm4, zmm16
    vpmaxsd zmm5, zmm5, zmm16
    vpmaxsd zmm6, zmm6, zmm16
    vpmaxsd zmm7, zmm7, zmm16
    vpmaxsd zmm8, zmm8, zmm16
    vpmaxsd zmm9, zmm9, zmm16
    vpmaxsd zmm10, zmm10, zmm16
    vpmaxsd zmm11, zmm11, zmm16
    vpmaxsd zmm12, zmm12, zmm16
    vpmaxsd zmm13, zmm13, zmm16
    vpmaxsd zmm14, zmm14, zmm16
    vpmaxsd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_add:
    vpbroadcastd    zmm16, dword ptr [rdi + 8]
    vpaddd zmm0, zmm0, zmm16
    vpaddd zmm1, zmm1, zmm16
    vpaddd zmm2, zmm2, zmm16
    vpaddd zmm3, zmm3, zmm16
    vpaddd zmm4, zmm4, zmm16
    vpaddd zmm5, zmm5, zmm16
    vpaddd zmm6, zmm6, zmm16
    vpaddd zmm7, zmm7, zmm16
    vpaddd zmm8, zmm8, zmm16
    vpaddd zmm9, zmm9, zmm16
    vpaddd zmm10, zmm10, zmm16
    vpaddd zmm11, zmm11, zmm16
    vpaddd zmm12, zmm12, zmm16
    vpaddd zmm13, zmm13, zmm16
    vpaddd zmm14, zmm14, zmm16
    vpaddd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_mul:
    vpbroadcastd    zmm16, dword ptr [rdi + 8]
    vpmulld zmm0, zmm0, zmm16
    vpmulld zmm1, zmm1, zmm16
    vpmulld zmm2, zmm2, zmm16
    vpmulld zmm3, zmm3, zmm16
    vpmulld zmm4, zmm4, zmm16
    vpmulld zmm5, zmm5, zmm16
    vpmulld zmm6, zmm6, zmm16
    vpmulld zmm7, zmm7, zmm16
    vpmulld zmm8, zmm8, zmm16
    vpmulld zmm9, zmm9, zmm16
    vpmulld zmm10, zmm10, zmm16
    vpmulld zmm11, zmm11, zmm16
    vpmulld zmm12, zmm12, zmm16
    vpmulld zmm13, zmm13, zmm16
    vpmulld zmm14, zmm14, zmm16
    vpmulld zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lscalar_sub:
    // non-flipped sub = operand - acc (matches fma_mmm_ymm_ops.j2 scalar macro)
    vpbroadcastd    zmm16, dword ptr [rdi + 8]
    vpsubd zmm0, zmm16, zmm0
    vpsubd zmm1, zmm16, zmm1
    vpsubd zmm2, zmm16, zmm2
    vpsubd zmm3, zmm16, zmm3
    vpsubd zmm4, zmm16, zmm4
    vpsubd zmm5, zmm16, zmm5
    vpsubd zmm6, zmm16, zmm6
    vpsubd zmm7, zmm16, zmm7
    vpsubd zmm8, zmm16, zmm8
    vpsubd zmm9, zmm16, zmm9
    vpsubd zmm10, zmm16, zmm10
    vpsubd zmm11, zmm16, zmm11
    vpsubd zmm12, zmm16, zmm12
    vpsubd zmm13, zmm16, zmm13
    vpsubd zmm14, zmm16, zmm14
    vpsubd zmm15, zmm16, zmm15
    
    jmp     .Lnon_linear_loop

.Lscalar_sub_flipped:
    // flipped sub = acc - operand
    vpbroadcastd    zmm16, dword ptr [rdi + 8]
    vpsubd zmm0, zmm0, zmm16
    vpsubd zmm1, zmm1, zmm16
    vpsubd zmm2, zmm2, zmm16
    vpsubd zmm3, zmm3, zmm16
    vpsubd zmm4, zmm4, zmm16
    vpsubd zmm5, zmm5, zmm16
    vpsubd zmm6, zmm6, zmm16
    vpsubd zmm7, zmm7, zmm16
    vpsubd zmm8, zmm8, zmm16
    vpsubd zmm9, zmm9, zmm16
    vpsubd zmm10, zmm10, zmm16
    vpsubd zmm11, zmm11, zmm16
    vpsubd zmm12, zmm12, zmm16
    vpsubd zmm13, zmm13, zmm16
    vpsubd zmm14, zmm14, zmm16
    vpsubd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lleaky_relu:
    // C[m, n] = (C[m, n] >= 0) ? C[m, n] : alpha * C[m, n]
    vpbroadcastd    zmm17, dword ptr [rdi + 8]    // alpha as i32 scale factor
    vpxorq          zmm16, zmm16, zmm16
    
    vpmulld         zmm18, zmm0, zmm17
    vpcmpgtd        k1,    zmm16, zmm0        // 1 where C < 0
    vpblendmd       zmm0{k1}, zmm0, zmm18
    
    vpmulld         zmm18, zmm1, zmm17
    vpcmpgtd        k1,    zmm16, zmm1        // 1 where C < 0
    vpblendmd       zmm1{k1}, zmm1, zmm18
    
    vpmulld         zmm18, zmm2, zmm17
    vpcmpgtd        k1,    zmm16, zmm2        // 1 where C < 0
    vpblendmd       zmm2{k1}, zmm2, zmm18
    
    vpmulld         zmm18, zmm3, zmm17
    vpcmpgtd        k1,    zmm16, zmm3        // 1 where C < 0
    vpblendmd       zmm3{k1}, zmm3, zmm18
    
    vpmulld         zmm18, zmm4, zmm17
    vpcmpgtd        k1,    zmm16, zmm4        // 1 where C < 0
    vpblendmd       zmm4{k1}, zmm4, zmm18
    
    vpmulld         zmm18, zmm5, zmm17
    vpcmpgtd        k1,    zmm16, zmm5        // 1 where C < 0
    vpblendmd       zmm5{k1}, zmm5, zmm18
    
    vpmulld         zmm18, zmm6, zmm17
    vpcmpgtd        k1,    zmm16, zmm6        // 1 where C < 0
    vpblendmd       zmm6{k1}, zmm6, zmm18
    
    vpmulld         zmm18, zmm7, zmm17
    vpcmpgtd        k1,    zmm16, zmm7        // 1 where C < 0
    vpblendmd       zmm7{k1}, zmm7, zmm18
    
    vpmulld         zmm18, zmm8, zmm17
    vpcmpgtd        k1,    zmm16, zmm8        // 1 where C < 0
    vpblendmd       zmm8{k1}, zmm8, zmm18
    
    vpmulld         zmm18, zmm9, zmm17
    vpcmpgtd        k1,    zmm16, zmm9        // 1 where C < 0
    vpblendmd       zmm9{k1}, zmm9, zmm18
    
    vpmulld         zmm18, zmm10, zmm17
    vpcmpgtd        k1,    zmm16, zmm10        // 1 where C < 0
    vpblendmd       zmm10{k1}, zmm10, zmm18
    
    vpmulld         zmm18, zmm11, zmm17
    vpcmpgtd        k1,    zmm16, zmm11        // 1 where C < 0
    vpblendmd       zmm11{k1}, zmm11, zmm18
    
    vpmulld         zmm18, zmm12, zmm17
    vpcmpgtd        k1,    zmm16, zmm12        // 1 where C < 0
    vpblendmd       zmm12{k1}, zmm12, zmm18
    
    vpmulld         zmm18, zmm13, zmm17
    vpcmpgtd        k1,    zmm16, zmm13        // 1 where C < 0
    vpblendmd       zmm13{k1}, zmm13, zmm18
    
    vpmulld         zmm18, zmm14, zmm17
    vpcmpgtd        k1,    zmm16, zmm14        // 1 where C < 0
    vpblendmd       zmm14{k1}, zmm14, zmm18
    
    vpmulld         zmm18, zmm15, zmm17
    vpcmpgtd        k1,    zmm16, zmm15        // 1 where C < 0
    vpblendmd       zmm15{k1}, zmm15, zmm18
    
    jmp     .Lnon_linear_loop

.Lper_row_min:
    mov rax, [rdi + 8]
    vpbroadcastd    zmm16, dword ptr [rax + 0]
    vpminsd         zmm0, zmm0, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 4]
    vpminsd         zmm1, zmm1, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 8]
    vpminsd         zmm2, zmm2, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 12]
    vpminsd         zmm3, zmm3, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 16]
    vpminsd         zmm4, zmm4, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 20]
    vpminsd         zmm5, zmm5, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 24]
    vpminsd         zmm6, zmm6, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 28]
    vpminsd         zmm7, zmm7, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 32]
    vpminsd         zmm8, zmm8, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 36]
    vpminsd         zmm9, zmm9, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 40]
    vpminsd         zmm10, zmm10, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 44]
    vpminsd         zmm11, zmm11, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 48]
    vpminsd         zmm12, zmm12, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 52]
    vpminsd         zmm13, zmm13, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 56]
    vpminsd         zmm14, zmm14, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 60]
    vpminsd         zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_max:
    mov rax, [rdi + 8]
    vpbroadcastd    zmm16, dword ptr [rax + 0]
    vpmaxsd         zmm0, zmm0, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 4]
    vpmaxsd         zmm1, zmm1, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 8]
    vpmaxsd         zmm2, zmm2, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 12]
    vpmaxsd         zmm3, zmm3, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 16]
    vpmaxsd         zmm4, zmm4, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 20]
    vpmaxsd         zmm5, zmm5, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 24]
    vpmaxsd         zmm6, zmm6, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 28]
    vpmaxsd         zmm7, zmm7, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 32]
    vpmaxsd         zmm8, zmm8, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 36]
    vpmaxsd         zmm9, zmm9, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 40]
    vpmaxsd         zmm10, zmm10, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 44]
    vpmaxsd         zmm11, zmm11, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 48]
    vpmaxsd         zmm12, zmm12, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 52]
    vpmaxsd         zmm13, zmm13, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 56]
    vpmaxsd         zmm14, zmm14, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 60]
    vpmaxsd         zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_add:
    mov rax, [rdi + 8]
    vpbroadcastd    zmm16, dword ptr [rax + 0]
    vpaddd          zmm0, zmm0, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 4]
    vpaddd          zmm1, zmm1, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 8]
    vpaddd          zmm2, zmm2, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 12]
    vpaddd          zmm3, zmm3, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 16]
    vpaddd          zmm4, zmm4, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 20]
    vpaddd          zmm5, zmm5, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 24]
    vpaddd          zmm6, zmm6, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 28]
    vpaddd          zmm7, zmm7, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 32]
    vpaddd          zmm8, zmm8, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 36]
    vpaddd          zmm9, zmm9, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 40]
    vpaddd          zmm10, zmm10, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 44]
    vpaddd          zmm11, zmm11, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 48]
    vpaddd          zmm12, zmm12, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 52]
    vpaddd          zmm13, zmm13, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 56]
    vpaddd          zmm14, zmm14, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 60]
    vpaddd          zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_mul:
    mov rax, [rdi + 8]
    vpbroadcastd    zmm16, dword ptr [rax + 0]
    vpmulld         zmm0, zmm0, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 4]
    vpmulld         zmm1, zmm1, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 8]
    vpmulld         zmm2, zmm2, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 12]
    vpmulld         zmm3, zmm3, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 16]
    vpmulld         zmm4, zmm4, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 20]
    vpmulld         zmm5, zmm5, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 24]
    vpmulld         zmm6, zmm6, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 28]
    vpmulld         zmm7, zmm7, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 32]
    vpmulld         zmm8, zmm8, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 36]
    vpmulld         zmm9, zmm9, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 40]
    vpmulld         zmm10, zmm10, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 44]
    vpmulld         zmm11, zmm11, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 48]
    vpmulld         zmm12, zmm12, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 52]
    vpmulld         zmm13, zmm13, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 56]
    vpmulld         zmm14, zmm14, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 60]
    vpmulld         zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_row_sub:
    // non-flipped sub = operand - acc
    mov rax, [rdi + 8]
    vpbroadcastd    zmm16, dword ptr [rax + 0]
    vpsubd          zmm0, zmm16, zmm0
    vpbroadcastd    zmm16, dword ptr [rax + 4]
    vpsubd          zmm1, zmm16, zmm1
    vpbroadcastd    zmm16, dword ptr [rax + 8]
    vpsubd          zmm2, zmm16, zmm2
    vpbroadcastd    zmm16, dword ptr [rax + 12]
    vpsubd          zmm3, zmm16, zmm3
    vpbroadcastd    zmm16, dword ptr [rax + 16]
    vpsubd          zmm4, zmm16, zmm4
    vpbroadcastd    zmm16, dword ptr [rax + 20]
    vpsubd          zmm5, zmm16, zmm5
    vpbroadcastd    zmm16, dword ptr [rax + 24]
    vpsubd          zmm6, zmm16, zmm6
    vpbroadcastd    zmm16, dword ptr [rax + 28]
    vpsubd          zmm7, zmm16, zmm7
    vpbroadcastd    zmm16, dword ptr [rax + 32]
    vpsubd          zmm8, zmm16, zmm8
    vpbroadcastd    zmm16, dword ptr [rax + 36]
    vpsubd          zmm9, zmm16, zmm9
    vpbroadcastd    zmm16, dword ptr [rax + 40]
    vpsubd          zmm10, zmm16, zmm10
    vpbroadcastd    zmm16, dword ptr [rax + 44]
    vpsubd          zmm11, zmm16, zmm11
    vpbroadcastd    zmm16, dword ptr [rax + 48]
    vpsubd          zmm12, zmm16, zmm12
    vpbroadcastd    zmm16, dword ptr [rax + 52]
    vpsubd          zmm13, zmm16, zmm13
    vpbroadcastd    zmm16, dword ptr [rax + 56]
    vpsubd          zmm14, zmm16, zmm14
    vpbroadcastd    zmm16, dword ptr [rax + 60]
    vpsubd          zmm15, zmm16, zmm15
    
    jmp     .Lnon_linear_loop

.Lper_row_sub_flipped:
    // flipped sub = acc - operand
    mov rax, [rdi + 8]
    vpbroadcastd    zmm16, dword ptr [rax + 0]
    vpsubd          zmm0, zmm0, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 4]
    vpsubd          zmm1, zmm1, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 8]
    vpsubd          zmm2, zmm2, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 12]
    vpsubd          zmm3, zmm3, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 16]
    vpsubd          zmm4, zmm4, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 20]
    vpsubd          zmm5, zmm5, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 24]
    vpsubd          zmm6, zmm6, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 28]
    vpsubd          zmm7, zmm7, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 32]
    vpsubd          zmm8, zmm8, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 36]
    vpsubd          zmm9, zmm9, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 40]
    vpsubd          zmm10, zmm10, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 44]
    vpsubd          zmm11, zmm11, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 48]
    vpsubd          zmm12, zmm12, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 52]
    vpsubd          zmm13, zmm13, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 56]
    vpsubd          zmm14, zmm14, zmm16
    vpbroadcastd    zmm16, dword ptr [rax + 60]
    vpsubd          zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_min:
    mov     rax, [rdi + 8]
    vmovdqu32       zmm16, [rax]
    vpminsd zmm0, zmm0, zmm16
    vpminsd zmm1, zmm1, zmm16
    vpminsd zmm2, zmm2, zmm16
    vpminsd zmm3, zmm3, zmm16
    vpminsd zmm4, zmm4, zmm16
    vpminsd zmm5, zmm5, zmm16
    vpminsd zmm6, zmm6, zmm16
    vpminsd zmm7, zmm7, zmm16
    vpminsd zmm8, zmm8, zmm16
    vpminsd zmm9, zmm9, zmm16
    vpminsd zmm10, zmm10, zmm16
    vpminsd zmm11, zmm11, zmm16
    vpminsd zmm12, zmm12, zmm16
    vpminsd zmm13, zmm13, zmm16
    vpminsd zmm14, zmm14, zmm16
    vpminsd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_max:
    mov     rax, [rdi + 8]
    vmovdqu32       zmm16, [rax]
    vpmaxsd zmm0, zmm0, zmm16
    vpmaxsd zmm1, zmm1, zmm16
    vpmaxsd zmm2, zmm2, zmm16
    vpmaxsd zmm3, zmm3, zmm16
    vpmaxsd zmm4, zmm4, zmm16
    vpmaxsd zmm5, zmm5, zmm16
    vpmaxsd zmm6, zmm6, zmm16
    vpmaxsd zmm7, zmm7, zmm16
    vpmaxsd zmm8, zmm8, zmm16
    vpmaxsd zmm9, zmm9, zmm16
    vpmaxsd zmm10, zmm10, zmm16
    vpmaxsd zmm11, zmm11, zmm16
    vpmaxsd zmm12, zmm12, zmm16
    vpmaxsd zmm13, zmm13, zmm16
    vpmaxsd zmm14, zmm14, zmm16
    vpmaxsd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_add:
    mov     rax, [rdi + 8]
    vmovdqu32       zmm16, [rax]
    vpaddd zmm0, zmm0, zmm16
    vpaddd zmm1, zmm1, zmm16
    vpaddd zmm2, zmm2, zmm16
    vpaddd zmm3, zmm3, zmm16
    vpaddd zmm4, zmm4, zmm16
    vpaddd zmm5, zmm5, zmm16
    vpaddd zmm6, zmm6, zmm16
    vpaddd zmm7, zmm7, zmm16
    vpaddd zmm8, zmm8, zmm16
    vpaddd zmm9, zmm9, zmm16
    vpaddd zmm10, zmm10, zmm16
    vpaddd zmm11, zmm11, zmm16
    vpaddd zmm12, zmm12, zmm16
    vpaddd zmm13, zmm13, zmm16
    vpaddd zmm14, zmm14, zmm16
    vpaddd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_mul:
    mov     rax, [rdi + 8]
    vmovdqu32       zmm16, [rax]
    vpmulld zmm0, zmm0, zmm16
    vpmulld zmm1, zmm1, zmm16
    vpmulld zmm2, zmm2, zmm16
    vpmulld zmm3, zmm3, zmm16
    vpmulld zmm4, zmm4, zmm16
    vpmulld zmm5, zmm5, zmm16
    vpmulld zmm6, zmm6, zmm16
    vpmulld zmm7, zmm7, zmm16
    vpmulld zmm8, zmm8, zmm16
    vpmulld zmm9, zmm9, zmm16
    vpmulld zmm10, zmm10, zmm16
    vpmulld zmm11, zmm11, zmm16
    vpmulld zmm12, zmm12, zmm16
    vpmulld zmm13, zmm13, zmm16
    vpmulld zmm14, zmm14, zmm16
    vpmulld zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lper_col_sub:
    // non-flipped sub = operand - acc
    mov     rax, [rdi + 8]
    vmovdqu32       zmm16, [rax]
    vpsubd zmm0, zmm16, zmm0
    vpsubd zmm1, zmm16, zmm1
    vpsubd zmm2, zmm16, zmm2
    vpsubd zmm3, zmm16, zmm3
    vpsubd zmm4, zmm16, zmm4
    vpsubd zmm5, zmm16, zmm5
    vpsubd zmm6, zmm16, zmm6
    vpsubd zmm7, zmm16, zmm7
    vpsubd zmm8, zmm16, zmm8
    vpsubd zmm9, zmm16, zmm9
    vpsubd zmm10, zmm16, zmm10
    vpsubd zmm11, zmm16, zmm11
    vpsubd zmm12, zmm16, zmm12
    vpsubd zmm13, zmm16, zmm13
    vpsubd zmm14, zmm16, zmm14
    vpsubd zmm15, zmm16, zmm15
    
    jmp     .Lnon_linear_loop

.Lper_col_sub_flipped:
    // flipped sub = acc - operand
    mov     rax, [rdi + 8]
    vmovdqu32       zmm16, [rax]
    vpsubd zmm0, zmm0, zmm16
    vpsubd zmm1, zmm1, zmm16
    vpsubd zmm2, zmm2, zmm16
    vpsubd zmm3, zmm3, zmm16
    vpsubd zmm4, zmm4, zmm16
    vpsubd zmm5, zmm5, zmm16
    vpsubd zmm6, zmm6, zmm16
    vpsubd zmm7, zmm7, zmm16
    vpsubd zmm8, zmm8, zmm16
    vpsubd zmm9, zmm9, zmm16
    vpsubd zmm10, zmm10, zmm16
    vpsubd zmm11, zmm11, zmm16
    vpsubd zmm12, zmm12, zmm16
    vpsubd zmm13, zmm13, zmm16
    vpsubd zmm14, zmm14, zmm16
    vpsubd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lload_tile:
    // Scratch layout is COL-MAJOR i32 from scratch.rs Store/AddUnicast remnant:
    //   tile[col][row] at offset (col*MR + row)*4 with MR=16
    //   = offset col*64 + row*4
    // For row-major accumulators we gather row m's 16 cols at index step 64.
    mov         r8, [rdi + 8]
    vmovdqa32   zmm16, [rip + .Llane_offsets_64]    // [0, 64, 128, ..., 15*64]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm0{k1}, [r8 + zmm16 + 0]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm1{k1}, [r8 + zmm16 + 4]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm2{k1}, [r8 + zmm16 + 8]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm3{k1}, [r8 + zmm16 + 12]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm4{k1}, [r8 + zmm16 + 16]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm5{k1}, [r8 + zmm16 + 20]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm6{k1}, [r8 + zmm16 + 24]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm7{k1}, [r8 + zmm16 + 28]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm8{k1}, [r8 + zmm16 + 32]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm9{k1}, [r8 + zmm16 + 36]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm10{k1}, [r8 + zmm16 + 40]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm11{k1}, [r8 + zmm16 + 44]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm12{k1}, [r8 + zmm16 + 48]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm13{k1}, [r8 + zmm16 + 52]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm14{k1}, [r8 + zmm16 + 56]
    
    mov         eax, 0xFFFF
    kmovw       k1, eax
    vpgatherdd  zmm15{k1}, [r8 + zmm16 + 60]
    
    jmp     .Lnon_linear_loop

.Ladd_unicast:
    mov     r10,    [rdi + 8]           // c ptr (base)
    mov     rsi,    [rdi + 16]          // row stride
    mov     rbx,    [rdi + 24]          // col stride
    mov     r8,     [rdi + 32]          // item size

    cmp     r8,    4
    je      .Lnon_linear_addc_i32

    // i8 path: read 16 i8 from [r10 + m*rsi + n*rbx] for n=0..15, sign-extend
    // to i32, add to zmm{m}. Use a stack scratch buffer (16 bytes per row).
    sub     rsp, 16
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm0, zmm0, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm1, zmm1, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm2, zmm2, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm3, zmm3, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm4, zmm4, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm5, zmm5, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm6, zmm6, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm7, zmm7, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm8, zmm8, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm9, zmm9, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm10, zmm10, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm11, zmm11, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm12, zmm12, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm13, zmm13, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm14, zmm14, zmm16
    add     r10, rsi
    
    mov     r8, r10
    
    mov     al, [r8]
    mov     byte ptr [rsp + 0], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 1], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 2], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 3], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 4], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 5], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 6], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 7], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 8], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 9], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 10], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 11], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 12], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 13], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 14], al
    add     r8, rbx
    
    mov     al, [r8]
    mov     byte ptr [rsp + 15], al
    add     r8, rbx
    
    vpmovsxbd       zmm16, [rsp]
    vpaddd          zmm15, zmm15, zmm16
    add     r10, rsi
    
    add     rsp, 16
    jmp     .Lnon_linear_loop

.Lnon_linear_addc_i32:
    // i32 strided read of external (or scratch) tile. Build per-lane index
    // vector [0, rbx, 2*rbx, ..., 15*rbx] once, then gather row by row.
    mov             eax, ebx
    vmovd xmm16, eax
    vpbroadcastd    zmm16, xmm16
    vpmulld         zmm16, zmm16, [rip + .Llane_indices]    // [0, rbx, 2*rbx, ...]

    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm0, zmm0, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm1, zmm1, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm2, zmm2, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm3, zmm3, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm4, zmm4, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm5, zmm5, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm6, zmm6, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm7, zmm7, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm8, zmm8, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm9, zmm9, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm10, zmm10, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm11, zmm11, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm12, zmm12, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm13, zmm13, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm14, zmm14, zmm17
    add             r10, rsi
    
    mov             eax, 0xFFFF
    kmovw           k1, eax
    vpgatherdd      zmm17{k1}, [r10 + zmm16]
    vpaddd          zmm15, zmm15, zmm17
    add             r10, rsi
    
    jmp    .Lnon_linear_loop

.Ladd_row_col_products:
    // bias[m, n] = row_data[m] * col_data[n], add to C[m, n].
    // For row-major regs: load 16 col_data values once into zmm16,
    //                     for each m: broadcast row_data[m], FMA add.
    mov             rax, [rdi + 8]
    mov             rbx, [rdi + 16]

    vmovdqu32       zmm16, [rax]                          // 16 row_data values
    vmovdqu32       zmm17, [rbx]                          // 16 col_data values

    
    vpbroadcastd    zmm18, dword ptr [rax + 0]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm0, zmm0, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 4]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm1, zmm1, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 8]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm2, zmm2, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 12]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm3, zmm3, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 16]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm4, zmm4, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 20]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm5, zmm5, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 24]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm6, zmm6, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 28]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm7, zmm7, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 32]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm8, zmm8, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 36]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm9, zmm9, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 40]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm10, zmm10, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 44]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm11, zmm11, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 48]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm12, zmm12, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 52]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm13, zmm13, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 56]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm14, zmm14, zmm19
    
    vpbroadcastd    zmm18, dword ptr [rax + 60]    // splat row_data[m]
    vpmulld         zmm19, zmm18, zmm17
    vpaddd          zmm15, zmm15, zmm19
    
    jmp     .Lnon_linear_loop

// ---- Q-scale (mult-shift with rounding) ---------------------------------

.Lq_scale:
    mov             r8, [rdi + 16]                  // policy
    vpbroadcastd    zmm16, dword ptr [rdi + 24]     // multi (broadcast i32)

    mov             rax, 1
    vmovq xmm17, rax
    vpbroadcastq    zmm17, xmm17                    // zmm17 <- 1 (i64 lanes)

    mov             rax, [rdi + 8]                  // shift
    add             rax, 31
    vmovq xmm18, rax
    vpbroadcastq    zmm18, xmm18                    // zmm18 <- (shift+31) (i64 lanes)

    vpsubq          zmm19, zmm18, zmm17
    vpsllvq         zmm19, zmm17, zmm19             // zmm19 <- 1 << (shift+31-1)  (i64)

    // Per-lane interleave mask for blending evens / shifted-odds.
    // bit i = 1 means take from "evens" source in vpblendmd; bit 0,2,4,...,14 set.
    mov             eax, 0x5555
    kmovw           k7, eax

    cmp     r8, 1
    je      .Lq_scale_rounding_zero
    cmp     r8, 2
    je      .Lq_scale_rounding_away
    cmp     r8, 3
    je      .Lq_scale_rounding_minus_inf
    cmp     r8, 4
    je      .Lq_scale_rounding_plus_inf
    cmp     r8, 5
    je      .Lq_scale_rounding_even
    cmp     r8, 6
    je      .Lq_scale_rounding_odd

    jmp     .Lunsupported

.Lq_scale_rounding_zero:           // signum * ( (abs + nudge - 1) >> shift )

    vpabsd      zmm20, zmm0
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16                 // even-lane i32 -> i64 mul
    vpmuldq     zmm21, zmm21, zmm16                 // odd-lane  i32 -> i64 mul

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsubq      zmm20, zmm20, zmm17
    vpsubq      zmm21, zmm21, zmm17

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20             // k7=0x5555: evens from zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_scale_rounding_away:           // signum * ( (abs + nudge) >> shift )

    vpabsd      zmm20, zmm0
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_scale_rounding_minus_inf:      // nudge by -1 where input was negative

    vpabsd      zmm20, zmm0
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm0, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm0, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm1, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm1, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm2, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm2, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm3, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm3, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm4, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm4, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm5, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm5, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm6, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm6, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm7, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm7, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm8, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm8, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm9, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm9, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm10, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm10, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm11, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm11, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm12, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm12, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm13, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm13, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm14, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm14, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpxorq      zmm22, zmm22, zmm22
    vpcmpgtd    k1,    zmm15, zmm22              // k1: 1 where input > 0 (we want the inverse, see below)
    knotw       k1,    k1                           // 1 where input <= 0  -- we want "input was negative => subtract 1"
    // For "<0": use compare against 0 with vpcmpltd
    vpxorq      zmm22, zmm22, zmm22
    vpcmpltd    k1,    zmm15, zmm22              // 1 where input < 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32] // (1 << 0) per neg lane, 0 elsewhere

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    // Subtract 1 from i64-evens / i64-odds where the original i32 input was < 0.
    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_scale_rounding_plus_inf:       // nudge by +1 where input was non-negative

    vpabsd      zmm20, zmm0
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm0              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm1              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm2              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm3              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm4              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm5              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm6              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm7              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm8              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm9              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm10              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm11              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm12              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm13              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm14              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpxorq      zmm22, zmm22, zmm22
    vpcmpled    k1,    zmm22, zmm15              // 1 where input >= 0
    vmovdqa32   zmm23{k1}{z}, [rip + .Lall_ones_i32]

    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrldq     zmm24, zmm23, 4
    vpmovsxdq   zmm25, ymm23
    vpmovsxdq   zmm26, ymm24
    vpsubq      zmm20, zmm20, zmm25
    vpsubq      zmm21, zmm21, zmm26

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_scale_rounding_even:           // banker's: round half to even

    vpabsd      zmm20, zmm0
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm20, zmm20, zmm22
    vpsubq      zmm20, zmm20, zmm17

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpaddq      zmm21, zmm21, zmm22
    vpsubq      zmm21, zmm21, zmm17

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_scale_rounding_odd:            // round half to odd

    vpabsd      zmm20, zmm0
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpsrldq     zmm21, zmm20, 4
    vpmuldq     zmm20, zmm20, zmm16
    vpmuldq     zmm21, zmm21, zmm16

    vpsrlq      zmm22, zmm20, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm20, zmm20, zmm22

    vpsrlq      zmm22, zmm21, xmm18
    vpandq      zmm22, zmm22, zmm17
    vpsubq      zmm21, zmm21, zmm22

    vpaddq      zmm20, zmm20, zmm19
    vpaddq      zmm21, zmm21, zmm19

    vpsrlq      zmm20, zmm20, xmm18
    vpsrlq      zmm21, zmm21, xmm18

    vpslldq     zmm21, zmm21, 4
    vpblendmd   zmm20{k7}, zmm21, zmm20
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_shl:
    mov             eax, [rdi + 8]                  // -shift (count: i32)
    vmovd xmm16, eax
    vpbroadcastd    zmm16, xmm16
    vpsllvd zmm0, zmm0, zmm16
    vpsllvd zmm1, zmm1, zmm16
    vpsllvd zmm2, zmm2, zmm16
    vpsllvd zmm3, zmm3, zmm16
    vpsllvd zmm4, zmm4, zmm16
    vpsllvd zmm5, zmm5, zmm16
    vpsllvd zmm6, zmm6, zmm16
    vpsllvd zmm7, zmm7, zmm16
    vpsllvd zmm8, zmm8, zmm16
    vpsllvd zmm9, zmm9, zmm16
    vpsllvd zmm10, zmm10, zmm16
    vpsllvd zmm11, zmm11, zmm16
    vpsllvd zmm12, zmm12, zmm16
    vpsllvd zmm13, zmm13, zmm16
    vpsllvd zmm14, zmm14, zmm16
    vpsllvd zmm15, zmm15, zmm16
    
    jmp     .Lnon_linear_loop

.Lq_shr:
    mov             r8,  [rdi + 16]                 // policy

    mov             eax, 1
    vmovd xmm16, eax
    vpbroadcastd    zmm16, xmm16                    // zmm16 <- 1 (i32 lanes)

    mov             eax, [rdi + 8]                  // shift
    vmovd xmm17, eax
    vpbroadcastd    zmm17, xmm17                    // zmm17 <- shift (i32 lanes)

    mov             ebx, 1
    mov             cl,  al
    sub             cl,  1
    sal             ebx, cl                         // ebx <- 1 << (shift - 1)
    vmovd xmm18, ebx
    vpbroadcastd    zmm18, xmm18                    // zmm18 <- "half"

    vpxorq          zmm19, zmm19, zmm19             // zeroes

    cmp     r8, 1
    je      .Lq_shr_rounding_zero
    cmp     r8, 2
    je      .Lq_shr_rounding_away
    cmp     r8, 3
    je      .Lq_shr_rounding_minus_inf
    cmp     r8, 4
    je      .Lq_shr_rounding_plus_inf
    cmp     r8, 5
    je      .Lq_shr_rounding_even
    cmp     r8, 6
    je      .Lq_shr_rounding_odd

    jmp     .Lunsupported

.Lq_shr_rounding_zero:

    vpabsd      zmm20, zmm0
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpsubd      zmm20, zmm20, zmm16
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_shr_rounding_away:

    vpabsd      zmm20, zmm0
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_shr_rounding_minus_inf:

    vpsubd      zmm0, zmm0, zmm16
    vpaddd      zmm0, zmm0, zmm18
    vpsravd     zmm0, zmm0, zmm17

    vpsubd      zmm1, zmm1, zmm16
    vpaddd      zmm1, zmm1, zmm18
    vpsravd     zmm1, zmm1, zmm17

    vpsubd      zmm2, zmm2, zmm16
    vpaddd      zmm2, zmm2, zmm18
    vpsravd     zmm2, zmm2, zmm17

    vpsubd      zmm3, zmm3, zmm16
    vpaddd      zmm3, zmm3, zmm18
    vpsravd     zmm3, zmm3, zmm17

    vpsubd      zmm4, zmm4, zmm16
    vpaddd      zmm4, zmm4, zmm18
    vpsravd     zmm4, zmm4, zmm17

    vpsubd      zmm5, zmm5, zmm16
    vpaddd      zmm5, zmm5, zmm18
    vpsravd     zmm5, zmm5, zmm17

    vpsubd      zmm6, zmm6, zmm16
    vpaddd      zmm6, zmm6, zmm18
    vpsravd     zmm6, zmm6, zmm17

    vpsubd      zmm7, zmm7, zmm16
    vpaddd      zmm7, zmm7, zmm18
    vpsravd     zmm7, zmm7, zmm17

    vpsubd      zmm8, zmm8, zmm16
    vpaddd      zmm8, zmm8, zmm18
    vpsravd     zmm8, zmm8, zmm17

    vpsubd      zmm9, zmm9, zmm16
    vpaddd      zmm9, zmm9, zmm18
    vpsravd     zmm9, zmm9, zmm17

    vpsubd      zmm10, zmm10, zmm16
    vpaddd      zmm10, zmm10, zmm18
    vpsravd     zmm10, zmm10, zmm17

    vpsubd      zmm11, zmm11, zmm16
    vpaddd      zmm11, zmm11, zmm18
    vpsravd     zmm11, zmm11, zmm17

    vpsubd      zmm12, zmm12, zmm16
    vpaddd      zmm12, zmm12, zmm18
    vpsravd     zmm12, zmm12, zmm17

    vpsubd      zmm13, zmm13, zmm16
    vpaddd      zmm13, zmm13, zmm18
    vpsravd     zmm13, zmm13, zmm17

    vpsubd      zmm14, zmm14, zmm16
    vpaddd      zmm14, zmm14, zmm18
    vpsravd     zmm14, zmm14, zmm17

    vpsubd      zmm15, zmm15, zmm16
    vpaddd      zmm15, zmm15, zmm18
    vpsravd     zmm15, zmm15, zmm17

    jmp     .Lnon_linear_loop

.Lq_shr_rounding_plus_inf:

    vpaddd      zmm0, zmm0, zmm18
    vpsravd     zmm0, zmm0, zmm17

    vpaddd      zmm1, zmm1, zmm18
    vpsravd     zmm1, zmm1, zmm17

    vpaddd      zmm2, zmm2, zmm18
    vpsravd     zmm2, zmm2, zmm17

    vpaddd      zmm3, zmm3, zmm18
    vpsravd     zmm3, zmm3, zmm17

    vpaddd      zmm4, zmm4, zmm18
    vpsravd     zmm4, zmm4, zmm17

    vpaddd      zmm5, zmm5, zmm18
    vpsravd     zmm5, zmm5, zmm17

    vpaddd      zmm6, zmm6, zmm18
    vpsravd     zmm6, zmm6, zmm17

    vpaddd      zmm7, zmm7, zmm18
    vpsravd     zmm7, zmm7, zmm17

    vpaddd      zmm8, zmm8, zmm18
    vpsravd     zmm8, zmm8, zmm17

    vpaddd      zmm9, zmm9, zmm18
    vpsravd     zmm9, zmm9, zmm17

    vpaddd      zmm10, zmm10, zmm18
    vpsravd     zmm10, zmm10, zmm17

    vpaddd      zmm11, zmm11, zmm18
    vpsravd     zmm11, zmm11, zmm17

    vpaddd      zmm12, zmm12, zmm18
    vpsravd     zmm12, zmm12, zmm17

    vpaddd      zmm13, zmm13, zmm18
    vpsravd     zmm13, zmm13, zmm17

    vpaddd      zmm14, zmm14, zmm18
    vpsravd     zmm14, zmm14, zmm17

    vpaddd      zmm15, zmm15, zmm18
    vpsravd     zmm15, zmm15, zmm17

    jmp     .Lnon_linear_loop

.Lq_shr_rounding_even:

    vpabsd      zmm20, zmm0
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm21, zmm16             // nudge = ((abs >>l shift) & 1) - 1
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

.Lq_shr_rounding_odd:

    vpabsd      zmm20, zmm0
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm0                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm0{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm1
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm1                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm1{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm2
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm2                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm2{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm3
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm3                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm3{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm4
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm4                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm4{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm5
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm5                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm5{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm6
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm6                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm6{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm7
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm7                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm7{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm8
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm8                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm8{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm9
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm9                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm9{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm10
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm10                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm10{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm11
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm11                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm11{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm12
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm12                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm12{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm13
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm13                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm13{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm14
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm14                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm14{k1}, zmm20, zmm27

    vpabsd      zmm20, zmm15
    vpsravd     zmm21, zmm20, zmm17
    vpandq      zmm21, zmm21, zmm16
    vpsubd      zmm21, zmm19, zmm21             // nudge = -((abs >>l shift) & 1)
    vpaddd      zmm20, zmm20, zmm21
    vpaddd      zmm20, zmm20, zmm18
    vpsravd     zmm20, zmm20, zmm17
    // emulate AVX2 vpsignd (no AVX-512 form): apply sign of original acc.
    vpxorq      zmm26, zmm26, zmm26
    vpcmpgtd    k1, zmm26, zmm15                 // k1 = 1 where acc < 0
    vpsubd      zmm27, zmm26, zmm20                 // zmm27 = -zmm20
    vpblendmd   zmm15{k1}, zmm20, zmm27

    jmp     .Lnon_linear_loop

// ---- Store ---------------------------------------------------------------

.Lstore:
    mov     r8,     [rdi + 8]               // c ptr
    mov     rsi,    [rdi + 16]              // row stride
    mov     rdx,    [rdi + 24]              // col stride
    mov     rcx,    [rdi + 32]              // item size

    cmp     rcx, 4
    je      .Lstore_strides_i32
    // else: i8 fallthrough

    cmp     rdx, 1
    je      .Lstore_strides_i8_row_contig

    // Generic i8 strided store: per row, per lane scalar byte stores
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm0
    vextracti32x4   xmm20, zmm0, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm1
    vextracti32x4   xmm20, zmm1, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm2
    vextracti32x4   xmm20, zmm2, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm3
    vextracti32x4   xmm20, zmm3, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm4
    vextracti32x4   xmm20, zmm4, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm5
    vextracti32x4   xmm20, zmm5, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm6
    vextracti32x4   xmm20, zmm6, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm7
    vextracti32x4   xmm20, zmm7, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm8
    vextracti32x4   xmm20, zmm8, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm9
    vextracti32x4   xmm20, zmm9, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm10
    vextracti32x4   xmm20, zmm10, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm11
    vextracti32x4   xmm20, zmm11, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm12
    vextracti32x4   xmm20, zmm12, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm13
    vextracti32x4   xmm20, zmm13, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm14
    vextracti32x4   xmm20, zmm14, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    // Extract from each 128-bit slice of zmm15
    vextracti32x4   xmm20, zmm15, 0
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 1
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 2
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 3
    
    vpextrd     ebx, xmm20, 0
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         byte ptr [r10], bl
    add         r10, rdx
    
    add     r8, rsi
    
    jmp     .Lnon_linear_loop

.Lstore_strides_i8_row_contig:
    // Each row is 16 i8 contiguous; one vpmovdb per row.
    
    vpmovdb     [r8], zmm0
    add         r8, rsi
    
    vpmovdb     [r8], zmm1
    add         r8, rsi
    
    vpmovdb     [r8], zmm2
    add         r8, rsi
    
    vpmovdb     [r8], zmm3
    add         r8, rsi
    
    vpmovdb     [r8], zmm4
    add         r8, rsi
    
    vpmovdb     [r8], zmm5
    add         r8, rsi
    
    vpmovdb     [r8], zmm6
    add         r8, rsi
    
    vpmovdb     [r8], zmm7
    add         r8, rsi
    
    vpmovdb     [r8], zmm8
    add         r8, rsi
    
    vpmovdb     [r8], zmm9
    add         r8, rsi
    
    vpmovdb     [r8], zmm10
    add         r8, rsi
    
    vpmovdb     [r8], zmm11
    add         r8, rsi
    
    vpmovdb     [r8], zmm12
    add         r8, rsi
    
    vpmovdb     [r8], zmm13
    add         r8, rsi
    
    vpmovdb     [r8], zmm14
    add         r8, rsi
    
    vpmovdb     [r8], zmm15
    add         r8, rsi
    
    jmp     .Lnon_linear_loop

.Lstore_strides_i32:
    cmp     rdx, 4
    je      .Lstore_strides_i32_row_contig

    // Generic i32 strided store
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm0, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm0, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm1, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm1, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm2, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm2, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm3, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm3, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm4, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm4, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm5, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm5, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm6, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm6, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm7, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm7, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm8, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm8, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm9, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm9, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm10, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm10, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm11, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm11, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm12, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm12, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm13, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm13, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm14, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm14, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    mov     r10, r8
    vextracti32x4   xmm20, zmm15, 0
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 1
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 2
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vextracti32x4   xmm20, zmm15, 3
    
    vpextrd     ebx, xmm20, 0
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 1
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 2
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    vpextrd     ebx, xmm20, 3
    mov         dword ptr [r10], ebx
    add         r10, rdx
    
    add     r8, rsi
    
    jmp     .Lnon_linear_loop

.Lstore_strides_i32_row_contig:
    // C is row-major in memory: each row's 16 i32 are contiguous; one
    // 64-byte aligned-or-unaligned store per row.
    
    vmovdqu32   [r8], zmm0
    add         r8, rsi
    
    vmovdqu32   [r8], zmm1
    add         r8, rsi
    
    vmovdqu32   [r8], zmm2
    add         r8, rsi
    
    vmovdqu32   [r8], zmm3
    add         r8, rsi
    
    vmovdqu32   [r8], zmm4
    add         r8, rsi
    
    vmovdqu32   [r8], zmm5
    add         r8, rsi
    
    vmovdqu32   [r8], zmm6
    add         r8, rsi
    
    vmovdqu32   [r8], zmm7
    add         r8, rsi
    
    vmovdqu32   [r8], zmm8
    add         r8, rsi
    
    vmovdqu32   [r8], zmm9
    add         r8, rsi
    
    vmovdqu32   [r8], zmm10
    add         r8, rsi
    
    vmovdqu32   [r8], zmm11
    add         r8, rsi
    
    vmovdqu32   [r8], zmm12
    add         r8, rsi
    
    vmovdqu32   [r8], zmm13
    add         r8, rsi
    
    vmovdqu32   [r8], zmm14
    add         r8, rsi
    
    vmovdqu32   [r8], zmm15
    add         r8, rsi
    
    jmp     .Lnon_linear_loop

.Lreturn:
    tilerelease
    add         rsp, 64

    ldmxcsr     [rsp + 4]
    add         rsp, 8

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx



    mov rsp, rbp
    pop rbp
    ret

// ---- Read-only data (RIP-relative) ---------------------------------------

.p2align 6
.Llane_offsets_64:
    .int  0,  64, 128, 192, 256, 320, 384, 448
    .int  512, 576, 640, 704, 768, 832, 896, 960

.p2align 6
.Llane_indices:
    .int  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

.p2align 6
.Lall_ones_i32:
    .int  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1


.cfi_endproc
